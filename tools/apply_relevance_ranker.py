#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CANONICAL_PROB_COLUMNS = [
    "prob_tokenizer_main",
    "prob_loco_aux",
    "prob_discard",
    "prob_diagnostic_candidate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained relevance ranker to take-level features.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keep-pre-ranker-probs",
        action="store_true",
        default=True,
        help="Preserve existing data-side suitability probabilities as pre_ranker_* columns before writing calibrated ranker probabilities.",
    )
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def canonical_probability_column(class_name: str) -> str | None:
    name = normalize_class_name(class_name)
    if name in {"tokenizer_main", "main", "fact_main", "a_interaction_rich", "interaction_rich"}:
        return "prob_tokenizer_main"
    if name in {"loco_aux", "b_loco_body", "loco_body"}:
        return "prob_loco_aux"
    if name in {"discard", "d_scene_only", "scene_only", "f_bad_or_unclear", "f_uncertain", "bad_or_unclear"}:
        return "prob_discard"
    if name in {"diagnostic_candidate", "c_active_view_only", "active_view_only", "e_fine_dexterous", "fine_dexterous"}:
        return "prob_diagnostic_candidate"
    return None


def model_classes(payload: dict, model) -> list[str]:
    if payload.get("classes"):
        return [str(value) for value in payload["classes"]]
    if hasattr(model, "classes_"):
        return [str(value) for value in model.classes_]
    if hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "classes_"):
        return [str(value) for value in model.named_steps["clf"].classes_]
    return []


def prepare_features(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    values = features.copy()
    for column in feature_columns:
        if column not in values.columns:
            values[column] = np.nan
    numeric = values[feature_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.ranker)
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    features = pd.read_csv(args.features)
    x = prepare_features(features, feature_columns)
    probabilities = model.predict_proba(x)
    classes = model_classes(payload, model)
    if not classes:
        raise ValueError("Could not determine ranker classes")
    output = features.copy()
    output["ranker_bucket"] = model.predict(x)
    for class_name, values in zip(classes, probabilities.T):
        output[f"ranker_prob_{normalize_class_name(class_name)}"] = values

    if args.keep_pre_ranker_probs:
        for column in CANONICAL_PROB_COLUMNS:
            if column in output.columns and f"pre_ranker_{column}" not in output.columns:
                output[f"pre_ranker_{column}"] = output[column]

    canonical_values = {column: pd.Series(0.0, index=output.index) for column in CANONICAL_PROB_COLUMNS}
    for class_name, values in zip(classes, probabilities.T):
        column = canonical_probability_column(class_name)
        if column is not None:
            canonical_values[column] = canonical_values[column] + values
    for column, values in canonical_values.items():
        if float(values.max()) > 0.0 or column in output.columns:
            output[column] = values.clip(0.0, 1.0)

    output["ranker_confidence"] = probabilities.max(axis=1)
    disagreements = []
    for column in CANONICAL_PROB_COLUMNS:
        pre = f"pre_ranker_{column}"
        if pre in output.columns and column in output.columns:
            disagreements.append((pd.to_numeric(output[column], errors="coerce") - pd.to_numeric(output[pre], errors="coerce")).abs())
    output["ranker_disagreement_score"] = pd.concat(disagreements, axis=1).max(axis=1).fillna(0.0) if disagreements else 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Saved ranked relevance rows to {args.out}")


if __name__ == "__main__":
    main()
