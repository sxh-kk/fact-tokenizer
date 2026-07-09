#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS = [
    "ego_motion_score",
    "exo_body_motion_score",
    "object_motion_proxy",
    "temporal_diversity_score",
    "metadata_interaction_prior",
    "metadata_loco_prior",
    "scene_only_score",
    "interaction_score",
    "loco_score",
    "fine_dexterous_score",
    "relevance_score",
    "ego_hand_score",
    "ego_hand_visibility_prob",
    "object_presence_score",
    "object_motion_score",
    "hand_object_contact_score",
    "interacting_object_score",
    "contact_state_change_score",
    "exo_body_visibility_score",
    "exo_pose_confidence",
    "exo_body_motion_score_v2",
    "body_phase_diversity_score",
    "loco_motion_score",
    "phase_diversity_score_v2",
    "motion_state_change_score",
    "pose_state_change_score",
    "vlm_prob_tokenizer_main",
    "vlm_prob_loco_aux",
    "vlm_prob_discard",
    "vlm_prob_diagnostic_candidate",
    "vlm_confidence",
    "auto_confidence",
    "auto_disagreement_score",
]

NEVER_FEATURE_COLUMNS = {
    "take_uid",
    "split",
    "parent_task_name",
    "task_name",
    "take_name",
    "contact_sheet_path",
    "feature_source_ego_hand_object",
    "feature_source_exo_pose_phase",
    "vlm_take_relevance",
    "vlm_usable_for",
    "vlm_reason",
    "auto_take_relevance",
    "auto_usable_for",
    "review_reasons",
    "needs_human_review",
    "take_relevance",
    "usable_for",
    "notes",
    "reason",
}

# The human-guided v1 ranker is data-side only. These patterns keep downstream
# tokenizer validation metrics from accidentally becoming filter training inputs.
TOKENIZER_METRIC_PATTERN = re.compile(
    r"(?:^|_)(?:loss|mse|probe|causality|gap|top1|ckpt|checkpoint|train_history|code_usage)(?:_|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight relevance ranker from manual calibration labels.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-column", default="usable_for")
    parser.add_argument(
        "--model",
        choices=["auto", "catboost", "lightgbm", "random_forest"],
        default="auto",
        help="Use CatBoost/LightGBM when installed, otherwise fall back to RandomForest.",
    )
    parser.add_argument(
        "--feature-columns",
        default="",
        help="Comma-separated feature column override. Defaults to numeric data-side columns.",
    )
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def nonempty_labels(labels: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if "take_uid" not in labels.columns:
        raise ValueError("Labels CSV must contain take_uid")
    if label_column not in labels.columns:
        raise ValueError(f"Labels CSV must contain label column {label_column!r}")
    values = labels[["take_uid", label_column]].copy()
    values[label_column] = values[label_column].astype(str).str.strip()
    values = values[values[label_column].notna() & (values[label_column] != "")]
    return values.drop_duplicates("take_uid", keep="last")


def numeric_feature_columns(features: pd.DataFrame, label_column: str, override: str) -> list[str]:
    if override:
        columns = [column.strip() for column in override.split(",") if column.strip()]
        missing = [column for column in columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing requested feature columns: {missing}")
        return columns

    columns: list[str] = []
    for column in DEFAULT_FEATURE_COLUMNS:
        if column in features.columns and column not in columns:
            columns.append(column)
    for column in features.columns:
        if column in columns or column in NEVER_FEATURE_COLUMNS or column == label_column:
            continue
        if TOKENIZER_METRIC_PATTERN.search(column):
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
            columns.append(column)
    if not columns:
        raise ValueError("No numeric data-side feature columns found")
    return columns


def frame_to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values


def build_model(kind: str, n_estimators: int, seed: int):
    requested = [kind] if kind != "auto" else ["catboost", "lightgbm", "random_forest"]
    errors: dict[str, str] = {}
    for candidate in requested:
        if candidate == "catboost":
            try:
                from catboost import CatBoostClassifier

                return (
                    "catboost",
                    CatBoostClassifier(
                        iterations=n_estimators,
                        depth=4,
                        learning_rate=0.05,
                        loss_function="MultiClass",
                        auto_class_weights="Balanced",
                        random_seed=seed,
                        verbose=False,
                    ),
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                errors[candidate] = str(exc)
        elif candidate == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                return (
                    "lightgbm",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            (
                                "clf",
                                LGBMClassifier(
                                    n_estimators=n_estimators,
                                    random_state=seed,
                                    class_weight="balanced",
                                    objective="multiclass",
                                ),
                            ),
                        ]
                    ),
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                errors[candidate] = str(exc)
        elif candidate == "random_forest":
            return (
                "random_forest",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=n_estimators,
                                min_samples_leaf=2,
                                random_state=seed,
                                class_weight="balanced",
                            ),
                        ),
                    ]
                ),
            )
    raise RuntimeError(f"Could not build requested ranker {kind!r}; dependency errors={errors}")


def model_classes(model) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(value) for value in model.classes_]
    if hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "classes_"):
        return [str(value) for value in model.named_steps["clf"].classes_]
    return []


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    labels = nonempty_labels(pd.read_csv(args.labels), args.label_column)
    merged = features.merge(labels, on="take_uid", how="inner")
    class_counts = merged[args.label_column].value_counts()
    keep_classes = class_counts[class_counts >= args.min_class_count].index
    merged = merged[merged[args.label_column].isin(keep_classes)]
    if merged[args.label_column].nunique() < 2:
        raise ValueError("Need at least two labeled classes to train a ranker")
    feature_columns = numeric_feature_columns(features, args.label_column, args.feature_columns)
    x = frame_to_numeric(merged, feature_columns)
    y = merged[args.label_column].astype(str)

    model_kind, model = build_model(args.model, args.n_estimators, args.seed)
    can_split = len(merged) >= 8 and y.value_counts().min() >= 2
    if can_split:
        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y
    model.fit(x_train, y_train)
    report = classification_report(y_test, model.predict(x_test), zero_division=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "model_kind": model_kind,
        "feature_columns": feature_columns,
        "label_column": args.label_column,
        "classes": model_classes(model),
        "data_side_only": True,
        "excluded_metric_pattern": TOKENIZER_METRIC_PATTERN.pattern,
    }
    joblib.dump(payload, args.out)
    report_path = args.out.with_suffix(".report.txt")
    report_path.write_text(report, encoding="utf-8")
    metadata_path = args.out.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_kind": model_kind,
                "label_column": args.label_column,
                "classes": payload["classes"],
                "feature_columns": feature_columns,
                "num_labeled_rows": int(len(merged)),
                "class_counts": {str(key): int(value) for key, value in class_counts.items()},
                "used_holdout_split": bool(can_split),
                "data_side_only": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"Saved {model_kind} ranker to {args.out}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
