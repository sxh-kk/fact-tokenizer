#!/usr/bin/env python3
"""Build a training transition-weight CSV from auto features and optional labels."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, to_float, write_csv


DEFAULT_LABEL_WEIGHTS = {
    "main_interaction": 1.0,
    "phase_context": 0.35,
    "loco_only": 0.10,
    "discard": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--label-column", default="transition_label")
    parser.add_argument("--auto-label-column", default="auto_label")
    parser.add_argument("--confidence-column", default="confidence")
    parser.add_argument("--take-weight-column", default="take_weight")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--unlabeled-policy", choices=["auto", "zero"], default="auto")
    parser.add_argument(
        "--label-weight",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT",
        help="Override default label weights.",
    )
    parser.add_argument("--normalize-max", type=float, default=1.0)
    return parser.parse_args()


def parse_label_weights(values: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_LABEL_WEIGHTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--label-weight must be LABEL=WEIGHT, got {value!r}")
        label, weight = value.split("=", 1)
        weights[label.strip()] = float(weight)
    return weights


def row_key(row: dict[str, str]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return f"sample_id:{sample_id}"
    return f"row_index:{str(row.get('row_index', '')).strip()}"


def load_annotations(path: Path | None, label_column: str) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    annotations: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        label = str(row.get(label_column, "")).strip()
        if not label:
            continue
        annotations[row_key(row)] = row
    return annotations


def main() -> None:
    args = parse_args()
    label_weights = parse_label_weights(args.label_weight)
    features = read_csv(args.features)
    annotations = load_annotations(args.annotations, args.label_column)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    matched_annotations = 0
    for feature in features:
        annotation = annotations.get(row_key(feature))
        source = "auto"
        confidence = 1.0
        if annotation is not None:
            label = str(annotation.get(args.label_column, "")).strip()
            confidence = to_float(annotation.get(args.confidence_column, ""), 1.0)
            if label and confidence >= args.min_confidence:
                source = "human"
                matched_annotations += 1
            else:
                label = ""
        else:
            label = ""
        if not label:
            label = str(feature.get(args.auto_label_column, "")).strip() if args.unlabeled_policy == "auto" else "discard"
        base_weight = float(label_weights.get(label, 0.0))
        take_weight = to_float(feature.get(args.take_weight_column, ""), 1.0)
        weight = max(0.0, base_weight * take_weight)
        rows.append(
            {
                "split": feature.get("split", ""),
                "row_index": feature.get("row_index", ""),
                "sample_id": feature.get("sample_id", ""),
                "take_uid": feature.get("take_uid", ""),
                "timestamp": feature.get("timestamp", ""),
                "parent_task_name": feature.get("parent_task_name", ""),
                "task_name": feature.get("task_name", ""),
                "auto_label": feature.get(args.auto_label_column, ""),
                "transition_label": label,
                "label_source": source,
                "confidence": f"{confidence:.3f}" if source == "human" else "",
                "interaction_score": feature.get("interaction_score", ""),
                "phase_change_score": feature.get("phase_change_score", ""),
                "ego_exo_sync_score": feature.get("ego_exo_sync_score", ""),
                "scene_only_score": feature.get("scene_only_score", ""),
                "take_weight": f"{take_weight:.6f}",
                "transition_weight": f"{weight:.9f}",
            }
        )
        counts[label] += 1
    if args.normalize_max > 0 and rows:
        max_weight = max(to_float(row["transition_weight"]) for row in rows)
        if max_weight > 0:
            scale = float(args.normalize_max) / max_weight
            for row in rows:
                row["transition_weight"] = f"{to_float(row['transition_weight']) * scale:.9f}"
    fieldnames = [
        "split",
        "row_index",
        "sample_id",
        "take_uid",
        "timestamp",
        "parent_task_name",
        "task_name",
        "auto_label",
        "transition_label",
        "label_source",
        "confidence",
        "interaction_score",
        "phase_change_score",
        "ego_exo_sync_score",
        "scene_only_score",
        "take_weight",
        "transition_weight",
    ]
    write_csv(args.out, rows, fieldnames)
    summary = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    positive = sum(1 for row in rows if to_float(row["transition_weight"]) > 0)
    print(
        f"Saved {len(rows)} transition weights to {args.out}; positive={positive}; "
        f"matched_human_annotations={matched_annotations}; labels: {summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
