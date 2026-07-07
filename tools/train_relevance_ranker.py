#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight relevance ranker from manual calibration labels.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-column", default="take_relevance")
    parser.add_argument("--min-class-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    merged = features.merge(labels[["take_uid", args.label_column]], on="take_uid", how="inner")
    merged = merged[merged[args.label_column].notna() & (merged[args.label_column].astype(str).str.len() > 0)]
    class_counts = merged[args.label_column].value_counts()
    keep_classes = class_counts[class_counts >= args.min_class_count].index
    merged = merged[merged[args.label_column].isin(keep_classes)]
    if merged[args.label_column].nunique() < 2:
        raise ValueError("Need at least two labeled classes to train a ranker")
    x = merged[FEATURE_COLUMNS].astype(float)
    y = merged[args.label_column].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123, stratify=stratify)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=123, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    report = classification_report(y_test, model.predict(x_test), zero_division=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "label_column": args.label_column}, args.out)
    report_path = args.out.with_suffix(".report.txt")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved ranker to {args.out}")


if __name__ == "__main__":
    main()
