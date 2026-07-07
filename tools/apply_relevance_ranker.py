#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained relevance ranker to take-level features.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.ranker)
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    features = pd.read_csv(args.features)
    probabilities = model.predict_proba(features[feature_columns].astype(float))
    classes = [str(value) for value in model.classes_]
    output = features.copy()
    output["ranker_bucket"] = model.predict(features[feature_columns].astype(float))
    for class_name, values in zip(classes, probabilities.T):
        output[f"prob_{normalize_class_name(class_name)}"] = values
    output["ranker_confidence"] = probabilities.max(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Saved ranked relevance rows to {args.out}")


if __name__ == "__main__":
    main()
