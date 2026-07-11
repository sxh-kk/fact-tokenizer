#!/usr/bin/env python3
"""Merge frozen feature shards while enforcing one checkpoint and unique sample IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = [json.loads((path / "feature_metadata.json").read_text(encoding="utf-8")) for path in args.input_dir]
    checkpoint_hashes = {row["encoder_checkpoint_sha256"] for row in metadata}
    feature_dims = {row["feature_dim"] for row in metadata}
    if len(checkpoint_hashes) != 1 or len(feature_dims) != 1:
        raise ValueError("feature shards use different encoder checkpoints or dimensions")
    arrays = {name: [] for name in ("features", "sample_id", "take_uid", "timestamp")}
    for path in args.input_dir:
        for name in arrays:
            arrays[name].append(np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False))
    merged = {name: np.concatenate(values) for name, values in arrays.items()}
    sample_ids = merged["sample_id"].astype(str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("feature shards contain overlapping sample IDs")
    order = np.argsort(sample_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in merged.items():
        np.save(args.output_dir / f"{name}.npy", value[order])
    report = {
        "schema": "fact-v7-effect-features-v1-merged",
        "encoder_frozen": True,
        "encoder_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "feature_dim": next(iter(feature_dims)),
        "samples": len(sample_ids),
        "input_shards": [str(path) for path in args.input_dir],
    }
    (args.output_dir / "feature_metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
