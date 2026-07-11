#!/usr/bin/env python3
"""Freeze 50 manifest rows stratified by quality, camera pose, and object mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import EffectCapability, read_manifest_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit selection: {args.output_dir}")
    records = read_manifest_jsonl(args.manifest)
    eligible = [
        (index, record)
        for index, record in enumerate(records)
        if record.training_valid
    ]
    if len(eligible) < args.sample_count:
        raise ValueError(f"only {len(eligible)} training-valid rows, need {args.sample_count}")
    groups: dict[tuple[str, bool, bool], list[int]] = {}
    for index, record in eligible:
        key = (
            record.quality_bucket or "unlabeled",
            record.has_capability(EffectCapability.CAMERA_POSE),
            record.has_capability(EffectCapability.OBJECT_MASK),
        )
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(args.seed)
    for values in groups.values():
        rng.shuffle(values)
    active = sorted(groups)
    selected: list[int] = []
    while len(selected) < args.sample_count and active:
        next_active = []
        for key in active:
            values = groups[key]
            if values and len(selected) < args.sample_count:
                selected.append(values.pop())
            if values:
                next_active.append(key)
        active = next_active
    args.output_dir.mkdir(parents=True)
    index_path = args.output_dir / "audit_sample_index.npy"
    np.save(index_path, np.asarray(selected, dtype=np.int64))
    selected_records = [records[index] for index in selected]
    strata: dict[str, int] = {}
    for record in selected_records:
        key = "|".join(
            [
                record.quality_bucket or "unlabeled",
                f"pose={int(record.has_capability(EffectCapability.CAMERA_POSE))}",
                f"mask={int(record.has_capability(EffectCapability.OBJECT_MASK))}",
            ]
        )
        strata[key] = strata.get(key, 0) + 1
    report = {
        "schema": "fact-target-audit-selection-v1",
        "sample_count": len(selected),
        "seed": args.seed,
        "sample_ids": [record.sample_id for record in selected_records],
        "strata_counts": dict(sorted(strata.items())),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "audit_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
