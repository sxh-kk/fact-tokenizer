#!/usr/bin/env python3
"""Freeze Assembly101 8-take/2-take probe split and a trainer-quarantined manifest."""

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

from fact_tokenizer.effect_manifest import EffectCapability, EffectSampleRecord, write_manifest_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-take", action="append", default=[])
    parser.add_argument("--test-take", action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen Assembly probe assets: {args.output_dir}")
    take_uid = np.load(args.input_dir / "take_uid.npy", mmap_mode="r", allow_pickle=False).astype(str)
    sample_id = np.load(args.input_dir / "sample_id.npy", mmap_mode="r", allow_pickle=False).astype(str)
    timestamp_path = args.input_dir / "timestamp.npy"
    timestamp = (
        np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
        if timestamp_path.is_file()
        else np.arange(len(take_uid), dtype=np.float32)
    )
    takes = sorted(set(take_uid.tolist()))
    if len(takes) != 10:
        raise ValueError(f"Assembly101 probe contract expects 10 takes, found {len(takes)}")
    if args.train_take or args.test_take:
        train_takes, test_takes = sorted(args.train_take), sorted(args.test_take)
    else:
        # This deterministic split is frozen before any model is evaluated.
        train_takes, test_takes = takes[:8], takes[8:]
    if len(train_takes) != 8 or len(test_takes) != 2 or set(train_takes) & set(test_takes) or set(train_takes) | set(test_takes) != set(takes):
        raise ValueError("Assembly split must partition all 10 takes into exactly 8 train and 2 test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "schema": "fact-assembly101-probe-v1",
        "train_takes": train_takes,
        "test_takes": test_takes,
        "model_selection_allowed": False,
        "threshold_selection_allowed": False,
        "run_after_model_freeze_only": True,
    }
    split_text = json.dumps(split, indent=2, sort_keys=True) + "\n"
    split_path = args.output_dir / "assembly101_split_frozen.json"
    split_path.write_text(split_text, encoding="utf-8")
    split_sha256 = sha256_file(split_path)
    records = [
        EffectSampleRecord(
            sample_id=str(sample_id[index]),
            take_uid=str(take_uid[index]),
            split="assembly_probe",
            row_index=index,
            source_dataset="assembly101",
            timestamp=float(timestamp[index]),
            capability_validity={EffectCapability.RGB_PAIRED: True},
            training_valid=False,
            provenance={"external_domain_probe": True, "model_selection_allowed": False},
        )
        for index in range(len(take_uid))
    ]
    write_manifest_jsonl(args.output_dir / "assembly101_probe_manifest.jsonl", records)
    (args.output_dir / "assembly101_prepare_report.json").write_text(
        json.dumps(
            {
                **split,
                "split_file_sha256": split_sha256,
                "samples": len(records),
                "takes": 10,
                "input_array_sha256": {
                    name: sha256_file(args.input_dir / f"{name}.npy")
                    for name in ("ego", "exo", "sample_id", "take_uid", "action_label")
                },
                "probe_manifest_sha256": sha256_file(
                    args.output_dir / "assembly101_probe_manifest.jsonl"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**split, "split_file_sha256": split_sha256, "samples": len(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
