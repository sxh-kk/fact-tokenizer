#!/usr/bin/env python3
"""Atomically unpack the frozen Assembly101 NPZ into pickle-free mmap NPY arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Assembly NPY store: {args.output_dir}")
    staging = args.output_dir.with_name(f".{args.output_dir.name}.staging-{uuid.uuid4().hex[:12]}")
    staging.mkdir(parents=True)
    try:
        with np.load(args.input_npz, allow_pickle=False) as payload:
            required = {"ego", "exo", "sample_id", "take_uid", "action_label"}
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"Assembly NPZ is missing arrays: {sorted(missing)}")
            lengths = {len(np.asarray(payload[name])) for name in required}
            if len(lengths) != 1:
                raise ValueError("Assembly NPZ arrays do not share one sample dimension")
            for name in payload.files:
                value = np.asarray(payload[name])
                if value.dtype == object:
                    raise ValueError(f"Assembly array {name} has forbidden object dtype")
                np.save(staging / f"{name}.npy", value)
        report = {
            "schema": "fact-assembly101-unpack-v1",
            "source_npz_sha256": sha256_file(args.input_npz),
            "samples": next(iter(lengths)),
            "array_sha256": {
                path.stem: sha256_file(path) for path in sorted(staging.glob("*.npy"))
            },
        }
        (staging / "unpack_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
