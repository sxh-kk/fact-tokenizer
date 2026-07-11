#!/usr/bin/env python3
"""Materialize a hash-bound row subset from a rebuilt FACT paired NPY source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np


CORE_ARRAYS = ("ego", "exo", "sample_id", "take_uid", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--row-index-npy", type=Path, required=True)
    parser.add_argument("--metadata-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(directory: Path, name: str) -> dict:
    path = directory / f"{name}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype)}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite NPY subset: {args.output_dir}")
    parent_report_path = args.parent_dir / "materialization_report.json"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    if (
        parent_report.get("schema") != "fact-npy-transition-rebuild-v1"
        or parent_report.get("color_space") != "RGB"
        or float(parent_report.get("transition_seconds", -1.0)) != 0.5
        or parent_report.get("endpoint_semantics") != ["t", "t+0.5s"]
        or float(parent_report.get("frame_rate_hz", -1.0)) != 30.0
        or int(parent_report.get("endpoint_offset_frames", -1)) != 15
    ):
        raise ValueError("parent is not a verified RGB t→t+0.5s rebuild")
    parent = {
        name: np.load(args.parent_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in CORE_ARRAYS
    }
    if parent_report.get("files") != {name: identity(args.parent_dir, name) for name in CORE_ARRAYS}:
        raise ValueError("parent materialization report no longer matches its arrays")
    parent_frame_index = np.load(
        args.parent_dir / "frame_index.npy", mmap_mode="r", allow_pickle=False
    )
    parent_frame_identity = identity(args.parent_dir, "frame_index")
    reported_frame_identity = parent_report.get("frame_index", {})
    if (
        parent_frame_index.ndim != 1
        or len(parent_frame_index) != len(parent["sample_id"])
        or not np.issubdtype(parent_frame_index.dtype, np.integer)
        or (parent_frame_index < 0).any()
        or any(
            reported_frame_identity.get(key) != value
            for key, value in parent_frame_identity.items()
        )
    ):
        raise ValueError("parent frame_index.npy is missing, invalid, or no longer hash-bound")
    indices = np.load(args.row_index_npy, mmap_mode="r", allow_pickle=False)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("row index must be a one-dimensional integer array")
    source_index_dtype = str(indices.dtype)
    indices = np.asarray(indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(parent["sample_id"])).any():
        raise ValueError("row index has duplicates or out-of-range values")
    reference = {
        name: np.load(args.metadata_reference_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("sample_id", "take_uid", "timestamp")
    }
    for name, array in reference.items():
        if not np.array_equal(array, parent[name][indices]):
            raise ValueError(f"metadata reference {name}.npy does not equal indexed parent rows")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for view in ("ego", "exo"):
            shape = (len(indices), *parent[view].shape[1:])
            output = np.lib.format.open_memmap(
                staging / f"{view}.npy", mode="w+", dtype=parent[view].dtype, shape=shape
            )
            for start in range(0, len(indices), 64):
                selected = indices[start : start + 64]
                output[start : start + len(selected)] = parent[view][selected]
            output.flush()
            del output
        for name, array in reference.items():
            np.save(staging / f"{name}.npy", np.asarray(array))
        np.save(
            staging / "frame_index.npy",
            np.asarray(parent_frame_index[indices], dtype=np.int64),
        )
        files = {name: identity(staging, name) for name in CORE_ARRAYS}
        report = {
            "schema": "fact-npy-transition-subset-v1",
            "parent_dir": str(args.parent_dir),
            "parent_report_sha256": sha256_file(parent_report_path),
            "row_index_npy": str(args.row_index_npy),
            "row_index_sha256": sha256_file(args.row_index_npy),
            "row_index_dtype": source_index_dtype,
            "metadata_reference_dir": str(args.metadata_reference_dir),
            "rows": len(indices),
            "color_space": "RGB",
            "transition_seconds": 0.5,
            "endpoint_semantics": ["t", "t+0.5s"],
            "frame_rate_hz": 30.0,
            "endpoint_offset_frames": 15,
            "frame_index": identity(staging, "frame_index"),
            "files": files,
            "subset_code_sha256": sha256_file(Path(__file__)),
        }
        (staging / "materialization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"rows": len(indices), "transition_seconds": 0.5}, indent=2))


if __name__ == "__main__":
    main()
