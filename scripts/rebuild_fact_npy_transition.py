#!/usr/bin/env python3
"""Rebuild paired FACT NPY endpoints from raw videos with an explicit transition duration."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_fact_npy_source_semantics import read_pairs_sequential  # noqa: E402
from prepare_fact_egoexo_npz import resolve_video  # noqa: E402


CORE_ARRAYS = ("ego", "exo", "sample_id", "take_uid", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Legacy NPY source providing frozen IDs/takes/timestamps.")
    parser.add_argument("--video-map-jsonl", type=Path, required=True)
    parser.add_argument("--egoexo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def file_identity(directory: Path, name: str) -> dict:
    path = directory / f"{name}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype)}


def main() -> None:
    args = parse_args()
    if args.transition_seconds <= 0.0:
        raise ValueError("--transition-seconds must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite rebuilt NPY data: {args.output_dir}")
    input_arrays = {
        name: np.load(args.input_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in CORE_ARRAYS
    }
    lengths = {name: len(array) for name, array in input_arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"legacy input arrays are not aligned: {lengths}")
    if input_arrays["ego"].ndim != 5 or input_arrays["exo"].shape != input_arrays["ego"].shape:
        raise ValueError("legacy paired arrays must have the same NxTxHxWxC shape")
    if input_arrays["ego"].shape[1] != 2 or input_arrays["ego"].dtype != np.uint8:
        raise ValueError("legacy paired arrays must be uint8 with two endpoints")
    sample_ids = [as_text(value) for value in input_arrays["sample_id"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("legacy sample IDs are not unique")
    video_rows = {str(row["take_uid"]): row for row in load_jsonl(args.video_map_jsonl)}
    by_take: dict[str, list[dict]] = defaultdict(list)
    for index, (sample_id, raw_take, raw_timestamp) in enumerate(
        zip(sample_ids, input_arrays["take_uid"], input_arrays["timestamp"])
    ):
        take_uid = as_text(raw_take)
        by_take[take_uid].append(
            {"sample_id": sample_id, "source_index": index, "timestamp": float(raw_timestamp)}
        )
    missing = sorted(set(by_take) - set(video_rows))
    if missing:
        raise ValueError(f"video map is missing {len(missing)} source takes: {missing[:10]}")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    shape = (len(sample_ids), 2, args.resize, args.resize, 3)
    output_arrays = {
        view: np.lib.format.open_memmap(
            staging / f"{view}.npy", mode="w+", dtype=np.uint8, shape=shape
        )
        for view in ("ego", "exo")
    }
    t0_exact_view_pairs = 0
    video_inventory = []
    try:
        for take_number, take_uid in enumerate(sorted(by_take), start=1):
            row = video_rows[take_uid]
            ego_path = resolve_video(args.egoexo_root, row, row.get("ego_relative_path"))
            exo_path = resolve_video(args.egoexo_root, row, row.get("exo_relative_path"))
            if ego_path is None or exo_path is None:
                raise FileNotFoundError(f"raw videos missing for {take_uid}: ego={ego_path}, exo={exo_path}")
            video_inventory.append(
                {
                    "take_uid": take_uid,
                    "ego_path": str(ego_path),
                    "ego_sha256": sha256_file(ego_path),
                    "exo_path": str(exo_path),
                    "exo_sha256": sha256_file(exo_path),
                }
            )
            for view, path in (("ego", ego_path), ("exo", exo_path)):
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open {view} video for {take_uid}")
                try:
                    decoded = read_pairs_sequential(
                        capture,
                        by_take[take_uid],
                        args.transition_seconds,
                        args.resize,
                    )
                finally:
                    capture.release()
                for sample in by_take[take_uid]:
                    index = int(sample["source_index"])
                    pair = decoded[index]
                    if not np.array_equal(pair[0], np.asarray(input_arrays[view][index, 0])):
                        raise ValueError(f"raw t0 frame differs from frozen sample {sample['sample_id']} {view}")
                    t0_exact_view_pairs += 1
                    output_arrays[view][index] = pair
            if take_number % 25 == 0:
                print(json.dumps({"processed_takes": take_number, "total_takes": len(by_take)}), flush=True)
        for array in output_arrays.values():
            array.flush()
        for name in ("sample_id", "take_uid", "timestamp"):
            np.save(staging / f"{name}.npy", np.asarray(input_arrays[name]))
        output_files = {name: file_identity(staging, name) for name in CORE_ARRAYS}
        input_files = {name: file_identity(args.input_dir, name) for name in CORE_ARRAYS}
        report = {
            "schema": "fact-npy-transition-rebuild-v1",
            "source_dir": str(args.input_dir),
            "source_files": input_files,
            "output_dir": str(args.output_dir),
            "files": output_files,
            "rows": len(sample_ids),
            "takes": len(by_take),
            "color_space": "RGB",
            "transition_seconds": args.transition_seconds,
            "endpoint_semantics": ["t", f"t+{args.transition_seconds:g}s"],
            "resize": args.resize,
            "t0_exact_view_matches": t0_exact_view_pairs,
            "expected_t0_exact_view_matches": 2 * len(sample_ids),
            "video_map_jsonl": str(args.video_map_jsonl),
            "video_map_sha256": sha256_file(args.video_map_jsonl),
            "video_inventory": video_inventory,
            "rebuild_code_sha256": sha256_file(Path(__file__)),
            "decoder_code_sha256": sha256_file(ROOT / "scripts" / "audit_fact_npy_source_semantics.py"),
        }
        report_path = staging / "materialization_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(args.output_dir)
    except BaseException:
        for array in output_arrays.values():
            try:
                array.flush()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({key: report[key] for key in ("rows", "takes", "transition_seconds", "t0_exact_view_matches")}, indent=2))


if __name__ == "__main__":
    main()
