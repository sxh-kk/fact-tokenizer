#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import group_indices_by_take, load_npz_metadata, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select transitions inside filtered takes.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--split-name", choices=["train", "heldout"], required=True)
    parser.add_argument("--filtered-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-transitions", type=int, default=48)
    parser.add_argument(
        "--bucket-num-transitions",
        action="append",
        default=[],
        metavar="BUCKET=COUNT",
        help="Override --num-transitions for a bucket, e.g. loco_aux=12 for a capped loco ablation.",
    )
    parser.add_argument("--include-buckets", nargs="+", default=["fact_main", "loco_aux"])
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument(
        "--score-mode",
        choices=["temporal", "motion"],
        default="motion",
        help="temporal avoids loading video tensors and selects evenly across timestamps; motion uses frame deltas within each bin.",
    )
    return parser.parse_args()


def motion_scores(video: np.ndarray) -> np.ndarray:
    values = video.astype(np.float32)
    if values.max(initial=0.0) > 1.5:
        values = values / 255.0
    if values.shape[-1] in (1, 3):
        delta = values[:, -1] - values[:, 0]
    else:
        delta = values[:, -1].transpose(0, 2, 3, 1) - values[:, 0].transpose(0, 2, 3, 1)
    return np.abs(delta).mean(axis=tuple(range(1, delta.ndim))).astype(np.float32)


def selected_takes(filtered_split: dict, split_name: str, include_buckets: list[str]) -> dict[str, str]:
    takes: dict[str, str] = {}
    split = filtered_split["splits"][split_name]
    for bucket in include_buckets:
        for row in split.get(bucket, []):
            takes[str(row["take_uid"])] = bucket
    return takes


def select_rows(indices: list[int], score: np.ndarray, timestamps: np.ndarray, count: int) -> list[int]:
    if len(indices) <= count:
        return list(indices)
    ordered = sorted(indices, key=lambda idx: float(timestamps[idx]))
    bins = np.array_split(np.asarray(ordered, dtype=np.int64), count)
    selected = []
    for values in bins:
        best = int(values[np.argmax(score[values])])
        selected.append(best)
    return sorted(set(selected), key=lambda idx: float(timestamps[idx]))


def select_rows_temporal(indices: list[int], timestamps: np.ndarray, count: int) -> list[int]:
    if len(indices) <= count:
        return list(indices)
    ordered = sorted(indices, key=lambda idx: float(timestamps[idx]))
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[int(pos)] for pos in positions]


def parse_bucket_counts(values: list[str], default_count: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--bucket-num-transitions must be BUCKET=COUNT, got {value!r}")
        bucket, count_text = value.split("=", 1)
        bucket = bucket.strip()
        count = int(count_text)
        if not bucket or count <= 0:
            raise ValueError(f"Invalid bucket transition cap: {value!r}")
        counts[bucket] = count
    counts.setdefault("default", default_count)
    return counts


def main() -> None:
    args = parse_args()
    metadata = load_npz_metadata(args.npz)
    filtered_split = json.loads(args.filtered_split.read_text(encoding="utf-8"))
    take_bucket = selected_takes(filtered_split, args.split_name, args.include_buckets)
    grouped = group_indices_by_take(metadata["take_uid"])
    bucket_counts = parse_bucket_counts(args.bucket_num_transitions, args.num_transitions)
    score = None
    if args.score_mode == "motion":
        with np.load(args.npz, allow_pickle=False) as data:
            score = sum(motion_scores(np.asarray(data[view_key])) for view_key in args.view_keys) / len(args.view_keys)
    rows = []
    for take_uid, indices in sorted(grouped.items()):
        if take_uid not in take_bucket:
            continue
        bucket = take_bucket[take_uid]
        count = bucket_counts.get(bucket, bucket_counts["default"])
        if args.score_mode == "motion":
            assert score is not None
            selected = set(select_rows(indices, score, metadata["timestamp"], count))
            reason = f"motion_aware_temporal_coverage_top{count}"
        else:
            selected = set(select_rows_temporal(indices, metadata["timestamp"], count))
            reason = f"temporal_coverage_top{count}"
        for idx in indices:
            is_selected = idx in selected
            rows.append(
                {
                    "split": args.split_name,
                    "bucket": bucket,
                    "take_uid": take_uid,
                    "row_index": idx,
                    "sample_id": str(metadata["sample_id"][idx]),
                    "timestamp": float(metadata["timestamp"][idx]),
                    "selected": int(is_selected),
                    "selection_reason": reason if is_selected else "",
                    "motion_score": float(score[idx]) if score is not None else "",
                }
            )
    fieldnames = ["split", "bucket", "take_uid", "row_index", "sample_id", "timestamp", "selected", "selection_reason", "motion_score"]
    write_csv(args.out, rows, fieldnames=fieldnames)
    print(f"Saved {len(rows)} transition rows to {args.out}")


if __name__ == "__main__":
    main()
