#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, write_csv


ANNOTATION_COLUMNS = [
    "take_relevance",
    "ego_hand_visibility",
    "exo_body_visibility",
    "object_interaction",
    "phase_diversity",
    "usable_for",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a stratified take annotation CSV from relevance scores.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument(
        "--strategy",
        choices=["stratified_auto_bucket", "v1_full_if_small"],
        default="stratified_auto_bucket",
        help="v1_full_if_small exports all rows when the score table has no more than --sample-size rows.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def score_band(row: dict[str, str]) -> str:
    try:
        value = float(row.get("relevance_score", "0"))
    except ValueError:
        value = 0.0
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "mid"
    return "low"


def group_key(row: dict[str, str], strategy: str) -> str:
    if strategy == "v1_full_if_small":
        return "|".join(
            [
                row.get("auto_bucket", "F_uncertain"),
                row.get("filtered_bucket", ""),
                row.get("parent_task_name", ""),
                score_band(row),
            ]
        )
    return row.get("auto_bucket", "F_uncertain")


def main() -> None:
    args = parse_args()
    rows = read_csv(args.scores)
    if args.strategy == "v1_full_if_small" and len(rows) <= args.sample_size:
        selected = sorted(rows, key=lambda row: (row.get("split", ""), row.get("parent_task_name", ""), row.get("take_uid", "")))
    else:
        selected = []
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[group_key(row, args.strategy)].append(row)
        rng = random.Random(args.seed)
        buckets = sorted(grouped)
        per_bucket = max(1, args.sample_size // max(1, len(buckets)))
        for bucket in buckets:
            values = grouped[bucket]
            rng.shuffle(values)
            selected.extend(values[:per_bucket])
        if len(selected) < args.sample_size:
            selected_ids = {row.get("take_uid", "") for row in selected}
            remaining = [row for row in rows if row.get("take_uid", "") not in selected_ids]
            rng.shuffle(remaining)
            selected.extend(remaining[: args.sample_size - len(selected)])
        selected = selected[: args.sample_size]
    output = []
    for row in selected:
        output.append(
            {
                "take_uid": row.get("take_uid", ""),
                "contact_sheet_path": row.get("contact_sheet_path", ""),
                "auto_bucket": row.get("auto_bucket", ""),
                "relevance_score": row.get("relevance_score", ""),
                "parent_task_name": row.get("parent_task_name", ""),
                "task_name": row.get("task_name", ""),
                **{column: "" for column in ANNOTATION_COLUMNS},
            }
        )
    write_csv(args.out, output)
    print(f"Saved {len(output)} annotation rows to {args.out}")


if __name__ == "__main__":
    main()
