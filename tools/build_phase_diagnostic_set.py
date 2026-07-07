#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample high-quality heldout takes for manual phase diagnostics.")
    parser.add_argument("--filtered-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--buckets", nargs="+", default=["fact_main"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = json.loads(args.filtered_split.read_text(encoding="utf-8"))
    candidates = []
    heldout = split["splits"].get("heldout", {})
    for bucket in args.buckets:
        for item in heldout.get(bucket, []):
            candidates.append({"bucket": bucket, **item})
    rng = random.Random(args.seed)
    candidates.sort(key=lambda row: (row.get("interaction_prob", 0.0), row.get("relevance_score", 0.0)), reverse=True)
    top_pool = candidates[: max(args.count * 3, args.count)]
    rng.shuffle(top_pool)
    selected = top_pool[: args.count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "take_uid",
            "bucket",
            "parent_task_name",
            "task_name",
            "relevance_score",
            "phase_segments_json",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "take_uid": row["take_uid"],
                    "bucket": row["bucket"],
                    "parent_task_name": row.get("parent_task_name", ""),
                    "task_name": row.get("task_name", ""),
                    "relevance_score": row.get("relevance_score", ""),
                    "phase_segments_json": "[]",
                    "notes": "",
                }
            )
    print(f"Saved {len(selected)} diagnostic candidates to {args.out}")


if __name__ == "__main__":
    main()
