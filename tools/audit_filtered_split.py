#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a markdown audit report for a filtered split JSON.")
    parser.add_argument("--filtered-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def bucket_counter(items: list[dict]) -> Counter:
    return Counter(str(item.get("parent_task_name", "")) for item in items)


def main() -> None:
    args = parse_args()
    split = json.loads(args.filtered_split.read_text(encoding="utf-8"))
    lines = ["# Filtering Audit Report", "", f"Version: `{split.get('version', '')}`", ""]
    for split_name, buckets in split["splits"].items():
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append("| bucket | takes | top parent tasks |")
        lines.append("| --- | ---: | --- |")
        for bucket, items in buckets.items():
            counts = bucket_counter(items)
            top = ", ".join(f"{name}:{count}" for name, count in counts.most_common(8))
            lines.append(f"| {bucket} | {len(items)} | {top} |")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved audit report to {args.out}")


if __name__ == "__main__":
    main()
