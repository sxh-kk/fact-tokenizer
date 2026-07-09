#!/usr/bin/env python3
"""Export a transition-level human review CSV from auto filter features."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-review", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--label-column", default="auto_label")
    parser.add_argument(
        "--label-quota",
        action="append",
        default=[],
        metavar="LABEL=COUNT",
        help="Optional per-label quota, e.g. main_interaction=180.",
    )
    parser.add_argument("--max-per-take", type=int, default=4)
    return parser.parse_args()


def parse_quotas(values: list[str], max_review: int) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--label-quota must be LABEL=COUNT, got {value!r}")
        label, count = value.split("=", 1)
        quotas[label.strip()] = int(count)
    if quotas:
        return quotas
    return {
        "main_interaction": int(max_review * 0.40),
        "phase_context": int(max_review * 0.25),
        "loco_only": int(max_review * 0.15),
        "discard": max_review,
    }


def take_limited_sample(rows: list[dict[str, str]], quota: int, max_per_take: int, rng: random.Random) -> list[dict[str, str]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected: list[dict[str, str]] = []
    take_counts: dict[str, int] = defaultdict(int)
    for row in shuffled:
        take_uid = str(row.get("take_uid", ""))
        if take_counts[take_uid] >= max_per_take:
            continue
        selected.append(row)
        take_counts[take_uid] += 1
        if len(selected) >= quota:
            break
    if len(selected) < quota:
        chosen = {id(row) for row in selected}
        for row in shuffled:
            if id(row) in chosen:
                continue
            selected.append(row)
            if len(selected) >= quota:
                break
    return selected


def main() -> None:
    args = parse_args()
    rows = read_csv(args.features)
    rng = random.Random(args.seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get(args.label_column, ""))].append(row)
    quotas = parse_quotas(args.label_quota, args.max_review)
    selected: list[dict[str, str]] = []
    for label, quota in quotas.items():
        if quota <= 0:
            continue
        selected.extend(take_limited_sample(by_label.get(label, []), quota, args.max_per_take, rng))
    if len(selected) < args.max_review:
        selected_ids = {str(row.get("row_index", "")) for row in selected}
        remaining = [row for row in rows if str(row.get("row_index", "")) not in selected_ids]
        selected.extend(take_limited_sample(remaining, args.max_review - len(selected), args.max_per_take, rng))
    selected = selected[: args.max_review]
    selected.sort(
        key=lambda row: (
            str(row.get("parent_task_name", "")),
            str(row.get("task_name", "")),
            str(row.get("take_uid", "")),
            float(row.get("timestamp", "0") or 0),
        )
    )

    out_rows: list[dict[str, Any]] = []
    for row in selected:
        review = dict(row)
        review.update(
            {
                "transition_label": "",
                "confidence": "",
                "reason": "",
                "notes": "",
            }
        )
        out_rows.append(review)

    base_fields = list(rows[0].keys()) if rows else []
    fieldnames = base_fields + ["transition_label", "confidence", "reason", "notes"]
    write_csv(args.out, out_rows, fieldnames)
    counts: dict[str, int] = defaultdict(int)
    for row in out_rows:
        counts[str(row.get(args.label_column, ""))] += 1
    summary = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    print(f"Saved {len(out_rows)} transition review rows to {args.out} ({summary})", flush=True)


if __name__ == "__main__":
    main()
