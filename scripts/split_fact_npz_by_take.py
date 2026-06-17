#!/usr/bin/env python3
"""Split a FACT paired-transition NPZ by take_uid for held-out validation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-name", default="train_by_take.npz")
    parser.add_argument("--heldout-name", default="heldout_by_take.npz")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--heldout-count", type=int, default=None)
    parser.add_argument("--heldout-uids", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--train-labels-name", default="train_labels.jsonl")
    parser.add_argument("--heldout-labels-name", default="heldout_labels.jsonl")
    parser.add_argument(
        "--label-columns",
        nargs="*",
        default=["parent_task_name", "task_name", "university_name"],
    )
    return parser.parse_args()


def load_uid_file(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                values.add(str(json.loads(line)["take_uid"]))
            else:
                values.add(line.split()[0])
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict[str, Any]], label_columns: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"takes": len(rows)}
    for column in label_columns:
        counts = Counter(str(row.get(column) or "") for row in rows)
        counts.pop("", None)
        summary[column] = {
            "unique": len(counts),
            "top": counts.most_common(12),
        }
    return summary


def split_uids(unique_uids: list[str], args: argparse.Namespace) -> tuple[set[str], set[str]]:
    if args.heldout_uids:
        heldout = load_uid_file(args.heldout_uids)
        unknown = sorted(heldout - set(unique_uids))
        if unknown:
            raise ValueError(f"Held-out UID file contains {len(unknown)} unknown takes, first={unknown[:3]}")
    else:
        if not 0.0 < args.heldout_fraction < 1.0 and args.heldout_count is None:
            raise ValueError("--heldout-fraction must be in (0, 1) unless --heldout-count is set")
        count = args.heldout_count
        if count is None:
            count = max(1, round(len(unique_uids) * args.heldout_fraction))
        count = min(max(count, 1), len(unique_uids) - 1)
        shuffled = list(unique_uids)
        random.Random(args.seed).shuffle(shuffled)
        heldout = set(shuffled[:count])
    train = set(unique_uids) - heldout
    if not train or not heldout:
        raise ValueError(f"Invalid split: train={len(train)} heldout={len(heldout)}")
    return train, heldout


def save_npz_split(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    sample_count: int,
) -> None:
    payload = {}
    for key, value in arrays.items():
        if value.shape[:1] == (sample_count,):
            payload[key] = value[sample_mask]
        else:
            payload[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_npz, allow_pickle=False) as data:
        if "take_uid" not in data:
            raise KeyError(f"{args.input_npz} does not contain required key 'take_uid'")
        arrays = {key: data[key] for key in data.files}

    take_uid = np.asarray(arrays["take_uid"]).astype(str)
    unique_uids = sorted(set(take_uid.tolist()))
    train_uids, heldout_uids = split_uids(unique_uids, args)
    train_mask = np.asarray([uid in train_uids for uid in take_uid], dtype=bool)
    heldout_mask = np.asarray([uid in heldout_uids for uid in take_uid], dtype=bool)

    train_npz = args.output_dir / args.train_name
    heldout_npz = args.output_dir / args.heldout_name
    save_npz_split(train_npz, arrays, train_mask, len(take_uid))
    save_npz_split(heldout_npz, arrays, heldout_mask, len(take_uid))

    (args.output_dir / "train_uids.txt").write_text("\n".join(sorted(train_uids)) + "\n", encoding="utf-8")
    (args.output_dir / "heldout_uids.txt").write_text("\n".join(sorted(heldout_uids)) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "input_npz": str(args.input_npz),
        "train_npz": str(train_npz),
        "heldout_npz": str(heldout_npz),
        "seed": args.seed,
        "num_takes": len(unique_uids),
        "num_train_takes": len(train_uids),
        "num_heldout_takes": len(heldout_uids),
        "num_samples": int(len(take_uid)),
        "num_train_samples": int(train_mask.sum()),
        "num_heldout_samples": int(heldout_mask.sum()),
    }

    if args.labels_jsonl:
        rows = load_jsonl(args.labels_jsonl)
        train_rows = [row for row in rows if str(row.get("take_uid")) in train_uids]
        heldout_rows = [row for row in rows if str(row.get("take_uid")) in heldout_uids]
        train_labels = args.output_dir / args.train_labels_name
        heldout_labels = args.output_dir / args.heldout_labels_name
        write_jsonl(train_labels, train_rows)
        write_jsonl(heldout_labels, heldout_rows)
        report.update(
            {
                "labels_jsonl": str(args.labels_jsonl),
                "train_labels_jsonl": str(train_labels),
                "heldout_labels_jsonl": str(heldout_labels),
                "train_label_summary": summarize_rows(train_rows, args.label_columns),
                "heldout_label_summary": summarize_rows(heldout_rows, args.label_columns),
            }
        )

    report_path = args.output_dir / "split_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote train split: {train_npz}")
    print(f"Wrote held-out split: {heldout_npz}")
    print(f"Wrote split report: {report_path}")


if __name__ == "__main__":
    main()
