#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize filtered train/heldout NPZ files from transition selections.")
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--heldout-npz", type=Path, required=True)
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--heldout-selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def selected_rows(path: Path) -> list[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [int(row["row_index"]) for row in csv.DictReader(handle) if str(row.get("selected", "0")) == "1"]


def write_filtered(source: Path, rows: list[int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    row_array = np.asarray(rows, dtype=np.int64)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(value)[row_array] for key, value in data.items()}
    payload["source_row_index"] = row_array
    np.savez_compressed(output, **payload)
    print(f"Saved {len(rows)} rows to {output}")


def main() -> None:
    args = parse_args()
    write_filtered(args.train_npz, selected_rows(args.train_selection), args.out_dir / "train_by_take.npz")
    write_filtered(args.heldout_npz, selected_rows(args.heldout_selection), args.out_dir / "heldout_by_take.npz")


if __name__ == "__main__":
    main()
