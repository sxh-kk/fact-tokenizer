#!/usr/bin/env python3
"""Prepare the dense transition48 FACT dataset and split it by held-out takes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--egoexo-root", type=Path, default=ROOT / "data/egoexo4d")
    parser.add_argument(
        "--selected-jsonl",
        type=Path,
        default=ROOT / "data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=ROOT / "data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz",
    )
    parser.add_argument(
        "--failed-jsonl",
        type=Path,
        default=ROOT / "data/fact_egoexo/failed_transition48_samples.jsonl",
    )
    parser.add_argument(
        "--prepare-report",
        type=Path,
        default=ROOT / "data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000_report.json",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20",
    )
    parser.add_argument("--samples-per-take", type=int, default=48)
    parser.add_argument("--transition-sec", type=float, default=0.5)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--allow-short-takes",
        action="store_true",
        help="Allow fewer than --samples-per-take transitions for short takes.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_npz_take_counts(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        take_uid = np.asarray(data["take_uid"]).astype(str)
        sample_id = np.asarray(data["sample_id"]).astype(str) if "sample_id" in data else None
        timestamps = np.asarray(data["timestamp"], dtype=np.float32) if "timestamp" in data else None
        keys = list(data.files)
        shapes = {key: list(data[key].shape) for key in data.files}
    counts = Counter(take_uid.tolist())
    values = list(counts.values())
    summary: dict[str, Any] = {
        "path": str(path),
        "keys": keys,
        "shapes": shapes,
        "samples": int(len(take_uid)),
        "takes": len(counts),
        "min_samples_per_take": int(min(values)) if values else 0,
        "max_samples_per_take": int(max(values)) if values else 0,
        "sample_count_histogram": dict(sorted(Counter(values).items())),
    }
    if sample_id is not None and len(sample_id):
        summary["first_sample_id"] = str(sample_id[0])
        summary["last_sample_id"] = str(sample_id[-1])
    if timestamps is not None and len(timestamps):
        summary["min_timestamp"] = float(timestamps.min())
        summary["max_timestamp"] = float(timestamps.max())
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.samples_per_take <= 0:
        raise ValueError("--samples-per-take must be positive")
    if args.transition_sec <= 0.0:
        raise ValueError("--transition-sec must be positive")
    if args.stride_sec <= 0.0:
        raise ValueError("--stride-sec must be positive")
    if args.output_npz.exists() and not args.force and not args.skip_prepare:
        raise FileExistsError(f"{args.output_npz} exists. Use --force or --skip-prepare.")

    if not args.skip_prepare:
        prepare_cmd = [
            str(args.python),
            "scripts/prepare_fact_egoexo_npz.py",
            "--egoexo-root",
            str(args.egoexo_root),
            "--selected-jsonl",
            str(args.selected_jsonl),
            "--output-npz",
            str(args.output_npz),
            "--failed-jsonl",
            str(args.failed_jsonl),
            "--report-json",
            str(args.prepare_report),
            "--samples-per-take",
            str(args.samples_per_take),
            "--transition-sec",
            str(args.transition_sec),
            "--stride-sec",
            str(args.stride_sec),
            "--resize",
            str(args.resize),
        ]
        if not args.allow_short_takes:
            prepare_cmd.append("--require-full-take")
        run_command(prepare_cmd)

    if not args.skip_split:
        split_cmd = [
            str(args.python),
            "scripts/split_fact_npz_by_take.py",
            "--input-npz",
            str(args.output_npz),
            "--output-dir",
            str(args.split_dir),
            "--heldout-fraction",
            str(args.heldout_fraction),
            "--seed",
            str(args.seed),
            "--labels-jsonl",
            str(args.selected_jsonl),
        ]
        run_command(split_cmd)

    train_npz = args.split_dir / "train_by_take.npz"
    heldout_npz = args.split_dir / "heldout_by_take.npz"
    report = {
        "config": {
            "selected_jsonl": str(args.selected_jsonl),
            "samples_per_take": args.samples_per_take,
            "transition_sec": args.transition_sec,
            "stride_sec": args.stride_sec,
            "resize": args.resize,
            "heldout_fraction": args.heldout_fraction,
            "seed": args.seed,
            "allow_short_takes": args.allow_short_takes,
        },
        "full": load_npz_take_counts(args.output_npz),
        "train": load_npz_take_counts(train_npz),
        "heldout": load_npz_take_counts(heldout_npz),
    }
    if not args.allow_short_takes:
        for split_name in ("full", "train", "heldout"):
            split = report[split_name]
            if split["min_samples_per_take"] != args.samples_per_take or split["max_samples_per_take"] != args.samples_per_take:
                raise RuntimeError(f"{split_name} split is not exactly {args.samples_per_take} samples/take: {split}")

    report_path = args.split_dir / "transition48_data_report.json"
    write_json(report_path, report)
    print(f"Wrote transition48 data report: {report_path}")


if __name__ == "__main__":
    main()
