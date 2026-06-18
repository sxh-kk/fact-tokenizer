#!/usr/bin/env python3
"""Print the latest FACT tokenizer training status from train_stdout.log."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--show-gpu", action="store_true")
    return parser.parse_args()


def load_latest_row(log_path: Path) -> tuple[dict[str, Any] | None, int]:
    latest = None
    count = 0
    if not log_path.exists():
        return None, 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not (text.startswith("{") and text.endswith("}")):
                continue
            try:
                latest = json.loads(text)
            except json.JSONDecodeError:
                continue
            count += 1
    return latest, count


def latest_checkpoint(run_dir: Path) -> str:
    checkpoints = sorted(run_dir.glob("fact_tokenizer_step_*.ckpt"))
    if checkpoints:
        return checkpoints[-1].name
    final_checkpoint = run_dir / "fact_tokenizer.ckpt"
    return final_checkpoint.name if final_checkpoint.exists() else "none"


def gpu_summary() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception as exc:  # pragma: no cover - best effort monitor.
        return f"gpu=unavailable:{exc}"
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, memory, util = [part.strip() for part in line.split(",")]
        rows.append(f"{index}:{memory}MiB/{util}%")
    return "gpu=" + " ".join(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    row, rows_seen = load_latest_row(run_dir / "train_stdout.log")
    now = time.strftime("%F %T")
    checkpoint = latest_checkpoint(run_dir)
    if row is None:
        parts = [
            f"time={now}",
            "state=waiting_for_first_step",
            f"rows={rows_seen}",
            f"ckpt={checkpoint}",
        ]
    else:
        step = int(row.get("step", -1))
        target_last = args.total_steps - 1
        done = max(step - args.start_step + 1, 0)
        total = max(args.total_steps - args.start_step, 1)
        remaining = max(target_last - step, 0)
        parts = [
            f"time={now}",
            f"step={step}/{target_last}",
            f"done={done}/{total}",
            f"progress={done / total * 100:.2f}%",
            f"remaining={remaining}",
            f"rows={rows_seen}",
            f"loss={float(row.get('loss', 0.0)):.4f}",
            f"self={float(row.get('self_loss', 0.0)):.4f}",
            f"swap={float(row.get('swap_loss', 0.0)):.4f}",
            f"top1={float(row.get('action_top1_agreement', 0.0)):.4f}",
            f"priv_drop={float(row.get('private_dropout', 0.0)):.3f}",
            f"slot_drop={float(row.get('action_slot_dropout', 0.0)):.3f}",
            f"ckpt={checkpoint}",
        ]
    if args.show_gpu:
        parts.append(gpu_summary())
    print(" | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
