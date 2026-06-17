#!/usr/bin/env python3
"""Write a compact live status file for a FACT tokenizer training run."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


JSON_RE = re.compile(r"^\{.*\}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%F %T")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gpu_text() -> str:
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
        return f"unavailable:{exc}"
    return " | ".join(", ".join(part.strip() for part in line.split(",")) for line in output.splitlines() if line.strip())


def latest_checkpoint(run_dir: Path) -> str:
    ckpts = sorted(run_dir.glob("fact_tokenizer_step_*.ckpt"))
    return ckpts[-1].name if ckpts else "none"


def latest_json_row(stdout_path: Path) -> tuple[dict | None, int]:
    if not stdout_path.exists():
        return None, 0
    row = None
    count = 0
    with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not JSON_RE.match(text):
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            count += 1
    return row, count


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    stdout_path = run_dir / "train_stdout.log"
    monitor_path = run_dir / "train_monitor.log"
    status_path = run_dir / "monitor_status.json"
    start_time = time.time()

    with monitor_path.open("a", encoding="utf-8") as monitor:
        monitor.write(f"[{now()}] monitor started run={run_dir} pid={args.pid}\n")
        monitor.flush()
        while True:
            row, rows_seen = latest_json_row(stdout_path)
            alive = process_alive(args.pid)
            state = "running" if alive else ("completed" if (run_dir / "fact_tokenizer.ckpt").exists() else "stopped")
            status = {
                "time": now(),
                "state": state,
                "pid": args.pid,
                "alive": alive,
                "json_rows_seen": rows_seen,
                "latest_checkpoint": latest_checkpoint(run_dir),
                "gpu": gpu_text(),
            }
            if row is not None:
                step = int(row.get("step", -1))
                elapsed = max(time.time() - start_time, 1e-6)
                finished = max(step - args.start_step + 1, 0)
                steps_per_second = finished / elapsed
                remaining = max(args.total_steps - step - 1, 0)
                eta_seconds = int(remaining / steps_per_second) if steps_per_second > 0.0 else None
                status.update(row)
                status["total_steps"] = args.total_steps
                status["steps_per_second"] = steps_per_second
                status["eta"] = None if eta_seconds is None else f"{eta_seconds // 3600}h{(eta_seconds % 3600) // 60:02d}m{eta_seconds % 60:02d}s"
                monitor.write(
                    f"[{now()}] state={state} step={step}/{args.total_steps} "
                    f"loss={float(row.get('loss', 0.0)):.4f} "
                    f"self={float(row.get('self_loss', 0.0)):.4f} "
                    f"swap={float(row.get('swap_loss', 0.0)):.4f} "
                    f"ent={float(row.get('assignment_entropy_loss', 0.0)):.4f} "
                    f"hard_bal={float(row.get('hard_usage_balance_loss', 0.0)):.4f} "
                    f"slot_div={float(row.get('slot_diversity_loss', 0.0)):.4f} "
                    f"slot_bal={float(row.get('slot_balance_loss', 0.0)):.4f} "
                    f"top1={float(row.get('action_top1_agreement', 0.0)):.4f} "
                    f"eta={status.get('eta')} ckpt={status['latest_checkpoint']} gpu={status['gpu']}\n"
                )
            else:
                monitor.write(
                    f"[{now()}] state={state} step=waiting ckpt={status['latest_checkpoint']} gpu={status['gpu']}\n"
                )
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
            monitor.flush()
            if not alive:
                return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
