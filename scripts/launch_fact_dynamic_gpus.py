#!/usr/bin/env python3
"""Dynamic GPU launcher for the FACT tokenizer training run.

This is not true DDP hot-plugging. PyTorch DDP fixes world size at process
startup, so adding GPUs requires a clean checkpoint/restart.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--current-run-file", type=Path, default=ROOT / "outputs" / "fact_tokenizer" / "current_train_run.txt")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--mem-threshold-mib", type=int, default=2000)
    parser.add_argument("--util-threshold", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--upgrade-check-seconds", type=int, default=300)
    parser.add_argument("--torchrun", type=Path, default=Path("/home/sxh/.conda/envs/fact_tokenizer/bin/torchrun"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--per-gpu-batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%F %T")


def log(msg: str, log_path: Path) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def query_gpus() -> dict[int, tuple[int, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        result[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return result


def idle_gpus(args: argparse.Namespace) -> list[int]:
    stats = query_gpus()
    idle = []
    for gpu_id in args.gpu_ids:
        mem, util = stats.get(gpu_id, (10**9, 100))
        if mem <= args.mem_threshold_mib and util < args.util_threshold:
            idle.append(gpu_id)
    return idle[: args.max_gpus]


def stats_text(args: argparse.Namespace) -> str:
    stats = query_gpus()
    chunks = []
    for gpu_id in args.gpu_ids:
        mem, util = stats.get(gpu_id, (-1, -1))
        chunks.append(f"{gpu_id}:mem={mem}MiB,util={util}%")
    return " | ".join(chunks)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"fact_tokenizer_step_(\d+)\.ckpt$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("fact_tokenizer_step_*.ckpt"), key=checkpoint_step)
    return ckpts[-1] if ckpts else None


def terminate_process_group(proc: subprocess.Popen, log_path: Path, reason: str) -> int:
    log(f"stopping training pid={proc.pid}: {reason}", log_path)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.wait()
    try:
        return proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        log(f"training pid={proc.pid} did not exit after SIGTERM; sending SIGKILL", log_path)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.wait()


def launch_training(
    args: argparse.Namespace,
    run_dir: Path,
    selected_gpus: list[int],
    resume_ckpt: Path | None,
    stdout_path: Path,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        str(args.torchrun),
        "--standalone",
        f"--nproc_per_node={len(selected_gpus)}",
        "scripts/train_fact_npz_debug.py",
        "--ddp",
        "--input-npz",
        "data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz",
        "--output-dir",
        str(run_dir),
        "--source-view-keys",
        "ego",
        "exo",
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.per_gpu_batch_size),
        "--resize",
        "224",
        "--backbone",
        "dino",
        "--device",
        "cuda",
        "--model-dim",
        "128",
        "--dino-dim",
        "768",
        "--latent-dim",
        "32",
        "--private-dim",
        "8",
        "--num-latents",
        "64",
        "--num-private-slots",
        "1",
        "--num-heads",
        "4",
        "--patch-size",
        "14",
        "--enc-blocks",
        "1",
        "--dec-blocks",
        "1",
        "--lr",
        "5e-5",
        "--vq-beta",
        "0.25",
        "--balance-weight",
        "0.05",
        "--private-reg-weight",
        "0.01",
        "--save-every",
        str(args.save_every),
    ]
    if resume_ckpt is not None:
        cmd.extend(["--resume-checkpoint", str(resume_ckpt)])

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in selected_gpus),
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SHM_DISABLE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    resume_txt = str(resume_ckpt) if resume_ckpt is not None else "fresh"
    log(
        f"starting training gpus={selected_gpus} nproc={len(selected_gpus)} resume={resume_txt}",
        log_path,
    )
    stdout_file = stdout_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=stdout_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )


def main() -> int:
    args = parse_args()
    if args.run_dir is None:
        if args.current_run_file.exists():
            args.run_dir = Path(args.current_run_file.read_text(encoding="utf-8").strip())
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_dir = ROOT / "outputs" / "fact_tokenizer" / f"egoexo_diverse_500takes_k64_dynamic_{stamp}"
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    args.current_run_file.parent.mkdir(parents=True, exist_ok=True)
    args.current_run_file.write_text(str(run_dir.relative_to(ROOT)), encoding="utf-8")

    log_path = run_dir / "launcher_dynamic.log"
    stdout_path = run_dir / "train_stdout.log"
    proc: subprocess.Popen | None = None
    current_gpus: list[int] = []
    last_upgrade_check = 0.0
    resume_ckpt = latest_checkpoint(run_dir)

    log(
        "dynamic launcher started "
        f"gpu_ids={args.gpu_ids} min_gpus={args.min_gpus} max_gpus={args.max_gpus} "
        f"idle=(mem<={args.mem_threshold_mib}MiB, util<{args.util_threshold}%)",
        log_path,
    )

    while True:
        if proc is None:
            free = idle_gpus(args)
            if len(free) < args.min_gpus:
                log(f"waiting for idle GPU; free={free}; {stats_text(args)}", log_path)
                time.sleep(args.poll_seconds)
                continue
            current_gpus = free
            resume_ckpt = latest_checkpoint(run_dir)
            proc = launch_training(args, run_dir, current_gpus, resume_ckpt, stdout_path, log_path)
            last_upgrade_check = time.time()
            time.sleep(args.poll_seconds)
            continue

        status = proc.poll()
        if status is not None:
            log(f"training exited status={status}", log_path)
            if status == 0 and (run_dir / "fact_tokenizer.ckpt").exists():
                log("training complete", log_path)
                return 0
            proc = None
            current_gpus = []
            time.sleep(args.poll_seconds)
            continue

        now_ts = time.time()
        if now_ts - last_upgrade_check >= args.upgrade_check_seconds:
            free = [g for g in idle_gpus(args) if g not in current_gpus]
            capacity = min(args.max_gpus, len(current_gpus) + len(free))
            if capacity > len(current_gpus):
                ckpt = latest_checkpoint(run_dir)
                if ckpt is not None and ckpt != resume_ckpt:
                    log(
                        f"found additional idle GPUs={free}; restarting from checkpoint {ckpt.name}",
                        log_path,
                    )
                    terminate_process_group(proc, log_path, "expand GPU count")
                    proc = None
                    current_gpus = []
                    resume_ckpt = ckpt
                else:
                    ckpt_txt = ckpt.name if ckpt is not None else "none"
                    log(
                        f"additional idle GPUs={free}, waiting for next checkpoint before restart; latest={ckpt_txt}",
                        log_path,
                    )
            else:
                log(f"running gpus={current_gpus}; no extra idle GPUs; {stats_text(args)}", log_path)
            last_upgrade_check = now_ts
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
