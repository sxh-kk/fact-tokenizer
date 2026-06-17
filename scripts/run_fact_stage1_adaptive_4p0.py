#!/usr/bin/env python3
"""Adaptive 8-GPU Stage-1 FACT tokenizer optimization using the 4.0 teacher recipe."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Recipe:
    name: str
    steps: int
    per_gpu_batch: int
    lr: float
    extra: tuple[str, ...]
    resume_checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--torchrun", type=Path, default=Path("/home/sxh/.conda/envs/fact_tokenizer/bin/torchrun"))
    parser.add_argument("--python", type=Path, default=Path("/home/sxh/.conda/envs/fact_tokenizer/bin/python"))
    parser.add_argument(
        "--train-npz",
        type=Path,
        default=ROOT / "data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz",
    )
    parser.add_argument(
        "--heldout-npz",
        type=Path,
        default=ROOT / "data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz",
    )
    parser.add_argument(
        "--heldout-labels",
        type=Path,
        default=ROOT / "data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "outputs/fact_tokenizer/v5k_4p0_usage_antitake_repair_8gpu_20260617_211752/fact_tokenizer.ckpt",
    )
    parser.add_argument("--initial-run-dir", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--reprobe-existing", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/fact_tokenizer")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/fact_tokenizer/stage1_optimization_manifest_transition48.json",
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--train-num-workers", type=int, default=6)
    parser.add_argument("--train-prefetch-factor", type=int, default=6)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=1024)
    parser.add_argument("--gpu-utilization-threshold", type=int, default=15)
    parser.add_argument("--gpu-wait-interval-sec", type=float, default=60.0)
    parser.add_argument(
        "--gpu-wait-timeout-sec",
        type=float,
        default=0.0,
        help="Maximum GPU wait time. Use 0 to wait indefinitely.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: Sequence[str], env: dict[str, str], log_path: Path, dry_run: bool = False) -> int:
    text = " ".join(shlex.quote(part) for part in command)
    print(text, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(text + "\n")
        log.flush()
        if dry_run:
            return 0
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def query_gpu_stats() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    stats = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, memory_used, utilization = [int(part.strip()) for part in line.split(",")]
        stats.append({"index": index, "memory_used_mib": memory_used, "utilization": utilization})
    return stats


def selected_gpus_idle(args: argparse.Namespace) -> tuple[bool, str]:
    selected = {int(gpu) for gpu in args.gpu_ids}
    stats = [row for row in query_gpu_stats() if row["index"] in selected]
    if len(stats) != len(selected):
        seen = {row["index"] for row in stats}
        return False, f"missing GPU stats for {sorted(selected - seen)}"
    busy = [
        row
        for row in stats
        if row["memory_used_mib"] > args.gpu_memory_threshold_mib
        or row["utilization"] > args.gpu_utilization_threshold
    ]
    summary = " | ".join(
        f"{row['index']}:mem={row['memory_used_mib']}MiB,util={row['utilization']}%" for row in stats
    )
    return not busy, summary


def wait_for_gpus_if_requested(args: argparse.Namespace, reason: str) -> None:
    if not args.wait_for_gpus or args.dry_run:
        return
    start = time.time()
    while True:
        idle, summary = selected_gpus_idle(args)
        if idle:
            print(f"GPUs idle for {reason}: {summary}", flush=True)
            return
        elapsed = time.time() - start
        if args.gpu_wait_timeout_sec > 0.0 and elapsed >= args.gpu_wait_timeout_sec:
            raise TimeoutError(f"Timed out waiting for GPUs for {reason}: {summary}")
        print(f"Waiting for GPUs for {reason}: {summary}", flush=True)
        time.sleep(max(args.gpu_wait_interval_sec, 1.0))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_manifest(args: argparse.Namespace, entry: dict[str, Any]) -> None:
    manifest = load_json(args.manifest, [])
    manifest.append(entry)
    save_json(args.manifest, manifest)


def checkpoint_step(path: Path, python: Path) -> int:
    code = (
        "import torch, sys; "
        "ckpt=torch.load(sys.argv[1], map_location='cpu'); "
        "print(int(ckpt.get('step', -1)))"
    )
    output = subprocess.check_output([str(python), "-c", code, str(path)], text=True)
    return int(output.strip().splitlines()[-1])


def gate_value(gate: dict[str, Any] | None, name: str, default: float = 0.0) -> float:
    if not gate:
        return default
    for item in gate.get("gates", []):
        if item.get("name") == name:
            return float(item.get("value", default))
    return default


def gate_failed(gate: dict[str, Any] | None, name: str) -> bool:
    if not gate:
        return False
    for item in gate.get("gates", []):
        if item.get("name") == name:
            return not bool(item.get("pass"))
    return False


def latest_completed_checkpoint(manifest: list[dict[str, Any]]) -> Path | None:
    for entry in reversed(manifest):
        if entry.get("stage") != "gate":
            continue
        checkpoint = Path(entry.get("run_dir", "")) / "fact_tokenizer.ckpt"
        if checkpoint.exists():
            return checkpoint
    return None


def latest_gate(manifest: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(manifest):
        if entry.get("stage") == "gate" and isinstance(entry.get("gate"), dict):
            return entry["gate"]
    return None


def evaluate_checkpoint(args: argparse.Namespace, env: dict[str, str], run_dir: Path, checkpoint: Path) -> tuple[int, dict[str, Any]]:
    probe_dir = run_dir / "action_token_probe_heldout"
    gate_path = probe_dir / "stage1_gate.json"
    probe_ready = (probe_dir / "probe_summary.json").exists() and (probe_dir / "semantic_probe.json").exists()
    if gate_path.exists() and not args.reprobe_existing:
        return 0, load_json(gate_path, {"passed": False, "missing": str(gate_path)})
    if not probe_ready or args.reprobe_existing:
        wait_for_gpus_if_requested(args, reason=f"heldout probe for {run_dir.name}")
        code = run_command(probe_command(args, run_dir, checkpoint), env=env, log_path=run_dir / "probe_heldout.log", dry_run=args.dry_run)
        if code != 0 or args.dry_run:
            return code, {"passed": False, "stage": "probe", "returncode": code}
    code = run_command(gate_command(args, run_dir), env=env, log_path=run_dir / "stage1_gate.log", dry_run=args.dry_run)
    gate = load_json(gate_path, {"passed": False, "missing": str(gate_path)})
    return code, gate


def recipe_for_round(round_index: int, previous_gate: dict[str, Any] | None, resume_checkpoint: Path) -> Recipe:
    usage = gate_value(previous_gate, "ego_used_codes_min", default=38.0)
    take_nmi = (previous_gate or {}).get("metrics", {}).get("tuple_take_nmi", 0.43)
    exo_weak = gate_failed(previous_gate, "exo_swap_random_take_delta") or gate_failed(previous_gate, "exo_swap_zero_delta")
    ego_weak = (
        gate_failed(previous_gate, "ego_swap_random_take_delta")
        or gate_failed(previous_gate, "ego_swap_random_code_delta")
        or gate_failed(previous_gate, "ego_action_without_private_saving")
    )

    common = (
        "--discard-resume-history",
        "--vq-beta",
        "0.38",
        "--private-reg-weight",
        "0.055",
        "--private-dropout",
        "0.55",
        "--private-dropout-start-fraction",
        "0.0",
        "--private-dropout-ramp-fraction",
        "0.05",
        "--action-only-motion-focus-weight",
        "0.045",
        "--motion-focus-weight",
        "0.04",
        "--motion-contrast-weight",
        "0.045",
        "--delta-focus-weight",
        "0.025",
        "--action-only-delta-focus-weight",
        "0.025",
        "--delta-contrast-weight",
        "0.025",
        "--action-contrast-margin",
        "0.008",
        "--action-aux-start-fraction",
        "0.0",
        "--action-aux-ramp-fraction",
        "0.05",
        "--teacher-ego-uncertainty-weight",
        "2.0",
        "--teacher-disagreement-weight",
        "3.0",
        "--teacher-base-bias",
        "-1.0",
    )

    if previous_gate is None:
        name = "v5f_4p0_teacher_usage_8gpu"
        extra = (
            "--vq-temperature",
            "0.055",
            "--kl-weight",
            "0.22",
            "--balance-weight",
            "0.12",
            "--hard-usage-balance-weight",
            "0.0045",
            "--slot-balance-weight",
            "0.0015",
            "--assignment-entropy-weight",
            "0.012",
            "--assignment-entropy-target",
            "0.82",
            "--action-slot-dropout",
            "0.20",
            "--action-slot-dropout-start-fraction",
            "0.0",
            "--action-slot-dropout-ramp-fraction",
            "0.05",
            "--action-only-weight",
            "0.20",
            "--action-contrast-weight",
            "0.12",
            "--no-private-contrast-weight",
            "0.20",
            "--action-consistency-weight",
            "0.055",
            "--exo-aux-multiplier",
            "3.5",
            *common,
        )
        return Recipe(name=name, steps=12000, per_gpu_batch=16, lr=8.0e-6, extra=extra, resume_checkpoint=resume_checkpoint)

    if usage < 45 and take_nmi <= 0.45:
        name = "v5g_4p0_usage_repair_8gpu"
        extra = (
            "--vq-temperature",
            "0.065",
            "--kl-weight",
            "0.20",
            "--balance-weight",
            "0.16",
            "--hard-usage-balance-weight",
            "0.007",
            "--slot-balance-weight",
            "0.0025",
            "--assignment-entropy-weight",
            "0.008",
            "--assignment-entropy-target",
            "0.86",
            "--action-slot-dropout",
            "0.16",
            "--action-slot-dropout-start-fraction",
            "0.0",
            "--action-slot-dropout-ramp-fraction",
            "0.05",
            "--action-only-weight",
            "0.18",
            "--action-contrast-weight",
            "0.11",
            "--no-private-contrast-weight",
            "0.18",
            "--action-consistency-weight",
            "0.045",
            "--exo-aux-multiplier",
            "3.2",
            *common,
        )
        return Recipe(name=name, steps=10000, per_gpu_batch=16, lr=7.0e-6, extra=extra, resume_checkpoint=resume_checkpoint)

    if usage < 45:
        if take_nmi > 0.43:
            name = "v5l_4p0_transition48_dense_take_repair_8gpu"
            extra = (
                "--take-grouped-batches",
                "--samples-per-take",
                "8",
                "--vq-temperature",
                "0.085",
                "--kl-weight",
                "0.18",
                "--balance-weight",
                "0.18",
                "--hard-usage-balance-weight",
                "0.012",
                "--slot-balance-weight",
                "0.006",
                "--assignment-entropy-weight",
                "0.004",
                "--assignment-entropy-target",
                "0.88",
                "--same-take-contrast-weight",
                "0.08",
                "--take-uniform-weight",
                "0.018",
                "--take-slot-uniform-weight",
                "0.024",
                "--take-pair-uniform-weight",
                "0.012",
                "--action-slot-dropout",
                "0.16",
                "--action-slot-dropout-start-fraction",
                "0.0",
                "--action-slot-dropout-ramp-fraction",
                "0.05",
                "--action-only-weight",
                "0.22",
                "--action-contrast-weight",
                "0.20",
                "--no-private-contrast-weight",
                "0.26",
                "--action-consistency-weight",
                "0.055",
                "--exo-aux-multiplier",
                "4.5",
                *common,
            )
            return Recipe(name=name, steps=14000, per_gpu_batch=32, lr=4.5e-6, extra=extra, resume_checkpoint=resume_checkpoint)

        name = "v5j_4p0_usage_take_repair_8gpu"
        extra = (
            "--take-grouped-batches",
            "--samples-per-take",
            "4",
            "--vq-temperature",
            "0.070",
            "--kl-weight",
            "0.18",
            "--balance-weight",
            "0.12",
            "--hard-usage-balance-weight",
            "0.005",
            "--slot-balance-weight",
            "0.002",
            "--assignment-entropy-weight",
            "0.005",
            "--assignment-entropy-target",
            "0.84",
            "--same-take-contrast-weight",
            "0.09",
            "--take-uniform-weight",
            "0.018",
            "--action-slot-dropout",
            "0.22",
            "--action-slot-dropout-start-fraction",
            "0.0",
            "--action-slot-dropout-ramp-fraction",
            "0.05",
            "--action-only-weight",
            "0.20",
            "--action-contrast-weight",
            "0.12",
            "--no-private-contrast-weight",
            "0.20",
            "--action-consistency-weight",
            "0.055",
            "--exo-aux-multiplier",
            "3.5",
            *common,
        )
        return Recipe(name=name, steps=12000, per_gpu_batch=16, lr=6.0e-6, extra=extra, resume_checkpoint=resume_checkpoint)

    if take_nmi > 0.35:
        name = "v5h_4p0_take_leakage_repair_8gpu"
        extra = (
            "--vq-temperature",
            "0.055",
            "--kl-weight",
            "0.24",
            "--balance-weight",
            "0.12",
            "--hard-usage-balance-weight",
            "0.004",
            "--slot-balance-weight",
            "0.002",
            "--assignment-entropy-weight",
            "0.010",
            "--assignment-entropy-target",
            "0.82",
            "--action-slot-dropout",
            "0.32",
            "--action-slot-dropout-start-fraction",
            "0.0",
            "--action-slot-dropout-ramp-fraction",
            "0.05",
            "--action-only-weight",
            "0.22",
            "--action-contrast-weight",
            "0.12",
            "--no-private-contrast-weight",
            "0.24",
            "--action-consistency-weight",
            "0.060",
            "--exo-aux-multiplier",
            "3.6",
            *common,
        )
        return Recipe(name=name, steps=10000, per_gpu_batch=16, lr=6.0e-6, extra=extra, resume_checkpoint=resume_checkpoint)

    name = "v5i_4p0_causality_repair_8gpu"
    extra = (
        "--vq-temperature",
        "0.052",
        "--kl-weight",
        "0.28" if exo_weak else "0.22",
        "--balance-weight",
        "0.10",
        "--hard-usage-balance-weight",
        "0.0035",
        "--slot-balance-weight",
        "0.0015",
        "--assignment-entropy-weight",
        "0.010",
        "--assignment-entropy-target",
        "0.82",
        "--action-slot-dropout",
        "0.18" if ego_weak else "0.24",
        "--action-slot-dropout-start-fraction",
        "0.0",
        "--action-slot-dropout-ramp-fraction",
        "0.05",
        "--action-only-weight",
        "0.22",
        "--action-contrast-weight",
        "0.14",
        "--no-private-contrast-weight",
        "0.22",
        "--action-consistency-weight",
        "0.070" if exo_weak else "0.055",
        "--exo-aux-multiplier",
        "4.0" if exo_weak else "3.2",
        *common,
    )
    return Recipe(name=name, steps=9000, per_gpu_batch=16, lr=5.0e-6, extra=extra, resume_checkpoint=resume_checkpoint)


def train_command(args: argparse.Namespace, recipe: Recipe, run_dir: Path) -> list[str]:
    target_steps = checkpoint_step(recipe.resume_checkpoint, args.python) + 1 + recipe.steps
    cmd = [
        str(args.torchrun),
        "--standalone",
        f"--nproc_per_node={len(args.gpu_ids)}",
        "scripts/train_fact_npz_debug.py",
        "--ddp",
        "--input-npz",
        str(args.train_npz),
        "--output-dir",
        str(run_dir),
        "--source-view-keys",
        "ego",
        "exo",
        "--steps",
        str(target_steps),
        "--batch-size",
        str(recipe.per_gpu_batch),
        "--num-workers",
        str(args.train_num_workers),
        "--prefetch-factor",
        str(args.train_prefetch_factor),
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
        "4",
        "--num-latents",
        "64",
        "--num-action-slots",
        "4",
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
        str(recipe.lr),
        "--save-every",
        str(args.save_every),
        "--resume-checkpoint",
        str(recipe.resume_checkpoint),
    ]
    cmd.extend(recipe.extra)
    return cmd


def probe_command(args: argparse.Namespace, run_dir: Path, checkpoint: Path) -> list[str]:
    return [
        str(args.python),
        "scripts/probe_fact_action_tokens.py",
        "--checkpoint",
        str(checkpoint),
        "--input-npz",
        str(args.heldout_npz),
        "--output-dir",
        str(run_dir / "action_token_probe_heldout"),
        "--labels",
        str(args.heldout_labels),
        "--label-columns",
        "parent_task_name",
        "task_name",
        "university_name",
        "--source-view-keys",
        "ego",
        "exo",
        "--resize",
        "224",
        "--batch-size",
        str(args.probe_batch_size),
        "--num-workers",
        str(args.probe_num_workers),
        "--device",
        "cuda",
    ]


def gate_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    probe_dir = run_dir / "action_token_probe_heldout"
    return [
        str(args.python),
        "scripts/evaluate_fact_stage1_gate.py",
        "--probe-dir",
        str(probe_dir),
        "--output-json",
        str(probe_dir / "stage1_gate.json"),
    ]


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu_ids)
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "0")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if len(args.gpu_ids) != 8:
        print(f"warning: requested {len(args.gpu_ids)} GPUs, user preference is 8 GPUs.", flush=True)
    if not args.train_npz.exists():
        raise FileNotFoundError(args.train_npz)
    if not args.heldout_npz.exists():
        raise FileNotFoundError(args.heldout_npz)
    if not args.heldout_labels.exists():
        raise FileNotFoundError(args.heldout_labels)

    previous_gate = None
    manifest = load_json(args.manifest, [])
    if manifest:
        previous_gate = latest_gate(manifest)
    resume_checkpoint = args.base_checkpoint
    completed_checkpoint = latest_completed_checkpoint(manifest)
    if completed_checkpoint is not None:
        resume_checkpoint = completed_checkpoint
    if not resume_checkpoint.exists():
        raise FileNotFoundError(resume_checkpoint)

    if args.initial_run_dir is not None:
        initial_checkpoint = args.initial_checkpoint or args.initial_run_dir / "fact_tokenizer.ckpt"
        if not initial_checkpoint.exists():
            raise FileNotFoundError(initial_checkpoint)
        code, gate = evaluate_checkpoint(args, env=env, run_dir=args.initial_run_dir, checkpoint=initial_checkpoint)
        append_manifest(
            args,
            {
                "run_dir": str(args.initial_run_dir),
                "recipe": args.initial_run_dir.name,
                "stage": "gate",
                "returncode": code,
                "gate": gate,
            },
        )
        if code not in (0, 1):
            return code
        if gate.get("passed"):
            print(f"Stage-1 gate passed: {args.initial_run_dir}", flush=True)
            return 0
        previous_gate = gate
        resume_checkpoint = initial_checkpoint

    for round_index in range(args.max_rounds):
        recipe = recipe_for_round(round_index, previous_gate, resume_checkpoint)
        run_dir = args.output_root / f"{recipe.name}_{now_tag()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "recipe": recipe.name,
            "stage1_plan": "4.0 reliability-aware exo teacher + private separation + usage balancing",
            "steps": recipe.steps,
            "per_gpu_batch": recipe.per_gpu_batch,
            "global_batch": recipe.per_gpu_batch * len(args.gpu_ids),
            "train_num_workers": args.train_num_workers,
            "train_prefetch_factor": args.train_prefetch_factor,
            "gpus": args.gpu_ids,
            "resume_checkpoint": str(recipe.resume_checkpoint),
            "extra": list(recipe.extra),
        }
        save_json(run_dir / "optimization_recipe.json", payload)

        train_cmd = train_command(args, recipe, run_dir)
        (run_dir / "train_command.txt").write_text(
            " ".join(shlex.quote(part) for part in train_cmd) + "\n",
            encoding="utf-8",
        )
        wait_for_gpus_if_requested(args, reason=f"training {run_dir.name}")
        code = run_command(train_cmd, env=env, log_path=run_dir / "train_stdout.log", dry_run=args.dry_run)
        if code != 0 or args.dry_run:
            append_manifest(args, {"run_dir": str(run_dir), "recipe": recipe.name, "stage": "train", "returncode": code})
            return code

        checkpoint = run_dir / "fact_tokenizer.ckpt"
        if not checkpoint.exists():
            append_manifest(args, {"run_dir": str(run_dir), "recipe": recipe.name, "stage": "checkpoint_missing"})
            return 1

        code, gate = evaluate_checkpoint(args, env=env, run_dir=run_dir, checkpoint=checkpoint)
        if code not in (0, 1):
            append_manifest(args, {"run_dir": str(run_dir), "recipe": recipe.name, "stage": "probe_or_gate", "returncode": code, "gate": gate})
            return code
        append_manifest(args, {"run_dir": str(run_dir), "recipe": recipe.name, "stage": "gate", "returncode": code, "gate": gate})
        if gate.get("passed"):
            print(f"Stage-1 gate passed: {run_dir}", flush=True)
            return 0
        previous_gate = gate
        resume_checkpoint = checkpoint
        time.sleep(2)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
