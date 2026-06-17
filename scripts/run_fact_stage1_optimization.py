#!/usr/bin/env python3
"""Run iterative 8-GPU FACT tokenizer Stage-1 optimization and heldout gates."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Recipe:
    name: str
    steps: int
    per_gpu_batch: int
    lr: float
    num_action_slots: int
    resume_checkpoint: Path | None
    extra: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--torchrun", type=Path, default=Path("/home/sxh/.conda/envs/fact_tokenizer/bin/torchrun"))
    parser.add_argument("--python", type=Path, default=Path("/home/sxh/.conda/envs/fact_tokenizer/bin/python"))
    parser.add_argument("--train-npz", type=Path, default=ROOT / "data/fact_egoexo/splits/diverse_500takes_seed123_80_20/train_by_take.npz")
    parser.add_argument("--heldout-npz", type=Path, default=ROOT / "data/fact_egoexo/splits/diverse_500takes_seed123_80_20/heldout_by_take.npz")
    parser.add_argument("--heldout-labels", type=Path, default=ROOT / "data/fact_egoexo/splits/diverse_500takes_seed123_80_20/heldout_labels.jsonl")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/fact_tokenizer/egoexo_diverse_500takes_k64_35k_v4d_gentle_usage_from_v2_20260616_215152/fact_tokenizer.ckpt",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/fact_tokenizer")
    parser.add_argument("--max-recipes", type=int, default=3)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--probe-num-workers", type=int, default=4)
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing-probe", action="store_true")
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: Sequence[str], env: dict[str, str], log_path: Path | None = None, dry_run: bool = False) -> int:
    text = " ".join(shlex.quote(part) for part in command)
    print(text, flush=True)
    if dry_run:
        return 0
    if log_path is None:
        return subprocess.run(command, cwd=ROOT, env=env).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(text + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def load_gate(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_step(path: Path, python: Path) -> int:
    code = (
        "import torch, sys; "
        "ckpt=torch.load(sys.argv[1], map_location='cpu'); "
        "print(int(ckpt.get('step', -1)))"
    )
    output = subprocess.check_output([str(python), "-c", code, str(path)], text=True)
    return int(output.strip().splitlines()[-1])


def recipes(args: argparse.Namespace) -> list[Recipe]:
    base = args.base_checkpoint
    return [
        Recipe(
            name="v5c_v4d_finetune_contrast_8gpu",
            steps=12000,
            per_gpu_batch=8,
            lr=1.0e-5,
            num_action_slots=4,
            resume_checkpoint=base,
            extra=(
                "--vq-temperature",
                "0.045",
                "--vq-beta",
                "0.40",
                "--balance-weight",
                "0.08",
                "--private-reg-weight",
                "0.04",
                "--private-dropout",
                "0.50",
                "--action-only-weight",
                "0.14",
                "--action-contrast-weight",
                "0.10",
                "--no-private-contrast-weight",
                "0.15",
                "--action-contrast-margin",
                "0.006",
                "--action-consistency-weight",
                "0.025",
                "--assignment-entropy-weight",
                "0.03",
                "--assignment-entropy-target",
                "0.72",
                "--hard-usage-balance-weight",
                "0.002",
                "--slot-diversity-weight",
                "0.02",
                "--motion-focus-weight",
                "0.04",
                "--action-only-motion-focus-weight",
                "0.04",
                "--motion-contrast-weight",
                "0.04",
                "--delta-focus-weight",
                "0.02",
                "--action-only-delta-focus-weight",
                "0.02",
                "--delta-contrast-weight",
                "0.02",
                "--exo-aux-multiplier",
                "2.5",
                "--action-aux-start-fraction",
                "0.10",
                "--action-aux-ramp-fraction",
                "0.20",
            ),
        ),
        Recipe(
            name="v5d_v4d_finetune_slot_dropout_8gpu",
            steps=12000,
            per_gpu_batch=8,
            lr=8.0e-6,
            num_action_slots=4,
            resume_checkpoint=base,
            extra=(
                "--vq-temperature",
                "0.045",
                "--vq-beta",
                "0.40",
                "--balance-weight",
                "0.08",
                "--private-reg-weight",
                "0.04",
                "--private-dropout",
                "0.50",
                "--action-slot-dropout",
                "0.25",
                "--action-slot-dropout-start-fraction",
                "0.10",
                "--action-slot-dropout-ramp-fraction",
                "0.20",
                "--action-only-weight",
                "0.16",
                "--action-contrast-weight",
                "0.10",
                "--no-private-contrast-weight",
                "0.16",
                "--action-contrast-margin",
                "0.006",
                "--action-consistency-weight",
                "0.02",
                "--assignment-entropy-weight",
                "0.025",
                "--assignment-entropy-target",
                "0.72",
                "--hard-usage-balance-weight",
                "0.002",
                "--slot-diversity-weight",
                "0.01",
                "--motion-focus-weight",
                "0.04",
                "--action-only-motion-focus-weight",
                "0.04",
                "--motion-contrast-weight",
                "0.04",
                "--delta-focus-weight",
                "0.02",
                "--action-only-delta-focus-weight",
                "0.02",
                "--delta-contrast-weight",
                "0.02",
                "--exo-aux-multiplier",
                "2.5",
                "--action-aux-start-fraction",
                "0.10",
                "--action-aux-ramp-fraction",
                "0.20",
            ),
        ),
        Recipe(
            name="v5e_slots2_capacity_check_8gpu",
            steps=20000,
            per_gpu_batch=8,
            lr=5.0e-5,
            num_action_slots=2,
            resume_checkpoint=None,
            extra=(
                "--vq-temperature",
                "0.06",
                "--vq-beta",
                "0.35",
                "--balance-weight",
                "0.08",
                "--private-reg-weight",
                "0.04",
                "--private-dropout",
                "0.50",
                "--action-only-weight",
                "0.18",
                "--action-contrast-weight",
                "0.08",
                "--no-private-contrast-weight",
                "0.18",
                "--action-contrast-margin",
                "0.006",
                "--action-consistency-weight",
                "0.025",
                "--assignment-entropy-weight",
                "0.03",
                "--assignment-entropy-target",
                "0.72",
                "--hard-usage-balance-weight",
                "0.002",
                "--motion-focus-weight",
                "0.04",
                "--action-only-motion-focus-weight",
                "0.04",
                "--motion-contrast-weight",
                "0.04",
                "--delta-focus-weight",
                "0.02",
                "--action-only-delta-focus-weight",
                "0.02",
                "--delta-contrast-weight",
                "0.02",
                "--exo-aux-multiplier",
                "2.5",
                "--action-aux-start-fraction",
                "0.15",
                "--action-aux-ramp-fraction",
                "0.25",
            ),
        ),
    ]


def train_command(args: argparse.Namespace, recipe: Recipe, run_dir: Path) -> list[str]:
    target_steps = recipe.steps
    if recipe.resume_checkpoint is not None:
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
        str(recipe.num_action_slots),
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
    ]
    if recipe.resume_checkpoint is not None:
        cmd.extend(["--resume-checkpoint", str(recipe.resume_checkpoint)])
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

    manifest = []
    selected = recipes(args)[: args.max_recipes]
    for recipe in selected:
        run_dir = args.output_root / f"{recipe.name}_{now_tag()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        recipe_payload = {
            "recipe": recipe.name,
            "steps": recipe.steps,
            "per_gpu_batch": recipe.per_gpu_batch,
            "gpus": args.gpu_ids,
            "train_npz": str(args.train_npz),
            "heldout_npz": str(args.heldout_npz),
            "heldout_labels": str(args.heldout_labels),
            "resume_checkpoint": str(recipe.resume_checkpoint) if recipe.resume_checkpoint else None,
            "extra": list(recipe.extra),
        }
        (run_dir / "optimization_recipe.json").write_text(json.dumps(recipe_payload, indent=2), encoding="utf-8")

        train_cmd = train_command(args, recipe, run_dir)
        (run_dir / "train_command.txt").write_text(" ".join(shlex.quote(part) for part in train_cmd) + "\n", encoding="utf-8")
        code = run_command(train_cmd, env=env, log_path=run_dir / "train_stdout.log", dry_run=args.dry_run)
        if args.dry_run:
            continue
        if code != 0:
            manifest.append({"run_dir": str(run_dir), "recipe": recipe.name, "stage": "train", "returncode": code})
            break

        checkpoint = run_dir / "fact_tokenizer.ckpt"
        if not checkpoint.exists() and not args.dry_run:
            manifest.append({"run_dir": str(run_dir), "recipe": recipe.name, "stage": "checkpoint_missing"})
            break

        probe_dir = run_dir / "action_token_probe_heldout"
        if not (args.skip_existing_probe and (probe_dir / "probe_summary.json").exists()):
            code = run_command(probe_command(args, run_dir, checkpoint), env=env, log_path=run_dir / "probe_heldout.log", dry_run=args.dry_run)
            if code != 0:
                manifest.append({"run_dir": str(run_dir), "recipe": recipe.name, "stage": "probe", "returncode": code})
                break

        code = run_command(gate_command(args, run_dir), env=env, log_path=run_dir / "stage1_gate.log", dry_run=args.dry_run)
        gate_path = probe_dir / "stage1_gate.json"
        gate = load_gate(gate_path) if gate_path.exists() else {"passed": False, "missing": str(gate_path)}
        manifest.append({"run_dir": str(run_dir), "recipe": recipe.name, "stage": "gate", "returncode": code, "gate": gate})
        (args.output_root / "stage1_optimization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if gate.get("passed"):
            print(f"Stage-1 gate passed: {run_dir}", flush=True)
            return 0
        time.sleep(2)

    (args.output_root / "stage1_optimization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
