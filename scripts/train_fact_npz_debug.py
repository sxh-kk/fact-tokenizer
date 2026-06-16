#!/usr/bin/env python3
"""Train a small FACT tokenizer on paired NPZ video transitions."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fact_tokenizer import FACTLossConfig, FACTPairedNPZDataset, FACTTokenizer, compute_fact_loss
from fact_tokenizer.utils import code_usage, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=ROOT / "outputs" / "lam_tokenizer" / "dummy_multiview.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fact_tokenizer" / "npz_debug_train")
    parser.add_argument("--source-view-keys", nargs="*", default=None)
    parser.add_argument("--view-names", nargs=2, default=["ego", "exo"])
    parser.add_argument("--videos-layout", default="VBTCHW")
    parser.add_argument("--frame-pair", choices=["first-last", "first-next"], default="first-last")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", choices=["mock", "dino"], default="mock")
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--dino-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--private-dim", type=int, default=16)
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument("--num-action-slots", type=int, default=4)
    parser.add_argument("--num-private-slots", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--enc-blocks", type=int, default=1)
    parser.add_argument("--dec-blocks", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--vq-temperature", type=float, default=0.1)
    parser.add_argument("--vq-beta", type=float, default=0.25)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--private-reg-weight", type=float, default=0.001)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    distributed = args.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not distributed:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, local_rank, rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(distributed: bool, rank: int) -> bool:
    return (not distributed) or rank == 0


def unwrap_model(model: torch.nn.Module) -> FACTTokenizer:
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


def make_model_config(args: argparse.Namespace) -> dict:
    return {
        "image_channels": 3,
        "model_dim": args.model_dim,
        "dino_dim": args.dino_dim,
        "latent_dim": args.latent_dim,
        "private_dim": args.private_dim,
        "num_latents": args.num_latents,
        "num_action_slots": args.num_action_slots,
        "num_private_slots": args.num_private_slots,
        "patch_size": args.patch_size,
        "enc_blocks": args.enc_blocks,
        "dec_blocks": args.dec_blocks,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "vq_temperature": args.vq_temperature,
        "backbone": args.backbone,
        "torch_home": str(ROOT / "checkpoints" / "torch_hub"),
        "view_names": tuple(args.view_names),
        "max_time": 8,
        "max_tokens": 1024,
    }


def main() -> None:
    args = parse_args()
    distributed, local_rank, rank, world_size = setup_distributed(args)
    main_process = is_main_process(distributed, rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    dataset = FACTPairedNPZDataset(
        input_npz=args.input_npz,
        source_view_keys=args.source_view_keys,
        output_view_names=args.view_names,
        videos_layout=args.videos_layout,
        frame_pair=args.frame_pair,
        start_index=args.start_index,
        resize=args.resize,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    model_config = make_model_config(args)
    model = FACTTokenizer(**model_config).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if args.data_parallel:
        if distributed:
            raise RuntimeError("--data-parallel and --ddp cannot be used together.")
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices.")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_config = FACTLossConfig(
        vq_beta=args.vq_beta,
        kl_weight=args.kl_weight,
        balance_weight=args.balance_weight,
        private_reg_weight=args.private_reg_weight,
    )

    history = []
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["state_dict"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        start_step = int(checkpoint.get("step", -1)) + 1
        if main_process:
            print(f"Resumed {args.resume_checkpoint} from step {start_step}", flush=True)

    model.train()
    iterator = iter(loader)
    epoch = start_step // max(1, len(loader))
    if sampler is not None:
        sampler.set_epoch(epoch)
    for step in range(start_step, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, logs = compute_fact_loss(outputs, step=step, total_steps=args.steps, config=loss_config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite FACT loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        row = {"step": step, **logs}
        if main_process:
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if main_process and args.save_every and (step + 1) % args.save_every == 0:
            model_to_save = unwrap_model(model)
            torch.save(
                {
                    "state_dict": model_to_save.state_dict(),
                    "model_config": model_config,
                    "loss_config": loss_config.__dict__,
                    "source_view_keys": dataset.source_view_keys,
                    "view_names": args.view_names,
                    "history": history,
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                },
                args.output_dir / f"fact_tokenizer_step_{step + 1:06d}.ckpt",
            )

    if distributed:
        dist.barrier()
    if not main_process:
        cleanup_distributed(distributed)
        return

    model_to_save = unwrap_model(model)
    model_to_save.eval()
    all_indices = []
    all_confidence = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            encoded = model_to_save.encode_shared_action(batch, view_name=args.view_names[0])
            all_indices.append(encoded["indices"].cpu())
            all_confidence.append(encoded["confidence"].cpu())
    indices = torch.cat(all_indices, dim=0)
    confidence = torch.cat(all_confidence, dim=0)

    checkpoint_path = args.output_dir / "fact_tokenizer.ckpt"
    torch.save(
        {
            "state_dict": model_to_save.state_dict(),
            "model_config": model_config,
            "loss_config": loss_config.__dict__,
            "source_view_keys": dataset.source_view_keys,
            "view_names": args.view_names,
            "history": history,
            "optimizer_state": optimizer.state_dict(),
            "step": args.steps - 1,
        },
        checkpoint_path,
    )
    save_json(args.output_dir / "train_history.json", history)
    save_json(args.output_dir / "code_usage.json", code_usage(indices, args.num_latents))
    save_json(
        args.output_dir / "metadata.json",
        {
            "checkpoint": str(checkpoint_path),
            "input_npz": str(args.input_npz),
            "source_view_keys": dataset.source_view_keys,
            "view_names": args.view_names,
            "token_shape": list(indices.shape),
            "confidence_mean": float(confidence.mean()),
        },
    )
    print(f"Saved FACT tokenizer checkpoint to {checkpoint_path}")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
