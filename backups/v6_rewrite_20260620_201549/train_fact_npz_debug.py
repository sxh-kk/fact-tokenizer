#!/usr/bin/env python3
"""Train a small FACT tokenizer on paired NPZ video transitions."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
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
    parser.add_argument(
        "--discard-resume-history",
        action="store_true",
        help="Do not copy old per-step logs from a resumed checkpoint into newly saved checkpoints.",
    )
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--take-grouped-batches", action="store_true")
    parser.add_argument("--samples-per-take", type=int, default=4)
    parser.add_argument("--no-tf32", action="store_true")
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
    parser.add_argument("--private-dropout", type=float, default=0.0)
    parser.add_argument("--private-dropout-start-fraction", type=float, default=0.1)
    parser.add_argument("--private-dropout-ramp-fraction", type=float, default=0.3)
    parser.add_argument("--action-slot-dropout", type=float, default=0.0)
    parser.add_argument("--action-slot-dropout-start-fraction", type=float, default=0.15)
    parser.add_argument("--action-slot-dropout-ramp-fraction", type=float, default=0.25)
    parser.add_argument("--action-only-weight", type=float, default=0.0)
    parser.add_argument("--action-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-contrast-weight", type=float, default=0.0)
    parser.add_argument("--random-code-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-random-code-contrast-weight", type=float, default=0.0)
    parser.add_argument("--zero-action-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-zero-action-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-same-take-contrast-weight", type=float, default=0.0)
    parser.add_argument("--temporal-offset-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-temporal-offset-contrast-weight", type=float, default=0.0)
    parser.add_argument("--temporal-offset", type=int, default=4)
    parser.add_argument("--action-aware-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-action-aware-contrast-weight", type=float, default=0.0)
    parser.add_argument("--action-aware-context-weight", type=float, default=0.35)
    parser.add_argument("--action-contrast-margin", type=float, default=0.01)
    parser.add_argument("--action-aux-start-fraction", type=float, default=0.2)
    parser.add_argument("--action-aux-ramp-fraction", type=float, default=0.2)
    parser.add_argument("--action-consistency-weight", type=float, default=0.0)
    parser.add_argument("--assignment-entropy-weight", type=float, default=0.0)
    parser.add_argument("--assignment-entropy-target", type=float, default=0.0)
    parser.add_argument("--slot-balance-weight", type=float, default=0.0)
    parser.add_argument("--hard-usage-balance-weight", type=float, default=0.0)
    parser.add_argument("--usage-capacity-weight", type=float, default=0.0)
    parser.add_argument("--usage-capacity-max-fraction", type=float, default=0.07)
    parser.add_argument("--slot-diversity-weight", type=float, default=0.0)
    parser.add_argument("--motion-focus-weight", type=float, default=0.0)
    parser.add_argument("--action-only-motion-focus-weight", type=float, default=0.0)
    parser.add_argument("--motion-contrast-weight", type=float, default=0.0)
    parser.add_argument("--delta-focus-weight", type=float, default=0.0)
    parser.add_argument("--action-only-delta-focus-weight", type=float, default=0.0)
    parser.add_argument("--delta-contrast-weight", type=float, default=0.0)
    parser.add_argument("--motion-focus-gamma", type=float, default=2.0)
    parser.add_argument("--motion-focus-max-weight", type=float, default=6.0)
    parser.add_argument("--exo-aux-multiplier", type=float, default=1.0)
    parser.add_argument("--teacher-ego-uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--teacher-disagreement-weight", type=float, default=0.0)
    parser.add_argument("--teacher-base-bias", type=float, default=0.0)
    parser.add_argument("--same-take-contrast-weight", type=float, default=0.0)
    parser.add_argument("--take-uniform-weight", type=float, default=0.0)
    parser.add_argument("--take-slot-uniform-weight", type=float, default=0.0)
    parser.add_argument("--take-pair-uniform-weight", type=float, default=0.0)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    distributed = args.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not distributed:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA.")
    timeout_sec = int(os.environ.get("FACT_DDP_TIMEOUT_SEC", "3600"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_sec))
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


def scheduled_scalar(step: int, total_steps: int, target: float, start: float, ramp: float) -> float:
    if target <= 0.0:
        return 0.0
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    if progress <= start:
        return 0.0
    if ramp <= 0.0:
        return float(target)
    return float(target) * min((progress - start) / ramp, 1.0)


def dataloader_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        kwargs["persistent_workers"] = not args.no_persistent_workers
    return kwargs


class TakeGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        take_indices: torch.Tensor,
        batch_size: int,
        samples_per_take: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if samples_per_take <= 0 or batch_size % samples_per_take != 0:
            raise ValueError("samples_per_take must be positive and divide batch_size")
        self.batch_size = int(batch_size)
        self.samples_per_take = int(samples_per_take)
        self.takes_per_batch = self.batch_size // self.samples_per_take
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.batches_per_epoch = max(1, (len(take_indices) + self.batch_size * self.world_size - 1) // (self.batch_size * self.world_size))
        self.take_to_indices: dict[int, torch.Tensor] = {}
        for take in torch.unique(take_indices).tolist():
            members = torch.nonzero(take_indices == int(take), as_tuple=False).flatten()
            self.take_to_indices[int(take)] = members
        self.take_ids = torch.tensor(sorted(self.take_to_indices), dtype=torch.long)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1009 + self.rank)
        for _ in range(self.batches_per_epoch):
            selected = self.take_ids[torch.randint(len(self.take_ids), (self.takes_per_batch,), generator=generator)]
            batch: list[int] = []
            for take in selected.tolist():
                members = self.take_to_indices[int(take)]
                choices = torch.randint(len(members), (self.samples_per_take,), generator=generator)
                batch.extend(int(index) for index in members[choices].tolist())
            yield batch


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available() and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
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
    batch_sampler = None
    sampler = None
    if args.take_grouped_batches:
        batch_sampler = TakeGroupedBatchSampler(
            dataset.take_indices,
            batch_size=args.batch_size,
            samples_per_take=args.samples_per_take,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
    elif distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs(args))
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            **dataloader_kwargs(args),
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
        action_only_weight=args.action_only_weight,
        action_contrast_weight=args.action_contrast_weight,
        no_private_contrast_weight=args.no_private_contrast_weight,
        random_code_contrast_weight=args.random_code_contrast_weight,
        no_private_random_code_contrast_weight=args.no_private_random_code_contrast_weight,
        zero_action_contrast_weight=args.zero_action_contrast_weight,
        no_private_zero_action_contrast_weight=args.no_private_zero_action_contrast_weight,
        no_private_same_take_contrast_weight=args.no_private_same_take_contrast_weight,
        temporal_offset_contrast_weight=args.temporal_offset_contrast_weight,
        no_private_temporal_offset_contrast_weight=args.no_private_temporal_offset_contrast_weight,
        action_aware_contrast_weight=args.action_aware_contrast_weight,
        no_private_action_aware_contrast_weight=args.no_private_action_aware_contrast_weight,
        action_aware_context_weight=args.action_aware_context_weight,
        action_contrast_margin=args.action_contrast_margin,
        action_aux_start_fraction=args.action_aux_start_fraction,
        action_aux_ramp_fraction=args.action_aux_ramp_fraction,
        action_consistency_weight=args.action_consistency_weight,
        assignment_entropy_weight=args.assignment_entropy_weight,
        assignment_entropy_target=args.assignment_entropy_target,
        slot_balance_weight=args.slot_balance_weight,
        hard_usage_balance_weight=args.hard_usage_balance_weight,
        usage_capacity_weight=args.usage_capacity_weight,
        usage_capacity_max_fraction=args.usage_capacity_max_fraction,
        slot_diversity_weight=args.slot_diversity_weight,
        motion_focus_weight=args.motion_focus_weight,
        action_only_motion_focus_weight=args.action_only_motion_focus_weight,
        motion_contrast_weight=args.motion_contrast_weight,
        delta_focus_weight=args.delta_focus_weight,
        action_only_delta_focus_weight=args.action_only_delta_focus_weight,
        delta_contrast_weight=args.delta_contrast_weight,
        motion_focus_gamma=args.motion_focus_gamma,
        motion_focus_max_weight=args.motion_focus_max_weight,
        exo_aux_multiplier=args.exo_aux_multiplier,
        teacher_ego_uncertainty_weight=args.teacher_ego_uncertainty_weight,
        teacher_disagreement_weight=args.teacher_disagreement_weight,
        teacher_base_bias=args.teacher_base_bias,
        same_take_contrast_weight=args.same_take_contrast_weight,
        take_uniform_weight=args.take_uniform_weight,
        take_slot_uniform_weight=args.take_slot_uniform_weight,
        take_pair_uniform_weight=args.take_pair_uniform_weight,
    )
    include_action_only = (
        args.action_only_weight > 0.0
        or args.no_private_contrast_weight > 0.0
        or args.no_private_random_code_contrast_weight > 0.0
        or args.no_private_zero_action_contrast_weight > 0.0
        or args.no_private_same_take_contrast_weight > 0.0
        or args.no_private_temporal_offset_contrast_weight > 0.0
        or args.no_private_action_aware_contrast_weight > 0.0
        or args.action_only_motion_focus_weight > 0.0
        or args.action_only_delta_focus_weight > 0.0
    )
    include_action_shuffle = (
        args.action_contrast_weight > 0.0
        or args.no_private_contrast_weight > 0.0
        or args.motion_contrast_weight > 0.0
        or args.delta_contrast_weight > 0.0
    )
    include_random_code_action = (
        args.random_code_contrast_weight > 0.0
        or args.no_private_random_code_contrast_weight > 0.0
    )
    include_same_take_action_shuffle = args.same_take_contrast_weight > 0.0 or args.no_private_same_take_contrast_weight > 0.0
    include_temporal_offset_action = (
        args.temporal_offset_contrast_weight > 0.0
        or args.no_private_temporal_offset_contrast_weight > 0.0
    )
    include_action_aware_action = (
        args.action_aware_contrast_weight > 0.0
        or args.no_private_action_aware_contrast_weight > 0.0
    )
    include_zero_action = args.zero_action_contrast_weight > 0.0 or args.no_private_zero_action_contrast_weight > 0.0

    history = []
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["state_dict"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
                group["weight_decay"] = args.weight_decay
        history = [] if args.discard_resume_history else list(checkpoint.get("history", []))
        start_step = int(checkpoint.get("step", -1)) + 1
        if main_process:
            print(f"Resumed {args.resume_checkpoint} from step {start_step}", flush=True)

    model.train()
    iterator = iter(loader)
    epoch = start_step // max(1, len(loader))
    if sampler is not None:
        sampler.set_epoch(epoch)
    if batch_sampler is not None:
        batch_sampler.set_epoch(epoch)
    for step in range(start_step, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        current_private_dropout = scheduled_scalar(
            step,
            args.steps,
            args.private_dropout,
            args.private_dropout_start_fraction,
            args.private_dropout_ramp_fraction,
        )
        current_action_slot_dropout = scheduled_scalar(
            step,
            args.steps,
            args.action_slot_dropout,
            args.action_slot_dropout_start_fraction,
            args.action_slot_dropout_ramp_fraction,
        )
        outputs = model(
            batch,
            private_dropout=current_private_dropout,
            action_slot_dropout=current_action_slot_dropout,
            include_action_only=include_action_only,
            include_action_shuffle=include_action_shuffle,
            include_same_take_action_shuffle=include_same_take_action_shuffle,
            include_temporal_offset_action=include_temporal_offset_action,
            temporal_offset=args.temporal_offset,
            include_action_aware_action=include_action_aware_action,
            action_aware_context_weight=args.action_aware_context_weight,
            include_zero_action=include_zero_action,
            include_random_code_action=include_random_code_action,
        )
        loss, logs = compute_fact_loss(outputs, step=step, total_steps=args.steps, config=loss_config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite FACT loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        logs["private_dropout"] = current_private_dropout
        logs["action_slot_dropout"] = current_action_slot_dropout
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
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **dataloader_kwargs(args)):
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
