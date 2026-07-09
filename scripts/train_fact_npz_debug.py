#!/usr/bin/env python3
"""Train a small FACT tokenizer on paired NPZ video transitions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional

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
    parser.add_argument("--take-weight-csv", type=Path, default=None)
    parser.add_argument(
        "--take-weight-column",
        default="sample_weight",
        help="Column in --take-weight-csv used for take-level sampling weights.",
    )
    parser.add_argument("--take-weight-uid-column", default="take_uid")
    parser.add_argument("--min-take-sampling-weight", type=float, default=0.0)
    parser.add_argument("--transition-weight-csv", type=Path, default=None)
    parser.add_argument(
        "--transition-weight-column",
        default="transition_weight",
        help="Column in --transition-weight-csv used for per-transition sampling weights.",
    )
    parser.add_argument("--transition-weight-sample-id-column", default="sample_id")
    parser.add_argument("--transition-weight-row-index-column", default="row_index")
    parser.add_argument("--transition-weight-take-uid-column", default="take_uid")
    parser.add_argument("--transition-weight-timestamp-column", default="timestamp")
    parser.add_argument("--transition-weight-timestamp-tolerance", type=float, default=1e-3)
    parser.add_argument("--min-transition-sampling-weight", type=float, default=0.0)
    parser.add_argument("--light-augment", action="store_true")
    parser.add_argument("--mined-negative-map", type=Path, default=None)
    parser.add_argument("--mined-negative-top-k", type=int, default=4)
    parser.add_argument("--mined-pair-batches", action="store_true")
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
    parser.add_argument("--current-context-mode", choices=["full", "bottleneck", "drop"], default="full")
    parser.add_argument("--current-context-tokens", type=int, default=0)
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
    parser.add_argument("--mined-same-take-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-mined-same-take-contrast-weight", type=float, default=0.0)
    parser.add_argument("--action-contrast-margin", type=float, default=0.01)
    parser.add_argument("--action-aux-start-fraction", type=float, default=0.2)
    parser.add_argument("--action-aux-ramp-fraction", type=float, default=0.2)
    parser.add_argument("--action-consistency-weight", type=float, default=0.0)
    parser.add_argument("--assignment-entropy-weight", type=float, default=0.0)
    parser.add_argument("--assignment-entropy-target", type=float, default=0.0)
    parser.add_argument("--slot-balance-weight", type=float, default=0.0)
    parser.add_argument("--hard-usage-balance-weight", type=float, default=0.0)
    parser.add_argument("--hard-usage-entropy-weight", type=float, default=0.0)
    parser.add_argument("--hard-usage-entropy-target-fraction", type=float, default=0.75)
    parser.add_argument("--usage-capacity-weight", type=float, default=0.0)
    parser.add_argument("--usage-capacity-max-fraction", type=float, default=0.07)
    parser.add_argument("--slot-diversity-weight", type=float, default=0.0)
    parser.add_argument("--motion-focus-weight", type=float, default=0.0)
    parser.add_argument("--action-only-motion-focus-weight", type=float, default=0.0)
    parser.add_argument("--motion-contrast-weight", type=float, default=0.0)
    parser.add_argument("--delta-focus-weight", type=float, default=0.0)
    parser.add_argument("--action-only-delta-focus-weight", type=float, default=0.0)
    parser.add_argument("--delta-contrast-weight", type=float, default=0.0)
    parser.add_argument("--no-private-delta-contrast-weight", type=float, default=0.0)
    parser.add_argument("--delta-direction-magnitude-weight", type=float, default=0.25)
    parser.add_argument("--motion-gated-usage-weight", type=float, default=0.0)
    parser.add_argument("--motion-gated-usage-gamma", type=float, default=2.0)
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
        "current_context_mode": args.current_context_mode,
        "current_context_tokens": args.current_context_tokens,
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
        take_sampling_weights: Optional[torch.Tensor] = None,
        sample_sampling_weights: Optional[torch.Tensor] = None,
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
        self.take_to_member_weights: dict[int, Optional[torch.Tensor]] = {}
        for take in torch.unique(take_indices).tolist():
            members = torch.nonzero(take_indices == int(take), as_tuple=False).flatten()
            self.take_to_indices[int(take)] = members
            if sample_sampling_weights is not None:
                weights = sample_sampling_weights[members].float().clamp_min(0.0)
                if float(weights.sum().item()) > 0.0:
                    self.take_to_member_weights[int(take)] = weights / weights.sum()
                else:
                    self.take_to_member_weights[int(take)] = None
        self.take_ids = torch.tensor(sorted(self.take_to_indices), dtype=torch.long)
        if take_sampling_weights is None:
            self.take_sampling_weights = None
        else:
            weights = take_sampling_weights[self.take_ids].float().clamp_min(0.0)
            if float(weights.sum()) <= 0.0:
                raise ValueError("take_sampling_weights must contain at least one positive weight")
            self.take_sampling_weights = weights / weights.sum()

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1009 + self.rank)
        for _ in range(self.batches_per_epoch):
            if self.take_sampling_weights is None:
                selected = self.take_ids[torch.randint(len(self.take_ids), (self.takes_per_batch,), generator=generator)]
            else:
                selected_offsets = torch.multinomial(
                    self.take_sampling_weights,
                    self.takes_per_batch,
                    replacement=True,
                    generator=generator,
                )
                selected = self.take_ids[selected_offsets]
            batch: list[int] = []
            for take in selected.tolist():
                members = self.take_to_indices[int(take)]
                member_weights = self.take_to_member_weights.get(int(take))
                if member_weights is None:
                    choices = torch.randint(len(members), (self.samples_per_take,), generator=generator)
                else:
                    choices = torch.multinomial(
                        member_weights,
                        self.samples_per_take,
                        replacement=True,
                        generator=generator,
                    )
                batch.extend(int(index) for index in members[choices].tolist())
            yield batch


class MinedPairBatchSampler(Sampler[list[int]]):
    """Yield anchor/donor pairs so mined negatives are always in-batch."""

    def __init__(
        self,
        donor_indices: torch.Tensor,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        top_k: int,
    ) -> None:
        if batch_size <= 0 or batch_size % 2 != 0:
            raise ValueError("--mined-pair-batches requires an even positive --batch-size")
        if donor_indices.ndim != 2:
            raise ValueError(f"donor_indices must be 2D, got shape {tuple(donor_indices.shape)}")
        self.donor_indices = donor_indices.long()
        self.num_samples = int(donor_indices.shape[0])
        self.top_k = max(1, min(int(top_k), int(donor_indices.shape[1])))
        self.batch_size = int(batch_size)
        self.anchors_per_batch = self.batch_size // 2
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.batches_per_epoch = max(
            1,
            (self.num_samples + self.anchors_per_batch * self.world_size - 1)
            // (self.anchors_per_batch * self.world_size),
        )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        anchor_generator = torch.Generator().manual_seed(self.seed + self.epoch * 1009)
        donor_generator = torch.Generator().manual_seed(self.seed + self.epoch * 2003 + self.rank)
        total_anchors = self.batches_per_epoch * self.anchors_per_batch * self.world_size
        anchors = torch.randperm(self.num_samples, generator=anchor_generator)
        if anchors.numel() < total_anchors:
            extra = torch.randint(
                self.num_samples,
                (total_anchors - anchors.numel(),),
                generator=anchor_generator,
            )
            anchors = torch.cat([anchors, extra], dim=0)
        anchors = anchors[:total_anchors].reshape(self.batches_per_epoch, self.world_size, self.anchors_per_batch)
        rank_anchors = anchors[:, self.rank]
        for batch_anchors in rank_anchors:
            batch: list[int] = []
            for anchor_tensor in batch_anchors:
                anchor = int(anchor_tensor.item())
                candidates = self.donor_indices[anchor, : self.top_k]
                valid = candidates[(candidates >= 0) & (candidates != anchor)]
                if valid.numel() == 0:
                    donor = anchor
                else:
                    donor = int(valid[torch.randint(valid.numel(), (1,), generator=donor_generator)].item())
                batch.extend([anchor, donor])
            yield batch


def load_mined_negative_map(path: Optional[Path], expected_len: int, top_k: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if path is None:
        return None, None
    with np.load(path, allow_pickle=False) as data:
        if "donor_index_topk" not in data:
            raise KeyError(f"{path} is missing donor_index_topk")
        donor_indices = np.asarray(data["donor_index_topk"], dtype=np.int64)
        donor_scores = np.asarray(data["donor_score_topk"], dtype=np.float32) if "donor_score_topk" in data else None
    if donor_indices.ndim != 2:
        raise ValueError(f"donor_index_topk must be 2D, got shape {donor_indices.shape}")
    if donor_indices.shape[0] != expected_len:
        raise ValueError(f"donor_index_topk has {donor_indices.shape[0]} rows, expected {expected_len}")
    if top_k <= 0 or top_k > donor_indices.shape[1]:
        raise ValueError(f"--mined-negative-top-k must be in [1, {donor_indices.shape[1]}], got {top_k}")
    if donor_scores is None:
        donor_scores = np.ones_like(donor_indices, dtype=np.float32)
    if donor_scores.shape != donor_indices.shape:
        raise ValueError(f"donor_score_topk shape {donor_scores.shape} does not match {donor_indices.shape}")
    return torch.from_numpy(donor_indices), torch.from_numpy(donor_scores)


def load_take_sampling_weights(
    path: Optional[Path],
    take_uids: list[str],
    take_indices: torch.Tensor,
    weight_column: str,
    uid_column: str,
    min_weight: float,
) -> tuple[Optional[torch.Tensor], dict]:
    if path is None:
        return None, {}
    weights_by_uid: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = [name for name in (uid_column, weight_column) if name not in reader.fieldnames]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")
        for row in reader:
            uid = str(row.get(uid_column, "")).strip()
            if not uid:
                continue
            try:
                weight = float(row.get(weight_column, "0") or 0.0)
            except ValueError:
                weight = 0.0
            weights_by_uid[uid] = max(float(min_weight), weight)
    num_takes = int(take_indices.max().item()) + 1 if take_indices.numel() else 0
    take_weights = torch.zeros(num_takes, dtype=torch.float32)
    take_index_by_uid: dict[str, int] = {}
    for uid, take_index in zip(take_uids, take_indices.tolist()):
        take_index_by_uid.setdefault(str(uid), int(take_index))
    matched = 0
    for uid, take_index in take_index_by_uid.items():
        if uid in weights_by_uid:
            take_weights[int(take_index)] = float(weights_by_uid[uid])
            matched += 1
    if matched == 0:
        raise ValueError(f"No take_uid values from dataset matched {path}")
    positive = int((take_weights > 0).sum().item())
    if positive == 0:
        raise ValueError(f"{path} produced zero positive take sampling weights")
    return take_weights, {
        "path": str(path),
        "weight_column": weight_column,
        "uid_column": uid_column,
        "matched_takes": matched,
        "positive_takes": positive,
        "total_takes": len(take_index_by_uid),
        "mean_positive_weight": float(take_weights[take_weights > 0].mean().item()),
    }


def _metadata_key(take_uid: str, timestamp: float, tolerance: float) -> tuple[str, int]:
    if tolerance <= 0:
        return (str(take_uid), int(round(float(timestamp) * 1_000_000)))
    return (str(take_uid), int(round(float(timestamp) / float(tolerance))))


def load_transition_sampling_weights(
    path: Optional[Path],
    sample_ids: list[str],
    take_uids: list[str],
    timestamps: torch.Tensor,
    weight_column: str,
    sample_id_column: str,
    row_index_column: str,
    take_uid_column: str,
    timestamp_column: str,
    timestamp_tolerance: float,
    min_weight: float,
) -> tuple[Optional[torch.Tensor], dict]:
    if path is None:
        return None, {}
    num_samples = len(sample_ids)
    weights = torch.zeros(num_samples, dtype=torch.float32)
    sample_id_to_index = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
    metadata_to_index = {
        _metadata_key(take_uid, float(timestamp), timestamp_tolerance): index
        for index, (take_uid, timestamp) in enumerate(zip(take_uids, timestamps.tolist()))
    }
    matched = 0
    duplicate_matches = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        if weight_column not in reader.fieldnames:
            raise KeyError(f"{path} is missing transition weight column {weight_column!r}")
        for row in reader:
            try:
                weight = float(row.get(weight_column, "0") or 0.0)
            except ValueError:
                weight = 0.0
            weight = max(float(min_weight), weight)
            index: Optional[int] = None
            sample_id = str(row.get(sample_id_column, "")).strip() if sample_id_column in reader.fieldnames else ""
            if sample_id:
                index = sample_id_to_index.get(sample_id)
            if index is None and row_index_column in reader.fieldnames:
                row_index = str(row.get(row_index_column, "")).strip()
                if row_index:
                    try:
                        candidate = int(row_index)
                    except ValueError:
                        candidate = -1
                    if 0 <= candidate < num_samples:
                        index = candidate
            if index is None and take_uid_column in reader.fieldnames and timestamp_column in reader.fieldnames:
                take_uid = str(row.get(take_uid_column, "")).strip()
                timestamp_text = str(row.get(timestamp_column, "")).strip()
                if take_uid and timestamp_text:
                    try:
                        key = _metadata_key(take_uid, float(timestamp_text), timestamp_tolerance)
                    except ValueError:
                        key = ("", 0)
                    index = metadata_to_index.get(key)
            if index is None:
                continue
            if float(weights[index].item()) > 0.0:
                duplicate_matches += 1
            weights[index] = float(weight)
            matched += 1
    if matched == 0:
        raise ValueError(f"No transition rows from dataset matched {path}")
    positive = int((weights > 0).sum().item())
    if positive == 0:
        raise ValueError(f"{path} produced zero positive transition sampling weights")
    return weights, {
        "path": str(path),
        "weight_column": weight_column,
        "sample_id_column": sample_id_column,
        "row_index_column": row_index_column,
        "take_uid_column": take_uid_column,
        "timestamp_column": timestamp_column,
        "timestamp_tolerance": float(timestamp_tolerance),
        "matched_rows": int(matched),
        "duplicate_matches": int(duplicate_matches),
        "positive_transitions": positive,
        "total_transitions": num_samples,
        "mean_positive_weight": float(weights[weights > 0].mean().item()),
    }


def derive_take_weights_from_transition_weights(
    take_indices: torch.Tensor,
    transition_weights: torch.Tensor,
) -> torch.Tensor:
    num_takes = int(take_indices.max().item()) + 1 if take_indices.numel() else 0
    take_weights = torch.zeros(num_takes, dtype=torch.float32)
    take_weights.scatter_add_(0, take_indices.long(), transition_weights.float().clamp_min(0.0))
    return take_weights


def attach_mined_negative_batch_fields(
    batch: Dict[str, Dict[str, torch.Tensor]],
    view_names: list[str],
    donor_indices: Optional[torch.Tensor],
    donor_scores: Optional[torch.Tensor],
    top_k: int,
) -> None:
    if donor_indices is None or donor_scores is None:
        return
    reference = batch[view_names[0]]["sample_id"].long().cpu()
    global_to_local = {int(sample_id): local for local, sample_id in enumerate(reference.tolist())}
    local_indices = torch.full((reference.numel(),), -1, dtype=torch.long)
    local_weights = torch.zeros((reference.numel(),), dtype=torch.float32)
    for local, sample_id in enumerate(reference.tolist()):
        sample_id = int(sample_id)
        candidates = donor_indices[sample_id, :top_k]
        scores = donor_scores[sample_id, :top_k]
        for donor, score in zip(candidates.tolist(), scores.tolist()):
            donor = int(donor)
            if donor == sample_id or donor not in global_to_local:
                continue
            local_indices[local] = int(global_to_local[donor])
            local_weights[local] = max(float(score), 0.0)
            break
    valid_weights = local_weights[local_weights > 0.0]
    if valid_weights.numel() > 0:
        local_weights = local_weights / valid_weights.mean().clamp_min(1e-6)
    for view_name in view_names:
        batch[view_name]["mined_negative_index"] = local_indices
        batch[view_name]["mined_negative_weight"] = local_weights


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
        augment=args.light_augment,
        augment_seed=args.seed + rank * 100003,
    )
    take_sampling_weights, take_weight_metadata = load_take_sampling_weights(
        args.take_weight_csv,
        dataset.take_uids,
        dataset.take_indices,
        args.take_weight_column,
        args.take_weight_uid_column,
        args.min_take_sampling_weight,
    )
    transition_sampling_weights, transition_weight_metadata = load_transition_sampling_weights(
        args.transition_weight_csv,
        dataset.sample_ids,
        dataset.take_uids,
        dataset.timestamps,
        args.transition_weight_column,
        args.transition_weight_sample_id_column,
        args.transition_weight_row_index_column,
        args.transition_weight_take_uid_column,
        args.transition_weight_timestamp_column,
        args.transition_weight_timestamp_tolerance,
        args.min_transition_sampling_weight,
    )
    if transition_sampling_weights is not None:
        positive_take_weights = derive_take_weights_from_transition_weights(dataset.take_indices, transition_sampling_weights)
        if take_sampling_weights is None:
            take_sampling_weights = positive_take_weights
            take_weight_metadata = {
                "derived_from_transition_weight_csv": str(args.transition_weight_csv),
                "positive_takes": int((take_sampling_weights > 0).sum().item()),
                "total_takes": int(take_sampling_weights.numel()),
                "mean_positive_weight": float(take_sampling_weights[take_sampling_weights > 0].mean().item()),
            }
        else:
            take_sampling_weights = take_sampling_weights.float().clone()
            take_sampling_weights[positive_take_weights <= 0] = 0.0
            if int((take_sampling_weights > 0).sum().item()) == 0:
                raise ValueError("Combining take and transition weights left zero positive takes")
            take_weight_metadata = dict(take_weight_metadata)
            take_weight_metadata["zeroed_takes_without_positive_transitions"] = int((positive_take_weights <= 0).sum().item())
    mined_donor_indices, mined_donor_scores = load_mined_negative_map(
        args.mined_negative_map,
        expected_len=len(dataset),
        top_k=args.mined_negative_top_k,
    )
    if args.mined_pair_batches and mined_donor_indices is None:
        raise ValueError("--mined-pair-batches requires --mined-negative-map")
    if args.mined_pair_batches and args.take_grouped_batches:
        raise ValueError("--mined-pair-batches and --take-grouped-batches are mutually exclusive")
    if args.take_weight_csv is not None and not args.take_grouped_batches:
        raise ValueError("--take-weight-csv currently requires --take-grouped-batches")
    if args.transition_weight_csv is not None and not args.take_grouped_batches:
        raise ValueError("--transition-weight-csv currently requires --take-grouped-batches")
    batch_sampler = None
    sampler = None
    if args.mined_pair_batches:
        batch_sampler = MinedPairBatchSampler(
            mined_donor_indices,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            top_k=args.mined_negative_top_k,
        )
    elif args.take_grouped_batches:
        batch_sampler = TakeGroupedBatchSampler(
            dataset.take_indices,
            batch_size=args.batch_size,
            samples_per_take=args.samples_per_take,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            take_sampling_weights=take_sampling_weights,
            sample_sampling_weights=transition_sampling_weights,
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
        mined_same_take_contrast_weight=args.mined_same_take_contrast_weight,
        no_private_mined_same_take_contrast_weight=args.no_private_mined_same_take_contrast_weight,
        action_contrast_margin=args.action_contrast_margin,
        action_aux_start_fraction=args.action_aux_start_fraction,
        action_aux_ramp_fraction=args.action_aux_ramp_fraction,
        action_consistency_weight=args.action_consistency_weight,
        assignment_entropy_weight=args.assignment_entropy_weight,
        assignment_entropy_target=args.assignment_entropy_target,
        slot_balance_weight=args.slot_balance_weight,
        hard_usage_balance_weight=args.hard_usage_balance_weight,
        hard_usage_entropy_weight=args.hard_usage_entropy_weight,
        hard_usage_entropy_target_fraction=args.hard_usage_entropy_target_fraction,
        usage_capacity_weight=args.usage_capacity_weight,
        usage_capacity_max_fraction=args.usage_capacity_max_fraction,
        slot_diversity_weight=args.slot_diversity_weight,
        motion_focus_weight=args.motion_focus_weight,
        action_only_motion_focus_weight=args.action_only_motion_focus_weight,
        motion_contrast_weight=args.motion_contrast_weight,
        delta_focus_weight=args.delta_focus_weight,
        action_only_delta_focus_weight=args.action_only_delta_focus_weight,
        delta_contrast_weight=args.delta_contrast_weight,
        no_private_delta_contrast_weight=args.no_private_delta_contrast_weight,
        delta_direction_magnitude_weight=args.delta_direction_magnitude_weight,
        motion_gated_usage_weight=args.motion_gated_usage_weight,
        motion_gated_usage_gamma=args.motion_gated_usage_gamma,
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
        or args.no_private_mined_same_take_contrast_weight > 0.0
        or args.no_private_delta_contrast_weight > 0.0
        or args.action_only_motion_focus_weight > 0.0
        or args.action_only_delta_focus_weight > 0.0
    )
    include_action_shuffle = (
        args.action_contrast_weight > 0.0
        or args.no_private_contrast_weight > 0.0
        or args.motion_contrast_weight > 0.0
        or args.delta_contrast_weight > 0.0
        or args.no_private_delta_contrast_weight > 0.0
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
    include_mined_same_take_action = (
        args.mined_same_take_contrast_weight > 0.0
        or args.no_private_mined_same_take_contrast_weight > 0.0
    )
    if include_mined_same_take_action and mined_donor_indices is None:
        raise ValueError("Mined same-take contrast weights require --mined-negative-map")
    include_zero_action = args.zero_action_contrast_weight > 0.0 or args.no_private_zero_action_contrast_weight > 0.0

    history = []
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        incompatible = unwrap_model(model).load_state_dict(checkpoint["state_dict"], strict=False)
        if main_process and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(
                "Checkpoint loaded with non-strict state dict. "
                f"missing_keys={incompatible.missing_keys} "
                f"unexpected_keys={incompatible.unexpected_keys}",
                flush=True,
            )
        if "optimizer_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                if main_process:
                    print(f"Skipped optimizer state from checkpoint: {exc}", flush=True)
            else:
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
        attach_mined_negative_batch_fields(
            batch,
            args.view_names,
            mined_donor_indices,
            mined_donor_scores,
            args.mined_negative_top_k,
        )
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
            include_mined_same_take_action=include_mined_same_take_action,
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
                    "mined_negative_map": str(args.mined_negative_map) if args.mined_negative_map else None,
                    "take_weight_csv": str(args.take_weight_csv) if args.take_weight_csv else None,
                    "transition_weight_csv": str(args.transition_weight_csv) if args.transition_weight_csv else None,
                    "light_augment": bool(args.light_augment),
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
            "mined_negative_map": str(args.mined_negative_map) if args.mined_negative_map else None,
            "take_weight_csv": str(args.take_weight_csv) if args.take_weight_csv else None,
            "transition_weight_csv": str(args.transition_weight_csv) if args.transition_weight_csv else None,
            "light_augment": bool(args.light_augment),
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
            "mined_negative_map": str(args.mined_negative_map) if args.mined_negative_map else None,
            "mined_negative_top_k": args.mined_negative_top_k if args.mined_negative_map else None,
            "mined_pair_batches": bool(args.mined_pair_batches),
            "take_weight_metadata": take_weight_metadata,
            "transition_weight_metadata": transition_weight_metadata,
            "light_augment": bool(args.light_augment),
        },
    )
    print(f"Saved FACT tokenizer checkpoint to {checkpoint_path}")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
