#!/usr/bin/env python3
"""Build same-take mined negative pairs for FACT transition training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fact_tokenizer import FACTPairedNPZDataset
from fact_tokenizer.model import DINOv2PatchFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--source-view-keys", nargs="*", default=["ego", "exo"])
    parser.add_argument("--view-names", nargs=2, default=["ego", "exo"])
    parser.add_argument("--videos-layout", default="VBTCHW")
    parser.add_argument("--frame-pair", choices=["first-last", "first-next"], default="first-last")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-gap", type=float, default=3.0)
    parser.add_argument("--max-gap", type=float, default=24.0)
    parser.add_argument("--mode", choices=["mined", "temporal"], default="mined")
    parser.add_argument("--temporal-offset", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dino-model", default="dinov2_vitb14_reg")
    parser.add_argument("--torch-home", type=Path, default=ROOT / "checkpoints" / "torch_hub")
    return parser.parse_args()


def read_npz_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        first_view = "ego" if "ego" in data else next(key for key, value in data.items() if value.ndim == 5)
        num_samples = int(data[first_view].shape[0])
        take_uid = np.asarray(data["take_uid"]).astype(str) if "take_uid" in data else np.asarray([str(i) for i in range(num_samples)])
        timestamp = (
            np.asarray(data["timestamp"], dtype=np.float32)
            if "timestamp" in data
            else np.arange(num_samples, dtype=np.float32)
        )
        sample_id = (
            [str(value) for value in np.asarray(data["sample_id"]).astype(str).tolist()]
            if "sample_id" in data
            else [str(index) for index in range(num_samples)]
        )
    if len(take_uid) != num_samples or len(timestamp) != num_samples:
        raise ValueError("take_uid/timestamp length does not match the first view length")
    return take_uid, timestamp, sample_id


def dataloader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": False,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return kwargs


@torch.inference_mode()
def extract_dino_transition_features(args: argparse.Namespace) -> dict[str, dict[str, torch.Tensor]]:
    dataset = FACTPairedNPZDataset(
        input_npz=args.input_npz,
        source_view_keys=args.source_view_keys,
        output_view_names=args.view_names,
        videos_layout=args.videos_layout,
        frame_pair=args.frame_pair,
        start_index=args.start_index,
        resize=args.resize,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, **dataloader_kwargs(args))
    extractor = DINOv2PatchFeatureExtractor(
        dino_model=args.dino_model,
        torch_home=str(args.torch_home),
    ).to(args.device)
    extractor.eval()
    features: dict[str, dict[str, list[torch.Tensor]]] = {
        view_name: {"context": [], "delta": []}
        for view_name in args.view_names
    }
    for batch in loader:
        for view_name in args.view_names:
            videos = batch[view_name]["videos"].to(args.device, non_blocking=True)
            patch_tokens = extractor(videos).mean(dim=2)
            current = patch_tokens[:, 0].float()
            target = patch_tokens[:, -1].float()
            features[view_name]["context"].append(F.normalize(current, dim=1).cpu())
            features[view_name]["delta"].append(F.normalize(target - current, dim=1).cpu())
    return {
        view_name: {
            name: torch.cat(values, dim=0)
            for name, values in view_features.items()
        }
        for view_name, view_features in features.items()
    }


def normalize_topk_weights(values: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    weights = torch.zeros_like(values, dtype=torch.float32)
    for row in range(values.shape[0]):
        row_finite = finite[row]
        if not bool(row_finite.any()):
            continue
        row_values = values[row, row_finite]
        if row_values.numel() == 1:
            weights[row, row_finite] = 1.0
            continue
        span = (row_values.max() - row_values.min()).clamp_min(1e-6)
        weights[row, row_finite] = 0.75 + 0.5 * (values[row, row_finite] - row_values.min()) / span
    return weights


def fill_topk_for_take(
    members: np.ndarray,
    timestamps: np.ndarray,
    donor_index_topk: np.ndarray,
    donor_score_topk: np.ndarray,
    donor_raw_score_topk: np.ndarray,
    top_k: int,
    score: torch.Tensor,
    min_gap: float,
    max_gap: float,
) -> None:
    local_timestamps = torch.from_numpy(timestamps[members].astype(np.float32))
    gap = (local_timestamps[:, None] - local_timestamps[None, :]).abs()
    not_self = ~torch.eye(len(members), dtype=torch.bool)
    valid = not_self & (gap >= float(min_gap)) & (gap <= float(max_gap))
    masked = score.masked_fill(~valid, -torch.inf)
    k = min(top_k, len(members))
    values, local_donors = torch.topk(masked, k=k, dim=1)
    finite = torch.isfinite(values)
    weights = normalize_topk_weights(values, finite)
    global_donors = torch.as_tensor(members, dtype=torch.long)[local_donors.clamp_min(0)]
    for row, anchor in enumerate(members.tolist()):
        valid_count = int(finite[row].sum().item())
        if valid_count <= 0:
            continue
        donor_index_topk[anchor, :valid_count] = global_donors[row, :valid_count].numpy()
        donor_score_topk[anchor, :valid_count] = weights[row, :valid_count].numpy()
        donor_raw_score_topk[anchor, :valid_count] = values[row, :valid_count].numpy()


def build_mined_pairs(
    take_uid: np.ndarray,
    timestamp: np.ndarray,
    features: dict[str, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = len(take_uid)
    donor_index_topk = np.full((num_samples, args.top_k), -1, dtype=np.int64)
    donor_score_topk = np.zeros((num_samples, args.top_k), dtype=np.float32)
    donor_raw_score_topk = np.full((num_samples, args.top_k), -np.inf, dtype=np.float32)
    take_to_members: dict[str, list[int]] = defaultdict(list)
    for index, take in enumerate(take_uid.tolist()):
        take_to_members[str(take)].append(index)

    ego_name, exo_name = args.view_names
    for members_list in take_to_members.values():
        members = np.asarray(members_list, dtype=np.int64)
        if len(members) <= 1:
            continue
        ego_delta = features[ego_name]["delta"][members]
        exo_delta = features[exo_name]["delta"][members]
        ego_context = features[ego_name]["context"][members]
        exo_context = features[exo_name]["context"][members]
        local_time = torch.from_numpy(timestamp[members].astype(np.float32))
        gap = (local_time[:, None] - local_time[None, :]).abs()
        ego_delta_dist = 1.0 - torch.matmul(ego_delta, ego_delta.T)
        exo_delta_dist = 1.0 - torch.matmul(exo_delta, exo_delta.T)
        ego_context_dist = 1.0 - torch.matmul(ego_context, ego_context.T)
        exo_context_dist = 1.0 - torch.matmul(exo_context, exo_context.T)
        time_bonus = (gap / 12.0).clamp(max=1.0)
        score = (
            0.45 * ego_delta_dist
            + 0.45 * exo_delta_dist
            - 0.25 * ego_context_dist
            - 0.15 * exo_context_dist
            + 0.10 * time_bonus
        )
        fill_topk_for_take(
            members,
            timestamp,
            donor_index_topk,
            donor_score_topk,
            donor_raw_score_topk,
            args.top_k,
            score,
            args.min_gap,
            args.max_gap,
        )
    return donor_index_topk, donor_score_topk, donor_raw_score_topk


def build_temporal_pairs(
    take_uid: np.ndarray,
    timestamp: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = len(take_uid)
    donor_index_topk = np.full((num_samples, args.top_k), -1, dtype=np.int64)
    donor_score_topk = np.zeros((num_samples, args.top_k), dtype=np.float32)
    donor_raw_score_topk = np.full((num_samples, args.top_k), -np.inf, dtype=np.float32)
    take_to_members: dict[str, list[int]] = defaultdict(list)
    for index, take in enumerate(take_uid.tolist()):
        take_to_members[str(take)].append(index)

    for members_list in take_to_members.values():
        members = np.asarray(members_list, dtype=np.int64)
        if len(members) <= 1:
            continue
        local_time = torch.from_numpy(timestamp[members].astype(np.float32))
        gap = (local_time[:, None] - local_time[None, :]).abs()
        not_self = ~torch.eye(len(members), dtype=torch.bool)
        valid = not_self & (gap >= float(args.min_gap)) & (gap <= float(args.max_gap))
        score = -torch.abs(gap - float(args.temporal_offset))
        fill_topk_for_take(
            members,
            timestamp,
            donor_index_topk,
            donor_score_topk,
            donor_raw_score_topk,
            args.top_k,
            score,
            args.min_gap,
            args.max_gap,
        )
    return donor_index_topk, donor_score_topk, donor_raw_score_topk


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    take_uid, timestamp, sample_id = read_npz_metadata(args.input_npz)
    if args.mode == "mined":
        features = extract_dino_transition_features(args)
        donor_index_topk, donor_score_topk, donor_raw_score_topk = build_mined_pairs(
            take_uid,
            timestamp,
            features,
            args,
        )
    else:
        donor_index_topk, donor_score_topk, donor_raw_score_topk = build_temporal_pairs(take_uid, timestamp, args)

    valid = donor_index_topk >= 0
    metadata = {
        "input_npz": str(args.input_npz),
        "mode": args.mode,
        "top_k": args.top_k,
        "min_gap": args.min_gap,
        "max_gap": args.max_gap,
        "temporal_offset": args.temporal_offset if args.mode == "temporal" else None,
        "num_samples": int(len(take_uid)),
        "valid_anchor_fraction": float(valid.any(axis=1).mean()),
        "valid_pair_fraction": float(valid.mean()),
        "score_formula": (
            "0.45*ego_delta_dist + 0.45*exo_delta_dist - 0.25*ego_context_dist "
            "- 0.15*exo_context_dist + 0.10*time_bonus"
            if args.mode == "mined"
            else "-abs(abs(delta_t)-temporal_offset)"
        ),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        donor_index_topk=donor_index_topk,
        donor_score_topk=donor_score_topk,
        donor_raw_score_topk=donor_raw_score_topk,
        take_uid=take_uid.astype(str),
        timestamp=timestamp.astype(np.float32),
        sample_id=np.asarray(sample_id).astype(str),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    print(f"Saved mined negative map to {args.output_npz}")


if __name__ == "__main__":
    main()
