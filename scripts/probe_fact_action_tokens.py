#!/usr/bin/env python3
"""Probe whether FACT shared tokens causally and semantically behave like action tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fact_tokenizer import FACTPairedNPZDataset, FACTTokenizer
from fact_tokenizer.utils import code_usage, load_checkpoint, save_json


DEFAULT_CAUSALITY_ACTION_MODES = (
    "correct",
    "global_shuffle",
    "same_take_shuffle",
    "random_take",
    "zero",
    "random_code",
)
DEFAULT_LEAKAGE_ACTION_MODES = ("correct", "global_shuffle", "zero")
DEFAULT_LEAKAGE_PRIVATE_MODES = ("correct", "global_shuffle", "zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-view-keys", nargs="*", default=None)
    parser.add_argument("--view-names", nargs=2, default=None)
    parser.add_argument("--videos-layout", default="VBTCHW")
    parser.add_argument("--frame-pair", choices=["first-last", "first-next"], default="first-last")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--eval-paths",
        nargs="+",
        default=["ego_self", "ego_swap", "exo_self", "exo_swap"],
        choices=["ego_self", "ego_swap", "exo_self", "exo_swap"],
    )
    parser.add_argument("--skip-causality", action="store_true")
    parser.add_argument("--skip-private-leakage", action="store_true")
    parser.add_argument("--skip-semantic-probe", action="store_true")
    parser.add_argument(
        "--causality-action-modes",
        nargs="+",
        default=list(DEFAULT_CAUSALITY_ACTION_MODES),
        choices=["correct", "global_shuffle", "same_take_shuffle", "random_take", "zero", "random_code"],
    )
    parser.add_argument(
        "--leakage-action-modes",
        nargs="+",
        default=list(DEFAULT_LEAKAGE_ACTION_MODES),
        choices=["correct", "global_shuffle", "same_take_shuffle", "random_take", "zero", "random_code"],
    )
    parser.add_argument(
        "--leakage-private-modes",
        nargs="+",
        default=list(DEFAULT_LEAKAGE_PRIVATE_MODES),
        choices=["correct", "global_shuffle", "same_take_shuffle", "random_take", "zero"],
    )
    parser.add_argument("--private-dropout-probs", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--temporal-offsets",
        nargs="*",
        type=int,
        default=[1, 4, 8],
        help="Same-take row offsets used as temporal action-token controls.",
    )
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--label-columns",
        nargs="*",
        default=["object", "contact", "action", "narration", "parent_task_name", "task_name"],
    )
    parser.add_argument("--condition-columns", nargs="*", default=["object", "contact"])
    parser.add_argument("--semantic-min-count", type=int, default=5)
    parser.add_argument("--semantic-top-k", type=int, default=8)
    return parser.parse_args()


def to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def stringify(value: Any) -> str:
    value = to_python(value)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def load_npz_metadata(path: Path, expected_len: int) -> dict:
    metadata: dict[str, list[Any]] = {
        "row_index": list(range(expected_len)),
        "sample_id": [str(index) for index in range(expected_len)],
        "take_uid": [""] * expected_len,
        "timestamp": [float("nan")] * expected_len,
    }
    with np.load(path, allow_pickle=False) as data:
        for key in ("sample_id", "take_uid", "timestamp"):
            if key not in data:
                continue
            array = data[key]
            if len(array) != expected_len:
                raise ValueError(f"Metadata key {key!r} has length {len(array)}, expected {expected_len}")
            if key == "timestamp":
                metadata[key] = [float(value) for value in array]
            else:
                metadata[key] = [stringify(value) for value in array]
    return metadata


def path_views(path_name: str, view_names: Sequence[str]) -> tuple[str, str]:
    ego_name, exo_name = view_names
    if path_name == "ego_self":
        return ego_name, ego_name
    if path_name == "ego_swap":
        return ego_name, exo_name
    if path_name == "exo_self":
        return exo_name, exo_name
    if path_name == "exo_swap":
        return exo_name, ego_name
    raise ValueError(f"Unknown eval path: {path_name}")


def per_sample_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (recon - target).square().reshape(recon.shape[0], -1).mean(dim=1)


def summarize_values(values: Iterable[torch.Tensor]) -> dict:
    tensor = torch.cat([value.detach().cpu().float().reshape(-1) for value in values], dim=0)
    if tensor.numel() == 0:
        return {"count": 0}
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(std),
        "stderr": float(std / math.sqrt(tensor.numel())),
        "median": float(tensor.median()),
        "p90": float(torch.quantile(tensor, 0.9)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def make_deranged_permutation(num_items: int, generator: torch.Generator) -> torch.Tensor:
    if num_items <= 1:
        return torch.arange(num_items, dtype=torch.long)
    permutation = torch.randperm(num_items, generator=generator)
    fixed = permutation == torch.arange(num_items)
    fixed_indices = fixed.nonzero(as_tuple=False).flatten()
    if fixed_indices.numel() == 1:
        index = int(fixed_indices.item())
        swap_index = 0 if index != 0 else 1
        permutation[index], permutation[swap_index] = permutation[swap_index].clone(), permutation[index].clone()
    elif fixed_indices.numel() > 1:
        rotated = fixed_indices.roll(1)
        permutation[fixed_indices] = permutation[rotated]
    return permutation


def make_group_shuffle(groups: Sequence[str], generator: torch.Generator) -> tuple[torch.Tensor, int]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[stringify(group)].append(index)

    donors = torch.arange(len(groups), dtype=torch.long)
    fallback = 0
    for indices in group_to_indices.values():
        if len(indices) <= 1:
            fallback += len(indices)
            continue
        local = torch.tensor(indices, dtype=torch.long)
        permutation = make_deranged_permutation(len(indices), generator)
        donors[local] = local[permutation]
    return donors, fallback


def make_different_group_shuffle(groups: Sequence[str], generator: torch.Generator) -> tuple[torch.Tensor, int]:
    num_items = len(groups)
    all_indices = torch.arange(num_items, dtype=torch.long)
    if num_items <= 1:
        return all_indices, num_items

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[stringify(group)].append(index)

    donors = torch.empty(num_items, dtype=torch.long)
    fallback = 0
    for index, group in enumerate(groups):
        group = stringify(group)
        valid = [candidate for candidate in range(num_items) if stringify(groups[candidate]) != group]
        if not valid:
            fallback += 1
            donors[index] = make_deranged_permutation(num_items, generator)[index]
            continue
        choice = torch.randint(len(valid), size=(1,), generator=generator).item()
        donors[index] = int(valid[choice])
    return donors, fallback


def make_temporal_offset_donors(
    metadata: Mapping[str, Sequence[Any]],
    offsets: Sequence[int],
) -> dict[str, torch.Tensor | int]:
    num_items = len(metadata["sample_id"])
    take_uids = [stringify(value) for value in metadata.get("take_uid", [""] * num_items)]
    timestamps = metadata.get("timestamp", [float("nan")] * num_items)
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, take_uid in enumerate(take_uids):
        group_to_indices[take_uid].append(index)

    sorted_groups = {}
    for take_uid, indices in group_to_indices.items():
        sorted_groups[take_uid] = sorted(
            indices,
            key=lambda index: (
                float(timestamps[index]) if isinstance(timestamps[index], (float, int)) else float("nan"),
                index,
            ),
        )

    donors: dict[str, torch.Tensor | int] = {}
    for raw_offset in offsets:
        offset = abs(int(raw_offset))
        if offset == 0:
            continue
        key = f"temporal_offset_{offset}"
        donor_tensor = torch.arange(num_items, dtype=torch.long)
        fallback = 0
        for indices in sorted_groups.values():
            position_by_index = {index: position for position, index in enumerate(indices)}
            for index in indices:
                position = position_by_index[index]
                if position + offset < len(indices):
                    donor_tensor[index] = indices[position + offset]
                elif position - offset >= 0:
                    donor_tensor[index] = indices[position - offset]
                else:
                    fallback += 1
        donors[key] = donor_tensor
        donors[f"{key}_fallback_count"] = fallback
    return donors


def make_donor_indices(
    metadata: Mapping[str, Sequence[Any]],
    seed: int,
    temporal_offsets: Sequence[int],
) -> dict[str, torch.Tensor | int]:
    generator = torch.Generator().manual_seed(seed)
    num_items = len(metadata["sample_id"])
    take_uids = [stringify(value) for value in metadata.get("take_uid", [""] * num_items)]

    global_shuffle = make_deranged_permutation(num_items, generator)
    same_take_shuffle, same_take_fallback = make_group_shuffle(take_uids, generator)
    random_take, random_take_fallback = make_different_group_shuffle(take_uids, generator)
    donors: dict[str, torch.Tensor | int] = {
        "global_shuffle": global_shuffle,
        "same_take_shuffle": same_take_shuffle,
        "random_take": random_take,
        "same_take_fallback_count": same_take_fallback,
        "random_take_fallback_count": random_take_fallback,
    }
    donors.update(make_temporal_offset_donors(metadata, temporal_offsets))
    return donors


def load_model_and_data(args: argparse.Namespace) -> tuple[FACTTokenizer, FACTPairedNPZDataset, list[str], torch.device, dict]:
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = dict(checkpoint["model_config"])
    if args.view_names:
        model_config["view_names"] = tuple(args.view_names)
    view_names = list(model_config.get("view_names", ("ego", "exo")))
    dataset = FACTPairedNPZDataset(
        input_npz=args.input_npz,
        source_view_keys=args.source_view_keys,
        output_view_names=view_names,
        videos_layout=args.videos_layout,
        frame_pair=args.frame_pair,
        start_index=args.start_index,
        resize=args.resize,
    )
    model = FACTTokenizer(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, dataset, view_names, device, checkpoint


def collect_latent_pools(
    model: FACTTokenizer,
    dataset: FACTPairedNPZDataset,
    view_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> dict[str, dict[str, torch.Tensor]]:
    pools: dict[str, dict[str, list[torch.Tensor]]] = {
        view: {"z_q": [], "r_priv": [], "indices": [], "confidence": []} for view in view_names
    }
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.inference_mode():
        for batch in loader:
            views = model.encode_views(batch)
            for view in view_names:
                pools[view]["z_q"].append(views[view]["z_q"].detach().cpu())
                pools[view]["r_priv"].append(views[view]["r_priv"].detach().cpu())
                pools[view]["indices"].append(views[view]["indices"].detach().cpu())
                pools[view]["confidence"].append(views[view]["confidence"].detach().cpu())

    return {
        view: {key: torch.cat(chunks, dim=0) for key, chunks in view_pool.items()}
        for view, view_pool in pools.items()
    }


def gather_pool_tensor(pool: torch.Tensor, row_indices: torch.Tensor, donor_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    donors = donor_indices[row_indices.cpu()].long()
    return pool.index_select(0, donors).to(device)


def make_action_tensor(
    mode: str,
    view: str,
    current: torch.Tensor,
    row_indices: torch.Tensor,
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    donor_indices: Mapping[str, torch.Tensor | int],
    model: FACTTokenizer,
    device: torch.device,
    random_code_indices: torch.Tensor,
) -> torch.Tensor:
    if mode == "correct":
        return current
    if mode == "zero":
        return torch.zeros_like(current)
    if mode in {"global_shuffle", "same_take_shuffle", "random_take"} or mode.startswith("temporal_offset_"):
        return gather_pool_tensor(pools[view]["z_q"], row_indices, donor_indices[mode], device)  # type: ignore[arg-type]
    if mode == "random_code":
        code_indices = random_code_indices.index_select(0, row_indices.cpu()).to(device)
        return model.action_vq.codebook(code_indices)
    raise ValueError(f"Unsupported action mode: {mode}")


def make_private_tensor(
    mode: str,
    view: str,
    current: torch.Tensor,
    row_indices: torch.Tensor,
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    donor_indices: Mapping[str, torch.Tensor | int],
    device: torch.device,
) -> torch.Tensor:
    if mode == "correct":
        return current
    if mode == "zero":
        return torch.zeros_like(current)
    if mode in {"global_shuffle", "same_take_shuffle", "random_take"}:
        return gather_pool_tensor(pools[view]["r_priv"], row_indices, donor_indices[mode], device)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported private mode: {mode}")


def decode_with_modes(
    model: FACTTokenizer,
    views: Mapping[str, Mapping[str, torch.Tensor]],
    path_name: str,
    view_names: Sequence[str],
    row_indices: torch.Tensor,
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    donor_indices: Mapping[str, torch.Tensor | int],
    action_mode: str,
    private_mode: str,
    device: torch.device,
    random_code_indices: torch.Tensor,
) -> torch.Tensor:
    obs_view, act_view = path_views(path_name, view_names)
    obs = dict(views[obs_view])
    act = dict(views[act_view])
    act["z_q"] = make_action_tensor(
        action_mode,
        act_view,
        act["z_q"],
        row_indices,
        pools,
        donor_indices,
        model,
        device,
        random_code_indices,
    )
    obs["r_priv"] = make_private_tensor(
        private_mode,
        obs_view,
        obs["r_priv"],
        row_indices,
        pools,
        donor_indices,
        device,
    )
    return model._decode_path(obs, act)


def build_random_code_indices(
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    view_names: Sequence[str],
    num_latents: int,
    seed: int,
) -> torch.Tensor:
    reference = pools[view_names[0]]["indices"]
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(num_latents, reference.shape, generator=generator, dtype=torch.long)


def build_private_dropout_pools(
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    view_names: Sequence[str],
    probabilities: Sequence[float],
    seed: int,
) -> dict[float, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    dropout_pools: dict[float, dict[str, torch.Tensor]] = {}
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Private dropout probability must be in [0, 1], got {probability}")
        dropout_pools[probability] = {}
        for view in view_names:
            private = pools[view]["r_priv"]
            mask = (torch.rand(private.shape, generator=generator) >= probability).to(private.dtype)
            dropout_pools[probability][view] = private * mask
    return dropout_pools


def add_metric(metrics: dict[tuple[Any, ...], list[torch.Tensor]], key: tuple[Any, ...], value: torch.Tensor) -> None:
    metrics.setdefault(key, []).append(value.detach().cpu())


def summarize_metric_table(
    metrics: Mapping[tuple[Any, ...], list[torch.Tensor]],
    columns: Sequence[str],
    correct_lookup: Mapping[str, torch.Tensor] | None = None,
) -> list[dict]:
    rows = []
    for key, values in sorted(metrics.items()):
        row = {column: key[index] if index < len(key) else None for index, column in enumerate(columns)}
        samples = torch.cat([value.float().reshape(-1) for value in values], dim=0)
        row.update(summarize_values([samples]))
        path_name = row.get("path")
        if correct_lookup is not None and isinstance(path_name, str) and path_name in correct_lookup:
            correct = correct_lookup[path_name]
            if correct.numel() == samples.numel():
                delta = samples - correct
                row["delta_vs_correct_mean"] = float(delta.mean())
                row["delta_vs_correct_median"] = float(delta.median())
                correct_mean = float(correct.mean())
                row["ratio_vs_correct_mean"] = float(samples.mean() / correct_mean) if correct_mean else None
                row["fraction_worse_than_correct"] = float((delta > 0).float().mean())
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "probe",
        "path",
        "action_mode",
        "private_mode",
        "private_dropout",
        "count",
        "mean",
        "delta_vs_correct_mean",
        "ratio_vs_correct_mean",
        "fraction_worse_than_correct",
        "stderr",
        "median",
        "p90",
        "std",
        "min",
        "max",
    ]
    fieldnames = [name for name in preferred if name in fieldnames] + [
        name for name in fieldnames if name not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: Mapping[str, Any], *keys: str) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in keys)


def index_rows(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {row_key(row, *keys): row for row in rows}


def compact_metric(row: Mapping[str, Any] | None) -> dict | None:
    if row is None:
        return None
    keys = [
        "mean",
        "delta_vs_correct_mean",
        "ratio_vs_correct_mean",
        "fraction_worse_than_correct",
        "stderr",
        "median",
        "p90",
    ]
    return {key: row.get(key) for key in keys if key in row}


def mean_gap(higher_mse: Mapping[str, Any] | None, lower_mse: Mapping[str, Any] | None) -> float | None:
    if higher_mse is None or lower_mse is None:
        return None
    if higher_mse.get("mean") is None or lower_mse.get("mean") is None:
        return None
    return float(higher_mse["mean"]) - float(lower_mse["mean"])


def build_probe_summary(
    causality_rows: Sequence[Mapping[str, Any]],
    leakage_rows: Sequence[Mapping[str, Any]],
    eval_paths: Sequence[str],
) -> dict:
    causality_by_key = index_rows(causality_rows, "path", "action_mode")
    leakage_by_key = index_rows(leakage_rows, "probe", "path", "action_mode", "private_mode")
    dropout_rows = [
        row
        for row in leakage_rows
        if row.get("probe") == "private_dropout" and row.get("private_dropout") is not None
    ]

    causality_summary = {}
    for path_name in eval_paths:
        path_rows = {
            action_mode: compact_metric(row)
            for (path, action_mode), row in causality_by_key.items()
            if path == path_name
        }
        negative_controls = {
            action_mode: metric
            for action_mode, metric in path_rows.items()
            if action_mode != "correct" and metric is not None
        }
        ranked_controls = sorted(
            negative_controls.items(),
            key=lambda item: float(item[1].get("delta_vs_correct_mean") or 0.0),
            reverse=True,
        )
        causality_summary[path_name] = {
            "correct": path_rows.get("correct"),
            "negative_controls": negative_controls,
            "strongest_degradation": ranked_controls[0] if ranked_controls else None,
            "reading": (
                "Positive delta_vs_correct_mean means the replacement token made reconstruction worse; "
                "same_take and temporal controls are the hardest negatives because scene/take context is preserved."
            ),
        }

    leakage_summary = {}
    for path_name in eval_paths:
        matrix = {}
        for action_mode in DEFAULT_LEAKAGE_ACTION_MODES:
            for private_mode in DEFAULT_LEAKAGE_PRIVATE_MODES:
                matrix[f"act={action_mode}|priv={private_mode}"] = compact_metric(
                    leakage_by_key.get(("private_leakage", path_name, action_mode, private_mode))
                )
        action_shuffled_private_correct = matrix.get("act=global_shuffle|priv=correct")
        action_correct_private_zero = matrix.get("act=correct|priv=zero")
        action_shuffled_private_zero = matrix.get("act=global_shuffle|priv=zero")
        action_zero_private_correct = matrix.get("act=zero|priv=correct")
        action_zero_private_zero = matrix.get("act=zero|priv=zero")
        leakage_summary[path_name] = {
            "matrix": matrix,
            "key_readouts": {
                "private_can_compensate_if_action_shuffled": action_shuffled_private_correct,
                "action_survives_without_private": action_correct_private_zero,
                "action_shuffled_without_private": action_shuffled_private_zero,
                "private_only_without_action": action_zero_private_correct,
                "both_removed": action_zero_private_zero,
                "mse_saved_by_correct_action_without_private_vs_zero_action": mean_gap(
                    action_zero_private_zero,
                    action_correct_private_zero,
                ),
                "mse_saved_by_correct_action_without_private_vs_shuffled_action": mean_gap(
                    action_shuffled_private_zero,
                    action_correct_private_zero,
                ),
                "mse_added_by_removing_private_with_correct_action": mean_gap(
                    action_correct_private_zero,
                    matrix.get("act=correct|priv=correct"),
                ),
            },
            "dropout_curve": [
                {
                    "action_mode": row.get("action_mode"),
                    "private_dropout": row.get("private_dropout"),
                    **(compact_metric(row) or {}),
                }
                for row in dropout_rows
                if row.get("path") == path_name
            ],
            "reading": (
                "If act=global_shuffle|priv=correct stays close to correct, private residual may be carrying "
                "transition information. If act=correct|priv=zero is still much better than act=zero|priv=zero, "
                "the action token remains useful after private removal."
            ),
        }

    return {
        "causality": causality_summary,
        "private_leakage": leakage_summary,
    }


def run_reconstruction_probes(
    args: argparse.Namespace,
    model: FACTTokenizer,
    dataset: FACTPairedNPZDataset,
    view_names: Sequence[str],
    device: torch.device,
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    donor_indices: Mapping[str, torch.Tensor | int],
) -> tuple[list[dict], list[dict]]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    random_code_indices = build_random_code_indices(pools, view_names, model.num_latents, args.seed + 17)
    private_dropout_pools = build_private_dropout_pools(pools, view_names, args.private_dropout_probs, args.seed + 29)
    temporal_modes = sorted(
        key
        for key, value in donor_indices.items()
        if key.startswith("temporal_offset_") and not key.endswith("_fallback_count")
    )
    causality_action_modes = list(args.causality_action_modes) + [
        mode for mode in temporal_modes if mode not in args.causality_action_modes
    ]

    causality_metrics: dict[tuple[Any, ...], list[torch.Tensor]] = {}
    leakage_metrics: dict[tuple[Any, ...], list[torch.Tensor]] = {}
    correct_by_path: dict[str, list[torch.Tensor]] = {path_name: [] for path_name in args.eval_paths}

    with torch.inference_mode():
        for batch in loader:
            row_indices = batch[view_names[0]]["sample_id"].long()
            views = model.encode_views(batch)

            for path_name in args.eval_paths:
                obs_view, _ = path_views(path_name, view_names)
                target = views[obs_view]["target_patches"]
                correct_recon = decode_with_modes(
                    model,
                    views,
                    path_name,
                    view_names,
                    row_indices,
                    pools,
                    donor_indices,
                    action_mode="correct",
                    private_mode="correct",
                    device=device,
                    random_code_indices=random_code_indices,
                )
                correct_mse = per_sample_mse(correct_recon, target)
                correct_by_path[path_name].append(correct_mse.detach().cpu())

                if not args.skip_causality:
                    for action_mode in causality_action_modes:
                        if action_mode == "correct":
                            mse = correct_mse
                        else:
                            recon = decode_with_modes(
                                model,
                                views,
                                path_name,
                                view_names,
                                row_indices,
                                pools,
                                donor_indices,
                                action_mode=action_mode,
                                private_mode="correct",
                                device=device,
                                random_code_indices=random_code_indices,
                            )
                            mse = per_sample_mse(recon, target)
                        add_metric(causality_metrics, ("causality", path_name, action_mode), mse)

                if not args.skip_private_leakage:
                    for action_mode in args.leakage_action_modes:
                        for private_mode in args.leakage_private_modes:
                            if action_mode == "correct" and private_mode == "correct":
                                mse = correct_mse
                            else:
                                recon = decode_with_modes(
                                    model,
                                    views,
                                    path_name,
                                    view_names,
                                    row_indices,
                                    pools,
                                    donor_indices,
                                    action_mode=action_mode,
                                    private_mode=private_mode,
                                    device=device,
                                    random_code_indices=random_code_indices,
                                )
                                mse = per_sample_mse(recon, target)
                            add_metric(leakage_metrics, ("private_leakage", path_name, action_mode, private_mode), mse)

                    for action_mode in args.leakage_action_modes:
                        for probability in args.private_dropout_probs:
                            obs_view, act_view = path_views(path_name, view_names)
                            obs = dict(views[obs_view])
                            act = dict(views[act_view])
                            act["z_q"] = make_action_tensor(
                                action_mode,
                                act_view,
                                act["z_q"],
                                row_indices,
                                pools,
                                donor_indices,
                                model,
                                device,
                                random_code_indices,
                            )
                            obs["r_priv"] = private_dropout_pools[probability][obs_view].index_select(
                                0, row_indices.cpu()
                            ).to(device)
                            recon = model._decode_path(obs, act)
                            mse = per_sample_mse(recon, target)
                            add_metric(
                                leakage_metrics,
                                (
                                    "private_dropout",
                                    path_name,
                                    action_mode,
                                    "dropout",
                                    f"{probability:.3f}",
                                ),
                                mse,
                            )

    correct_lookup = {
        path_name: torch.cat(chunks, dim=0).float() for path_name, chunks in correct_by_path.items()
    }
    causality_rows = summarize_metric_table(
        causality_metrics,
        columns=["probe", "path", "action_mode"],
        correct_lookup=correct_lookup,
    )
    leakage_rows = summarize_metric_table(
        leakage_metrics,
        columns=["probe", "path", "action_mode", "private_mode", "private_dropout"],
        correct_lookup=correct_lookup,
    )
    return causality_rows, leakage_rows


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("rows", "annotations", "data", "samples"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError(f"Could not find a row list in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        lengths = {len(value) for value in arrays.values() if value.ndim >= 1}
        if len(lengths) != 1:
            raise ValueError(f"NPZ label arrays must share one row length, got {sorted(lengths)}")
        row_count = lengths.pop()
        return [
            {key: to_python(value[index]) for key, value in arrays.items() if value.ndim >= 1}
            for index in range(row_count)
        ]
    raise ValueError(f"Unsupported label file extension: {path.suffix}")


def first_present(row: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in row and stringify(row[key]) != "":
            return key
    return None


def align_label_rows(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Sequence[Any]],
    label_columns: Sequence[str],
) -> dict[str, list[str]]:
    num_samples = len(metadata["sample_id"])
    labels = {column: [""] * num_samples for column in label_columns}
    if not rows:
        return labels

    row0 = rows[0]
    start_key = first_present(
        row0,
        ("start_sec", "start_time", "start", "video_start_sec", "timestamp_start", "task_start_sec"),
    )
    end_key = first_present(
        row0,
        ("end_sec", "end_time", "end", "video_end_sec", "timestamp_end", "task_end_sec"),
    )
    has_interval = "take_uid" in row0 and start_key is not None and end_key is not None

    if "sample_id" in row0 and not has_interval:
        row_by_id = {stringify(row.get("sample_id")): row for row in rows}
        for index, sample_id in enumerate(metadata["sample_id"]):
            row = row_by_id.get(stringify(sample_id))
            if row is None:
                continue
            for column in label_columns:
                labels[column][index] = stringify(row.get(column))
        return labels

    if len(rows) == num_samples and not has_interval:
        for index, row in enumerate(rows):
            for column in label_columns:
                labels[column][index] = stringify(row.get(column))
        return labels

    if has_interval:
        by_take: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_take[stringify(row.get("take_uid"))].append(row)
        for take_rows in by_take.values():
            take_rows.sort(key=lambda row: float(row.get(start_key, 0.0)))  # type: ignore[arg-type]
        for index, (take_uid, timestamp) in enumerate(zip(metadata["take_uid"], metadata["timestamp"])):
            if not isinstance(timestamp, float) or math.isnan(timestamp):
                continue
            for row in by_take.get(stringify(take_uid), []):
                start = float(row.get(start_key, -math.inf))  # type: ignore[arg-type]
                end = float(row.get(end_key, math.inf))  # type: ignore[arg-type]
                if start <= timestamp <= end:
                    for column in label_columns:
                        labels[column][index] = stringify(row.get(column))
                    break
        return labels

    raise ValueError(
        "Labels must either have one row per sample, a sample_id column, or interval columns "
        "(take_uid plus start_sec/end_sec or equivalent)."
    )


def label_coverage(labels: Mapping[str, Sequence[str]]) -> dict:
    coverage = {}
    for column, values in labels.items():
        nonempty = sum(1 for value in values if stringify(value) != "")
        coverage[column] = {
            "count": int(nonempty),
            "total": int(len(values)),
            "fraction": float(nonempty / len(values)) if values else 0.0,
            "unique_values": int(len({value for value in values if stringify(value) != ""})),
        }
    return coverage


def entropy_from_counts(counts: Iterable[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def cluster_label_metrics(tokens: Sequence[str], labels: Sequence[str], min_count: int) -> dict:
    pairs = [(token, label) for token, label in zip(tokens, labels) if token != "" and label != ""]
    if len(pairs) < min_count:
        return {"count": len(pairs), "skipped": True, "reason": "too_few_labeled_samples"}

    token_counts = Counter(token for token, _ in pairs)
    label_counts = Counter(label for _, label in pairs)
    joint_counts = Counter(pairs)
    total = len(pairs)
    mutual_info = 0.0
    for (token, label), count in joint_counts.items():
        mutual_info += (count / total) * math.log((count * total) / (token_counts[token] * label_counts[label]))

    token_entropy = entropy_from_counts(token_counts.values())
    label_entropy = entropy_from_counts(label_counts.values())
    nmi = mutual_info / math.sqrt(token_entropy * label_entropy) if token_entropy and label_entropy else 0.0

    by_token: dict[str, Counter] = defaultdict(Counter)
    by_label: dict[str, Counter] = defaultdict(Counter)
    for token, label in pairs:
        by_token[token][label] += 1
        by_label[label][token] += 1

    purity = sum(max(counter.values()) for counter in by_token.values()) / total
    inverse_purity = sum(max(counter.values()) for counter in by_label.values()) / total
    label_given_token_entropy = sum(
        (sum(counter.values()) / total) * entropy_from_counts(counter.values()) for counter in by_token.values()
    )
    token_given_label_entropy = sum(
        (sum(counter.values()) / total) * entropy_from_counts(counter.values()) for counter in by_label.values()
    )
    return {
        "count": total,
        "num_token_values": len(token_counts),
        "num_label_values": len(label_counts),
        "mutual_info": mutual_info,
        "nmi_sqrt": nmi,
        "purity_label_given_token": purity,
        "purity_token_given_label": inverse_purity,
        "label_entropy": label_entropy,
        "token_entropy": token_entropy,
        "label_given_token_entropy": label_given_token_entropy,
        "token_given_label_entropy": token_given_label_entropy,
        "normalized_label_given_token_entropy": label_given_token_entropy / label_entropy if label_entropy else 0.0,
        "normalized_token_given_label_entropy": token_given_label_entropy / token_entropy if token_entropy else 0.0,
    }


def top_conditional_histograms(
    tokens: Sequence[str],
    labels: Sequence[str],
    top_k: int,
    min_count: int,
) -> dict:
    by_token: dict[str, Counter] = defaultdict(Counter)
    by_label: dict[str, Counter] = defaultdict(Counter)
    for token, label in zip(tokens, labels):
        if token == "" or label == "":
            continue
        by_token[token][label] += 1
        by_label[label][token] += 1

    def summarize(counter_by_key: Mapping[str, Counter]) -> dict:
        rows = {}
        for key, counter in counter_by_key.items():
            total = sum(counter.values())
            if total < min_count:
                continue
            rows[key] = [
                {"value": value, "count": int(count), "probability": float(count / total)}
                for value, count in counter.most_common(top_k)
            ]
        return dict(sorted(rows.items(), key=lambda item: sum(row["count"] for row in item[1]), reverse=True)[:top_k])

    return {
        "label_given_token_top": summarize(by_token),
        "token_given_label_top": summarize(by_label),
    }


def conditional_metrics(
    tokens: Sequence[str],
    labels: Sequence[str],
    conditions: Sequence[str],
    min_count: int,
) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, condition in enumerate(conditions):
        if condition != "" and labels[index] != "" and tokens[index] != "":
            groups[condition].append(index)

    weighted: dict[str, float] = defaultdict(float)
    total = 0
    used_groups = 0
    for indices in groups.values():
        if len(indices) < min_count:
            continue
        group_tokens = [tokens[index] for index in indices]
        group_labels = [labels[index] for index in indices]
        metrics = cluster_label_metrics(group_tokens, group_labels, min_count=min_count)
        if metrics.get("skipped"):
            continue
        weight = len(indices)
        total += weight
        used_groups += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in {"count", "num_token_values", "num_label_values"}:
                weighted[key] += float(value) * weight
    if total == 0:
        return {"count": 0, "condition_groups": 0, "skipped": True, "reason": "too_few_condition_groups"}
    return {
        "count": total,
        "condition_groups": used_groups,
        **{key: value / total for key, value in weighted.items()},
    }


def token_strings(indices: torch.Tensor) -> dict[str, list[str]]:
    flat = indices.reshape(indices.shape[0], -1).cpu().numpy()
    tokens = {"tuple": ["|".join(str(int(value)) for value in row) for row in flat]}
    slot_count = flat.shape[1]
    for slot in range(slot_count):
        tokens[f"slot_{slot}"] = [str(int(row[slot])) for row in flat]
    return tokens


def automatic_token_controls(
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    view_names: Sequence[str],
    metadata: Mapping[str, Sequence[Any]],
    min_count: int,
) -> dict:
    report: dict[str, Any] = {"view_invariance": {}, "take_leakage": {}}
    per_view_tokens = {view: token_strings(pools[view]["indices"]) for view in view_names}

    for token_name in per_view_tokens[view_names[0]].keys():
        combined_tokens = []
        view_labels = []
        for view in view_names:
            values = per_view_tokens[view][token_name]
            combined_tokens.extend(values)
            view_labels.extend([view] * len(values))
        report["view_invariance"][token_name] = cluster_label_metrics(
            combined_tokens,
            view_labels,
            min_count=min_count,
        )

    take_labels = [stringify(value) for value in metadata.get("take_uid", [])]
    if take_labels:
        for view in view_names:
            report["take_leakage"][view] = {}
            for token_name, values in per_view_tokens[view].items():
                report["take_leakage"][view][token_name] = cluster_label_metrics(
                    values,
                    take_labels,
                    min_count=min_count,
                )
    return report


def run_semantic_probe(
    args: argparse.Namespace,
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    view_names: Sequence[str],
    metadata: Mapping[str, Sequence[Any]],
    num_latents: int,
) -> dict:
    automatic_controls = automatic_token_controls(
        pools,
        view_names,
        metadata,
        min_count=args.semantic_min_count,
    )
    if args.skip_semantic_probe:
        return {"skipped": True, "reason": "disabled", "automatic_controls": automatic_controls}
    if args.labels is None:
        return {
            "skipped": True,
            "reason": "no_labels_file",
            "automatic_controls": automatic_controls,
            "expected_label_formats": [
                "CSV/JSONL/NPZ with one row per sample",
                "CSV/JSONL/NPZ with sample_id plus label columns",
                "CSV/JSONL with take_uid, start_sec/end_sec or task_start_sec/task_end_sec plus label columns",
            ],
        }

    rows = load_label_rows(args.labels)
    labels = align_label_rows(rows, metadata, args.label_columns)
    report: dict[str, Any] = {
        "labels_file": str(args.labels),
        "label_columns": list(args.label_columns),
        "condition_columns": list(args.condition_columns),
        "label_coverage": label_coverage(labels),
        "automatic_controls": automatic_controls,
        "views": {},
    }

    for view in view_names:
        view_report: dict[str, Any] = {
            "usage": code_usage(pools[view]["indices"], num_latents),
            "tokens": {},
        }
        for token_name, token_values in token_strings(pools[view]["indices"]).items():
            token_report: dict[str, Any] = {}
            for label_column, label_values in labels.items():
                token_report[label_column] = {
                    "metrics": cluster_label_metrics(token_values, label_values, args.semantic_min_count),
                    "histograms": top_conditional_histograms(
                        token_values,
                        label_values,
                        top_k=args.semantic_top_k,
                        min_count=args.semantic_min_count,
                    ),
                }
                for condition_column in args.condition_columns:
                    if condition_column == label_column or condition_column not in labels:
                        continue
                    conditional_key = f"{label_column}|{condition_column}"
                    token_report[label_column].setdefault("conditional_metrics", {})[conditional_key] = (
                        conditional_metrics(
                            token_values,
                            label_values,
                            labels[condition_column],
                            min_count=args.semantic_min_count,
                        )
                    )
            view_report["tokens"][token_name] = token_report
        report["views"][view] = view_report
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, dataset, view_names, device, checkpoint = load_model_and_data(args)
    metadata = load_npz_metadata(args.input_npz, expected_len=len(dataset))
    donor_indices = make_donor_indices(metadata, seed=args.seed, temporal_offsets=args.temporal_offsets)

    pools = collect_latent_pools(model, dataset, view_names, args.batch_size, args.num_workers)

    token_usage = {
        view: {
            "usage": code_usage(pools[view]["indices"], model.num_latents),
            "confidence_mean": float(pools[view]["confidence"].float().mean()),
            "confidence_min": float(pools[view]["confidence"].float().min()),
            "confidence_max": float(pools[view]["confidence"].float().max()),
        }
        for view in view_names
    }

    causality_rows: list[dict] = []
    leakage_rows: list[dict] = []
    if not args.skip_causality or not args.skip_private_leakage:
        causality_rows, leakage_rows = run_reconstruction_probes(
            args,
            model,
            dataset,
            view_names,
            device,
            pools,
            donor_indices,
        )

    if causality_rows:
        save_json(args.output_dir / "causality_ablation.json", {"rows": causality_rows})
        write_csv(args.output_dir / "causality_ablation.csv", causality_rows)
    if leakage_rows:
        save_json(args.output_dir / "private_leakage_ablation.json", {"rows": leakage_rows})
        write_csv(args.output_dir / "private_leakage_ablation.csv", leakage_rows)

    probe_summary = build_probe_summary(causality_rows, leakage_rows, args.eval_paths)
    save_json(args.output_dir / "probe_summary.json", probe_summary)

    semantic_report = run_semantic_probe(args, pools, view_names, metadata, model.num_latents)
    save_json(args.output_dir / "semantic_probe.json", semantic_report)

    temporal_fallback_counts = {
        key: int(value)
        for key, value in donor_indices.items()
        if key.startswith("temporal_offset_") and key.endswith("_fallback_count")
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "input_npz": str(args.input_npz),
        "view_names": view_names,
        "num_samples": len(dataset),
        "eval_paths": args.eval_paths,
        "donor_sampling": {
            "same_take_fallback_count": int(donor_indices["same_take_fallback_count"]),
            "random_take_fallback_count": int(donor_indices["random_take_fallback_count"]),
            "temporal_fallback_counts": temporal_fallback_counts,
            "note": "Fallback means a sample had no eligible same-take or different-take donor.",
        },
        "token_usage": token_usage,
        "outputs": {
            "causality_ablation_json": str(args.output_dir / "causality_ablation.json") if causality_rows else None,
            "causality_ablation_csv": str(args.output_dir / "causality_ablation.csv") if causality_rows else None,
            "private_leakage_ablation_json": str(args.output_dir / "private_leakage_ablation.json")
            if leakage_rows
            else None,
            "private_leakage_ablation_csv": str(args.output_dir / "private_leakage_ablation.csv")
            if leakage_rows
            else None,
            "probe_summary_json": str(args.output_dir / "probe_summary.json"),
            "semantic_probe_json": str(args.output_dir / "semantic_probe.json"),
        },
        "semantic_probe_skipped": bool(semantic_report.get("skipped")),
        "checkpoint_step": int(checkpoint.get("step", -1)),
    }
    save_json(args.output_dir / "action_token_probe_report.json", report)
    print(f"Saved FACT action token probe report to {args.output_dir / 'action_token_probe_report.json'}")


if __name__ == "__main__":
    main()
