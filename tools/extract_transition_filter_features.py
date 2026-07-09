#!/usr/bin/env python3
"""Extract transition-level filtering proxies without VLM dependencies."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import load_labels_by_take, load_npz_metadata, read_csv, write_csv


LABEL_WEIGHTS = {
    "main_interaction": 1.0,
    "phase_context": 0.35,
    "loco_only": 0.10,
    "discard": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", "--npz", dest="input_npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--take-weight-csv", type=Path, default=None)
    parser.add_argument("--take-weight-column", default="sample_weight")
    parser.add_argument("--take-weight-uid-column", default="take_uid")
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--spatial-stride", type=int, default=4)
    parser.add_argument("--main-threshold", type=float, default=0.58)
    parser.add_argument("--phase-threshold", type=float, default=0.36)
    parser.add_argument("--loco-threshold", type=float, default=0.24)
    parser.add_argument("--min-main-sync", type=float, default=0.35)
    parser.add_argument("--max-main-scene-risk", type=float, default=0.68)
    parser.add_argument("--min-weight", type=float, default=0.0)
    return parser.parse_args()


def load_view(input_path: Path, key: str) -> np.ndarray:
    if input_path.is_dir():
        return np.load(input_path / f"{key}.npy", mmap_mode="r")
    data = np.load(input_path, allow_pickle=False)
    return data[key]


def frame_delta_features(video: np.ndarray, spatial_stride: int) -> dict[str, np.ndarray]:
    if video.ndim != 5:
        raise ValueError(f"Expected view array with shape (N,T,H,W,C) or (N,T,C,H,W), got {video.shape}")
    stride = max(1, int(spatial_stride))
    if video.shape[-1] in (1, 3):
        first = video[:, 0, ::stride, ::stride, :].astype(np.float32)
        last = video[:, -1, ::stride, ::stride, :].astype(np.float32)
        delta = np.abs(last - first)
    elif video.shape[2] in (1, 3):
        first = video[:, 0, :, ::stride, ::stride].astype(np.float32)
        last = video[:, -1, :, ::stride, ::stride].astype(np.float32)
        delta = np.abs(np.transpose(last - first, (0, 2, 3, 1)))
    else:
        raise ValueError(f"Cannot infer channels for view shape {video.shape}")
    if np.issubdtype(video.dtype, np.integer) or delta.max(initial=0.0) > 1.5:
        delta = delta / 255.0
    global_motion = delta.mean(axis=(1, 2, 3))
    gray = delta.mean(axis=3)
    n, height, width = gray.shape
    grid_h = max(1, height // 4)
    grid_w = max(1, width // 4)
    cells = []
    for y in range(4):
        y0 = y * grid_h
        y1 = height if y == 3 else min(height, (y + 1) * grid_h)
        for x in range(4):
            x0 = x * grid_w
            x1 = width if x == 3 else min(width, (x + 1) * grid_w)
            cells.append(gray[:, y0:y1, x0:x1].mean(axis=(1, 2)))
    cell_motion = np.stack(cells, axis=1)
    local_peak = np.percentile(cell_motion, 90, axis=1)
    local_contrast = local_peak / (global_motion + 1e-6)
    center = gray[:, height // 4 : max(height // 4 + 1, 3 * height // 4), width // 4 : max(width // 4 + 1, 3 * width // 4)]
    center_motion = center.mean(axis=(1, 2)) if center.size else global_motion
    return {
        "global_motion": global_motion.astype(np.float32),
        "local_peak_motion": local_peak.astype(np.float32),
        "local_contrast": local_contrast.astype(np.float32),
        "center_motion": center_motion.astype(np.float32),
    }


def robust_norm(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def take_phase_change(take_uid: np.ndarray, timestamp: np.ndarray, signal: np.ndarray) -> np.ndarray:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, take in enumerate(take_uid.tolist()):
        grouped[str(take)].append(index)
    phase = np.zeros_like(signal, dtype=np.float32)
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda idx: float(timestamp[idx]))
        if len(ordered) < 3:
            phase[ordered] = signal[ordered]
            continue
        values = signal[ordered]
        prev_values = np.r_[values[0], values[:-1]]
        next_values = np.r_[values[1:], values[-1]]
        phase[ordered] = np.maximum(np.abs(values - prev_values), np.abs(next_values - values))
    return robust_norm(phase)


def load_take_weights(path: Path | None, uid_column: str, weight_column: str) -> dict[str, float]:
    if path is None:
        return {}
    weights: dict[str, float] = {}
    for row in read_csv(path):
        uid = str(row.get(uid_column, "")).strip()
        if not uid:
            continue
        try:
            weights[uid] = float(row.get(weight_column, "1") or 1.0)
        except ValueError:
            weights[uid] = 1.0
    return weights


def auto_label(row: dict[str, float], args: argparse.Namespace) -> str:
    if row["scene_only_score"] >= 0.82 or row["interaction_score"] < args.loco_threshold * 0.65:
        return "discard"
    if (
        row["interaction_score"] >= args.main_threshold
        and row["ego_exo_sync_score"] >= args.min_main_sync
        and row["scene_only_score"] <= args.max_main_scene_risk
    ):
        return "main_interaction"
    if row["interaction_score"] >= args.phase_threshold or row["phase_change_score"] >= 0.55:
        return "phase_context"
    if max(row["ego_motion_score"], row["exo_motion_score"]) >= args.loco_threshold:
        return "loco_only"
    return "discard"


def main() -> None:
    args = parse_args()
    metadata = load_npz_metadata(args.input_npz)
    labels = load_labels_by_take(args.labels_jsonl)
    take_weights = load_take_weights(args.take_weight_csv, args.take_weight_uid_column, args.take_weight_column)
    views = {key: load_view(args.input_npz, key) for key in args.view_keys}
    num_rows = len(metadata["sample_id"])
    if any(len(view) != num_rows for view in views.values()):
        raise ValueError("View arrays do not match metadata length")

    feature_blocks: dict[str, dict[str, list[np.ndarray]]] = {
        key: {"global_motion": [], "local_peak_motion": [], "local_contrast": [], "center_motion": []}
        for key in args.view_keys
    }
    for start in range(0, num_rows, args.chunk_size):
        end = min(num_rows, start + args.chunk_size)
        for key, array in views.items():
            features = frame_delta_features(np.asarray(array[start:end]), args.spatial_stride)
            for feature_name, values in features.items():
                feature_blocks[key][feature_name].append(values)

    flat: dict[str, dict[str, np.ndarray]] = {}
    for key in args.view_keys:
        flat[key] = {
            feature_name: np.concatenate(parts, axis=0)
            for feature_name, parts in feature_blocks[key].items()
        }

    ego_key, exo_key = args.view_keys
    ego_motion = robust_norm(flat[ego_key]["global_motion"])
    exo_motion = robust_norm(flat[exo_key]["global_motion"])
    ego_local = robust_norm(flat[ego_key]["local_peak_motion"] * np.minimum(flat[ego_key]["local_contrast"], 4.0))
    exo_local = robust_norm(flat[exo_key]["local_peak_motion"] * np.minimum(flat[exo_key]["local_contrast"], 4.0))
    object_motion = robust_norm(0.7 * flat[ego_key]["center_motion"] + 0.3 * flat[ego_key]["local_peak_motion"])
    both_active = np.sqrt(np.clip(ego_motion * exo_motion, 0.0, 1.0))
    sync = np.clip(1.0 - np.abs(ego_motion - exo_motion), 0.0, 1.0) * both_active
    combined_motion = 0.55 * ego_local + 0.25 * exo_local + 0.20 * object_motion
    phase = take_phase_change(metadata["take_uid"], metadata["timestamp"], combined_motion)
    scene_only = np.clip(ego_motion * (1.0 - np.clip(flat[ego_key]["local_contrast"] / 2.0, 0.0, 1.0)), 0.0, 1.0)
    interaction = np.clip(
        0.35 * ego_local
        + 0.20 * exo_local
        + 0.20 * object_motion
        + 0.15 * sync
        + 0.10 * phase
        - 0.20 * scene_only,
        0.0,
        1.0,
    )

    rows: list[dict[str, Any]] = []
    label_counts: dict[str, int] = defaultdict(int)
    for index in range(num_rows):
        take_uid = str(metadata["take_uid"][index])
        label = labels.get(take_uid, {})
        scores = {
            "ego_motion_score": float(ego_motion[index]),
            "exo_motion_score": float(exo_motion[index]),
            "ego_local_motion_score": float(ego_local[index]),
            "exo_local_motion_score": float(exo_local[index]),
            "object_motion_score": float(object_motion[index]),
            "phase_change_score": float(phase[index]),
            "ego_exo_sync_score": float(sync[index]),
            "scene_only_score": float(scene_only[index]),
            "interaction_score": float(interaction[index]),
        }
        label_name = auto_label(scores, args)
        label_counts[label_name] += 1
        take_weight = float(take_weights.get(take_uid, 1.0))
        transition_weight = max(args.min_weight, LABEL_WEIGHTS[label_name] * take_weight)
        rows.append(
            {
                "split": args.split_name,
                "row_index": index,
                "sample_id": str(metadata["sample_id"][index]),
                "take_uid": take_uid,
                "timestamp": f"{float(metadata['timestamp'][index]):.6f}",
                "parent_task_name": label.get("parent_task_name", ""),
                "task_name": label.get("task_name", ""),
                "take_name": label.get("take_name", ""),
                **{key: f"{value:.6f}" for key, value in scores.items()},
                "take_weight": f"{take_weight:.6f}",
                "auto_label": label_name,
                "transition_weight": f"{transition_weight:.6f}",
            }
        )

    fieldnames = [
        "split",
        "row_index",
        "sample_id",
        "take_uid",
        "timestamp",
        "parent_task_name",
        "task_name",
        "take_name",
        "ego_motion_score",
        "exo_motion_score",
        "ego_local_motion_score",
        "exo_local_motion_score",
        "object_motion_score",
        "phase_change_score",
        "ego_exo_sync_score",
        "scene_only_score",
        "interaction_score",
        "take_weight",
        "auto_label",
        "transition_weight",
    ]
    write_csv(args.out, rows, fieldnames)
    summary = ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items()))
    print(f"Saved {len(rows)} transition feature rows to {args.out} ({summary})", flush=True)


if __name__ == "__main__":
    main()
