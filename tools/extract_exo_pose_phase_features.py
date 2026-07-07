#!/usr/bin/env python3
"""Extract lightweight exo body/loco/phase proxy features for filtering_v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import evenly_spaced_indices, group_indices_by_take, load_npz_metadata, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", "--npz", dest="npz", type=Path, required=True)
    parser.add_argument("--split-name", choices=["train", "heldout", "unknown"], default="unknown")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view-key", default="exo")
    parser.add_argument("--sample-transitions-per-take", type=int, default=24)
    return parser.parse_args()


def to_float_video(video: np.ndarray) -> np.ndarray:
    values = video.astype(np.float32)
    if values.max(initial=0.0) > 1.5:
        values = values / 255.0
    if values.shape[-1] in (1, 3):
        return values
    if values.shape[2] in (1, 3):
        return np.transpose(values, (0, 1, 3, 4, 2))
    raise ValueError(f"Unsupported video layout {video.shape}; expected [N,T,H,W,C] or [N,T,C,H,W]")


def normalize(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def transition_proxy(video: np.ndarray) -> dict[str, np.ndarray]:
    values = to_float_video(video)
    first = values[:, 0]
    last = values[:, -1]
    delta = np.abs(last - first).mean(axis=-1)
    h, w = delta.shape[1:3]
    center = delta[:, int(0.10 * h) : int(0.92 * h), int(0.08 * w) : int(0.92 * w)]
    global_motion = delta.mean(axis=(1, 2))
    center_motion = center.mean(axis=(1, 2))
    texture = last.std(axis=(1, 2, 3))

    grid_h = max(1, h // 5)
    grid_w = max(1, w // 5)
    grid_values = []
    for gy in range(5):
        for gx in range(5):
            crop = delta[:, gy * grid_h : min(h, (gy + 1) * grid_h), gx * grid_w : min(w, (gx + 1) * grid_w)]
            if crop.size:
                grid_values.append(crop.mean(axis=(1, 2)))
    grid = np.stack(grid_values, axis=1)
    spatial_extent = (grid > (grid.mean(axis=1, keepdims=True) + 0.35 * grid.std(axis=1, keepdims=True))).mean(axis=1)

    body_motion = np.clip(0.65 * normalize(center_motion, 0.055) + 0.35 * normalize(global_motion, 0.070), 0.0, 1.0)
    body_visibility = np.clip(0.50 * normalize(texture, 0.30) + 0.30 * body_motion + 0.20 * normalize(spatial_extent, 0.35), 0.0, 1.0)
    pose_confidence = np.clip(0.55 * body_visibility + 0.45 * normalize(spatial_extent, 0.35), 0.0, 1.0)
    return {
        "exo_body_visibility_score": body_visibility,
        "exo_pose_confidence": pose_confidence,
        "exo_body_motion_score_v2": body_motion,
        "body_phase_diversity_score": normalize(np.abs(body_motion - np.median(body_motion)), 0.24),
        "loco_motion_score": body_motion,
        "pose_state_change_score": normalize(np.abs(spatial_extent - np.median(spatial_extent)), 0.22),
    }


def summarize(values: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.mean(values[indices]))


def sampled_index_map(grouped: dict[str, list[int]], count: int) -> tuple[list[int], dict[str, list[int]]]:
    sampled_by_take = {
        take_uid: evenly_spaced_indices(indices, count)
        for take_uid, indices in grouped.items()
    }
    sampled_indices = sorted({index for indices in sampled_by_take.values() for index in indices})
    sampled_position = {index: position for position, index in enumerate(sampled_indices)}
    sampled_positions_by_take = {
        take_uid: [sampled_position[index] for index in indices]
        for take_uid, indices in sampled_by_take.items()
    }
    return sampled_indices, sampled_positions_by_take


def main() -> None:
    args = parse_args()
    metadata = load_npz_metadata(args.npz)
    grouped = group_indices_by_take(metadata["take_uid"])
    sampled_indices, sampled_positions_by_take = sampled_index_map(grouped, args.sample_transitions_per_take)
    with np.load(args.npz, allow_pickle=False) as data:
        if args.view_key not in data:
            raise KeyError(f"{args.npz} does not contain view key {args.view_key!r}")
        features = transition_proxy(np.asarray(data[args.view_key][sampled_indices]))

    rows = []
    for take_uid, indices in sorted(grouped.items()):
        sampled = sampled_positions_by_take[take_uid]
        rows.append(
            {
                "take_uid": take_uid,
                "split": args.split_name,
                "exo_body_visibility_score": round(summarize(features["exo_body_visibility_score"], sampled), 6),
                "exo_pose_confidence": round(summarize(features["exo_pose_confidence"], sampled), 6),
                "exo_body_motion_score_v2": round(summarize(features["exo_body_motion_score_v2"], sampled), 6),
                "body_phase_diversity_score": round(summarize(features["body_phase_diversity_score"], sampled), 6),
                "loco_motion_score": round(summarize(features["loco_motion_score"], sampled), 6),
                "pose_state_change_score": round(summarize(features["pose_state_change_score"], sampled), 6),
                "feature_source_exo_pose_phase": "npz_motion_proxy_mvp",
            }
        )
    write_csv(args.out, rows)
    print(f"Saved {len(rows)} exo pose/phase proxy rows to {args.out}")


if __name__ == "__main__":
    main()
