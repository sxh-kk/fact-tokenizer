#!/usr/bin/env python3
"""Extract lightweight ego hand/object/contact proxy features for filtering_v2.

This MVP intentionally uses only frames already present in FACT NPZ files. The
columns match the v2 contract so they can later be replaced by EgoHOS,
GroundingDINO/SAM2, or contact-aware models without changing downstream tools.
"""

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
    parser.add_argument("--view-key", default="ego")
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


def transition_proxy(video: np.ndarray) -> dict[str, np.ndarray]:
    values = to_float_video(video)
    first = values[:, 0]
    last = values[:, -1]
    delta = np.abs(last - first).mean(axis=-1)
    h, w = delta.shape[1:3]
    y0, y1 = int(0.35 * h), int(0.95 * h)
    x0, x1 = int(0.12 * w), int(0.88 * w)
    center_delta = delta[:, y0:y1, x0:x1]
    global_motion = delta.mean(axis=(1, 2))
    center_motion = center_delta.mean(axis=(1, 2))
    edge_motion = np.concatenate(
        [
            delta[:, : max(1, h // 8), :].reshape(delta.shape[0], -1),
            delta[:, -max(1, h // 8) :, :].reshape(delta.shape[0], -1),
            delta[:, :, : max(1, w // 8)].reshape(delta.shape[0], -1),
            delta[:, :, -max(1, w // 8) :].reshape(delta.shape[0], -1),
        ],
        axis=1,
    ).mean(axis=1)
    texture = last.std(axis=(1, 2, 3))

    grid_h = max(1, h // 4)
    grid_w = max(1, w // 4)
    grid_values = []
    for gy in range(4):
        for gx in range(4):
            crop = delta[:, gy * grid_h : min(h, (gy + 1) * grid_h), gx * grid_w : min(w, (gx + 1) * grid_w)]
            if crop.size:
                grid_values.append(crop.mean(axis=(1, 2)))
    grid = np.stack(grid_values, axis=1)
    concentration = grid.max(axis=1) / np.maximum(grid.mean(axis=1), 1e-6)

    # Egocentric hands/objects often occupy the lower-center region and produce
    # localized motion. Camera shake tends to move the whole frame, including edges.
    localized_motion = np.clip(center_motion - 0.55 * edge_motion, 0.0, None)
    ego_hand_score = np.clip(0.55 * normalize_array(localized_motion, 0.055) + 0.45 * normalize_array(concentration - 1.0, 2.4), 0.0, 1.0)
    object_presence = np.clip(0.60 * normalize_array(texture, 0.28) + 0.40 * normalize_array(center_motion, 0.055), 0.0, 1.0)
    object_motion = np.clip(0.70 * normalize_array(center_motion, 0.060) + 0.30 * normalize_array(global_motion, 0.075), 0.0, 1.0)
    contact = np.clip(0.45 * ego_hand_score + 0.35 * object_motion + 0.20 * object_presence, 0.0, 1.0)
    interacting_object = np.clip(0.50 * object_presence + 0.50 * contact, 0.0, 1.0)
    return {
        "ego_hand_score": ego_hand_score,
        "ego_hand_visibility_prob": ego_hand_score,
        "object_presence_score": object_presence,
        "object_motion_score": object_motion,
        "hand_object_contact_score": contact,
        "interacting_object_score": interacting_object,
        "contact_state_change_score": normalize_array(np.abs(contact - np.median(contact)), 0.22),
    }


def normalize_array(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


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
                "ego_hand_score": round(summarize(features["ego_hand_score"], sampled), 6),
                "ego_hand_visibility_prob": round(summarize(features["ego_hand_visibility_prob"], sampled), 6),
                "object_presence_score": round(summarize(features["object_presence_score"], sampled), 6),
                "object_motion_score": round(summarize(features["object_motion_score"], sampled), 6),
                "hand_object_contact_score": round(summarize(features["hand_object_contact_score"], sampled), 6),
                "interacting_object_score": round(summarize(features["interacting_object_score"], sampled), 6),
                "contact_state_change_score": round(summarize(features["contact_state_change_score"], sampled), 6),
                "feature_source_ego_hand_object": "npz_motion_proxy_mvp",
            }
        )
    write_csv(args.out, rows)
    print(f"Saved {len(rows)} ego hand/object proxy rows to {args.out}")


if __name__ == "__main__":
    main()
