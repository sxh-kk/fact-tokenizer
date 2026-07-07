#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import (
    evenly_spaced_indices,
    group_indices_by_take,
    load_labels_by_take,
    load_npz_metadata,
    normalize_score,
    write_csv,
)


TASK_PRIORS = {
    "Cooking": {"interaction": 0.85, "loco": 0.20, "scene": 0.10, "fine": 0.35},
    "Bike Repair": {"interaction": 0.85, "loco": 0.25, "scene": 0.10, "fine": 0.30},
    "Health": {"interaction": 0.75, "loco": 0.20, "scene": 0.15, "fine": 0.25},
    "Music": {"interaction": 0.50, "loco": 0.15, "scene": 0.20, "fine": 0.70},
    "Basketball": {"interaction": 0.25, "loco": 0.85, "scene": 0.35, "fine": 0.05},
    "Soccer": {"interaction": 0.25, "loco": 0.85, "scene": 0.35, "fine": 0.05},
    "Dance": {"interaction": 0.15, "loco": 0.80, "scene": 0.45, "fine": 0.05},
    "Rock Climbing": {"interaction": 0.35, "loco": 0.65, "scene": 0.35, "fine": 0.10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cheap take-level relevance proxy features from FACT NPZ splits.")
    parser.add_argument("--split", "--npz", dest="npz", type=Path, required=True)
    parser.add_argument("--split-name", choices=["train", "heldout", "unknown"], default="unknown")
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--contact-sheet-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument("--sample-transitions-per-take", type=int, default=16)
    return parser.parse_args()


def per_transition_motion(video: np.ndarray) -> np.ndarray:
    frames = video.astype(np.float32)
    if frames.max(initial=0.0) > 1.5:
        frames = frames / 255.0
    if frames.shape[-1] in (1, 3):
        delta = frames[:, -1] - frames[:, 0]
    else:
        delta = frames[:, -1].transpose(0, 2, 3, 1) - frames[:, 0].transpose(0, 2, 3, 1)
    return np.abs(delta).mean(axis=tuple(range(1, delta.ndim))).astype(np.float32)


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


def load_contact_paths(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["take_uid"]: row.get("contact_sheet_path", "") for row in csv.DictReader(handle)}


def bucket_from_scores(interaction: float, loco: float, scene: float, fine: float) -> str:
    if scene >= 0.68 and interaction < 0.45 and loco < 0.55:
        return "D_scene_only"
    if interaction >= 0.62:
        return "A_interaction_rich"
    if loco >= 0.62:
        return "B_loco_body"
    if fine >= 0.62:
        return "E_fine_dexterous"
    if scene >= 0.48:
        return "C_active_view_only"
    return "F_uncertain"


def main() -> None:
    args = parse_args()
    metadata = load_npz_metadata(args.npz)
    labels = load_labels_by_take(args.labels_jsonl)
    contact_paths = load_contact_paths(args.contact_sheet_manifest)
    grouped = group_indices_by_take(metadata["take_uid"])
    sampled_indices, sampled_positions_by_take = sampled_index_map(grouped, args.sample_transitions_per_take)
    with np.load(args.npz, allow_pickle=False) as data:
        motion = {
            view_key: per_transition_motion(np.asarray(data[view_key][sampled_indices]))
            for view_key in args.view_keys
        }

    rows = []
    for take_uid, indices in sorted(grouped.items()):
        sampled = sampled_positions_by_take[take_uid]
        label = labels.get(take_uid, {})
        parent = str(label.get("parent_task_name", ""))
        priors = TASK_PRIORS.get(parent, {"interaction": 0.35, "loco": 0.35, "scene": 0.45, "fine": 0.20})
        ego_values = motion[args.view_keys[0]][sampled]
        exo_values = motion[args.view_keys[1]][sampled]
        ego_motion = normalize_score(float(np.mean(ego_values)), 0.10)
        exo_motion = normalize_score(float(np.mean(exo_values)), 0.08)
        body_motion = exo_motion
        object_motion_proxy = ego_motion
        temporal_diversity = normalize_score(float(np.std(ego_values) + np.std(exo_values)), 0.08)
        low_motion_scene = 1.0 - max(ego_motion, exo_motion)
        scene_only = max(0.0, min(1.0, 0.55 * priors["scene"] + 0.45 * low_motion_scene))
        interaction_score = max(
            0.0,
            min(
                1.0,
                0.45 * priors["interaction"]
                + 0.20 * object_motion_proxy
                + 0.20 * temporal_diversity
                + 0.15 * (1.0 - scene_only),
            ),
        )
        loco_score = max(0.0, min(1.0, 0.50 * priors["loco"] + 0.35 * body_motion + 0.15 * temporal_diversity))
        fine_score = float(priors["fine"])
        relevance = max(
            0.0,
            min(
                1.0,
                0.35 * interaction_score
                + 0.20 * loco_score
                + 0.15 * temporal_diversity
                + 0.15 * object_motion_proxy
                + 0.15 * body_motion
                - 0.25 * scene_only,
            ),
        )
        rows.append(
            {
                "take_uid": take_uid,
                "split": args.split_name,
                "parent_task_name": parent,
                "task_name": label.get("task_name", ""),
                "take_name": label.get("take_name", ""),
                "num_transitions": len(indices),
                "ego_motion_score": round(ego_motion, 6),
                "exo_body_motion_score": round(body_motion, 6),
                "object_motion_proxy": round(object_motion_proxy, 6),
                "temporal_diversity_score": round(temporal_diversity, 6),
                "metadata_interaction_prior": priors["interaction"],
                "metadata_loco_prior": priors["loco"],
                "scene_only_score": round(scene_only, 6),
                "interaction_score": round(interaction_score, 6),
                "loco_score": round(loco_score, 6),
                "fine_dexterous_score": round(fine_score, 6),
                "relevance_score": round(relevance, 6),
                "auto_bucket": bucket_from_scores(interaction_score, loco_score, scene_only, fine_score),
                "contact_sheet_path": contact_paths.get(take_uid, ""),
            }
        )
    fieldnames = [
        "take_uid",
        "split",
        "parent_task_name",
        "task_name",
        "take_name",
        "num_transitions",
        "ego_motion_score",
        "exo_body_motion_score",
        "object_motion_proxy",
        "temporal_diversity_score",
        "metadata_interaction_prior",
        "metadata_loco_prior",
        "scene_only_score",
        "interaction_score",
        "loco_score",
        "fine_dexterous_score",
        "relevance_score",
        "auto_bucket",
        "contact_sheet_path",
    ]
    write_csv(args.out, rows, fieldnames=fieldnames)
    print(f"Saved {len(rows)} take-level feature rows to {args.out}")


if __name__ == "__main__":
    main()
