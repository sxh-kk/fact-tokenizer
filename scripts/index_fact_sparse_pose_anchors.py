#!/usr/bin/env python3
"""Index sparse EgoExo body-wrist and hand-joint 3D deltas as probe-only anchors."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_index import joint_delta, nearest_pose_annotation  # noqa: E402
from fact_tokenizer.effect_manifest import EffectCapability, read_manifest_jsonl, write_manifest_jsonl  # noqa: E402


BODY_JOINTS = ("left-wrist", "right-wrist")
HAND_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("wrist", *(f"{finger}_{index}" for finger in ("thumb", "index", "middle", "ring", "pinky") for index in range(1, 5)))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--body-dir", type=Path, action="append", default=[])
    parser.add_argument("--hand-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-skew-frames", type=int, default=1)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    return parser.parse_args()


def find_json(directories: list[Path], take_uid: str) -> Path | None:
    for directory in directories:
        path = directory / f"{take_uid}.json"
        if path.is_file():
            return path
    return None


def index_kind(
    records: list,
    directories: list[Path],
    joint_names: tuple[str, ...],
    minimum_joints: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[str]]:
    values = np.zeros((len(records), len(joint_names), 3), dtype=np.float32)
    joint_valid = np.zeros((len(records), len(joint_names)), dtype=bool)
    sample_valid = np.zeros(len(records), dtype=bool)
    used_takes: set[str] = set()
    cache: dict[str, dict | None] = {}
    for index, record in enumerate(records):
        if record.take_uid not in cache:
            path = find_json(directories, record.take_uid)
            cache[record.take_uid] = json.loads(path.read_text(encoding="utf-8")) if path else None
        annotation = cache[record.take_uid]
        if annotation is None:
            continue
        start_frame = int(round(float(record.timestamp or 0.0) * args.fps))
        end_frame = int(round((float(record.timestamp or 0.0) + args.transition_seconds) * args.fps))
        first = nearest_pose_annotation(annotation, start_frame, max_skew_frames=args.max_skew_frames)
        second = nearest_pose_annotation(annotation, end_frame, max_skew_frames=args.max_skew_frames)
        if first is None or second is None:
            continue
        delta, valid, usable = joint_delta(
            first[0],
            second[0],
            joint_names,
            minimum_common_joints=minimum_joints,
        )
        if usable:
            values[index] = delta
            joint_valid[index] = valid
            sample_valid[index] = True
            used_takes.add(record.take_uid)
    return values, joint_valid, sample_valid, used_takes


def main() -> None:
    args = parse_args()
    records = read_manifest_jsonl(args.manifest)
    body_delta, body_joint_valid, body_valid, body_takes = index_kind(
        records, args.body_dir, BODY_JOINTS, 1, args
    )
    hand_delta, hand_joint_valid, hand_valid, hand_takes = index_kind(
        records, args.hand_dir, HAND_JOINTS, 5, args
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "body_joint_3d_delta": body_delta,
        "body_joint_3d_joint_valid": body_joint_valid,
        "body_joint_3d_valid": body_valid,
        "hand_joint_3d_delta": hand_delta,
        "hand_joint_3d_joint_valid": hand_joint_valid,
        "hand_joint_3d_valid": hand_valid,
    }
    for name, value in arrays.items():
        np.save(output / f"{name}.npy", value)
    updated = []
    for index, record in enumerate(records):
        capabilities = dict(record.capability_validity)
        capabilities[EffectCapability.BODY_POSE] = bool(body_valid[index])
        capabilities[EffectCapability.HAND_POSE] = bool(hand_valid[index])
        refs = dict(record.annotation_refs)
        if body_valid[index]:
            refs["body_pose_anchor"] = {"row_index": index, "probe_only": True}
        if hand_valid[index]:
            refs["hand_pose_anchor"] = {"row_index": index, "probe_only": True}
        updated.append(replace(record, capability_validity=capabilities, annotation_refs=refs))
    manifest_path = output / "effect_manifest_sparse_pose_indexed.jsonl"
    write_manifest_jsonl(manifest_path, updated)
    report = {
        "samples": len(records),
        "body_valid_samples": int(body_valid.sum()),
        "body_valid_takes": len(body_takes),
        "hand_valid_samples": int(hand_valid.sum()),
        "hand_valid_takes": len(hand_takes),
        "body_joint_order": list(BODY_JOINTS),
        "hand_joint_order": list(HAND_JOINTS),
        "probe_only": True,
        "dense_3d_geometry_claim_allowed": False,
        "manifest": str(manifest_path),
    }
    (output / "sparse_pose_index_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
