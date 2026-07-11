#!/usr/bin/env python3
"""Index EgoExo camera pose, atomic text and phase annotations onto an effect manifest."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_index import (  # noqa: E402
    atomic_by_take,
    build_weak_sample_index,
    camera_pose_pair_from_egoexo_json,
    phases_by_take,
)
from fact_tokenizer.effect_manifest import EffectCapability, read_manifest_jsonl, write_manifest_jsonl  # noqa: E402
from fact_tokenizer.effect_targets import scale_intrinsics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-pose-dir", type=Path, action="append", default=[])
    parser.add_argument("--camera-map-jsonl", type=Path)
    parser.add_argument("--atomic-json", type=Path, action="append", default=[])
    parser.add_argument("--phase-json", type=Path, action="append", default=[])
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument("--pose-fps", type=float, default=30.0)
    parser.add_argument("--max-pose-skew-frames", type=int, default=1)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--ego-camera", default="aria01")
    parser.add_argument("--default-exo-camera", help="Only safe for a single-camera subset; normally use --camera-map-jsonl.")
    return parser.parse_args()


def load_camera_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            take_uid = str(row.get("take_uid", ""))
            camera = row.get("exo_camera") or row.get("exo_camera_name") or row.get("camera_name")
            if not camera:
                relative = str(row.get("exo_relative_path", ""))
                match = re.search(r"(?:^|[/_])(cam\d+)(?:[/_.]|$)", relative)
                camera = match.group(1) if match else None
            if not take_uid or not camera:
                raise ValueError(f"camera map row {line_number} needs take_uid and exo camera")
            if take_uid in result and result[take_uid] != camera:
                raise ValueError(f"conflicting exo cameras for take {take_uid}")
            result[take_uid] = str(camera)
    return result


def find_pose_json(directories: list[Path], take_uid: str) -> Path | None:
    for directory in directories:
        candidate = directory / f"{take_uid}.json"
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    records = read_manifest_jsonl(args.manifest)
    count = len(records)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    camera_map = load_camera_map(args.camera_map_jsonl)
    pose_cache: dict[str, tuple[dict | None, Path | None]] = {}
    sidecars = {
        view: {
            "intrinsics": np.zeros((count, 2, 3, 3), dtype=np.float64),
            "world_to_camera": np.zeros((count, 2, 4, 4), dtype=np.float64),
            "source_size": np.zeros((count, 2), dtype=np.int32),
            "valid": np.zeros(count, dtype=bool),
            "aligned_frames": np.full((count, 2), -1, dtype=np.int64),
            "camera_context": np.zeros((count, 18), dtype=np.float32),
        }
        for view in ("ego", "exo")
    }
    pose_paths: set[Path] = set()
    for index, record in enumerate(records):
        take_uid = record.take_uid
        if take_uid not in pose_cache:
            path = find_pose_json(args.camera_pose_dir, take_uid)
            payload = json.loads(path.read_text(encoding="utf-8")) if path else None
            pose_cache[take_uid] = payload, path
        payload, pose_path = pose_cache[take_uid]
        if payload is None:
            continue
        pose_paths.add(pose_path)
        cameras = {
            "ego": args.ego_camera,
            "exo": camera_map.get(take_uid, args.default_exo_camera),
        }
        for view, camera_name in cameras.items():
            if not camera_name:
                continue
            pair = camera_pose_pair_from_egoexo_json(
                payload,
                camera_name,
                float(record.timestamp or 0.0),
                float(record.timestamp or 0.0) + args.transition_seconds,
                fps=args.pose_fps,
                max_skew_frames=args.max_pose_skew_frames,
            )
            if pair is None:
                continue
            for key in ("intrinsics", "world_to_camera", "source_size", "aligned_frames"):
                sidecars[view][key][index] = pair[key]
            sidecars[view]["valid"][index] = True
            scaled_k = scale_intrinsics(
                pair["intrinsics"][0],
                tuple(int(value) for value in pair["source_size"]),
                (args.output_size, args.output_size),
            )
            sidecars[view]["camera_context"][index] = np.concatenate(
                [scaled_k.reshape(-1), pair["world_to_camera"][0, :3, :3].reshape(-1)]
            )

    for view, arrays in sidecars.items():
        directory = output / "camera" / view
        directory.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            if name == "camera_context":
                np.save(output / f"{view}_camera_context.npy", array)
            else:
                np.save(directory / f"{name}.npy", array)

    selected_takes = {record.take_uid for record in records}
    atomic: dict[str, list[dict]] = {take: [] for take in selected_takes}
    for path in args.atomic_json:
        indexed = atomic_by_take(path, selected_takes)
        for take, values in indexed.items():
            atomic[take].extend(values)
    phases = phases_by_take(args.phase_json, selected_takes)
    sample_records = [
        {"sample_id": record.sample_id, "take_uid": record.take_uid, "timestamp": record.timestamp or 0.0}
        for record in records
    ]
    weak_rows = build_weak_sample_index(
        sample_records,
        atomic,
        phases,
        transition_seconds=args.transition_seconds,
    )
    weak_path = output / "weak_index.jsonl"
    with weak_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in weak_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_valid = np.asarray([row["atomic_valid"] for row in weak_rows], dtype=bool)
    phase_valid = np.asarray([row["phase_valid"] for row in weak_rows], dtype=bool)
    camera_valid = sidecars["ego"]["valid"] | sidecars["exo"]["valid"]
    np.save(output / "camera_pose_valid.npy", camera_valid)
    np.save(output / "atomic_text_valid.npy", atomic_valid)
    np.save(output / "phase_valid.npy", phase_valid)

    updated = []
    for index, record in enumerate(records):
        capabilities = dict(record.capability_validity)
        capabilities[EffectCapability.CAMERA_POSE] = bool(camera_valid[index])
        capabilities[EffectCapability.ATOMIC_TEXT] = bool(atomic_valid[index])
        capabilities[EffectCapability.PHASE] = bool(phase_valid[index])
        refs = dict(record.annotation_refs)
        refs.update(
            {
                "weak_index": f"{weak_path}#{record.sample_id}",
                "ego_camera_sidecar": str(output / "camera" / "ego"),
                "exo_camera_sidecar": str(output / "camera" / "exo"),
            }
        )
        updated.append(replace(record, capability_validity=capabilities, annotation_refs=refs))
    manifest_path = output / "effect_manifest_indexed.jsonl"
    write_manifest_jsonl(manifest_path, updated)
    report = {
        "samples": count,
        "takes": len(selected_takes),
        "camera_pose_valid": int(camera_valid.sum()),
        "ego_camera_pose_valid": int(sidecars["ego"]["valid"].sum()),
        "exo_camera_pose_valid": int(sidecars["exo"]["valid"].sum()),
        "atomic_text_valid": int(atomic_valid.sum()),
        "phase_valid": int(phase_valid.sum()),
        "pose_files": len(pose_paths),
        "source_hashes": {
            str(path): sha256_file(path)
            for path in [*args.atomic_json, *args.phase_json]
        },
        "pose_convention": "world_to_camera as stored in EgoExo camera_extrinsics",
        "max_pose_skew_frames": args.max_pose_skew_frames,
        "output_manifest": str(manifest_path),
    }
    (output / "index_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
