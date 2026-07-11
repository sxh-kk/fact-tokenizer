"""Canonical indexing of EgoExo camera poses and weak semantic annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from fact_tokenizer.effect_targets import align_atomic_descriptions, align_phase_segments


def homogeneous_extrinsic(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = matrix
        return result
    raise ValueError(f"camera extrinsic must be 3x4 or 4x4, got {matrix.shape}")


def nearest_dynamic_extrinsic(
    frame_to_extrinsic: Mapping[str, Any],
    timestamp_seconds: float,
    fps: float = 30.0,
    max_skew_frames: int = 1,
) -> Optional[tuple[np.ndarray, int]]:
    if not frame_to_extrinsic:
        return None
    requested = timestamp_seconds * fps
    frames = np.asarray([int(key) for key in frame_to_extrinsic], dtype=np.int64)
    position = int(np.argmin(np.abs(frames - requested)))
    frame = int(frames[position])
    if abs(frame - requested) > max_skew_frames + 1e-9:
        return None
    return homogeneous_extrinsic(frame_to_extrinsic[str(frame)]), frame


def camera_pose_pair_from_egoexo_json(
    payload: Mapping[str, Any],
    camera_name: str,
    start_seconds: float,
    end_seconds: float,
    *,
    fps: float = 30.0,
    max_skew_frames: int = 1,
) -> Optional[dict[str, Any]]:
    camera = payload.get(camera_name)
    if not isinstance(camera, Mapping):
        return None
    intrinsics = np.asarray(camera.get("camera_intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        return None
    extrinsics = camera.get("camera_extrinsics")
    if isinstance(extrinsics, Mapping):
        first = nearest_dynamic_extrinsic(extrinsics, start_seconds, fps, max_skew_frames)
        second = nearest_dynamic_extrinsic(extrinsics, end_seconds, fps, max_skew_frames)
        if first is None or second is None:
            return None
        e0, frame0 = first
        e1, frame1 = second
        dynamic = True
    else:
        try:
            e0 = e1 = homogeneous_extrinsic(extrinsics)
        except (TypeError, ValueError):
            return None
        frame0, frame1 = int(round(start_seconds * fps)), int(round(end_seconds * fps))
        dynamic = False
    source_size = (512, 512) if camera_name.startswith("aria") else (2160, 3840)
    return {
        "intrinsics": np.stack([intrinsics, intrinsics]),
        "world_to_camera": np.stack([e0, e1]),
        "source_size": np.asarray(source_size, dtype=np.int32),
        "aligned_frames": np.asarray([frame0, frame1], dtype=np.int64),
        "dynamic": dynamic,
        "camera_name": camera_name,
    }


def iter_annotation_items(path: Path | str) -> Iterator[tuple[str, Any]]:
    """Stream the top-level ``annotations`` object, including 1.2GB Relations files."""

    source = Path(path)
    try:
        import ijson
    except ImportError:
        if source.stat().st_size > 256 * 1024 * 1024:
            raise RuntimeError(f"ijson is required to stream large annotation file {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        yield from payload.get("annotations", {}).items()
        return
    with source.open("rb") as handle:
        # ``use_float`` avoids Decimal values leaking into JSONL sidecars.
        yield from ijson.kvitems(handle, "annotations", use_float=True)


def atomic_by_take(path: Path | str, selected_takes: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    selected = set(selected_takes)
    result = {take: [] for take in selected}
    for take_uid, annotation_jobs in iter_annotation_items(path):
        if take_uid not in selected:
            continue
        for job in annotation_jobs if isinstance(annotation_jobs, list) else [annotation_jobs]:
            if job.get("rejected"):
                continue
            for description in job.get("descriptions", []):
                row = dict(description)
                row["source"] = "egoexo_atomic_description"
                row["annotation_uid"] = job.get("annotation_uid")
                result[take_uid].append(row)
    return result


def phases_by_take(paths: Sequence[Path | str], selected_takes: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    selected = set(selected_takes)
    result = {take: [] for take in selected}
    for path in paths:
        source_name = Path(path).stem
        for take_uid, annotation in iter_annotation_items(path):
            if take_uid not in selected or not isinstance(annotation, Mapping):
                continue
            for segment in annotation.get("segments", []):
                row = dict(segment)
                row.update(
                    {
                        "start_sec": segment.get("start_time", segment.get("start_sec")),
                        "end_sec": segment.get("end_time", segment.get("end_sec")),
                        "step": segment.get("step_name", segment.get("step_description")),
                        "source": source_name,
                    }
                )
                result[take_uid].append(row)
    return result


def build_weak_sample_index(
    sample_records: Sequence[Mapping[str, Any]],
    atomic: Mapping[str, Sequence[Mapping[str, Any]]],
    phases: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    transition_seconds: float = 0.5,
    atomic_tolerance_seconds: float = 0.75,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in sample_records:
        sample_id = str(sample["sample_id"])
        take_uid = str(sample["take_uid"])
        start = float(sample["timestamp"])
        end = start + transition_seconds
        aligned_atomic = align_atomic_descriptions(
            atomic.get(take_uid, []), start, end, atomic_tolerance_seconds
        )
        aligned_phase = align_phase_segments(phases.get(take_uid, []), start, end)
        rows.append(
            {
                "sample_id": sample_id,
                "take_uid": take_uid,
                "timestamp": start,
                "atomic_descriptions": [row["raw"] for row in aligned_atomic],
                "phase_segments": [row["raw"] for row in aligned_phase],
                "atomic_valid": bool(aligned_atomic),
                "phase_valid": bool(aligned_phase),
                # Filled only after the 60-sample dev mapping audit.
                "verb_map_dev_precision": None,
                "weak_loss_enabled": False,
            }
        )
    return rows


def select_relation_mask_entries(
    relation_take: Mapping[str, Any],
    camera_prefix: str,
    target_frame: int,
    *,
    max_frame_distance: int = 15,
    include_hands: bool = False,
) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Select one nearby annotated frame and all object masks available on it."""

    object_masks = relation_take.get("object_masks", {})
    candidates: list[tuple[int, str, str, Mapping[str, Any]]] = []
    for object_id, cameras in object_masks.items():
        normalized_object = str(object_id).lower().replace("-", "_").replace(" ", "_")
        if not include_hands and ("hand" in normalized_object or "wrist" in normalized_object):
            continue
        if not isinstance(cameras, Mapping):
            continue
        for camera_name, camera in cameras.items():
            if not str(camera_name).startswith(camera_prefix) or not isinstance(camera, Mapping):
                continue
            for frame_text, annotation in camera.get("annotation", {}).items():
                if isinstance(annotation, Mapping) and annotation.get("encodedMask"):
                    candidates.append((int(frame_text), str(object_id), str(camera_name), annotation))
    if not candidates:
        return [], None
    selected_frame = min({frame for frame, _, _, _ in candidates}, key=lambda frame: (abs(frame - target_frame), frame))
    if abs(selected_frame - target_frame) > max_frame_distance:
        return [], None
    selected = [
        {
            "frame": frame,
            "object_id": object_id,
            "camera_name": camera_name,
            "width": int(annotation.get("width", 0)),
            "height": int(annotation.get("height", 0)),
            "encoded_mask": annotation["encodedMask"],
        }
        for frame, object_id, camera_name, annotation in candidates
        if frame == selected_frame
    ]
    return selected, selected_frame


def decode_relation_union(
    entries: Sequence[Mapping[str, Any]],
    *,
    output_size: tuple[int, int],
    decoder: Any,
) -> np.ndarray:
    """Decode and nearest-resize a union mask; decoder is injectable for tests."""

    import torch
    import torch.nn.functional as F

    output = np.zeros(output_size, dtype=bool)
    for entry in entries:
        encoded = {
            "width": int(entry["width"]),
            "height": int(entry["height"]),
            "encodedMask": entry["encoded_mask"],
        }
        decoded = np.asarray(decoder(encoded), dtype=bool)
        if decoded.ndim != 2:
            raise ValueError(f"decoded relation mask must be 2D, got {decoded.shape}")
        if decoded.shape != output_size:
            tensor = torch.from_numpy(decoded.astype(np.float32))[None, None]
            decoded = F.interpolate(tensor, size=output_size, mode="nearest")[0, 0].bool().numpy()
        output |= decoded
    return output


def nearest_pose_annotation(
    frame_annotations: Mapping[str, Any],
    target_frame: int,
    *,
    max_skew_frames: int = 1,
) -> Optional[tuple[Mapping[str, Any], int]]:
    if not frame_annotations:
        return None
    frames = [int(frame) for frame in frame_annotations]
    selected = min(frames, key=lambda frame: (abs(frame - target_frame), frame))
    if abs(selected - target_frame) > max_skew_frames:
        return None
    people = frame_annotations[str(selected)]
    if not isinstance(people, list):
        people = [people]
    valid_people = [person for person in people if isinstance(person, Mapping) and isinstance(person.get("annotation3D"), Mapping)]
    if not valid_people:
        return None
    person = max(valid_people, key=lambda value: len(value["annotation3D"]))
    return person["annotation3D"], selected


def joint_delta(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    joint_names: Sequence[str],
    *,
    minimum_common_joints: int = 1,
) -> tuple[np.ndarray, np.ndarray, bool]:
    delta = np.zeros((len(joint_names), 3), dtype=np.float32)
    valid = np.zeros(len(joint_names), dtype=bool)
    for index, name in enumerate(joint_names):
        if name not in first or name not in second:
            continue
        try:
            start = np.asarray([first[name][axis] for axis in ("x", "y", "z")], dtype=np.float32)
            end = np.asarray([second[name][axis] for axis in ("x", "y", "z")], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(start).all() and np.isfinite(end).all():
            delta[index] = end - start
            valid[index] = True
    return delta, valid, bool(valid.sum() >= minimum_common_joints)
