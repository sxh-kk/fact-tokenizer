#!/usr/bin/env python3
"""Decode the frozen short73 manifest into atomic mmap NPY arrays."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_fact_egoexo_npz import resolve_video  # noqa: E402
from fact_tokenizer.effect_manifest import (  # noqa: E402
    EffectCapability,
    EffectSampleRecord,
    write_manifest_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoexo-root", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument(
        "--audit-only-provisional",
        action="store_true",
        help="Required to decode a provisional freeze solely for perceptual leakage audit.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_frame_with_index(
    capture: cv2.VideoCapture,
    timestamp_seconds: float,
    resize: int,
) -> tuple[int, np.ndarray]:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or abs(fps - 30.0) > 1e-3:
        raise ValueError(f"short73 materialization requires 30Hz video, got {fps}")
    if not capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_seconds, 0.0) * 1000.0):
        raise RuntimeError("decoder rejected short73 timestamp seek")
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("short73 timestamp decode failed")
    frame_index = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES)))) - 1
    rgb = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        (resize, resize),
        interpolation=cv2.INTER_AREA,
    )
    return frame_index, rgb


def array_identity(directory: Path, name: str) -> dict:
    path = directory / f"{name}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite locked data: {args.output_dir}")
    freeze = json.loads((args.locked_dir / "freeze.json").read_text(encoding="utf-8"))
    freeze_stage = str(freeze.get("freeze_stage", ""))
    if freeze_stage == "provisional" and not args.audit_only_provisional:
        raise ValueError("provisional locked data may only be decoded with --audit-only-provisional")
    if freeze_stage == "final" and args.audit_only_provisional:
        raise ValueError("--audit-only-provisional cannot be used with a final freeze")
    if freeze_stage not in {"provisional", "final"}:
        raise ValueError("locked freeze has no recognized provisional/final stage")
    sample_path = args.locked_dir / "samples.jsonl"
    selected_path = args.locked_dir / "selected_takes.jsonl"
    if sha256_file(sample_path) != freeze["samples_sha256"] or sha256_file(selected_path) != freeze["selected_takes_sha256"]:
        raise ValueError("locked manifest hash differs from freeze.json")
    samples = load_jsonl(sample_path)
    takes = {row["take_uid"]: row for row in load_jsonl(selected_path)}
    if len(samples) != 584 or len(takes) != 73:
        raise ValueError("short73 materialization requires exactly 73 takes and 584 samples")
    transition_seconds = float(freeze.get("config", {}).get("transition_seconds", -1.0))
    if transition_seconds != 0.5 or any(
        abs((float(row["end_timestamp"]) - float(row["timestamp"])) - transition_seconds) > 1e-6
        for row in samples
    ):
        raise ValueError("short73 source contract requires exact t→t+0.5s endpoints")
    by_take: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_take[sample["take_uid"]].append((index, sample))
    staging = args.output_dir.with_name(f".{args.output_dir.name}.staging-{uuid.uuid4().hex[:12]}")
    staging.mkdir(parents=True)
    shape = (len(samples), 2, args.resize, args.resize, 3)
    ego = np.lib.format.open_memmap(staging / "ego.npy", mode="w+", dtype=np.uint8, shape=shape)
    exo = np.lib.format.open_memmap(staging / "exo.npy", mode="w+", dtype=np.uint8, shape=shape)
    frame_indices = np.full(len(samples), -1, dtype=np.int64)
    video_inventory = []
    for take_uid in sorted(by_take):
        row = takes[take_uid]
        ego_path = resolve_video(args.egoexo_root, row, row["ego_relative_path"])
        exo_path = resolve_video(args.egoexo_root, row, row["exo_relative_path"])
        if ego_path is None or exo_path is None:
            raise FileNotFoundError(f"missing locked videos for {take_uid}: ego={ego_path}, exo={exo_path}")
        video_inventory.append(
            {
                "take_uid": take_uid,
                "ego_path": str(ego_path),
                "ego_sha256": sha256_file(ego_path),
                "exo_path": str(exo_path),
                "exo_sha256": sha256_file(exo_path),
            }
        )
        ego_cap = cv2.VideoCapture(str(ego_path))
        exo_cap = cv2.VideoCapture(str(exo_path))
        if not ego_cap.isOpened() or not exo_cap.isOpened():
            raise RuntimeError(f"cannot open locked videos for {take_uid}")
        try:
            for index, sample in sorted(by_take[take_uid]):
                endpoint_indices = {}
                pairs = {}
                for view, capture in (("ego", ego_cap), ("exo", exo_cap)):
                    start_index, start = read_frame_with_index(
                        capture, float(sample["timestamp"]), args.resize
                    )
                    end_index, end = read_frame_with_index(
                        capture, float(sample["end_timestamp"]), args.resize
                    )
                    endpoint_indices[view] = (start_index, end_index)
                    pairs[view] = np.stack([start, end], axis=0)
                if (
                    endpoint_indices["ego"] != endpoint_indices["exo"]
                    or endpoint_indices["ego"][1] - endpoint_indices["ego"][0] != 15
                    or abs(
                        endpoint_indices["ego"][0]
                        - float(sample["timestamp"]) * 30.0
                    )
                    > 0.501
                ):
                    raise ValueError(
                        f"short73 endpoint frame contract failed for {sample['sample_id']}: "
                        f"{endpoint_indices}"
                    )
                frame_indices[index] = endpoint_indices["ego"][0]
                ego[index] = pairs["ego"]
                exo[index] = pairs["exo"]
        finally:
            ego_cap.release()
            exo_cap.release()
    ego.flush()
    exo.flush()
    del ego, exo
    np.save(staging / "sample_id.npy", np.asarray([row["sample_id"] for row in samples]))
    np.save(staging / "take_uid.npy", np.asarray([row["take_uid"] for row in samples]))
    np.save(staging / "timestamp.npy", np.asarray([row["timestamp"] for row in samples], dtype=np.float32))
    if (frame_indices < 0).any():
        raise AssertionError("some short73 rows have no exact raw frame index")
    np.save(staging / "frame_index.npy", frame_indices)
    np.save(staging / "role.npy", np.asarray(["locked_test"] * len(samples)))
    np.save(staging / "training_valid.npy", np.zeros(len(samples), dtype=bool))
    locked_manifest = [
        EffectSampleRecord(
            sample_id=row["sample_id"],
            take_uid=row["take_uid"],
            split="locked_test",
            row_index=index,
            source_dataset="egoexo_short73",
            timestamp=float(row["timestamp"]),
            capability_validity={EffectCapability.RGB_PAIRED: True},
            training_valid=False,
            provenance={
                "locked_set_id": freeze["locked_set_id"],
                "freeze_stage": freeze_stage,
                "final_inference_only": True,
            },
        )
        for index, row in enumerate(samples)
    ]
    write_manifest_jsonl(staging / "effect_manifest_locked.jsonl", locked_manifest)
    report = {
        "schema": "fact-short73-materialization-v3",
        "locked_set_id": freeze["locked_set_id"],
        "samples": len(samples),
        "takes": len(takes),
        "shape": list(shape),
        "resize": args.resize,
        "color_space": "RGB",
        "transition_seconds": transition_seconds,
        "endpoint_semantics": ["t", "t+0.5s"],
        "frame_rate_hz": 30.0,
        "endpoint_offset_frames": 15,
        "maximum_t0_timestamp_distance_frames": float(
            np.max(
                np.abs(
                    frame_indices
                    - np.asarray([row["timestamp"] for row in samples], dtype=np.float64)
                    * 30.0
                )
            )
        ),
        "frame_index": array_identity(staging, "frame_index"),
        "files": {
            name: array_identity(staging, name)
            for name in ("ego", "exo", "sample_id", "take_uid", "timestamp")
        },
        "source_freeze_sha256": sha256_file(args.locked_dir / "freeze.json"),
        "array_sha256": {
            name: sha256_file(staging / f"{name}.npy")
            for name in (
                "ego",
                "exo",
                "sample_id",
                "take_uid",
                "timestamp",
                "role",
                "training_valid",
                "frame_index",
            )
        },
        "array_identity": {
            name: {
                "shape": list(np.load(staging / f"{name}.npy", mmap_mode="r", allow_pickle=False).shape),
                "dtype": str(np.load(staging / f"{name}.npy", mmap_mode="r", allow_pickle=False).dtype),
            }
            for name in ("ego", "exo", "sample_id", "take_uid", "timestamp")
        },
        "final_inference_only": True,
        "freeze_stage": freeze_stage,
        "evaluation_allowed": freeze_stage == "final",
        "effect_manifest_sha256": sha256_file(staging / "effect_manifest_locked.jsonl"),
        "video_inventory": video_inventory,
        "materializer_code_sha256": sha256_file(Path(__file__)),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    (staging / "materialization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
