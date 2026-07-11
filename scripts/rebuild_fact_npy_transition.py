#!/usr/bin/env python3
"""Rebuild paired FACT NPY endpoints from raw videos with an explicit transition duration."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_fact_npy_source_semantics import frame_timestamp  # noqa: E402
from prepare_fact_egoexo_npz import candidate_paths  # noqa: E402


CORE_ARRAYS = ("ego", "exo", "sample_id", "take_uid", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Legacy NPY source providing frozen IDs/takes/timestamps.")
    parser.add_argument("--video-map-jsonl", type=Path, required=True)
    parser.add_argument("--egoexo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument(
        "--source-transition-seconds",
        type=float,
        default=1.0,
        help="Observed endpoint spacing in the frozen input arrays (1.0 for legacy dense NPY).",
    )
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def file_identity(directory: Path, name: str) -> dict:
    path = directory / f"{name}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype)}


def resolve_video_strict(root: Path, row: dict, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    direct = {path.resolve() for path in candidate_paths(root, row, relative_path) if path.is_file()}
    if len(direct) > 1:
        raise ValueError(f"ambiguous direct video paths for {row['take_uid']}: {sorted(map(str, direct))}")
    if direct:
        return next(iter(direct))
    filename = Path(relative_path).name
    fallback = {
        path.resolve()
        for path in (root / "takes" / str(row["take_name"])).glob(f"**/{filename}")
        if path.is_file()
    }
    if len(fallback) > 1:
        raise ValueError(f"ambiguous fallback video paths for {row['take_uid']}: {sorted(map(str, fallback))}")
    return next(iter(fallback)) if fallback else None


def file_stat(path: Path) -> dict:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": getattr(stat, "st_ino", None),
    }


def decode_selected_frames(
    path: Path,
    frame_indices: set[int],
    resize: int,
    *,
    allow_eof_for_candidate_frames: bool = False,
) -> dict[int, np.ndarray]:
    if not frame_indices:
        raise ValueError("at least one source frame must be requested")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or abs(fps - 30.0) > 1e-3:
        capture.release()
        raise ValueError(f"transition rebuild requires 30Hz video, got {fps}: {path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        for frame_index in range(max(frame_indices) + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                if allow_eof_for_candidate_frames:
                    break
                raise RuntimeError(f"video ended before frame {frame_index}: {path}")
            if frame_index not in frame_indices:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[frame_index] = cv2.resize(
                frame, (resize, resize), interpolation=cv2.INTER_AREA
            )
    finally:
        capture.release()
    missing = frame_indices - set(decoded)
    if missing and not allow_eof_for_candidate_frames:
        raise RuntimeError(f"decoder missed requested frames for {path}: {sorted(missing)[:5]}")
    return decoded


def seek_historical_t0(path: Path, timestamp: float, resize: int) -> tuple[int, np.ndarray]:
    """Replay the legacy CAP_PROP_POS_MSEC read and expose its decoded frame index."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for historical seek: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or abs(fps - 30.0) > 1e-3:
            raise ValueError(f"historical seek requires 30Hz video, got {fps}: {path}")
        if not capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000.0):
            raise RuntimeError(f"decoder rejected timestamp seek for {path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"decoder failed historical timestamp seek for {path}")
        next_index = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        frame_index = int(round(next_index)) - 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_AREA)
        return frame_index, rgb
    finally:
        capture.release()


def main() -> None:
    args = parse_args()
    if args.transition_seconds <= 0.0:
        raise ValueError("--transition-seconds must be positive")
    if args.resize <= 0:
        raise ValueError("--resize must be positive")
    frame_rate_hz = 30.0
    endpoint_offset_frames = int(round(args.transition_seconds * frame_rate_hz))
    if abs(endpoint_offset_frames / frame_rate_hz - args.transition_seconds) > 1e-9:
        raise ValueError("transition duration must be an exact integer number of 30Hz frames")
    source_endpoint_offset_frames = int(
        round(args.source_transition_seconds * frame_rate_hz)
    )
    if (
        args.source_transition_seconds <= 0.0
        or abs(
            source_endpoint_offset_frames / frame_rate_hz
            - args.source_transition_seconds
        )
        > 1e-9
    ):
        raise ValueError("source transition duration must be a positive integer number of 30Hz frames")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite rebuilt NPY data: {args.output_dir}")
    input_arrays = {
        name: np.load(args.input_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in CORE_ARRAYS
    }
    lengths = {name: len(array) for name, array in input_arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"legacy input arrays are not aligned: {lengths}")
    if input_arrays["ego"].ndim != 5 or input_arrays["exo"].shape != input_arrays["ego"].shape:
        raise ValueError("legacy paired arrays must have the same NxTxHxWxC shape")
    if (
        input_arrays["ego"].shape[1] != 2
        or input_arrays["ego"].shape[-1] != 3
        or input_arrays["ego"].dtype != np.uint8
        or input_arrays["exo"].dtype != np.uint8
    ):
        raise ValueError("legacy paired arrays must be uint8 with two endpoints")
    for name in ("sample_id", "take_uid", "timestamp"):
        if input_arrays[name].ndim != 1:
            raise ValueError(f"legacy {name}.npy must be one-dimensional")
    if not np.isfinite(np.asarray(input_arrays["timestamp"], dtype=np.float64)).all():
        raise ValueError("legacy timestamps must all be finite")
    sample_ids = [as_text(value) for value in input_arrays["sample_id"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("legacy sample IDs are not unique")
    raw_video_rows = load_jsonl(args.video_map_jsonl)
    video_rows = {str(row["take_uid"]): row for row in raw_video_rows}
    if len(video_rows) != len(raw_video_rows):
        raise ValueError("video map contains duplicate take_uid rows")
    by_take: dict[str, list[dict]] = defaultdict(list)
    for index, (sample_id, raw_take, raw_timestamp) in enumerate(
        zip(sample_ids, input_arrays["take_uid"], input_arrays["timestamp"])
    ):
        take_uid = as_text(raw_take)
        if sample_id.rsplit(":", 1)[0] != take_uid:
            raise ValueError(f"sample_id/take_uid mismatch: {sample_id!r} versus {take_uid!r}")
        by_take[take_uid].append(
            {
                "sample_id": sample_id,
                "source_index": index,
                "timestamp": float(raw_timestamp),
                "frame_timestamp": frame_timestamp(sample_id, float(raw_timestamp)),
            }
        )
    missing = sorted(set(by_take) - set(video_rows))
    if missing:
        raise ValueError(f"video map is missing {len(missing)} source takes: {missing[:10]}")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    shape = (len(sample_ids), 2, args.resize, args.resize, 3)
    output_arrays = {
        view: np.lib.format.open_memmap(
            staging / f"{view}.npy", mode="w+", dtype=np.uint8, shape=shape
        )
        for view in ("ego", "exo")
    }
    frame_indices = np.full(len(sample_ids), -1, dtype=np.int64)
    t0_exact_view_pairs = 0
    source_t1_exact_view_pairs = 0
    ambiguous_t0_matches = 0
    seek_disambiguated_t0_matches = 0
    video_inventory = []
    try:
        for take_number, take_uid in enumerate(sorted(by_take), start=1):
            row = video_rows[take_uid]
            ego_path = resolve_video_strict(args.egoexo_root, row, row.get("ego_relative_path"))
            exo_path = resolve_video_strict(args.egoexo_root, row, row.get("exo_relative_path"))
            if ego_path is None or exo_path is None:
                raise FileNotFoundError(f"raw videos missing for {take_uid}: ego={ego_path}, exo={exo_path}")
            source_stats = {"ego": file_stat(ego_path), "exo": file_stat(exo_path)}
            source_hashes = {"ego": sha256_file(ego_path), "exo": sha256_file(exo_path)}
            video_inventory.append(
                {
                    "take_uid": take_uid,
                    "ego_path": str(ego_path),
                    "ego_sha256": source_hashes["ego"],
                    "ego_stat": source_stats["ego"],
                    "exo_path": str(exo_path),
                    "exo_sha256": source_hashes["exo"],
                    "exo_stat": source_stats["exo"],
                }
            )
            candidate_indices: dict[int, list[int]] = {}
            needed: set[int] = set()
            for sample in by_take[take_uid]:
                approximate = int(
                    np.floor(float(sample["frame_timestamp"]) * frame_rate_hz + 0.5)
                )
                candidates = list(range(max(0, approximate - 2), approximate + 3))
                candidate_indices[int(sample["source_index"])] = candidates
                needed.update(candidates)
                needed.update(value + endpoint_offset_frames for value in candidates)
                needed.update(value + source_endpoint_offset_frames for value in candidates)
            decoded = {
                "ego": decode_selected_frames(
                    ego_path,
                    needed,
                    args.resize,
                    allow_eof_for_candidate_frames=True,
                ),
                "exo": decode_selected_frames(
                    exo_path,
                    needed,
                    args.resize,
                    allow_eof_for_candidate_frames=True,
                ),
            }
            for sample in by_take[take_uid]:
                index = int(sample["source_index"])
                matches = [
                    candidate
                    for candidate in candidate_indices[index]
                    if candidate in decoded["ego"]
                    and candidate in decoded["exo"]
                    and candidate + source_endpoint_offset_frames in decoded["ego"]
                    and candidate + source_endpoint_offset_frames in decoded["exo"]
                    and np.array_equal(
                        decoded["ego"][candidate], np.asarray(input_arrays["ego"][index, 0])
                    )
                    and np.array_equal(
                        decoded["exo"][candidate], np.asarray(input_arrays["exo"][index, 0])
                    )
                    and np.array_equal(
                        decoded["ego"][candidate + source_endpoint_offset_frames],
                        np.asarray(input_arrays["ego"][index, 1]),
                    )
                    and np.array_equal(
                        decoded["exo"][candidate + source_endpoint_offset_frames],
                        np.asarray(input_arrays["exo"][index, 1]),
                    )
                ]
                if not matches:
                    raise ValueError(
                        "cannot jointly anchor both frozen source endpoints in raw videos: "
                        f"{sample['sample_id']}"
                    )
                if len(matches) > 1:
                    ambiguous_t0_matches += 1
                    seeked = {
                        view: seek_historical_t0(
                            path,
                            float(sample["frame_timestamp"]),
                            args.resize,
                        )
                        for view, path in (("ego", ego_path), ("exo", exo_path))
                    }
                    if (
                        seeked["ego"][0] != seeked["exo"][0]
                        or seeked["ego"][0] not in matches
                        or not np.array_equal(
                            seeked["ego"][1], np.asarray(input_arrays["ego"][index, 0])
                        )
                        or not np.array_equal(
                            seeked["exo"][1], np.asarray(input_arrays["exo"][index, 0])
                        )
                    ):
                        raise ValueError(
                            "ambiguous frozen endpoint anchor cannot be resolved by the historical "
                            f"timestamp seek: {sample['sample_id']} candidates={matches}"
                        )
                    anchor = seeked["ego"][0]
                    seek_disambiguated_t0_matches += 1
                else:
                    anchor = matches[0]
                target_frame = float(sample["frame_timestamp"]) * frame_rate_hz
                if abs(anchor - target_frame) > 0.501:
                    raise ValueError(
                        f"raw anchor is more than half a frame from timestamp: {sample['sample_id']} "
                        f"anchor={anchor} nominal={target_frame}"
                    )
                frame_indices[index] = anchor
                for view in ("ego", "exo"):
                    pair = np.stack(
                        [
                            decoded[view][anchor],
                            decoded[view][anchor + endpoint_offset_frames],
                        ],
                        axis=0,
                    )
                    if not np.array_equal(pair[0], np.asarray(input_arrays[view][index, 0])):
                        raise AssertionError("selected raw anchor no longer equals frozen t0")
                    if not np.array_equal(
                        decoded[view][anchor + source_endpoint_offset_frames],
                        np.asarray(input_arrays[view][index, 1]),
                    ):
                        raise AssertionError("selected raw anchor no longer reproduces frozen source t1")
                    t0_exact_view_pairs += 1
                    source_t1_exact_view_pairs += 1
                    output_arrays[view][index] = pair
            if file_stat(ego_path) != source_stats["ego"] or file_stat(exo_path) != source_stats["exo"]:
                raise RuntimeError(f"raw video changed while rebuilding take {take_uid}")
            if take_number % 25 == 0:
                print(json.dumps({"processed_takes": take_number, "total_takes": len(by_take)}), flush=True)
        for array in output_arrays.values():
            array.flush()
        for name in ("sample_id", "take_uid", "timestamp"):
            np.save(staging / f"{name}.npy", np.asarray(input_arrays[name]))
        if (frame_indices < 0).any():
            raise AssertionError("some source rows have no frozen raw frame anchor")
        np.save(staging / "frame_index.npy", frame_indices)
        output_files = {name: file_identity(staging, name) for name in CORE_ARRAYS}
        input_files = {name: file_identity(args.input_dir, name) for name in CORE_ARRAYS}
        report = {
            "schema": "fact-npy-transition-rebuild-v1",
            "source_dir": str(args.input_dir),
            "source_files": input_files,
            "output_dir": str(args.output_dir),
            "files": output_files,
            "rows": len(sample_ids),
            "takes": len(by_take),
            "color_space": "RGB",
            "transition_seconds": args.transition_seconds,
            "endpoint_semantics": ["t", f"t+{args.transition_seconds:g}s"],
            "resize": args.resize,
            "t0_exact_view_matches": t0_exact_view_pairs,
            "expected_t0_exact_view_matches": 2 * len(sample_ids),
            "source_t1_exact_view_matches": source_t1_exact_view_pairs,
            "expected_source_t1_exact_view_matches": 2 * len(sample_ids),
            "source_transition_seconds": args.source_transition_seconds,
            "source_endpoint_offset_frames": source_endpoint_offset_frames,
            "ambiguous_t0_anchor_matches": ambiguous_t0_matches,
            "seek_disambiguated_t0_anchor_matches": seek_disambiguated_t0_matches,
            "frame_rate_hz": frame_rate_hz,
            "endpoint_offset_frames": endpoint_offset_frames,
            "anchor_search_radius_frames": 2,
            "maximum_t0_timestamp_distance_frames": float(
                np.max(
                    np.abs(
                        frame_indices
                        - np.asarray(input_arrays["timestamp"], dtype=np.float64)
                        * frame_rate_hz
                    )
                )
            ),
            "frame_index": {
                "sha256": sha256_file(staging / "frame_index.npy"),
                "shape": list(frame_indices.shape),
                "dtype": str(frame_indices.dtype),
                "meaning": (
                    "exact raw t0 frame; t1 is frame_index + "
                    f"{endpoint_offset_frames} at {frame_rate_hz:g}Hz"
                ),
            },
            "video_map_jsonl": str(args.video_map_jsonl),
            "video_map_sha256": sha256_file(args.video_map_jsonl),
            "video_inventory": video_inventory,
            "rebuild_code_sha256": sha256_file(Path(__file__)),
            "video_resolution_code_sha256": sha256_file(
                ROOT / "scripts" / "prepare_fact_egoexo_npz.py"
            ),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        }
        report_path = staging / "materialization_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(args.output_dir)
    except BaseException:
        for array in output_arrays.values():
            try:
                array.flush()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({key: report[key] for key in ("rows", "takes", "transition_seconds", "t0_exact_view_matches")}, indent=2))


if __name__ == "__main__":
    main()
