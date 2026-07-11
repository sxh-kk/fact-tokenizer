#!/usr/bin/env python3
"""Verify selected FACT NPY endpoints exactly against raw RGB videos at t/t+0.5s."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeze_fact_npy_source_contract import inspect_source, sha256_file  # noqa: E402
from prepare_fact_egoexo_npz import resolve_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--include-split", action="append", required=True)
    parser.add_argument("--egoexo-root", type=Path, required=True)
    parser.add_argument("--video-map-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--frame-index-npy",
        type=Path,
        help=(
            "Exact raw t0 frame for every source row; defaults to "
            "INPUT_DIR/frame_index.npy when present."
        ),
    )
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def frame_timestamp(sample_id: str, stored_timestamp: float) -> float:
    try:
        parsed = float(sample_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"sample_id has no frozen decimal timestamp: {sample_id!r}") from error
    if abs(parsed - stored_timestamp) > 0.002:
        raise ValueError(f"sample_id/stored timestamp mismatch for {sample_id}")
    return stored_timestamp


def read_pairs_sequential(
    capture: cv2.VideoCapture, requests: list[dict], transition_seconds: float, resize: int
) -> dict[int, np.ndarray]:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or abs(fps - 30.0) > 1e-3:
        raise ValueError(f"semantic audit requires a 30Hz aligned video, got {fps}")
    needed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    results: dict[int, list[np.ndarray | None]] = {
        int(row["source_index"]): [None, None] for row in requests
    }
    offset_frames = int(round(transition_seconds * fps))
    if abs(offset_frames / fps - transition_seconds) > 1e-9:
        raise ValueError("transition duration must be an exact integer number of source frames")
    for row in requests:
        if "frame_index" in row:
            endpoint_indices = (
                int(row["frame_index"]),
                int(row["frame_index"]) + offset_frames,
            )
        else:
            # Diagnostic fallback for legacy arrays without a frozen frame sidecar.
            start = int(
                np.floor(
                    float(row.get("frame_timestamp", row["timestamp"])) * fps
                    + 0.5
                    + 1e-3
                )
            )
            endpoint_indices = (start, start + offset_frames)
        for endpoint, frame_index in enumerate(endpoint_indices):
            needed[frame_index].append((int(row["source_index"]), endpoint))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_index in range(max(needed) + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"raw video ended before required frame {frame_index}")
        if frame_index not in needed:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_AREA)
        for source_index, endpoint in needed[frame_index]:
            results[source_index][endpoint] = rgb.copy()
    if any(frame is None for pair in results.values() for frame in pair):
        raise RuntimeError("sequential raw decoder did not materialize every requested endpoint")
    return {
        source_index: np.stack([pair[0], pair[1]], axis=0)
        for source_index, pair in results.items()
    }


def main() -> None:
    args = parse_args()
    if args.transition_seconds <= 0.0 or args.resize <= 0:
        raise ValueError("transition seconds and resize must be positive")
    endpoint_offset_frames = int(round(args.transition_seconds * 30.0))
    if abs(endpoint_offset_frames / 30.0 - args.transition_seconds) > 1e-9:
        raise ValueError("transition duration must be an exact integer number of 30Hz frames")
    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite semantic audit: {args.output_report}")
    arrays, files = inspect_source(args.input_dir)
    frame_index_path = args.frame_index_npy
    if frame_index_path is None and (args.input_dir / "frame_index.npy").is_file():
        frame_index_path = args.input_dir / "frame_index.npy"
    frame_indices: np.ndarray | None = None
    frame_index_identity: dict | None = None
    materialization_report_path = args.input_dir / "materialization_report.json"
    materialization_report: dict | None = None
    if frame_index_path is not None:
        if frame_index_path.resolve() != (args.input_dir / "frame_index.npy").resolve():
            raise ValueError("frame index must be the sidecar inside --input-dir")
        frame_indices = np.load(frame_index_path, mmap_mode="r", allow_pickle=False)
        if (
            frame_indices.ndim != 1
            or len(frame_indices) != len(arrays["sample_id"])
            or not np.issubdtype(frame_indices.dtype, np.integer)
            or (frame_indices < 0).any()
        ):
            raise ValueError("frame_index.npy must be a nonnegative integer vector aligned to source rows")
        frame_index_identity = {
            "sha256": sha256_file(frame_index_path),
            "shape": list(frame_indices.shape),
            "dtype": str(frame_indices.dtype),
        }
        if not materialization_report_path.is_file():
            raise ValueError("frame-index audit requires the source materialization report")
        materialization_report = json.loads(
            materialization_report_path.read_text(encoding="utf-8")
        )
        reported_frame_identity = materialization_report.get("frame_index", {})
        if (
            materialization_report.get("schema")
            not in {
                "fact-npy-transition-rebuild-v1",
                "fact-npy-transition-subset-v1",
                "fact-short73-materialization-v3",
            }
            or
            materialization_report.get("files") != files
            or any(
                reported_frame_identity.get(key) != value
                for key, value in frame_index_identity.items()
            )
            or materialization_report.get("color_space") != "RGB"
            or float(materialization_report.get("transition_seconds", -1.0))
            != args.transition_seconds
            or float(materialization_report.get("frame_rate_hz", -1.0)) != 30.0
            or int(materialization_report.get("endpoint_offset_frames", -1))
            != int(round(args.transition_seconds * 30.0))
        ):
            raise ValueError("materialization report does not bind the source arrays and frame index")
    source_lookup = {as_text(value): index for index, value in enumerate(arrays["sample_id"])}
    if len(source_lookup) != len(arrays["sample_id"]):
        raise ValueError("source sample IDs are not unique")
    splits = set(args.include_split)
    requested_rows = [
        row for row in load_jsonl(args.gold_manifest) if str(row.get("gold_split")) in splits
    ]
    missing_requested_ids = sorted(
        str(row.get("sample_id"))
        for row in requested_rows
        if str(row.get("sample_id")) not in source_lookup
    )
    if missing_requested_ids:
        raise ValueError(
            f"source is missing {len(missing_requested_ids)} requested gold samples: "
            f"{missing_requested_ids[:5]}"
        )
    gold_rows = requested_rows
    if not gold_rows:
        raise ValueError("no selected gold samples belong to this source")
    selected_ids = [str(row["sample_id"]) for row in gold_rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected gold sample IDs are not unique")
    raw_video_rows = load_jsonl(args.video_map_jsonl)
    video_rows = {str(row["take_uid"]): row for row in raw_video_rows}
    if len(video_rows) != len(raw_video_rows):
        raise ValueError("video map contains duplicate take_uid rows")
    by_take: dict[str, list[dict]] = defaultdict(list)
    for row in gold_rows:
        sample_id = str(row["sample_id"])
        index = source_lookup[sample_id]
        take_uid = as_text(arrays["take_uid"][index])
        timestamp = float(arrays["timestamp"][index])
        if take_uid != str(row["take_uid"]) or abs(timestamp - float(row["timestamp"])) > 1e-3:
            raise ValueError(f"gold/source metadata mismatch for {sample_id}")
        by_take[take_uid].append(
            {
                **row,
                "source_index": index,
                "timestamp": timestamp,
                "frame_timestamp": frame_timestamp(sample_id, timestamp),
                **(
                    {"frame_index": int(frame_indices[index])}
                    if frame_indices is not None
                    else {}
                ),
            }
        )
    missing_maps = sorted(set(by_take) - set(video_rows))
    if missing_maps:
        raise ValueError(f"video map is missing {len(missing_maps)} takes: {missing_maps[:5]}")
    video_inventory = []
    exact_matches = 0
    mismatch_details = []
    for take_uid in sorted(by_take):
        video_row = video_rows[take_uid]
        ego_path = resolve_video(args.egoexo_root, video_row, video_row.get("ego_relative_path"))
        exo_path = resolve_video(args.egoexo_root, video_row, video_row.get("exo_relative_path"))
        if ego_path is None or exo_path is None:
            raise FileNotFoundError(f"raw videos missing for {take_uid}: ego={ego_path}, exo={exo_path}")
        video_inventory.append(
            {
                "take_uid": take_uid,
                "ego_path": str(ego_path),
                "ego_sha256": sha256_file(ego_path),
                "exo_path": str(exo_path),
                "exo_sha256": sha256_file(exo_path),
            }
        )
        captures = {
            "ego": cv2.VideoCapture(str(ego_path)),
            "exo": cv2.VideoCapture(str(exo_path)),
        }
        if not all(capture.isOpened() for capture in captures.values()):
            raise RuntimeError(f"cannot open raw videos for {take_uid}")
        try:
            decoded_by_view = {
                view: read_pairs_sequential(
                    capture,
                    by_take[take_uid],
                    args.transition_seconds,
                    args.resize,
                )
                for view, capture in captures.items()
            }
            for row in sorted(by_take[take_uid], key=lambda value: value["timestamp"]):
                for view in captures:
                    decoded = decoded_by_view[view][row["source_index"]]
                    stored = np.asarray(arrays[view][row["source_index"]])
                    if np.array_equal(decoded, stored):
                        exact_matches += 1
                    else:
                        delta = np.abs(decoded.astype(np.int16) - stored.astype(np.int16))
                        mismatch_details.append(
                            {
                                "sample_id": row["sample_id"],
                                "view": view,
                                "mean_absolute_error": float(delta.mean()),
                                "maximum_absolute_error": int(delta.max()),
                            }
                        )
        finally:
            for capture in captures.values():
                capture.release()
    expected_matches = 2 * len(gold_rows)
    if mismatch_details or exact_matches != expected_matches:
        raise ValueError(
            f"raw semantic audit found {len(mismatch_details)} endpoint-pair mismatches: {mismatch_details[:5]}"
        )
    sorted_ids = sorted(selected_ids)
    ids_text = "".join(value + "\n" for value in sorted_ids).encode("utf-8")
    audited_frame_rows = (
        sorted(
            (
                {
                    "sample_id": sample_id,
                    "take_uid": as_text(arrays["take_uid"][source_lookup[sample_id]]),
                    "frame_index": int(frame_indices[source_lookup[sample_id]]),
                }
                for sample_id in selected_ids
            ),
            key=lambda row: row["sample_id"],
        )
        if frame_indices is not None
        else []
    )
    audited_frame_rows_sha256 = (
        hashlib.sha256(
            "".join(
                f"{row['sample_id']}\t{row['take_uid']}\t{row['frame_index']}\n"
                for row in audited_frame_rows
            ).encode("utf-8")
        ).hexdigest()
        if audited_frame_rows
        else None
    )
    selected_frame_distances = (
        [
            abs(
                int(frame_indices[source_lookup[sample_id]])
                - float(arrays["timestamp"][source_lookup[sample_id]]) * 30.0
            )
            for sample_id in selected_ids
        ]
        if frame_indices is not None
        else []
    )
    if selected_frame_distances and max(selected_frame_distances) > 0.501:
        raise ValueError("a frozen t0 frame is more than half a frame from its sample timestamp")
    report = {
        "schema": "fact-npy-source-semantic-audit-v1",
        "passed": True,
        "source_dir": str(args.input_dir.resolve()),
        "rows": len(arrays["sample_id"]),
        "files": files,
        "gold_manifest": str(args.gold_manifest),
        "gold_manifest_sha256": sha256_file(args.gold_manifest),
        "included_splits": sorted(splits),
        "audited_sample_ids": sorted_ids,
        "audited_sample_ids_sha256": hashlib.sha256(ids_text).hexdigest(),
        "audited_samples": len(sorted_ids),
        "exact_view_pair_matches": exact_matches,
        "color_space": "RGB",
        "transition_seconds": args.transition_seconds,
        "endpoint_semantics": ["t", f"t+{args.transition_seconds:g}s"],
        "resize": args.resize,
        "frame_rate_hz": 30.0,
        "endpoint_offset_frames": int(round(args.transition_seconds * 30.0)),
        "frame_selection_mode": (
            "frozen_frame_index_sidecar"
            if frame_indices is not None
            else "legacy_timestamp_nearest_half_up"
        ),
        "frame_selection": (
            "frame_index.npy exact t0 and frame_index+offset exact t1; sequential decode from frame zero"
            if frame_indices is not None
            else "floor(timestamp_seconds * 30Hz + 0.5), sequential decode from frame zero"
        ),
        "timestamp_alignment": "sample_id suffix within 2ms of timestamp.npy",
        "maximum_t0_timestamp_distance_frames": (
            max(selected_frame_distances) if selected_frame_distances else None
        ),
        "frame_index_npy": (
            str(frame_index_path.resolve()) if frame_index_path is not None else None
        ),
        "frame_index": frame_index_identity,
        "audited_frame_rows": audited_frame_rows,
        "audited_frame_rows_sha256": audited_frame_rows_sha256,
        "materialization_report": (
            str(materialization_report_path.resolve())
            if materialization_report is not None
            else None
        ),
        "materialization_report_sha256": (
            sha256_file(materialization_report_path)
            if materialization_report is not None
            else None
        ),
        "video_inventory": video_inventory,
        "video_map_jsonl": str(args.video_map_jsonl),
        "video_map_sha256": sha256_file(args.video_map_jsonl),
        "audit_code_sha256": sha256_file(Path(__file__)),
        "decoder_code_sha256": sha256_file(ROOT / "scripts" / "prepare_fact_egoexo_npz.py"),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.output_report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
