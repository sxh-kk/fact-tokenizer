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
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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
    for row in requests:
        for endpoint, value in enumerate(
            (float(row["timestamp"]), float(row["timestamp"]) + transition_seconds)
        ):
            needed[int(round(value * fps))].append((int(row["source_index"]), endpoint))
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
    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite semantic audit: {args.output_report}")
    arrays, files = inspect_source(args.input_dir)
    source_lookup = {as_text(value): index for index, value in enumerate(arrays["sample_id"])}
    if len(source_lookup) != len(arrays["sample_id"]):
        raise ValueError("source sample IDs are not unique")
    splits = set(args.include_split)
    gold_rows = [
        row
        for row in load_jsonl(args.gold_manifest)
        if str(row.get("gold_split")) in splits and str(row.get("sample_id")) in source_lookup
    ]
    if not gold_rows:
        raise ValueError("no selected gold samples belong to this source")
    selected_ids = [str(row["sample_id"]) for row in gold_rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected gold sample IDs are not unique")
    video_rows = {str(row["take_uid"]): row for row in load_jsonl(args.video_map_jsonl)}
    by_take: dict[str, list[dict]] = defaultdict(list)
    for row in gold_rows:
        sample_id = str(row["sample_id"])
        index = source_lookup[sample_id]
        take_uid = as_text(arrays["take_uid"][index])
        timestamp = float(arrays["timestamp"][index])
        if take_uid != str(row["take_uid"]) or abs(timestamp - float(row["timestamp"])) > 1e-3:
            raise ValueError(f"gold/source metadata mismatch for {sample_id}")
        by_take[take_uid].append({**row, "source_index": index, "timestamp": timestamp})
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
        "frame_selection": "round(timestamp_seconds * 30Hz), sequential decode from frame zero",
        "video_inventory": video_inventory,
        "video_map_jsonl": str(args.video_map_jsonl),
        "video_map_sha256": sha256_file(args.video_map_jsonl),
        "audit_code_sha256": sha256_file(Path(__file__)),
        "decoder_code_sha256": sha256_file(ROOT / "scripts" / "prepare_fact_egoexo_npz.py"),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.output_report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
