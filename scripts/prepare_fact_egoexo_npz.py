#!/usr/bin/env python3
"""Convert downloaded EgoExo4D aligned videos into FACT paired NPZ shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def candidate_paths(root: Path, row: dict, relative_path: str) -> list[Path]:
    root_dir = Path(row["root_dir"])
    rel = Path(relative_path)
    downscaled_relative_path = rel.parent / "downscaled" / "448" / rel.name
    candidates = [
        root / root_dir / relative_path,
        root / root_dir / downscaled_relative_path,
        root / "downscaled_takes" / "448" / root_dir / relative_path,
        root / "takes" / root_dir.name / relative_path,
        root / "takes" / root_dir.name / downscaled_relative_path,
    ]
    return candidates


def resolve_video(root: Path, row: dict, relative_path: str) -> Optional[Path]:
    for path in candidate_paths(root, row, relative_path):
        if path.exists():
            return path
    filename = Path(relative_path).name
    matches = list((root / "takes" / row["take_name"]).glob(f"**/{filename}"))
    if matches:
        return matches[0]
    return None


def read_frame_pair_from_capture(
    cap: cv2.VideoCapture,
    timestamp_sec: float,
    delta_sec: float,
    resize: int,
) -> Optional[np.ndarray]:
    frames = []
    for ts in (timestamp_sec, timestamp_sec + delta_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(ts, 0.0) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return np.stack(frames, axis=0)


def read_frame_pair(path: Path, timestamp_sec: float, delta_sec: float, resize: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        return read_frame_pair_from_capture(cap, timestamp_sec, delta_sec, resize)
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoexo-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--selected-jsonl", type=Path, default=Path("data/egoexo4d/fact_debug/selected_takes.jsonl"))
    parser.add_argument("--output-npz", type=Path, default=Path("data/fact_egoexo/shards/train_debug_000000.npz"))
    parser.add_argument("--failed-jsonl", type=Path, default=Path("data/fact_egoexo/failed_samples.jsonl"))
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--samples-per-take", type=int, default=8, help="Maximum transitions sampled per take.")
    parser.add_argument(
        "--require-full-take",
        action="store_true",
        help="Skip a take unless exactly --samples-per-take transitions can be decoded.",
    )
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--transition-sec", type=float, default=0.5)
    parser.add_argument("--resize", type=int, default=224)
    args = parser.parse_args()
    if args.samples_per_take <= 0:
        raise ValueError("--samples-per-take must be positive")
    if args.stride_sec <= 0.0:
        raise ValueError("--stride-sec must be positive")
    if args.transition_sec <= 0.0:
        raise ValueError("--transition-sec must be positive")

    rows = load_jsonl(args.selected_jsonl)
    ego_clips = []
    exo_clips = []
    sample_ids = []
    take_uids = []
    timestamps = []
    failed = []
    take_reports = []

    for row in tqdm(rows, desc="takes"):
        ego_path = resolve_video(args.egoexo_root, row, row["ego_relative_path"])
        exo_path = resolve_video(args.egoexo_root, row, row["exo_relative_path"])
        if not ego_path or not exo_path:
            failed.append({**row, "reason": "missing_video", "ego_found": bool(ego_path), "exo_found": bool(exo_path)})
            take_reports.append(
                {
                    "take_uid": row["take_uid"],
                    "take_name": row.get("take_name"),
                    "status": "missing_video",
                    "available_transitions": 0,
                    "written_transitions": 0,
                }
            )
            continue

        start = float(row.get("task_start_sec") or 0.0)
        end = float(row.get("task_end_sec") or row.get("duration_sec") or start)
        max_start = max(start, end - args.transition_sec)
        available = int(np.floor((max_start - start) / args.stride_sec)) + 1
        available = max(0, available)
        if args.require_full_take and available < args.samples_per_take:
            failed.append(
                {
                    **row,
                    "reason": "insufficient_duration",
                    "available_transitions": available,
                    "target_transitions": args.samples_per_take,
                }
            )
            take_reports.append(
                {
                    "take_uid": row["take_uid"],
                    "take_name": row.get("take_name"),
                    "status": "insufficient_duration",
                    "available_transitions": available,
                    "written_transitions": 0,
                }
            )
            continue

        num_samples = min(args.samples_per_take, available)
        if num_samples <= 0:
            failed.append({**row, "reason": "no_valid_transition", "available_transitions": available})
            take_reports.append(
                {
                    "take_uid": row["take_uid"],
                    "take_name": row.get("take_name"),
                    "status": "no_valid_transition",
                    "available_transitions": available,
                    "written_transitions": 0,
                }
            )
            continue

        ego_cap = cv2.VideoCapture(str(ego_path))
        exo_cap = cv2.VideoCapture(str(exo_path))
        if not ego_cap.isOpened() or not exo_cap.isOpened():
            failed.append({**row, "reason": "open_failed", "ego_open": ego_cap.isOpened(), "exo_open": exo_cap.isOpened()})
            ego_cap.release()
            exo_cap.release()
            take_reports.append(
                {
                    "take_uid": row["take_uid"],
                    "take_name": row.get("take_name"),
                    "status": "open_failed",
                    "available_transitions": available,
                    "written_transitions": 0,
                }
            )
            continue

        take_ego_clips = []
        take_exo_clips = []
        take_sample_ids = []
        take_uid_values = []
        take_timestamps = []
        take_failed = []
        for index in range(num_samples):
            timestamp = start + index * args.stride_sec
            ego = read_frame_pair_from_capture(ego_cap, timestamp, args.transition_sec, args.resize)
            exo = read_frame_pair_from_capture(exo_cap, timestamp, args.transition_sec, args.resize)
            if ego is None or exo is None:
                take_failed.append({**row, "reason": "decode_failed", "timestamp_sec": timestamp})
                continue
            take_ego_clips.append(ego)
            take_exo_clips.append(exo)
            take_sample_ids.append(f"{row['take_uid']}:{timestamp:.3f}")
            take_uid_values.append(row["take_uid"])
            take_timestamps.append(timestamp)
        ego_cap.release()
        exo_cap.release()

        if args.require_full_take and len(take_ego_clips) != args.samples_per_take:
            failed.extend(take_failed)
            failed.append(
                {
                    **row,
                    "reason": "incomplete_take_decode",
                    "written_transitions": len(take_ego_clips),
                    "target_transitions": args.samples_per_take,
                }
            )
            take_reports.append(
                {
                    "take_uid": row["take_uid"],
                    "take_name": row.get("take_name"),
                    "status": "incomplete_take_decode",
                    "available_transitions": available,
                    "written_transitions": len(take_ego_clips),
                }
            )
            continue

        failed.extend(take_failed)
        ego_clips.extend(take_ego_clips)
        exo_clips.extend(take_exo_clips)
        sample_ids.extend(take_sample_ids)
        take_uids.extend(take_uid_values)
        timestamps.extend(take_timestamps)
        take_reports.append(
            {
                "take_uid": row["take_uid"],
                "take_name": row.get("take_name"),
                "status": "converted",
                "available_transitions": available,
                "written_transitions": len(take_ego_clips),
            }
        )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.failed_jsonl.open("w", encoding="utf-8") as handle:
        for row in failed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not ego_clips:
        raise RuntimeError(f"No clips were converted. See {args.failed_jsonl}")

    np.savez_compressed(
        args.output_npz,
        ego=np.stack(ego_clips, axis=0),
        exo=np.stack(exo_clips, axis=0),
        sample_id=np.asarray(sample_ids),
        take_uid=np.asarray(take_uids),
        timestamp=np.asarray(timestamps, dtype=np.float32),
    )
    if args.report_json:
        counts = Counter(take_uids)
        report = {
            "selected_jsonl": str(args.selected_jsonl),
            "output_npz": str(args.output_npz),
            "failed_jsonl": str(args.failed_jsonl),
            "samples_per_take": args.samples_per_take,
            "require_full_take": args.require_full_take,
            "transition_sec": args.transition_sec,
            "stride_sec": args.stride_sec,
            "resize": args.resize,
            "selected_takes": len(rows),
            "converted_takes": len(counts),
            "clips": len(ego_clips),
            "failed_records": len(failed),
            "sample_count_histogram": dict(sorted(Counter(counts.values()).items())),
            "take_reports": take_reports,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output_npz}")
    print(f"clips: {len(ego_clips)} failed: {len(failed)}")
    if args.report_json:
        print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
