#!/usr/bin/env python3
"""Convert downloaded EgoExo4D aligned videos into FACT paired NPZ shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def read_frame_pair(path: Path, timestamp_sec: float, delta_sec: float, resize: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    for ts in (timestamp_sec, timestamp_sec + delta_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(ts, 0.0) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoexo-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--selected-jsonl", type=Path, default=Path("data/egoexo4d/fact_debug/selected_takes.jsonl"))
    parser.add_argument("--output-npz", type=Path, default=Path("data/fact_egoexo/shards/train_debug_000000.npz"))
    parser.add_argument("--failed-jsonl", type=Path, default=Path("data/fact_egoexo/failed_samples.jsonl"))
    parser.add_argument("--samples-per-take", type=int, default=8)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--transition-sec", type=float, default=0.5)
    parser.add_argument("--resize", type=int, default=224)
    args = parser.parse_args()

    rows = load_jsonl(args.selected_jsonl)
    ego_clips = []
    exo_clips = []
    sample_ids = []
    take_uids = []
    timestamps = []
    failed = []

    for row in tqdm(rows, desc="takes"):
        ego_path = resolve_video(args.egoexo_root, row, row["ego_relative_path"])
        exo_path = resolve_video(args.egoexo_root, row, row["exo_relative_path"])
        if not ego_path or not exo_path:
            failed.append({**row, "reason": "missing_video", "ego_found": bool(ego_path), "exo_found": bool(exo_path)})
            continue

        start = float(row.get("task_start_sec") or 0.0)
        end = float(row.get("task_end_sec") or row.get("duration_sec") or start)
        max_start = max(start, end - args.transition_sec)
        for index in range(args.samples_per_take):
            timestamp = min(start + index * args.stride_sec, max_start)
            ego = read_frame_pair(ego_path, timestamp, args.transition_sec, args.resize)
            exo = read_frame_pair(exo_path, timestamp, args.transition_sec, args.resize)
            if ego is None or exo is None:
                failed.append({**row, "reason": "decode_failed", "timestamp_sec": timestamp})
                continue
            ego_clips.append(ego)
            exo_clips.append(exo)
            sample_ids.append(f"{row['take_uid']}:{timestamp:.3f}")
            take_uids.append(row["take_uid"])
            timestamps.append(timestamp)

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
    print(f"Wrote {args.output_npz}")
    print(f"clips: {len(ego_clips)} failed: {len(failed)}")


if __name__ == "__main__":
    main()
