#!/usr/bin/env python3
"""Fail on sample, take, or transition-frame overlap across effect manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import read_manifest_jsonl  # noqa: E402


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("manifests must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("manifests must use non-empty NAME=PATH")
    return name.strip(), Path(path)


def frame_keys(records, fps: float, transition_seconds: float) -> set[str]:
    values: set[str] = set()
    for record in records:
        if record.timestamp is None:
            continue
        start = int(round(record.timestamp * fps))
        end = int(round((record.timestamp + transition_seconds) * fps))
        values.update(f"{record.take_uid}:{frame}" for frame in range(start, end + 1))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=named_path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if len(args.manifest) < 2 or len({name for name, _ in args.manifest}) != len(args.manifest):
        raise ValueError("split audit requires at least two uniquely named manifests")
    groups = {name: read_manifest_jsonl(path) for name, path in args.manifest}
    summaries = {
        name: {
            "samples": len(records),
            "takes": len({record.take_uid for record in records}),
            "training_valid": sum(record.training_valid for record in records),
            "manifest_sha256": hashlib.sha256(dict(args.manifest)[name].read_bytes()).hexdigest(),
        }
        for name, records in groups.items()
    }
    overlap_reports = []
    failures = []
    names = list(groups)
    for left_index, left_name in enumerate(names):
        left = groups[left_name]
        for right_name in names[left_index + 1 :]:
            right = groups[right_name]
            sample_overlap = sorted({record.sample_id for record in left} & {record.sample_id for record in right})
            take_overlap = sorted({record.take_uid for record in left} & {record.take_uid for record in right})
            frames = sorted(
                frame_keys(left, args.fps, args.transition_seconds)
                & frame_keys(right, args.fps, args.transition_seconds)
            )
            row = {
                "left": left_name,
                "right": right_name,
                "sample_id_overlap": sample_overlap,
                "take_uid_overlap": take_overlap,
                "frame_overlap": frames,
            }
            overlap_reports.append(row)
            if sample_overlap or take_overlap or frames:
                failures.append(
                    f"{left_name}/{right_name}: sample={len(sample_overlap)}, "
                    f"take={len(take_overlap)}, frame={len(frames)}"
                )
    report = {
        "schema": "fact-effect-split-audit-v1",
        "passed": not failures,
        "failures": failures,
        "groups": summaries,
        "pairwise_overlaps": overlap_reports,
        "fps": args.fps,
        "transition_seconds": args.transition_seconds,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
