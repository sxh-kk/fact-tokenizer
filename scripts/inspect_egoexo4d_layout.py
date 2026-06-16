#!/usr/bin/env python3
"""Inspect a local EgoExo4D download for FACT tokenizer readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_valid_takes(takes: Iterable[dict]) -> Iterable[dict]:
    for take in takes:
        videos = take.get("frame_aligned_videos") or {}
        best_exo = take.get("best_exo")
        ego_rgb = videos.get("aria01", {}).get("rgb")
        exo = videos.get(best_exo, {}).get("0") if best_exo else None
        if (
            not take.get("is_dropped")
            and take.get("validated", True)
            and ego_rgb
            and ego_rgb.get("relative_path")
            and exo
            and exo.get("relative_path")
        ):
            yield take


def local_candidates(root: Path, take: dict, relative_path: str) -> list[Path]:
    root_dir = Path(take["root_dir"])
    rel = Path(relative_path)
    downscaled_relative_path = rel.parent / "downscaled" / "448" / rel.name
    return [
        root / root_dir / relative_path,
        root / root_dir / downscaled_relative_path,
        root / "downscaled_takes" / "448" / root_dir / relative_path,
        root / "takes" / root_dir.name / relative_path,
        root / "takes" / root_dir.name / downscaled_relative_path,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoexo-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    root = args.egoexo_root
    takes_path = root / "takes.json"
    captures_path = root / "captures.json"
    if not takes_path.exists():
        raise FileNotFoundError(f"Missing {takes_path}. Download EgoExo metadata first.")

    takes = load_json(takes_path)
    captures = load_json(captures_path) if captures_path.exists() else []
    valid = list(iter_valid_takes(takes))
    downloaded_mp4s = sorted(root.rglob("*.mp4"))

    print(f"egoexo_root: {root}")
    print(f"takes: {len(takes)}")
    print(f"captures: {len(captures)}")
    print(f"fact_ready_takes: {len(valid)}")
    print(f"downloaded_mp4s: {len(downloaded_mp4s)}")
    if downloaded_mp4s:
        print("first_downloaded_mp4s:")
        for path in downloaded_mp4s[: args.show]:
            print(f"  {path}")

    print("sample_valid_takes:")
    for take in valid[: args.show]:
        videos = take["frame_aligned_videos"]
        ego = videos["aria01"]["rgb"]
        exo = videos[take["best_exo"]]["0"]
        ego_exists = any(path.exists() for path in local_candidates(root, take, ego["relative_path"]))
        exo_exists = any(path.exists() for path in local_candidates(root, take, exo["relative_path"]))
        print(
            json.dumps(
                {
                    "take_uid": take["take_uid"],
                    "take_name": take["take_name"],
                    "duration_sec": take.get("duration_sec"),
                    "best_exo": take.get("best_exo"),
                    "ego_relative_path": ego["relative_path"],
                    "exo_relative_path": exo["relative_path"],
                    "ego_local_exists": ego_exists,
                    "exo_local_exists": exo_exists,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
