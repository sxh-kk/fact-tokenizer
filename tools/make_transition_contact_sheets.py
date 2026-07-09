#!/usr/bin/env python3
"""Generate small Ego/Exo contact sheets for transition-level review rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", "--npz", dest="input_npz", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def load_view(input_path: Path, key: str) -> np.ndarray:
    if input_path.is_dir():
        return np.load(input_path / f"{key}.npy", mmap_mode="r")
    data = np.load(input_path, allow_pickle=False)
    return data[key]


def frame_to_image(video: np.ndarray, frame_index: int, thumb_size: int) -> Image.Image:
    frame_index = min(max(frame_index, 0), video.shape[0] - 1)
    frame = video[frame_index]
    if frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.max(initial=0.0) <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame[..., :3]).resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)


def build_sheet(row: dict[str, str], views: dict[str, np.ndarray], view_keys: list[str], thumb_size: int) -> Image.Image:
    label_height = 82
    width = 2 * thumb_size
    height = label_height + len(view_keys) * thumb_size
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"{row.get('take_uid', '')} @ {row.get('timestamp', '')}s"
    subtitle = f"{row.get('parent_task_name', '')} | {row.get('task_name', '')}"
    scores = (
        f"auto={row.get('auto_label', '')} "
        f"inter={row.get('interaction_score', '')} "
        f"sync={row.get('ego_exo_sync_score', '')} "
        f"scene={row.get('scene_only_score', '')}"
    )
    draw.text((6, 4), title[:180], fill=(0, 0, 0))
    draw.text((6, 24), subtitle[:180], fill=(0, 0, 0))
    draw.text((6, 44), scores[:180], fill=(0, 0, 0))
    row_index = int(str(row.get("row_index", "0") or 0))
    for view_offset, view_key in enumerate(view_keys):
        y = label_height + view_offset * thumb_size
        video = views[view_key][row_index]
        sheet.paste(frame_to_image(video, 0, thumb_size), (0, y))
        sheet.paste(frame_to_image(video, video.shape[0] - 1, thumb_size), (thumb_size, y))
        draw.text((4, y + 4), f"{view_key} t0", fill=(255, 255, 0))
        draw.text((thumb_size + 4, y + 4), f"{view_key} t1", fill=(255, 255, 0))
    return sheet


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.review_csv)
    views = {key: load_view(args.input_npz, key) for key in args.view_keys}
    suffix = "jpg" if args.image_format == "jpg" else "png"
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sheet = build_sheet(row, views, list(args.view_keys), args.thumb_size)
        row_index = str(row.get("row_index", index - 1))
        sample_id = str(row.get("sample_id", row_index)).replace("/", "_").replace(":", "_")
        image_path = args.out_dir / f"{int(row_index):06d}_{sample_id}.{suffix}"
        save_kwargs = {"quality": 92} if suffix == "jpg" else {}
        sheet.save(image_path, **save_kwargs)
        merged = dict(row)
        merged["contact_sheet_path"] = str(image_path)
        out_rows.append(merged)
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
            print(f"Saved {index}/{len(rows)} transition sheets", flush=True)
    out_csv = args.out_csv or (args.out_dir / "transition_review_with_contact_sheets.csv")
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    write_csv(out_csv, out_rows, fieldnames)
    print(f"Saved {len(out_rows)} contact sheets to {args.out_dir}; CSV: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
