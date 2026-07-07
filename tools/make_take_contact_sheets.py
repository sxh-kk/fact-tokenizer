#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import evenly_spaced_indices, group_indices_by_take, load_labels_by_take, load_npz_metadata, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-take Ego/Exo contact sheets for relevance annotation.")
    parser.add_argument("--split", "--npz", dest="npz", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--thumb-size", type=int, default=112)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    return parser.parse_args()


def frame_to_image(array: np.ndarray, row: int, frame_index: int, thumb_size: int) -> Image.Image:
    video = array[row]
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


def build_sheet(
    views: dict[str, np.ndarray],
    rows: list[int],
    take_uid: str,
    timestamps: np.ndarray,
    label: dict,
    view_keys: list[str],
    thumb_size: int,
) -> Image.Image:
    label_height = 58
    row_height = 2 * thumb_size
    width = len(rows) * thumb_size
    height = label_height + len(view_keys) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"{take_uid} | {label.get('parent_task_name', '')} | {label.get('task_name', '')}"
    draw.text((6, 4), title[:220], fill=(0, 0, 0))
    draw.text((6, 24), f"rows={len(rows)} sampled timestamps={[round(float(timestamps[r]), 2) for r in rows[:8]]}", fill=(0, 0, 0))
    for col, row in enumerate(rows):
        x = col * thumb_size
        draw.text((x + 3, 42), f"{timestamps[row]:.1f}s", fill=(0, 0, 0))
        for view_idx, view_key in enumerate(view_keys):
            y = label_height + view_idx * row_height
            sheet.paste(frame_to_image(views[view_key], row, 0, thumb_size), (x, y))
            sheet.paste(frame_to_image(views[view_key], row, 1, thumb_size), (x, y + thumb_size))
            draw.text((x + 3, y + 3), f"{view_key} t0", fill=(255, 255, 0))
            draw.text((x + 3, y + thumb_size + 3), f"{view_key} t1", fill=(255, 255, 0))
    return sheet


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = load_npz_metadata(args.npz)
    labels = load_labels_by_take(args.labels_jsonl)
    grouped = group_indices_by_take(metadata["take_uid"])
    with np.load(args.npz, allow_pickle=False) as data:
        views = {view_key: np.asarray(data[view_key]) for view_key in args.view_keys}

    rows = []
    for take_uid, indices in sorted(grouped.items()):
        sampled = evenly_spaced_indices(indices, args.num_frames)
        label = labels.get(take_uid, {})
        sheet = build_sheet(
            views=views,
            rows=sampled,
            take_uid=take_uid,
            timestamps=metadata["timestamp"],
            label=label,
            view_keys=list(args.view_keys),
            thumb_size=args.thumb_size,
        )
        suffix = "jpg" if args.image_format == "jpg" else "png"
        image_path = args.out / f"{take_uid}.{suffix}"
        save_kwargs = {"quality": 92} if suffix == "jpg" else {}
        sheet.save(image_path, **save_kwargs)
        rows.append(
            {
                "take_uid": take_uid,
                "contact_sheet_path": str(image_path),
                "num_transitions": len(indices),
                "sampled_transitions": len(sampled),
                "parent_task_name": label.get("parent_task_name", ""),
                "task_name": label.get("task_name", ""),
                "take_name": label.get("take_name", ""),
            }
        )
    write_csv(args.out / "contact_sheet_manifest.csv", rows)
    print(f"Saved {len(rows)} contact sheets to {args.out}")


if __name__ == "__main__":
    main()
