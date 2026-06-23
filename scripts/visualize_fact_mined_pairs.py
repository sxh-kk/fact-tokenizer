#!/usr/bin/env python3
"""Visualize FACT mined same-take pairs as contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--pair-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--top-rank", type=int, default=0)
    parser.add_argument("--thumb-size", type=int, default=112)
    parser.add_argument("--pairs-per-page", type=int, default=25)
    return parser.parse_args()


def load_metadata(path: Path, expected_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        take_uid = np.asarray(data["take_uid"]).astype(str) if "take_uid" in data else np.asarray([str(i) for i in range(expected_len)])
        timestamp = (
            np.asarray(data["timestamp"], dtype=np.float32)
            if "timestamp" in data
            else np.arange(expected_len, dtype=np.float32)
        )
        sample_id = (
            np.asarray(data["sample_id"]).astype(str)
            if "sample_id" in data
            else np.asarray([str(i) for i in range(expected_len)])
        )
    return take_uid, timestamp, sample_id


def frame_to_image(video: np.ndarray, index: int, frame_index: int, thumb_size: int) -> Image.Image:
    sample = video[index]
    frame_index = min(max(frame_index, 0), sample.shape[0] - 1)
    frame = sample[frame_index]
    if frame.ndim != 3:
        raise ValueError(f"Expected per-frame image with 3 dims, got shape {frame.shape}")
    if frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.max(initial=0.0) <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    image = Image.fromarray(frame[..., :3])
    return image.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)


def make_pair_tile(
    views: dict[str, np.ndarray],
    anchor: int,
    donor: int,
    score: float,
    take_uid: str,
    anchor_time: float,
    donor_time: float,
    sample_id: str,
    donor_sample_id: str,
    view_keys: list[str],
    thumb_size: int,
) -> Image.Image:
    label_height = 44
    width = 4 * thumb_size
    height = 2 * thumb_size + label_height
    tile = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(tile)
    labels = [
        ("anchor t0", anchor, 0),
        ("anchor t1", anchor, 1),
        ("donor t0", donor, 0),
        ("donor t1", donor, 1),
    ]
    for row, view_key in enumerate(view_keys):
        y = label_height + row * thumb_size
        for col, (_, row_index, frame_index) in enumerate(labels):
            x = col * thumb_size
            tile.paste(frame_to_image(views[view_key], row_index, frame_index, thumb_size), (x, y))
    for col, (label, _, _) in enumerate(labels):
        draw.text((col * thumb_size + 4, 4), label, fill=(0, 0, 0))
    draw.text(
        (4, 20),
        f"take={take_uid} a={anchor}/{sample_id}@{anchor_time:.2f} d={donor}/{donor_sample_id}@{donor_time:.2f} score={score:.3f}",
        fill=(0, 0, 0),
    )
    draw.text((4, label_height + 4), view_keys[0], fill=(255, 255, 0))
    draw.text((4, label_height + thumb_size + 4), view_keys[1], fill=(255, 255, 0))
    return tile


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.pairs_per_page <= 0:
        raise ValueError("--pairs-per-page must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_npz, allow_pickle=False) as data:
        views = {view_key: np.asarray(data[view_key]) for view_key in args.view_keys}
    num_samples = int(next(iter(views.values())).shape[0])
    take_uid, timestamp, sample_id = load_metadata(args.input_npz, num_samples)

    with np.load(args.pair_map, allow_pickle=False) as pair_data:
        donor_index_topk = np.asarray(pair_data["donor_index_topk"], dtype=np.int64)
        donor_score_topk = np.asarray(pair_data["donor_score_topk"], dtype=np.float32)
        metadata_json = str(pair_data["metadata_json"]) if "metadata_json" in pair_data else "{}"
    if donor_index_topk.shape[0] != num_samples:
        raise ValueError(f"Pair map rows {donor_index_topk.shape[0]} do not match NPZ rows {num_samples}")
    if args.top_rank < 0 or args.top_rank >= donor_index_topk.shape[1]:
        raise ValueError(f"--top-rank must be in [0, {donor_index_topk.shape[1] - 1}]")

    valid_anchors = np.flatnonzero(donor_index_topk[:, args.top_rank] >= 0)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(valid_anchors)
    anchors = valid_anchors[: args.count]

    tiles = []
    rows = []
    for anchor in anchors.tolist():
        donor = int(donor_index_topk[anchor, args.top_rank])
        score = float(donor_score_topk[anchor, args.top_rank])
        tiles.append(
            make_pair_tile(
                views,
                anchor,
                donor,
                score,
                str(take_uid[anchor]),
                float(timestamp[anchor]),
                float(timestamp[donor]),
                str(sample_id[anchor]),
                str(sample_id[donor]),
                list(args.view_keys),
                args.thumb_size,
            )
        )
        rows.append(
            {
                "anchor_index": anchor,
                "donor_index": donor,
                "score": score,
                "take_uid": str(take_uid[anchor]),
                "anchor_timestamp": float(timestamp[anchor]),
                "donor_timestamp": float(timestamp[donor]),
                "sample_id": str(sample_id[anchor]),
                "donor_sample_id": str(sample_id[donor]),
            }
        )

    tile_width, tile_height = tiles[0].size if tiles else (4 * args.thumb_size, 2 * args.thumb_size + 44)
    columns = 5
    rows_per_page = max(1, (args.pairs_per_page + columns - 1) // columns)
    for page_index, start in enumerate(range(0, len(tiles), args.pairs_per_page)):
        page_tiles = tiles[start : start + args.pairs_per_page]
        page_rows = max(1, (len(page_tiles) + columns - 1) // columns)
        page = Image.new("RGB", (columns * tile_width, page_rows * tile_height), "white")
        for local, tile in enumerate(page_tiles):
            x = (local % columns) * tile_width
            y = (local // columns) * tile_height
            page.paste(tile, (x, y))
        page.save(args.output_dir / f"pairs_page_{page_index:03d}.png")

    with (args.output_dir / "pairs_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["anchor_index", "donor_index"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_npz": str(args.input_npz),
                "pair_map": str(args.pair_map),
                "pair_map_metadata": metadata_json,
                "count": len(rows),
                "top_rank": args.top_rank,
                "pages": int((len(tiles) + args.pairs_per_page - 1) // args.pairs_per_page),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved {len(rows)} visualized pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
