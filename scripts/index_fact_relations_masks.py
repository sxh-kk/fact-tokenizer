#!/usr/bin/env python3
"""Stream EgoExo Relations and build sparse target-only object ROI sidecars."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fact_tokenizer.effect_index import (  # noqa: E402
    decode_relation_union,
    iter_annotation_items,
    select_relation_mask_entries,
)
from fact_tokenizer.effect_manifest import EffectCapability, read_manifest_jsonl, write_manifest_jsonl  # noqa: E402
from prepare_fact_egoexo_npz import resolve_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relations-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-prefix", default="aria01")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frame-distance", type=int, default=15)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--egoexo-root", type=Path)
    parser.add_argument(
        "--video-map-jsonl",
        type=Path,
        help="Selected-take rows containing root_dir/take_name/ego_relative_path.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if (args.egoexo_root is None) != (args.video_map_jsonl is None):
        raise ValueError("Relations anchor images require both --egoexo-root and --video-map-jsonl")
    try:
        from ego4d.research.util.masks import decode_mask
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Relations indexing requires ego4d plus pycocotools") from exc
    records = read_manifest_jsonl(args.manifest)
    by_take: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_take.setdefault(record.take_uid, []).append(index)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output / "ego_object_mask.npy"
    masks = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(records), args.output_size, args.output_size),
    )
    valid = np.zeros(len(records), dtype=bool)
    anchor_frame = np.full(len(records), -1, dtype=np.int64)
    anchor_offset = np.full(len(records), np.iinfo(np.int16).max, dtype=np.int16)
    refs: list[dict] = []
    video_rows = {}
    if args.video_map_jsonl is not None:
        with args.video_map_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    video_rows[str(row["take_uid"])] = row
    anchor_images = None
    anchor_image_valid = np.zeros(len(records), dtype=bool)
    if video_rows:
        anchor_images = np.lib.format.open_memmap(
            output / "ego_object_mask_anchor_image.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(records), args.output_size, args.output_size, 3),
        )
    seen_takes: set[str] = set()
    for relation_path in args.relations_json:
        for take_uid, relation_take in iter_annotation_items(relation_path):
            if take_uid not in by_take:
                continue
            if take_uid in seen_takes:
                raise ValueError(f"take {take_uid} occurs in multiple Relations files")
            seen_takes.add(take_uid)
            capture = None
            if anchor_images is not None:
                video_row = video_rows.get(take_uid)
                if video_row is None:
                    raise ValueError(f"video map is missing Relations take {take_uid}")
                video_path = resolve_video(
                    args.egoexo_root,
                    video_row,
                    str(video_row["ego_relative_path"]),
                )
                if video_path is None:
                    raise FileNotFoundError(f"cannot resolve Relations anchor video for {take_uid}")
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open Relations anchor video {video_path}")
            for index in by_take[take_uid]:
                target_frame = int(round(float(records[index].timestamp or 0.0) * args.fps))
                entries, selected_frame = select_relation_mask_entries(
                    relation_take,
                    args.camera_prefix,
                    target_frame,
                    max_frame_distance=args.max_frame_distance,
                    include_hands=args.include_hands,
                )
                if not entries or selected_frame is None:
                    continue
                union = decode_relation_union(
                    entries,
                    output_size=(args.output_size, args.output_size),
                    decoder=decode_mask,
                )
                if not union.any():
                    continue
                masks[index] = union.astype(np.uint8)
                valid[index] = True
                anchor_frame[index] = selected_frame
                anchor_offset[index] = selected_frame - target_frame
                if capture is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"cannot decode Relations anchor frame {selected_frame} for {take_uid}"
                        )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    anchor_images[index] = cv2.resize(
                        frame,
                        (args.output_size, args.output_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    anchor_image_valid[index] = True
                refs.append(
                    {
                        "sample_id": records[index].sample_id,
                        "take_uid": take_uid,
                        "source": str(relation_path),
                        "camera_prefix": args.camera_prefix,
                        "target_frame": target_frame,
                        "anchor_frame": selected_frame,
                        "anchor_offset_frames": selected_frame - target_frame,
                        "objects": [
                            {
                                "object_id": entry["object_id"],
                                "camera_name": entry["camera_name"],
                                "encoded_mask_sha256": hashlib.sha256(
                                    str(entry["encoded_mask"]).encode("utf-8")
                                ).hexdigest(),
                            }
                            for entry in entries
                        ],
                        "target_only": True,
                        "allowed_in_history_private_wam_input": False,
                    }
                )
            if capture is not None:
                capture.release()
    masks.flush()
    del masks
    if anchor_images is not None:
        anchor_images.flush()
        del anchor_images
    np.save(output / "object_mask_valid.npy", valid)
    np.save(output / "object_mask_anchor_frame.npy", anchor_frame)
    np.save(output / "object_mask_anchor_offset_frames.npy", anchor_offset)
    np.save(output / "object_mask_anchor_image_valid.npy", anchor_image_valid)
    refs_path = output / "relations_refs.jsonl"
    with refs_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(refs, key=lambda value: value["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ref_by_id = {row["sample_id"]: row for row in refs}
    updated = []
    for index, record in enumerate(records):
        capabilities = dict(record.capability_validity)
        capabilities[EffectCapability.OBJECT_MASK] = bool(valid[index])
        annotation_refs = dict(record.annotation_refs)
        if record.sample_id in ref_by_id:
            annotation_refs["relations_object_mask"] = {
                "index": index,
                "mask_npy": str(mask_path),
                "ref_jsonl": f"{refs_path}#{record.sample_id}",
                "target_only": True,
            }
        updated.append(replace(record, capability_validity=capabilities, annotation_refs=annotation_refs))
    manifest_path = output / "effect_manifest_relations_indexed.jsonl"
    write_manifest_jsonl(manifest_path, updated)
    offsets = np.abs(anchor_offset[valid].astype(np.int64))
    report = {
        "samples": len(records),
        "takes": len(by_take),
        "relations_takes_found": len(seen_takes),
        "object_mask_valid": int(valid.sum()),
        "object_mask_fraction": float(valid.mean()) if len(valid) else 0.0,
        "camera_prefix": args.camera_prefix,
        "max_frame_distance": args.max_frame_distance,
        "absolute_anchor_offset_frames": {
            "max": int(offsets.max()) if len(offsets) else None,
            "median": float(np.median(offsets)) if len(offsets) else None,
            "p90": float(np.quantile(offsets, 0.9)) if len(offsets) else None,
        },
        "source_sha256": {str(path): sha256_file(path) for path in args.relations_json},
        "mask_npy": str(mask_path),
        "anchor_image_npy": (
            str(output / "ego_object_mask_anchor_image.npy") if video_rows else None
        ),
        "anchor_image_valid": int(anchor_image_valid.sum()),
        "manifest": str(manifest_path),
        "target_only": True,
    }
    (output / "relations_index_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
