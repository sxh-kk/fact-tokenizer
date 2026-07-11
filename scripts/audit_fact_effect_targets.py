#!/usr/bin/env python3
"""Create/validate the mandatory 50-sample target alignment visual audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_experiments import EffectTargetCache  # noqa: E402
from fact_tokenizer.effect_manifest import EffectCapability, read_manifest_jsonl  # noqa: E402


REVIEW_COLUMNS = ("time_direction_ok", "coordinate_orientation_ok", "mask_flow_alignment_ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--validate-review", type=Path)
    parser.add_argument("--minimum-pass-fraction", type=float, default=0.90)
    return parser.parse_args()


def frame_rgb(sample: np.ndarray, endpoint: int) -> np.ndarray:
    frame = np.asarray(sample[endpoint])
    if frame.ndim != 3:
        raise ValueError("video frame must be 3D")
    if frame.shape[0] in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def flow_rgb(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import cv2

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    scale = np.quantile(magnitude[valid], 0.95) if valid.any() else 1.0
    hsv[..., 2] = np.clip(magnitude / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~valid] = 0
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2

    result = image.copy()
    resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    result[resized] = (0.35 * result[resized] + 0.65 * np.asarray([255, 40, 40])).astype(np.uint8)
    return result


def validate(path: Path, minimum: float) -> dict:
    if minimum < 0.90:
        raise ValueError("formal target audit minimum pass fraction cannot be below 0.90")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise ValueError(f"formal target audit requires exactly 50 rows, found {len(rows)}")
    passed = 0
    invalid = []
    for row_number, row in enumerate(rows, start=2):
        values = [str(row.get(column, "")).strip().lower() for column in REVIEW_COLUMNS]
        if any(value not in {"yes", "no"} for value in values):
            invalid.append(row_number)
        elif all(value == "yes" for value in values):
            passed += 1
    if invalid:
        raise ValueError(f"audit rows need yes/no in all review fields: {invalid[:5]}")
    fraction = passed / len(rows)
    pack_path = path.parent / "audit_pack.json"
    if not pack_path.is_file():
        raise ValueError("review CSV is missing its immutable sibling audit_pack.json")
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    if [row["sample_id"] for row in rows] != pack.get("selected_sample_ids"):
        raise ValueError("review rows differ from the frozen stratified audit pack")
    report = {
        "rows": len(rows),
        "fully_aligned": passed,
        "pass_fraction": fraction,
        "minimum_pass_fraction": minimum,
        "passed": fraction >= minimum,
        "review_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "audit_pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "target_identity_sha256": pack["target_identity_sha256"],
        "target_config_sha256": pack["target_config_sha256"],
        "strata_counts": pack["strata_counts"],
    }
    if not report["passed"]:
        raise RuntimeError(f"target audit gate failed: {fraction:.3f} < {minimum:.3f}")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_review:
        report = validate(args.validate_review, args.minimum_pass_fraction)
        (args.output_dir / "target_visual_audit_gate.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.input_dir is None or args.target_cache is None:
        raise ValueError("audit pack creation requires --input-dir and --target-cache")
    if args.manifest is None:
        raise ValueError("stratified audit pack creation requires --manifest")
    cache = EffectTargetCache(args.target_cache)
    import cv2
    from PIL import Image, ImageDraw
    manifest_records = read_manifest_jsonl(args.manifest)
    manifest_by_id = {record.sample_id: record for record in manifest_records}
    sample_ids = np.load(args.input_dir / "sample_id.npy", mmap_mode="r", allow_pickle=False).astype(str)
    videos = {
        view: np.load(args.input_dir / f"{view}.npy", mmap_mode="r", allow_pickle=False)
        for view in ("ego", "exo")
        if (args.input_dir / f"{view}.npy").is_file()
    }
    available = np.asarray([index for index, sample_id in enumerate(sample_ids) if sample_id in cache.records])
    if len(available) < args.sample_count:
        raise ValueError(f"only {len(available)} cached samples, need {args.sample_count}")
    groups: dict[tuple[str, bool, bool], list[int]] = {}
    for index in available:
        sample_id = str(sample_ids[index])
        record = manifest_by_id.get(sample_id)
        if record is None:
            raise ValueError(f"manifest is missing cached sample_id={sample_id}")
        stratum = (
            record.quality_bucket or "unlabeled",
            record.has_capability(EffectCapability.CAMERA_POSE),
            record.has_capability(EffectCapability.OBJECT_MASK),
        )
        groups.setdefault(stratum, []).append(int(index))
    rng = np.random.default_rng(args.seed)
    for indices in groups.values():
        rng.shuffle(indices)
    selected_list: list[int] = []
    active = sorted(groups, key=lambda value: (value[0], value[1], value[2]))
    while len(selected_list) < args.sample_count and active:
        next_active = []
        for stratum in active:
            values = groups[stratum]
            if values and len(selected_list) < args.sample_count:
                selected_list.append(values.pop())
            if values:
                next_active.append(stratum)
        active = next_active
    selected = np.asarray(selected_list, dtype=np.int64)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    rows = []
    for review_index, source_index in enumerate(selected, start=1):
        sample_id = str(sample_ids[source_index])
        target_views = cache.load(sample_id, tuple(videos))
        panels = []
        coverage = []
        for view, array in videos.items():
            current = frame_rgb(array[source_index], 0)
            future = frame_rgb(array[source_index], -1)
            target = target_views[view]
            flow = target.get("rotation_compensated_flow_2d", np.zeros((*current.shape[:2], 2)))
            valid = target.get("rotation_compensated_flow_2d_valid", np.zeros(current.shape[:2], dtype=bool))
            flow = np.asarray(flow)
            valid = np.asarray(valid).astype(bool)
            mask = np.asarray(target.get("relations_mask_t0", np.zeros(flow.shape[:2], dtype=bool))).astype(bool)
            size = (current.shape[1], current.shape[0])
            flow_panel = cv2.resize(flow_rgb(flow, valid), size, interpolation=cv2.INTER_NEAREST)
            panels.extend([current, future, flow_panel, overlay_mask(current, mask)])
            coverage.append({"view": view, "flow_valid_fraction": float(valid.mean()), "mask_pixels": int(mask.sum())})
        height = max(panel.shape[0] for panel in panels)
        panels = [cv2.resize(panel, (height, height)) for panel in panels]
        canvas = np.concatenate(panels, axis=1)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
        draw.text((5, 5), f"{review_index:02d} {sample_id} | per view: t0, t1, compensated flow, ROI", fill=(255, 255, 255))
        filename = f"{review_index:02d}_{hashlib.sha256(sample_id.encode()).hexdigest()[:8]}.png"
        image.save(image_dir / filename)
        rows.append(
            {
                "review_id": f"{review_index:02d}",
                "sample_id": sample_id,
                "image": f"images/{filename}",
                "coverage_json": json.dumps(coverage, sort_keys=True),
                **{column: "" for column in REVIEW_COLUMNS},
                "notes": "",
            }
        )
    review_path = args.output_dir / "target_visual_audit.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    strata_counts: dict[str, int] = {}
    for source_index in selected:
        record = manifest_by_id[str(sample_ids[source_index])]
        key = "|".join(
            [
                record.quality_bucket or "unlabeled",
                f"pose={int(record.has_capability(EffectCapability.CAMERA_POSE))}",
                f"mask={int(record.has_capability(EffectCapability.OBJECT_MASK))}",
            ]
        )
        strata_counts[key] = strata_counts.get(key, 0) + 1
    pack = {
        "schema": "fact-target-visual-audit-pack-v1",
        "selection": "round_robin_quality_pose_mask_strata",
        "seed": args.seed,
        "selected_sample_ids": [row["sample_id"] for row in rows],
        "strata_counts": dict(sorted(strata_counts.items())),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "input_sample_id_sha256": hashlib.sha256((args.input_dir / "sample_id.npy").read_bytes()).hexdigest(),
        "target_manifest_sha256": hashlib.sha256(
            (args.target_cache / "target_manifest.jsonl").read_bytes()
        ).hexdigest(),
        "target_config_sha256": cache.config["config_sha256"],
        "target_identity_sha256": cache.config["identity_sha256"],
    }
    (args.output_dir / "audit_pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} visual audit rows to {review_path}")


if __name__ == "__main__":
    main()
