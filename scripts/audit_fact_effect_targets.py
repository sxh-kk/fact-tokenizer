#!/usr/bin/env python3
"""Create/validate the mandatory 50-sample target-alignment visual audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_experiments import EffectTargetCache  # noqa: E402
from fact_tokenizer.effect_manifest import read_manifest_jsonl  # noqa: E402
from fact_tokenizer.target_audit import (  # noqa: E402
    AUDIT_PACK_SCHEMA,
    IMMUTABLE_REVIEW_COLUMNS,
    REVIEW_COLUMNS,
    canonical_json_sha256,
    immutable_review_row_hashes,
    immutable_review_rows_sha256,
    snapshot_file,
    validate_review_artifacts,
)
from fact_tokenizer.target_audit_selection import validate_audit_selection_contract  # noqa: E402


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
    resized = cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    result[resized] = (0.35 * result[resized] + 0.65 * np.asarray([255, 40, 40])).astype(np.uint8)
    return result


def labeled_panel(panel: np.ndarray, label: str, size: int) -> np.ndarray:
    import cv2
    from PIL import Image, ImageDraw

    resized = cv2.resize(panel, (size, size), interpolation=cv2.INTER_NEAREST)
    image = Image.new("RGB", (size, size + 24), (0, 0, 0))
    image.paste(Image.fromarray(resized), (0, 24))
    ImageDraw.Draw(image).text((4, 5), label, fill=(255, 255, 255))
    return np.asarray(image)


def _load_json(snapshot, label: str) -> Mapping[str, Any]:
    assert snapshot.data is not None
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _load_selection_indices(snapshot) -> np.ndarray:
    assert snapshot.data is not None
    try:
        values = np.load(io.BytesIO(snapshot.data), allow_pickle=False)
    except Exception as exc:
        raise ValueError("selection index is not a safe NPY") from exc
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("selection index must be a one-dimensional integer NPY")
    return values


def validate(path: Path, minimum: float) -> dict[str, Any]:
    """Compatibility entry point used by tests and the CLI."""

    return validate_review_artifacts(path, minimum)


def _create_pack(args: argparse.Namespace) -> Path:
    if args.input_dir is None or args.target_cache is None or args.manifest is None:
        raise ValueError("audit pack creation requires --input-dir, --target-cache, and --manifest")
    if args.sample_count != 50 or args.seed != 20260711:
        raise ValueError("formal visual audit requires sample-count=50 and seed=20260711")
    if (args.output_dir / "audit_pack.json").exists() or (args.output_dir / "target_visual_audit.csv").exists():
        raise FileExistsError("refusing to overwrite an existing visual audit pack")

    cache = EffectTargetCache(args.target_cache)
    config_snapshot = snapshot_file(args.target_cache / "target_config.json", keep_bytes=True)
    target_manifest_snapshot = snapshot_file(
        args.target_cache / "target_manifest.jsonl", keep_bytes=True
    )
    provenance_snapshot = snapshot_file(
        args.target_cache / "builder_provenance.json", keep_bytes=True
    )
    source_manifest_snapshot = snapshot_file(args.manifest, keep_bytes=True)
    config = _load_json(config_snapshot, "target_config.json")
    provenance = _load_json(provenance_snapshot, "builder_provenance.json")
    config_payload = config.get("config")
    identity = config.get("identity")
    if not isinstance(config_payload, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("target config lacks config/identity")
    config_sha = canonical_json_sha256(config_payload)
    identity_sha = canonical_json_sha256({"config_sha256": config_sha, "sources": identity})
    if config.get("config_sha256") != config_sha or config.get("identity_sha256") != identity_sha:
        raise ValueError("target config contains a derived-hash mismatch")
    provenance_identity = {
        key: value
        for key, value in provenance.items()
        if key not in {"selection_contract", "target_identity_sha256", "formal_release"}
    }
    if provenance_identity != identity or provenance.get("target_identity_sha256") != identity_sha:
        raise ValueError("builder provenance does not reproduce the target identity")

    selection_provenance = provenance.get("selection_contract")
    if not isinstance(selection_provenance, Mapping):
        raise ValueError("audit cache lacks frozen selection provenance")
    selection_path = Path(str(selection_provenance.get("selection_contract", "")))
    selection_index_path = Path(
        str(
            selection_provenance.get(
                "sample_index",
                selection_provenance.get("sample_index_npy", ""),
            )
        )
    )
    selection_snapshot = snapshot_file(selection_path, keep_bytes=True)
    selection_index_snapshot = snapshot_file(selection_index_path, keep_bytes=True)
    selection = _load_json(selection_snapshot, "audit_selection.json")
    selected_ids = [str(value) for value in selection.get("sample_ids", [])]
    selected_indices = _load_selection_indices(selection_index_snapshot)
    if (
        selection.get("schema") != "fact-target-audit-selection-v1"
        or int(selection.get("sample_count", -1)) != args.sample_count
        or int(selection.get("seed", -1)) != args.seed
        or len(selected_ids) != args.sample_count
        or len(set(selected_ids)) != args.sample_count
        or len(selected_indices) != args.sample_count
        or selection.get("manifest_sha256") != source_manifest_snapshot.sha256
        or selection.get("index_sha256") != selection_index_snapshot.sha256
        or selection_provenance.get("selection_contract_sha256") != selection_snapshot.sha256
        or selection_provenance.get("sample_index_sha256") != selection_index_snapshot.sha256
    ):
        raise ValueError("selection provenance does not bind the required deterministic selection")
    if set(cache.records) != set(selected_ids) or len(cache.records) != args.sample_count:
        raise ValueError("audit target cache must contain exactly the frozen 50 selected samples")

    records = read_manifest_jsonl(args.manifest)
    canonical_selection = validate_audit_selection_contract(
        records,
        manifest_path=args.manifest,
        index_path=selection_index_path,
        contract_path=selection_path,
    )
    if (
        canonical_selection.get("manifest_sha256") != source_manifest_snapshot.sha256
        or canonical_selection.get("sample_index_sha256") != selection_index_snapshot.sha256
        or canonical_selection.get("selection_contract_sha256") != selection_snapshot.sha256
        or canonical_selection.get("sample_ids") != selected_ids
        or canonical_selection.get("strata_counts") != selection.get("strata_counts")
    ):
        raise ValueError("selection artifacts differ from the canonical deterministic selector")
    if (
        (selected_indices < 0).any()
        or (selected_indices >= len(records)).any()
        or len(set(int(index) for index in selected_indices)) != args.sample_count
    ):
        raise ValueError("selection index contains duplicate/out-of-range rows")
    selected_records = [records[int(index)] for index in selected_indices]
    if [record.sample_id for record in selected_records] != selected_ids:
        raise ValueError("selection index does not reproduce the frozen sample IDs")
    if not all(record.training_valid for record in selected_records):
        raise ValueError("visual audit selection contains a training-invalid sample")

    source_hashes = identity.get("source_hashes")
    input_contract = identity.get("input_transition_contract")
    if not isinstance(source_hashes, Mapping) or not isinstance(input_contract, Mapping):
        raise ValueError("target identity lacks formal input evidence")
    identity_views = source_hashes.get("views")
    contract_files = input_contract.get("files")
    if not isinstance(identity_views, Mapping) or not identity_views or not isinstance(contract_files, Mapping):
        raise ValueError("target identity lacks declared views/input files")
    if source_hashes.get("manifest") != source_manifest_snapshot.sha256:
        raise ValueError("--manifest hash differs from target identity")
    source_contract_path = Path(str(input_contract.get("source_contract", "")))
    source_contract_snapshot = snapshot_file(source_contract_path, keep_bytes=True)
    if (
        source_contract_path.resolve() != source_contract_snapshot.path
        or input_contract.get("source_contract_sha256") != source_contract_snapshot.sha256
        or input_contract.get("source_contract_schema") != "fact-npy-source-contract-v1"
    ):
        raise ValueError("source contract artifact differs from the target input identity")

    sample_id_snapshot = snapshot_file(args.input_dir / "sample_id.npy", keep_bytes=True)
    if not isinstance(contract_files.get("sample_id"), Mapping) or contract_files["sample_id"].get("sha256") != sample_id_snapshot.sha256:
        raise ValueError("input sample_id.npy differs from target transition contract")
    input_view_snapshots = {}
    videos = {}
    for view in sorted(identity_views):
        path = args.input_dir / f"{view}.npy"
        snapshot = snapshot_file(path)
        contract_view = contract_files.get(view)
        if identity_views.get(view) != snapshot.sha256 or not isinstance(contract_view, Mapping) or contract_view.get("sha256") != snapshot.sha256:
            raise ValueError(f"input {view}.npy differs from the target identity/transition contract")
        input_view_snapshots[view] = snapshot
        videos[view] = np.load(path, mmap_mode="r", allow_pickle=False)
    if not videos:
        raise ValueError("target identity declares no visual views")

    assert sample_id_snapshot.data is not None
    sample_ids = np.load(io.BytesIO(sample_id_snapshot.data), allow_pickle=False).astype(str)
    if len(sample_ids) != len(records):
        raise ValueError("input sample_id row count differs from manifest")
    input_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(input_index) != len(sample_ids) or any(sample_id not in input_index for sample_id in selected_ids):
        raise ValueError("input sample_id.npy lacks unique selected sample IDs")

    from PIL import Image

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, str]] = []
    image_refs: list[dict[str, Any]] = []
    for review_index, sample_id in enumerate(selected_ids, start=1):
        source_index = input_index[sample_id]
        target_views = cache.load(sample_id, tuple(videos))
        panels = []
        coverage = []
        for view, array in videos.items():
            current = frame_rgb(array[source_index], 0)
            future = frame_rgb(array[source_index], -1)
            target = target_views[view]
            flow = np.asarray(
                target.get(
                    "rotation_compensated_flow_2d",
                    np.zeros((*current.shape[:2], 2), dtype=np.float32),
                )
            )
            valid = np.asarray(
                target.get(
                    "rotation_compensated_flow_2d_valid",
                    np.zeros(flow.shape[:2], dtype=bool),
                )
            ).astype(bool)
            mask_t0 = np.asarray(
                target.get("relations_mask_t0", np.zeros(flow.shape[:2], dtype=bool))
            ).astype(bool)
            mask_t1 = np.asarray(
                target.get("relations_mask_t1", np.zeros(flow.shape[:2], dtype=bool))
            ).astype(bool)
            size = current.shape[0]
            panels.extend(
                [
                    labeled_panel(current, f"{view} t0", size),
                    labeled_panel(future, f"{view} t+0.5s", size),
                    labeled_panel(flow_rgb(flow, valid), f"{view} flow: hue=dir, value=mag", size),
                    labeled_panel(overlay_mask(current, mask_t0), f"{view} ROI t0", size),
                    labeled_panel(overlay_mask(future, mask_t1), f"{view} ROI t+0.5s", size),
                ]
            )
            coverage.append(
                {
                    "view": view,
                    "flow_valid_fraction": float(valid.mean()),
                    "mask_t0_pixels": int(mask_t0.sum()),
                    "mask_t1_pixels": int(mask_t1.sum()),
                }
            )
        canvas = np.concatenate(panels, axis=1)
        filename = f"{review_index:02d}_{hashlib.sha256(sample_id.encode()).hexdigest()[:8]}.png"
        image_path = image_dir / filename
        Image.fromarray(canvas).save(image_path)
        image_snapshot = snapshot_file(image_path)
        relative_path = image_path.resolve().relative_to(args.output_dir.resolve()).as_posix()
        image_refs.append(
            {
                "review_id": f"{review_index:02d}",
                "sample_id": sample_id,
                "relative_path": relative_path,
                **image_snapshot.reference(),
            }
        )
        rows.append(
            {
                "review_id": f"{review_index:02d}",
                "sample_id": sample_id,
                "image": relative_path,
                "coverage_json": json.dumps(coverage, sort_keys=True, separators=(",", ":")),
                "time_direction_ok": "",
                "coordinate_orientation_ok": "",
                "mask_flow_alignment_ok": "",
                "notes": "",
            }
        )

    review_path = args.output_dir / "target_visual_audit.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    pack = {
        "schema": AUDIT_PACK_SCHEMA,
        "selection": "round_robin_quality_pose_mask_strata",
        "seed": args.seed,
        "sample_count": args.sample_count,
        "selected_sample_ids": selected_ids,
        "strata_counts": selection.get("strata_counts"),
        "view_order": list(videos),
        "target_config_sha256": config_sha,
        "target_identity_sha256": identity_sha,
        "review_contract": {
            "columns": list(REVIEW_COLUMNS),
            "immutable_columns": list(IMMUTABLE_REVIEW_COLUMNS),
            "immutable_rows_sha256": immutable_review_rows_sha256(rows),
            "immutable_row_hashes": immutable_review_row_hashes(rows),
        },
        "artifacts": {
            "source_manifest": source_manifest_snapshot.reference(),
            "input": {
                "sample_id": sample_id_snapshot.reference(),
                "views": {
                    view: snapshot.reference() for view, snapshot in input_view_snapshots.items()
                },
            },
            "target_config": config_snapshot.reference(),
            "target_manifest": target_manifest_snapshot.reference(),
            "builder_provenance": provenance_snapshot.reference(),
            "source_contract": source_contract_snapshot.reference(),
            "selection_contract": selection_snapshot.reference(),
            "selection_index": selection_index_snapshot.reference(),
            "images": image_refs,
        },
    }
    (args.output_dir / "audit_pack.json").write_text(
        json.dumps(pack, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} hash-bound visual audit rows to {review_path}")
    return review_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_review:
        report = validate(args.validate_review, args.minimum_pass_fraction)
        gate_path = args.output_dir / "target_visual_audit_gate.json"
        if gate_path.exists():
            raise FileExistsError(f"refusing to overwrite visual audit gate: {gate_path}")
        gate_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    _create_pack(args)


if __name__ == "__main__":
    main()
