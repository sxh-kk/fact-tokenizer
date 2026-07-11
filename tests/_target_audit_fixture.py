"""Test-only builder for a complete FACT target visual-audit v2 evidence chain."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fact_tokenizer.effect_manifest import EffectSampleRecord, write_manifest_jsonl
from fact_tokenizer.effect_targets import EffectTargetCacheWriter, EffectTargetConfig
from fact_tokenizer.target_audit import (
    AUDIT_PACK_SCHEMA,
    IMMUTABLE_REVIEW_COLUMNS,
    REVIEW_COLUMNS,
    canonical_json_sha256,
    immutable_review_row_hashes,
    immutable_review_rows_sha256,
    snapshot_file,
    validate_review_artifacts,
)
from fact_tokenizer.target_audit_selection import audit_strata_counts, select_audit_indices


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_visual_audit_v2(root: Path, passed_rows: int = 45) -> dict[str, Any]:
    root.mkdir(parents=True)
    source_manifest = root / "effect_manifest.jsonl"
    records = [
        EffectSampleRecord(
            sample_id=f"s{index}",
            take_uid=f"t{index // 2}",
            split="train",
            row_index=index,
            timestamp=float(index) / 30.0,
        )
        for index in range(50)
    ]
    write_manifest_jsonl(source_manifest, records)
    source_manifest_sha = snapshot_file(source_manifest).sha256

    input_dir = root / "input"
    input_dir.mkdir()
    np.save(input_dir / "sample_id.npy", np.asarray([record.sample_id for record in records]))
    for view in ("ego", "exo"):
        np.save(input_dir / f"{view}.npy", np.zeros((50, 2, 1, 1, 3), dtype=np.uint8))
    input_snapshots = {
        name: snapshot_file(input_dir / f"{name}.npy")
        for name in ("sample_id", "ego", "exo")
    }
    source_contract_path = root / "source_contract.json"
    _write_json(
        source_contract_path,
        {
            "schema": "fact-npy-source-contract-v1",
            "source_dir": str(input_dir.resolve()),
            "rows": 50,
            "files": {
                name: {"sha256": snapshot.sha256}
                for name, snapshot in input_snapshots.items()
            },
        },
    )
    source_contract_snapshot = snapshot_file(source_contract_path, keep_bytes=True)
    identity = {
        "claim": "test 2D target",
        "depth_available": False,
        "flow_3d_available": False,
        "input_transition_contract": {
            "source_contract": str(source_contract_path.resolve()),
            "source_contract_sha256": source_contract_snapshot.sha256,
            "source_contract_schema": "fact-npy-source-contract-v1",
            "files": {
                name: {"sha256": snapshot.sha256}
                for name, snapshot in input_snapshots.items()
            }
        },
        "source_hashes": {
            "manifest": source_manifest_sha,
            "views": {
                view: input_snapshots[view].sha256 for view in ("ego", "exo")
            },
        },
    }

    selection_dir = root / "selection"
    selection_dir.mkdir()
    selection_index = selection_dir / "audit_sample_index.npy"
    selected_indices = select_audit_indices(records)
    selected_records = [records[index] for index in selected_indices]
    np.save(selection_index, np.asarray(selected_indices, dtype=np.int64))
    selection_index_snapshot = snapshot_file(selection_index, keep_bytes=True)
    strata = audit_strata_counts(records, selected_indices)
    selection = {
        "schema": "fact-target-audit-selection-v1",
        "sample_count": 50,
        "seed": 20260711,
        "sample_ids": [record.sample_id for record in selected_records],
        "strata_counts": strata,
        "manifest_sha256": source_manifest_sha,
        "index_sha256": selection_index_snapshot.sha256,
    }
    selection_contract = selection_dir / "audit_selection.json"
    _write_json(selection_contract, selection)
    selection_snapshot = snapshot_file(selection_contract, keep_bytes=True)

    target_cache = root / "target_cache"
    writer = EffectTargetCacheWriter(target_cache, EffectTargetConfig(), identity=identity)
    for record in records:
        writer.write(
            record.sample_id,
            {
                "depth_valid": False,
                "flow_3d_valid": False,
                "ego_full_dino_delta": np.zeros((1, 2), dtype=np.float32),
                "ego_full_dino_delta_valid": True,
            },
            metadata={
                "sample_key": record.sample_key,
                "take_uid": record.take_uid,
                "split": record.split,
                "row_index": record.row_index,
                "timestamp": record.timestamp,
            },
        )
    writer.finalize()
    config_snapshot = snapshot_file(target_cache / "target_config.json", keep_bytes=True)
    config = json.loads(config_snapshot.data.decode("utf-8"))
    provenance = {
        **identity,
        "selection_contract": {
            "schema": selection["schema"],
            "sample_index": str(selection_index.resolve()),
            "sample_index_sha256": selection_index_snapshot.sha256,
            "rows": 50,
            "selection_contract": str(selection_contract.resolve()),
            "selection_contract_sha256": selection_snapshot.sha256,
        },
        "target_identity_sha256": config["identity_sha256"],
        "formal_release": None,
    }
    builder_provenance = target_cache / "builder_provenance.json"
    _write_json(builder_provenance, provenance)

    review_dir = root / "review"
    images_dir = review_dir / "images"
    images_dir.mkdir(parents=True)
    rows = []
    image_refs = []
    for index, record in enumerate(selected_records, start=1):
        relative = f"images/{index:02d}.png"
        image_path = review_dir / relative
        Image.new("RGB", (2, 2), (index % 255, 0, 0)).save(image_path)
        image_snapshot = snapshot_file(image_path)
        image_refs.append(
            {
                "review_id": f"{index:02d}",
                "sample_id": record.sample_id,
                "relative_path": relative,
                **image_snapshot.reference(),
            }
        )
        rows.append(
            {
                "review_id": f"{index:02d}",
                "sample_id": record.sample_id,
                "image": relative,
                "coverage_json": json.dumps([{"view": "ego"}], separators=(",", ":")),
                "time_direction_ok": "yes",
                "coordinate_orientation_ok": "yes",
                "mask_flow_alignment_ok": "yes" if index <= passed_rows else "no",
                "notes": "",
            }
        )
    review_csv = review_dir / "target_visual_audit.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    pack = {
        "schema": AUDIT_PACK_SCHEMA,
        "selection": "round_robin_quality_pose_mask_strata",
        "seed": 20260711,
        "sample_count": 50,
        "selected_sample_ids": [record.sample_id for record in selected_records],
        "strata_counts": strata,
        "view_order": ["ego", "exo"],
        "target_config_sha256": config["config_sha256"],
        "target_identity_sha256": config["identity_sha256"],
        "review_contract": {
            "columns": list(REVIEW_COLUMNS),
            "immutable_columns": list(IMMUTABLE_REVIEW_COLUMNS),
            "immutable_rows_sha256": immutable_review_rows_sha256(rows),
            "immutable_row_hashes": immutable_review_row_hashes(rows),
        },
        "artifacts": {
            "source_manifest": snapshot_file(source_manifest).reference(),
            "input": {
                "sample_id": input_snapshots["sample_id"].reference(),
                "views": {
                    view: input_snapshots[view].reference() for view in ("ego", "exo")
                },
            },
            "target_config": config_snapshot.reference(),
            "target_manifest": snapshot_file(
                target_cache / "target_manifest.jsonl", keep_bytes=True
            ).reference(),
            "builder_provenance": snapshot_file(builder_provenance, keep_bytes=True).reference(),
            "source_contract": source_contract_snapshot.reference(),
            "selection_contract": selection_snapshot.reference(),
            "selection_index": selection_index_snapshot.reference(),
            "images": image_refs,
        },
    }
    _write_json(review_dir / "audit_pack.json", pack)
    gate = validate_review_artifacts(review_csv, 0.90)
    gate_path = root / "target_visual_audit_gate.json"
    _write_json(gate_path, gate)
    return {
        "gate": gate_path,
        "identity": identity,
        "identity_sha256": config["identity_sha256"],
        "manifest": source_manifest,
        "records": records,
        "review_csv": review_csv,
        "review_dir": review_dir,
    }
