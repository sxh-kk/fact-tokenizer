from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import (
    EFFECT_MANIFEST_SCHEMA_VERSION,
    DiagnosticCodelabel,
    EffectCapability,
    EffectSampleRecord,
    audit_effect_manifest,
    build_records_from_npy_directory,
    read_manifest_jsonl,
    write_manifest_jsonl,
)


def make_npy_store(path: Path, *, sample_ids: list[str] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    sample_ids = sample_ids or ["take_a:0.000", "take_a:1.000", "take_b:0.000"]
    count = len(sample_ids)
    rng = np.random.default_rng(7)
    np.save(path / "ego.npy", rng.integers(0, 255, size=(count, 4, 8, 8, 3), dtype=np.uint8))
    np.save(path / "exo.npy", rng.integers(0, 255, size=(count, 4, 8, 8, 3), dtype=np.uint8))
    np.save(path / "sample_id.npy", np.asarray(sample_ids))
    np.save(path / "take_uid.npy", np.asarray([value.split(":", 1)[0] for value in sample_ids]))
    np.save(path / "timestamp.npy", np.arange(count, dtype=np.float32))
    return path


def test_record_jsonl_roundtrip_and_diagnostic_quarantine(tmp_path: Path) -> None:
    record = EffectSampleRecord(
        sample_id="take_a:0.000",
        take_uid="take_a",
        split="train",
        row_index=0,
        source_dataset="intern02",
        timestamp=0.0,
        capability_validity={
            EffectCapability.RGB_PAIRED: True,
            EffectCapability.CAMERA_POSE: True,
        },
        quality_bucket="usable",
        quality_weight=0.75,
        annotation_refs={"camera_pose": "poses/take_a.json"},
        content_hashes={"rgb_store": "a" * 64},
        diagnostic_codelabel=DiagnosticCodelabel(value="17", source="failed_500"),
        provenance={"data_hash": "abc"},
    )
    assert record.sample_key == "intern02/train/take_a:0.000"
    assert record.has_capability("camera_pose")
    assert not record.has_capability("depth")
    assert record.diagnostic_codelabel is not None
    assert record.diagnostic_codelabel.training_valid is False
    assert record.training_valid is False

    path = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(path, [record])
    loaded = read_manifest_jsonl(path)

    assert loaded == [record]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == EFFECT_MANIFEST_SCHEMA_VERSION
    assert payload["diagnostic_codelabel"]["training_valid"] is False
    assert set(payload["capability_validity"]) == {capability.value for capability in EffectCapability}


def test_diagnostic_codelabel_can_never_be_training_valid() -> None:
    with pytest.raises(ValueError, match="training_valid=False"):
        DiagnosticCodelabel(value="5", training_valid=True)

    payload = {"value": "5", "training_valid": True}
    with pytest.raises(ValueError, match="training_valid=False"):
        DiagnosticCodelabel.from_dict(payload)
    with pytest.raises(ValueError, match="64-character SHA256"):
        EffectSampleRecord(
            sample_id="s",
            take_uid="t",
            split="train",
            row_index=0,
            content_hashes={"source": "abc123"},
        )


def test_v7_forces_depth_and_flow_3d_invalid() -> None:
    camera_only = EffectSampleRecord(
        sample_id="sample",
        take_uid="take",
        split="train",
        row_index=0,
        capability_validity={EffectCapability.CAMERA_POSE: True},
    )
    assert camera_only.has_capability(EffectCapability.CAMERA_POSE)
    assert not camera_only.has_capability(EffectCapability.DEPTH)
    assert not camera_only.has_capability(EffectCapability.FLOW_3D)

    with pytest.raises(ValueError, match="depth_valid=false"):
        EffectSampleRecord(
            sample_id="bad",
            take_uid="take",
            split="train",
            row_index=1,
            capability_validity={EffectCapability.CAMERA_POSE: True, EffectCapability.FLOW_3D: True},
        )

    with pytest.raises(ValueError, match="depth_valid=false"):
        EffectSampleRecord(
            sample_id="also-bad",
            take_uid="take",
            split="train",
            row_index=2,
            capability_validity={EffectCapability.DEPTH: True},
        )

    with pytest.raises(ValueError, match="depth_valid=false"):
        EffectSampleRecord(
            sample_id="both-bad",
            take_uid="take",
            split="train",
            row_index=3,
            capability_validity={EffectCapability.DEPTH: True, EffectCapability.FLOW_3D: True},
        )


def test_build_records_from_mmap_npy_metadata_and_audit(tmp_path: Path) -> None:
    store = make_npy_store(tmp_path / "store")
    camera_pose = np.asarray([True, False, True], dtype=np.bool_)
    camera_path = tmp_path / "camera_pose_valid.npy"
    np.save(camera_path, camera_pose)
    diagnostics = {
        "take_a:1.000": DiagnosticCodelabel(value="23", source="failed_500"),
        "not_in_store": DiagnosticCodelabel(value="99", source="failed_500"),
    }

    records, audit = build_records_from_npy_directory(
        store,
        split="heldout",
        source_dataset="intern02",
        capability_arrays={EffectCapability.CAMERA_POSE: camera_path},
        diagnostic_codelabels=diagnostics,
    )

    assert [record.row_index for record in records] == [0, 1, 2]
    assert [record.sample_key for record in records] == [
        "intern02/heldout/take_a:0.000",
        "intern02/heldout/take_a:1.000",
        "intern02/heldout/take_b:0.000",
    ]
    assert all(record.has_capability(EffectCapability.RGB_PAIRED) for record in records)
    assert [record.has_capability(EffectCapability.CAMERA_POSE) for record in records] == [True, False, True]
    assert records[1].diagnostic_codelabel is not None
    assert records[1].diagnostic_codelabel.training_valid is False
    assert records[1].training_valid is False
    assert audit["sample_count"] == 3
    assert audit["take_count"] == 2
    assert audit["capability_valid_counts"]["camera_pose"] == 2
    assert audit["capability_valid_counts"]["depth"] == 0
    assert audit["capability_valid_counts"]["flow_3d"] == 0
    assert audit["no_depth_guard"] == {
        "depth_valid_count": 0,
        "flow_3d_valid_count": 0,
        "flow_3d_without_depth_count": 0,
    }
    assert all(entry["memory_mapped"] for entry in audit["npy_arrays"].values())
    assert audit["capability_sidecars"]["camera_pose"]["memory_mapped"] is True
    assert audit["diagnostic_codelabel_join"] == {
        "provided": 2,
        "matched": 1,
        "unmatched": ["not_in_store"],
        "training_valid_count": 0,
    }


def test_builder_rejects_bad_alignment_duplicate_keys_and_flow_3d_without_depth(tmp_path: Path) -> None:
    mismatched = make_npy_store(tmp_path / "mismatched")
    np.save(mismatched / "exo.npy", np.zeros((2, 4, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="different lengths"):
        build_records_from_npy_directory(mismatched, split="train")

    duplicate = make_npy_store(tmp_path / "duplicate", sample_ids=["same", "same"])
    with pytest.raises(ValueError, match="Duplicate effect manifest sample_key"):
        build_records_from_npy_directory(duplicate, split="train")

    no_depth = make_npy_store(tmp_path / "no_depth")
    flow_path = tmp_path / "flow_3d_valid.npy"
    np.save(flow_path, np.ones(3, dtype=np.bool_))
    with pytest.raises(ValueError, match="depth_valid=false"):
        build_records_from_npy_directory(
            no_depth,
            split="train",
            capability_arrays={EffectCapability.FLOW_3D: flow_path},
        )


def test_jsonl_reader_rejects_tampered_or_duplicate_primary_keys(tmp_path: Path) -> None:
    record = EffectSampleRecord(sample_id="s", take_uid="t", split="train", row_index=0)
    tampered = record.to_dict()
    tampered["sample_key"] = "wrong/key"
    path = tmp_path / "tampered.jsonl"
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample_key mismatch"):
        read_manifest_jsonl(path)

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(record.to_json() + "\n" + record.to_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate effect manifest sample_key"):
        read_manifest_jsonl(duplicate_path)


def test_build_manifest_cli_writes_quarantined_diagnostics_and_audit(tmp_path: Path) -> None:
    store = make_npy_store(tmp_path / "store")
    camera_path = tmp_path / "camera.npy"
    np.save(camera_path, np.asarray([1, 1, 0], dtype=np.uint8))
    diagnostics = tmp_path / "diagnostics.csv"
    diagnostics.write_text(
        "sample_id,codelabel,training_valid,note\n"
        "take_a:0.000,12,true,failed experiment\n",
        encoding="utf-8",
    )
    output = tmp_path / "effect.jsonl"
    audit_path = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_fact_effect_manifest.py"),
            "--input-dir",
            str(store),
            "--split",
            "train",
            "--source-dataset",
            "intern02",
            "--output-jsonl",
            str(output),
            "--audit-json",
            str(audit_path),
            "--capability-npy",
            f"camera_pose={camera_path}",
            "--diagnostic-codelabels",
            str(diagnostics),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote 3 effect manifest records" in completed.stdout
    records = read_manifest_jsonl(output)
    assert records[0].diagnostic_codelabel is not None
    assert records[0].diagnostic_codelabel.training_valid is False
    assert records[0].diagnostic_codelabel.metadata == {"note": "failed experiment"}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["capability_valid_counts"]["camera_pose"] == 2
    assert audit["diagnostic_codelabel_training_valid_count"] == 0
    assert audit["diagnostic_codelabel_join"]["training_valid_count"] == 0


def test_audit_rejects_duplicate_records() -> None:
    record = EffectSampleRecord(sample_id="s", take_uid="t", split="train", row_index=0)
    with pytest.raises(ValueError, match="Duplicate effect manifest sample_key"):
        audit_effect_manifest([record, record])


def test_take_quality_join_keeps_human_bucket_and_derived_weight_separate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(
        manifest,
        [
            EffectSampleRecord(sample_id=f"s{index}", take_uid=f"t{index}", split="train", row_index=index)
            for index in range(3)
        ],
    )
    human = tmp_path / "human.csv"
    human.write_text(
        "take_uid,usable_for,take_relevance,confidence\n"
        "t0,tokenizer_main,A_interaction_rich,high\n"
        "t1,diagnostic_candidate,A_interaction_rich,high\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.csv"
    weights.write_text(
        "take_uid,bucket,sample_weight\n"
        "t0,diagnostic_candidate,0.125\n"
        "t2,discard,0.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "quality"
    subprocess.run(
        [
            sys.executable,
            "scripts/index_fact_take_quality.py",
            "--manifest",
            str(manifest),
            "--human-quality-csv",
            str(human),
            "--derived-weight-csv",
            str(weights),
            "--output-dir",
            str(output),
            "--allow-noncanonical-human-count",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    records = read_manifest_jsonl(output / "effect_manifest_take_quality_indexed.jsonl")
    assert records[0].quality_bucket == "tokenizer_main"
    assert records[0].quality_weight == pytest.approx(0.125)
    assert records[0].has_capability(EffectCapability.TAKE_QUALITY)
    assert records[1].quality_bucket == "diagnostic_candidate"
    assert records[1].training_valid is True
    assert records[2].quality_bucket == "unlabeled"
    assert records[2].quality_weight == 0.0
    assert not records[2].has_capability(EffectCapability.TAKE_QUALITY)


def test_cross_manifest_split_audit_detects_interval_overlap(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest_jsonl(
        first,
        [EffectSampleRecord(sample_id="a", take_uid="take", split="train", row_index=0, timestamp=0.0)],
    )
    write_manifest_jsonl(
        second,
        [EffectSampleRecord(sample_id="b", take_uid="take", split="heldout", row_index=0, timestamp=0.25)],
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_fact_effect_splits.py",
            "--manifest",
            f"train={first}",
            "--manifest",
            f"heldout={second}",
            "--output-json",
            str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    report = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert report["pairwise_overlaps"][0]["frame_overlap"]
