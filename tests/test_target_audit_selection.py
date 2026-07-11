from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fact_tokenizer.effect_manifest import EffectCapability, EffectSampleRecord, write_manifest_jsonl
from fact_tokenizer.target_audit_selection import validate_audit_selection_contract


def _records() -> list[EffectSampleRecord]:
    return [
        EffectSampleRecord(
            sample_id=f"sample-{index:03d}",
            take_uid=f"take-{index // 2:03d}",
            split="train",
            row_index=index,
            quality_bucket=("good", "poor", "unlabeled")[index % 3],
            capability_validity={
                EffectCapability.CAMERA_POSE: index % 2 == 0,
                EffectCapability.OBJECT_MASK: index % 5 == 0,
            },
            training_valid=index != 59,
        )
        for index in range(60)
    ]


def test_canonical_target_audit_selection_is_recomputed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(manifest, _records())
    output = tmp_path / "selection"
    subprocess.run(
        [
            sys.executable,
            "scripts/select_fact_target_audit_samples.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    validated = validate_audit_selection_contract(
        _records(),
        manifest_path=manifest,
        index_path=output / "audit_sample_index.npy",
        contract_path=output / "audit_selection.json",
    )
    assert validated["sample_count"] == 50
    assert validated["seed"] == 20260711


def test_target_audit_selection_rejects_noncanonical_index_and_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = _records()
    write_manifest_jsonl(manifest, records)
    index = tmp_path / "index.npy"
    np.save(index, np.arange(50, dtype=np.int64))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"schema": "fact-target-audit-selection-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical deterministic"):
        validate_audit_selection_contract(
            records,
            manifest_path=manifest,
            index_path=index,
            contract_path=contract,
        )


def test_target_audit_selection_rejects_custom_formal_parameters() -> None:
    from fact_tokenizer.target_audit_selection import select_audit_indices

    with pytest.raises(ValueError, match="fixed to 50"):
        select_audit_indices(_records(), sample_count=49)
