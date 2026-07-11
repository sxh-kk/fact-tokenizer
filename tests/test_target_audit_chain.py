from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fact_tokenizer.target_audit import (
    AUDIT_GATE_SCHEMA,
    validate_review_artifacts,
    validate_visual_audit_gate,
)
from tests._target_audit_fixture import make_visual_audit_v2


def test_v2_gate_revalidates_the_complete_evidence_chain(tmp_path: Path) -> None:
    fixture = make_visual_audit_v2(tmp_path / "audit")
    gate = validate_visual_audit_gate(fixture["gate"], fixture["identity_sha256"])
    assert gate["schema"] == AUDIT_GATE_SCHEMA
    assert gate["rows"] == 50
    assert gate["fully_aligned"] == 45

    first_image = fixture["review_dir"] / "images" / "01.png"
    first_image.write_bytes(first_image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash/size"):
        validate_visual_audit_gate(fixture["gate"], fixture["identity_sha256"])


def test_review_immutable_fields_are_bound_but_notes_remain_reviewable(tmp_path: Path) -> None:
    fixture = make_visual_audit_v2(tmp_path / "audit", passed_rows=50)
    review = fixture["review_csv"]
    with review.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["sample_id"] = "substituted"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="frozen selected|immutable"):
        validate_review_artifacts(review)


def test_self_attested_v1_gate_is_rejected(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "rows": 50,
                "fully_aligned": 50,
                "pass_fraction": 1.0,
                "minimum_pass_fraction": 0.9,
                "target_identity_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must use"):
        validate_visual_audit_gate(gate)

