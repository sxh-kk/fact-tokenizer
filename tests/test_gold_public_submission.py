from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.validate_fact_gold300_submission import PUBLIC_FIELDS, validate_submission


def rows() -> list[dict[str, str]]:
    return [
        {
            "review_id": "review_a",
            "image": "images/review_a.png",
            "effect_label": "",
            "contact_label": "",
            "ambiguous_reason": "",
            "annotator_id": "",
            "notes": "",
        },
        {
            "review_id": "review_b",
            "image": "images/review_b.png",
            "effect_label": "",
            "contact_label": "",
            "ambiguous_reason": "",
            "annotator_id": "",
            "notes": "",
        },
    ]


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def completed(template: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **template[0],
            "effect_label": "acquire_control",
            "contact_label": "onset",
            "annotator_id": "ann_a01",
        },
        {
            **template[1],
            "effect_label": "ambiguous",
            "contact_label": "unknown",
            "ambiguous_reason": "occlusion",
            "annotator_id": "ann_a01",
            "notes": "target hidden",
        },
    ]


def test_public_submission_validation_accepts_complete_blind_rows(tmp_path: Path) -> None:
    template = rows()
    template_path = tmp_path / "task.template.csv"
    write_csv(template_path, template)
    manifest = {
        "schema": "fact-gold300-annotator-delivery-v2",
        "tasks": 2,
        "task_template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
    }
    report = validate_submission(
        PUBLIC_FIELDS,
        completed(template),
        PUBLIC_FIELDS,
        template,
        manifest=manifest,
        template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        expected_annotator_id="ann_a01",
    )
    assert report["passed"] is True
    assert report["annotator_id"] == "ann_a01"
    assert report["rows"] == 2


def test_public_submission_validation_rejects_identity_edits_and_invalid_labels() -> None:
    template = rows()
    invalid = completed(template)
    invalid[0]["image"] = "images/other.png"
    invalid[0]["effect_label"] = "grasp"
    invalid[0]["annotator_id"] = "different"
    invalid[1]["ambiguous_reason"] = ""
    report = validate_submission(PUBLIC_FIELDS, invalid, PUBLIC_FIELDS, template)
    assert report["passed"] is False
    joined = "\n".join(report["validation_errors"])
    assert "changed immutable image" in joined
    assert "invalid effect_label" in joined
    assert "multiple annotator_id" in joined
    assert "ambiguous_reason is blank" in joined


def test_public_submission_cli_writes_auditable_report(tmp_path: Path) -> None:
    template = rows()
    template_path = tmp_path / "task.template.csv"
    submission_path = tmp_path / "task.csv"
    manifest_path = tmp_path / "delivery_manifest.json"
    report_path = tmp_path / "submission_validation_report.json"
    write_csv(template_path, template)
    write_csv(submission_path, completed(template))
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "fact-gold300-annotator-delivery-v2",
                "tasks": 2,
                "task_template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "validate_fact_gold300_submission.py"),
            "--submission",
            str(submission_path),
            "--template",
            str(template_path),
            "--delivery-manifest",
            str(manifest_path),
            "--expected-annotator-id",
            "ann_a01",
            "--output-report",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["submission_sha256"] == hashlib.sha256(submission_path.read_bytes()).hexdigest()
    assert report["template_sha256"] == hashlib.sha256(template_path.read_bytes()).hexdigest()
