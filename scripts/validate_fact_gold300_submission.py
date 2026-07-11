#!/usr/bin/env python3
"""Validate a blind FACT gold300 annotator submission without revealing sample identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


IMMUTABLE_FIELDS = ("review_id", "image")
EDITABLE_FIELDS = (
    "effect_label",
    "contact_label",
    "ambiguous_reason",
    "annotator_id",
    "notes",
)
PUBLIC_FIELDS = (*IMMUTABLE_FIELDS, *EDITABLE_FIELDS)
EFFECT_LABELS = (
    "no_effect",
    "approach_align",
    "acquire_control",
    "state_change_or_manipulate",
    "transport_reposition",
    "release_complete",
    "recover_abort",
    "ambiguous",
)
CONTACT_LABELS = ("none", "onset", "stable", "release", "unknown")
ANNOTATOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--delivery-manifest", type=Path)
    parser.add_argument("--expected-annotator-id")
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_public_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def _duplicate_values(rows: Sequence[Mapping[str, str]], field: str) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = str(row.get(field, ""))
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_submission(
    submission_fields: Sequence[str],
    submission_rows: Sequence[Mapping[str, str]],
    template_fields: Sequence[str],
    template_rows: Sequence[Mapping[str, str]],
    *,
    manifest: Mapping[str, Any] | None = None,
    template_sha256: str | None = None,
    expected_annotator_id: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if tuple(template_fields) != PUBLIC_FIELDS:
        errors.append(
            f"template columns must be exactly {list(PUBLIC_FIELDS)}, got {list(template_fields)}"
        )
    if tuple(submission_fields) != PUBLIC_FIELDS:
        errors.append(
            f"submission columns must be exactly {list(PUBLIC_FIELDS)}, got {list(submission_fields)}"
        )

    for row_number, row in enumerate(template_rows, start=2):
        if any(str(row.get(field, "")).strip() for field in EDITABLE_FIELDS):
            errors.append(f"template row {row_number} is not blank in editable fields")

    for label, rows in (("template", template_rows), ("submission", submission_rows)):
        blank_ids = [index for index, row in enumerate(rows, start=2) if not str(row.get("review_id", "")).strip()]
        if blank_ids:
            errors.append(f"{label} has blank review_id at rows {blank_ids[:5]}")
        duplicates = _duplicate_values(rows, "review_id")
        if duplicates:
            errors.append(f"{label} has duplicate review_id values: {duplicates[:5]}")

    template_by_id = {str(row.get("review_id", "")): row for row in template_rows}
    submission_by_id = {str(row.get("review_id", "")): row for row in submission_rows}
    missing = sorted(set(template_by_id) - set(submission_by_id))
    unexpected = sorted(set(submission_by_id) - set(template_by_id))
    if missing:
        errors.append(f"submission is missing {len(missing)} review_id values: {missing[:5]}")
    if unexpected:
        errors.append(f"submission has {len(unexpected)} unexpected review_id values: {unexpected[:5]}")

    annotator_ids: set[str] = set()
    for review_id in sorted(set(template_by_id) & set(submission_by_id)):
        template = template_by_id[review_id]
        row = submission_by_id[review_id]
        if str(row.get("image", "")) != str(template.get("image", "")):
            errors.append(f"review_id {review_id!r} changed immutable image path")

        for field in EDITABLE_FIELDS:
            value = str(row.get(field, ""))
            if value != value.strip():
                errors.append(f"review_id {review_id!r} has leading/trailing whitespace in {field}")
        effect = str(row.get("effect_label", "")).strip()
        contact = str(row.get("contact_label", "")).strip()
        reason = str(row.get("ambiguous_reason", "")).strip()
        annotator_id = str(row.get("annotator_id", "")).strip()
        notes = str(row.get("notes", ""))

        if effect not in EFFECT_LABELS:
            errors.append(f"review_id {review_id!r} has invalid effect_label={effect!r}")
        if contact not in CONTACT_LABELS:
            errors.append(f"review_id {review_id!r} has invalid contact_label={contact!r}")
        if effect == "ambiguous" and not reason:
            errors.append(f"review_id {review_id!r} is ambiguous but ambiguous_reason is blank")
        if effect != "ambiguous" and reason:
            errors.append(f"review_id {review_id!r} must leave ambiguous_reason blank")
        if not ANNOTATOR_ID_RE.fullmatch(annotator_id):
            errors.append(
                f"review_id {review_id!r} has invalid annotator_id; use 2-64 letters, digits, '.', '_' or '-'"
            )
        else:
            annotator_ids.add(annotator_id)
        if len(reason) > 256:
            errors.append(f"review_id {review_id!r} ambiguous_reason exceeds 256 characters")
        if len(notes) > 1000:
            errors.append(f"review_id {review_id!r} notes exceeds 1000 characters")

    if len(annotator_ids) > 1:
        errors.append(f"submission contains multiple annotator_id values: {sorted(annotator_ids)}")
    actual_annotator_id = next(iter(annotator_ids), None) if len(annotator_ids) == 1 else None
    if expected_annotator_id and actual_annotator_id != expected_annotator_id:
        errors.append(
            f"submission annotator_id={actual_annotator_id!r} does not match expected {expected_annotator_id!r}"
        )

    if manifest is not None:
        if manifest.get("schema") not in {
            "fact-gold300-annotator-delivery-v1",
            "fact-gold300-annotator-delivery-v2",
        }:
            errors.append("delivery manifest has an unsupported schema")
        expected_tasks = manifest.get("tasks")
        if expected_tasks is not None and int(expected_tasks) != len(template_rows):
            errors.append(
                f"delivery manifest tasks={expected_tasks} does not match template rows={len(template_rows)}"
            )
        expected_template_sha = manifest.get("task_template_sha256", manifest.get("task_sha256"))
        if expected_template_sha and template_sha256 != str(expected_template_sha):
            errors.append("task.template.csv hash does not match delivery manifest")

    return {
        "schema": "fact-gold300-public-submission-validation-v1",
        "rows": len(submission_rows),
        "annotator_id": actual_annotator_id,
        "effect_labels": list(EFFECT_LABELS),
        "contact_labels": list(CONTACT_LABELS),
        "validation_errors": errors,
        "passed": not errors,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    submission_fields, submission_rows = load_public_csv(args.submission)
    template_fields, template_rows = load_public_csv(args.template)
    manifest = None
    if args.delivery_manifest is not None:
        manifest = json.loads(args.delivery_manifest.read_text(encoding="utf-8"))
    report = validate_submission(
        submission_fields,
        submission_rows,
        template_fields,
        template_rows,
        manifest=manifest,
        template_sha256=sha256_file(args.template),
        expected_annotator_id=args.expected_annotator_id,
    )
    report.update(
        {
            "submission": str(args.submission),
            "submission_sha256": sha256_file(args.submission),
            "template": str(args.template),
            "template_sha256": sha256_file(args.template),
            "delivery_manifest": str(args.delivery_manifest) if args.delivery_manifest else None,
            "delivery_manifest_sha256": (
                sha256_file(args.delivery_manifest) if args.delivery_manifest is not None else None
            ),
        }
    )
    if args.output_report is not None:
        write_json_atomic(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
