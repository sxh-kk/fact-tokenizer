from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from fact_tokenizer.gold_annotations import validate_gold_rows
from scripts.import_fact_gold300_submissions import (
    CANONICAL_FIELDS,
    MAPPING_REQUIRED_FIELDS,
    PUBLIC_FIELDS,
    import_submissions,
    load_mapping,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_rows() -> list[dict[str, object]]:
    return [
        {
            "review_id": f"review-{index}",
            "image": f"images/review-{index}.png",
            "source_name": "egoexo",
            "source_index": index,
            "sample_id": f"sample-{index}",
            "take_uid": f"take-{index}",
            "gold_split": "probe_train" if index < 2 else "calibration_dev",
            "timestamp": float(index),
            "source_dataset": "egoexo",
            "dual_annotation": index in {0, 2},
            "representation_training_valid": False,
        }
        for index in range(3)
    ]


def _public_rows(mapping: list[dict[str, object]], annotator_id: str = "") -> list[dict[str, object]]:
    return [
        {
            "review_id": row["review_id"],
            "image": row["image"],
            "effect_label": "" if not annotator_id else ("ambiguous" if index == 1 else "no_effect"),
            "contact_label": "" if not annotator_id else "none",
            "ambiguous_reason": "" if index != 1 or not annotator_id else "effect evidence is occluded",
            "annotator_id": annotator_id,
            "notes": "",
        }
        for index, row in enumerate(mapping)
    ]


def _delivery_manifest(
    path: Path,
    *,
    role: str,
    tasks: int,
    template: Path,
    mapping: Path,
    version: int = 2,
) -> Path:
    payload = {
        "schema": f"fact-gold300-annotator-delivery-v{version}",
        "artifact_role": role,
        "tasks": tasks,
        "task_sha256": _sha256(template),
        "admin_binding": {
            "gold_manifest_sha256": "a" * 64,
            "sealed_mapping_sha256": _sha256(mapping),
        },
        "contains_frozen_sample_mapping": False,
        "contains_other_annotator_task": False,
        "weak_or_model_labels_in_pack": False,
    }
    if version == 2:
        payload["task_template_sha256"] = _sha256(template)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def test_imports_a_and_dual_b_with_v2_delivery_bindings_and_atomic_hashes(tmp_path: Path) -> None:
    mapping_rows = _mapping_rows()
    mapping = _write_csv(tmp_path / "sealed" / "review_mapping.csv", MAPPING_REQUIRED_FIELDS, mapping_rows)
    template_a = _write_csv(tmp_path / "a" / "task.template.csv", PUBLIC_FIELDS, _public_rows(mapping_rows))
    submission_a = _write_csv(tmp_path / "a" / "task.csv", PUBLIC_FIELDS, _public_rows(mapping_rows, "ann-a"))
    dual_rows = [row for row in mapping_rows if row["dual_annotation"]]
    template_b = _write_csv(tmp_path / "b" / "task.template.csv", PUBLIC_FIELDS, _public_rows(dual_rows))
    submission_b = _write_csv(tmp_path / "b" / "task.csv", PUBLIC_FIELDS, _public_rows(dual_rows, "ann-b"))
    manifest_a = _delivery_manifest(
        tmp_path / "a" / "delivery_manifest.json",
        role="annotator_a_all_selected",
        tasks=3,
        template=template_a,
        mapping=mapping,
    )
    manifest_b = _delivery_manifest(
        tmp_path / "b" / "delivery_manifest.json",
        role="annotator_b_dual_only",
        tasks=2,
        template=template_b,
        mapping=mapping,
    )
    output = tmp_path / "canonical"

    report = import_submissions(
        review_mapping=mapping,
        template_a=[template_a],
        submission_a=submission_a,
        submission_b=submission_b,
        # Exercise v2 task.template.csv inference for B.
        template_b=[],
        delivery_manifest_a=manifest_a,
        delivery_manifest_b=manifest_b,
        output_dir=output,
    )

    fields_a, rows_a = _read_csv(output / "annotations_a_canonical.csv")
    fields_b, rows_b = _read_csv(output / "annotations_b_dual_canonical.csv")
    assert fields_a == fields_b == CANONICAL_FIELDS
    assert [row["sample_id"] for row in rows_a] == ["sample-0", "sample-1", "sample-2"]
    assert [row["sample_id"] for row in rows_b] == ["sample-0", "sample-2"]
    _, probe_a = _read_csv(output / "annotations_a_probe_train.csv")
    _, dev_a = _read_csv(output / "annotations_a_calibration_dev.csv")
    _, probe_b = _read_csv(output / "annotations_b_dual_probe_train.csv")
    _, dev_b = _read_csv(output / "annotations_b_dual_calibration_dev.csv")
    assert len(probe_a) == 2 and len(dev_a) == 1
    assert len(probe_b) == 1 and len(dev_b) == 1
    assert validate_gold_rows(rows_a, require_complete=True) == []
    assert validate_gold_rows(rows_b, require_complete=True) == []
    assert report["annotator_a_id"] == "ann-a"
    assert report["annotator_b_id"] == "ann-b"
    assert report["delivery_manifests"]["annotator_b"]["template_hash_verified"] is True
    assert report["output_files"]["annotations_a_probe_train.csv"]["rows"] == 2
    assert (output / "result_manifest.sha256").read_text(encoding="ascii").startswith(
        _sha256(output / "result_manifest.json")
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        import_submissions(
            review_mapping=mapping,
            template_a=[template_a],
            submission_a=submission_a,
            output_dir=output,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", "missing 1 required review_ids"),
        ("unknown", "unknown review_ids"),
        ("duplicate", "duplicate review_id"),
        ("image", "image changed"),
        ("label", "invalid effect_label"),
        ("ambiguous", "ambiguous requires ambiguous_reason"),
        ("annotator", "annotator_id must be stable"),
        ("schema", "invalid public schema"),
    ],
)
def test_rejects_incomplete_or_mutated_public_submissions(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    mapping_rows = _mapping_rows()
    mapping = _write_csv(tmp_path / "review_mapping.csv", MAPPING_REQUIRED_FIELDS, mapping_rows)
    template = _write_csv(tmp_path / "task.template.csv", PUBLIC_FIELDS, _public_rows(mapping_rows))
    rows = _public_rows(mapping_rows, "ann-a")
    fields = PUBLIC_FIELDS
    if mutation == "missing":
        rows.pop()
    elif mutation == "unknown":
        rows[-1]["review_id"] = "not-in-mapping"
    elif mutation == "duplicate":
        rows[-1]["review_id"] = rows[0]["review_id"]
    elif mutation == "image":
        rows[0]["image"] = "images/replaced.png"
    elif mutation == "label":
        rows[0]["effect_label"] = "invented"
    elif mutation == "ambiguous":
        rows[0]["effect_label"] = "ambiguous"
        rows[0]["ambiguous_reason"] = ""
    elif mutation == "annotator":
        rows[-1]["annotator_id"] = "ann-other"
    elif mutation == "schema":
        fields = (*PUBLIC_FIELDS, "sample_id")
        for row in rows:
            row["sample_id"] = "untrusted"
    submission = _write_csv(tmp_path / "task.csv", fields, rows)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=error):
        import_submissions(
            review_mapping=mapping,
            template_a=[template],
            submission_a=submission,
            output_dir=output,
        )
    assert not output.exists()


def test_rejects_dual_scope_identity_frozen_mutation_and_locked_mixing(tmp_path: Path) -> None:
    mapping_rows = _mapping_rows()
    mapping = _write_csv(tmp_path / "review_mapping.csv", MAPPING_REQUIRED_FIELDS, mapping_rows)
    public_template = _write_csv(tmp_path / "task.template.csv", PUBLIC_FIELDS, _public_rows(mapping_rows))
    submission_a = _write_csv(tmp_path / "task.csv", PUBLIC_FIELDS, _public_rows(mapping_rows, "same-id"))
    dual_rows = [row for row in mapping_rows if row["dual_annotation"]]
    template_b = _write_csv(tmp_path / "task_b.template.csv", PUBLIC_FIELDS, _public_rows(dual_rows))
    submission_b = _write_csv(tmp_path / "task_b.csv", PUBLIC_FIELDS, _public_rows(dual_rows, "same-id"))
    with pytest.raises(ValueError, match="different annotator IDs"):
        import_submissions(
            review_mapping=mapping,
            template_a=[public_template],
            submission_a=submission_a,
            submission_b=submission_b,
            template_b=[template_b],
            output_dir=tmp_path / "same-id-output",
        )

    wrong_b_rows = _public_rows([mapping_rows[0], mapping_rows[1]], "ann-b")
    wrong_b = _write_csv(tmp_path / "wrong_b.csv", PUBLIC_FIELDS, wrong_b_rows)
    with pytest.raises(ValueError, match="unknown review_ids"):
        import_submissions(
            review_mapping=mapping,
            template_a=[public_template],
            submission_a=submission_a,
            submission_b=wrong_b,
            template_b=[template_b],
            output_dir=tmp_path / "wrong-b-output",
        )

    canonical_template_rows = [
        {
            **{field: row[field] for field in CANONICAL_FIELDS if field in row},
            "effect_label": "",
            "contact_label": "",
            "ambiguous_reason": "",
            "annotator_id": "",
            "notes": "",
        }
        for row in mapping_rows
    ]
    canonical_template_rows[0]["take_uid"] = "tampered-take"
    canonical_template = _write_csv(
        tmp_path / "canonical_blank.csv", CANONICAL_FIELDS, canonical_template_rows
    )
    with pytest.raises(ValueError, match="changed frozen field take_uid"):
        import_submissions(
            review_mapping=mapping,
            template_a=[canonical_template],
            submission_a=submission_a,
            output_dir=tmp_path / "frozen-output",
        )

    mixed = [dict(row) for row in mapping_rows]
    mixed[-1]["gold_split"] = "locked_test"
    mixed_mapping = _write_csv(tmp_path / "mixed_mapping.csv", MAPPING_REQUIRED_FIELDS, mixed)
    with pytest.raises(ValueError, match="must be imported separately"):
        load_mapping(mixed_mapping)


def test_accepts_v1_manifest_bound_to_public_blank_template(tmp_path: Path) -> None:
    mapping_rows = _mapping_rows()
    mapping = _write_csv(tmp_path / "review_mapping.csv", MAPPING_REQUIRED_FIELDS, mapping_rows)
    template = _write_csv(tmp_path / "task.template.csv", PUBLIC_FIELDS, _public_rows(mapping_rows))
    submission = _write_csv(tmp_path / "task.csv", PUBLIC_FIELDS, _public_rows(mapping_rows, "ann-a"))
    manifest = _delivery_manifest(
        tmp_path / "delivery_manifest.json",
        role="annotator_a_all_selected",
        tasks=3,
        template=template,
        mapping=mapping,
        version=1,
    )
    report = import_submissions(
        review_mapping=mapping,
        template_a=[template],
        submission_a=submission,
        delivery_manifest_a=manifest,
        output_dir=tmp_path / "output",
    )
    assert report["delivery_manifests"]["annotator_a"]["template_hash_field"] == "task_sha256"
    assert report["delivery_manifests"]["annotator_a"]["template_hash_verified"] is True


def test_manifest_must_bind_exact_sealed_mapping_and_locked_output_is_private(tmp_path: Path) -> None:
    rows = _mapping_rows()
    mapping = _write_csv(tmp_path / "review_mapping.csv", MAPPING_REQUIRED_FIELDS, rows)
    template = _write_csv(tmp_path / "task.template.csv", PUBLIC_FIELDS, _public_rows(rows))
    submission = _write_csv(tmp_path / "task.csv", PUBLIC_FIELDS, _public_rows(rows, "ann-a"))
    manifest = _delivery_manifest(
        tmp_path / "delivery_manifest.json",
        role="annotator_a_all_selected",
        tasks=3,
        template=template,
        mapping=mapping,
    )
    rows[0]["source_name"] = "tampered-source"
    _write_csv(mapping, MAPPING_REQUIRED_FIELDS, rows)
    with pytest.raises(ValueError, match="not bound to the supplied sealed review mapping"):
        import_submissions(
            review_mapping=mapping,
            template_a=[template],
            submission_a=submission,
            delivery_manifest_a=manifest,
            output_dir=tmp_path / "tampered-output",
        )

    locked_rows = [dict(_mapping_rows()[0])]
    locked_rows[0]["gold_split"] = "locked_test"
    locked_mapping = _write_csv(
        tmp_path / "locked" / "review_mapping.csv", MAPPING_REQUIRED_FIELDS, locked_rows
    )
    locked_template = _write_csv(
        tmp_path / "locked" / "task.template.csv", PUBLIC_FIELDS, _public_rows(locked_rows)
    )
    locked_submission = _write_csv(
        tmp_path / "locked" / "task.csv", PUBLIC_FIELDS, _public_rows(locked_rows, "ann-a")
    )
    locked_output = tmp_path / "locked-output"
    report = import_submissions(
        review_mapping=locked_mapping,
        template_a=[locked_template],
        submission_a=locked_submission,
        output_dir=locked_output,
    )
    assert report["locked_isolation"] is True
    if os.name == "posix":
        assert stat.S_IMODE(locked_output.stat().st_mode) == 0o700
        assert report["output_security"]["platform_enforced"] is True
