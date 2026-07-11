#!/usr/bin/env python3
"""Validate blind FACT gold300 submissions and atomically publish canonical CSVs.

The public annotation sheets intentionally contain only opaque review IDs.  This
admin-side importer is the sole join back to the frozen sample mapping.  It
never trusts sample identity or split information supplied by an annotator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import uuid
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.gold_annotations import CONTACT_LABELS, EFFECT_LABELS  # noqa: E402
from scripts.materialize_fact_gold300_review import FROZEN_FIELDS  # noqa: E402


EDITABLE_FIELDS = (
    "effect_label",
    "contact_label",
    "ambiguous_reason",
    "annotator_id",
    "notes",
)
PUBLIC_FIELDS = ("review_id", "image", *EDITABLE_FIELDS)
CANONICAL_FIELDS = (*FROZEN_FIELDS, *EDITABLE_FIELDS)
MAPPING_REQUIRED_FIELDS = (
    "review_id",
    "image",
    "source_name",
    "source_index",
    *FROZEN_FIELDS,
)
DELIVERY_SCHEMAS = {
    "fact-gold300-annotator-delivery-v1",
    "fact-gold300-annotator-delivery-v2",
}
GOLD_SPLITS = {"probe_train", "calibration_dev", "locked_test"}
ANNOTATOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-mapping", type=Path, required=True)
    parser.add_argument(
        "--template-a",
        type=Path,
        action="append",
        required=True,
        help=(
            "Blank annotator-A task template or sealed canonical blank template. "
            "Repeat when the sealed canonical template is split across CSV files."
        ),
    )
    parser.add_argument("--submission-a", type=Path, required=True)
    parser.add_argument("--submission-b", type=Path)
    parser.add_argument(
        "--template-b",
        type=Path,
        action="append",
        default=[],
        help="Optional blank dual-only template for annotator B.",
    )
    parser.add_argument("--delivery-manifest-a", type=Path)
    parser.add_argument("--delivery-manifest-b", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = tuple(reader.fieldnames)
        if len(fields) != len(set(fields)):
            raise ValueError(f"CSV has duplicate columns: {path}")
        rows = list(reader)
    return fields, rows


def _require_exact_header(path: Path, actual: Sequence[str], expected: Sequence[str]) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(
            f"{path} has an invalid public schema; expected {list(expected)}, got {list(actual)}"
        )


def _bool_value(value: object, *, field: str, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{context}: invalid boolean {field}={value!r}")


def _normalized_frozen(field: str, value: object, *, context: str) -> object:
    if field == "timestamp":
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid timestamp={value!r}") from exc
        if not math.isfinite(result):
            raise ValueError(f"{context}: timestamp must be finite")
        return result
    if field in {"dual_annotation", "representation_training_valid"}:
        return _bool_value(value, field=field, context=context)
    return str(value)


def _validate_relative_image(value: object, *, context: str) -> str:
    image = str(value)
    pure = PurePosixPath(image)
    if (
        not image
        or "\\" in image
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{context}: image must be a normalized relative POSIX path")
    return image


def load_mapping(path: Path) -> list[dict[str, str]]:
    fields, rows = _read_csv(path)
    missing = set(MAPPING_REQUIRED_FIELDS) - set(fields)
    if missing:
        raise ValueError(f"review mapping is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("review mapping is empty")
    seen_review: set[str] = set()
    seen_sample: set[str] = set()
    splits: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        context = f"{path}:{row_number}"
        review_id = str(row["review_id"]).strip()
        sample_id = str(row["sample_id"]).strip()
        take_uid = str(row["take_uid"]).strip()
        if not review_id or not sample_id or not take_uid:
            raise ValueError(f"{context}: review_id, sample_id and take_uid must be non-empty")
        if review_id in seen_review:
            raise ValueError(f"{context}: duplicate review_id={review_id!r}")
        if sample_id in seen_sample:
            raise ValueError(f"{context}: duplicate sample_id={sample_id!r}")
        seen_review.add(review_id)
        seen_sample.add(sample_id)
        row["review_id"] = review_id
        row["sample_id"] = sample_id
        row["take_uid"] = take_uid
        row["image"] = _validate_relative_image(row["image"], context=context)
        split = str(row["gold_split"])
        if split not in GOLD_SPLITS:
            raise ValueError(f"{context}: invalid gold_split={split!r}")
        splits.add(split)
        for field in FROZEN_FIELDS:
            _normalized_frozen(field, row[field], context=context)
        if _bool_value(
            row["representation_training_valid"],
            field="representation_training_valid",
            context=context,
        ):
            raise ValueError(f"{context}: gold mapping must prohibit representation training")
    if "locked_test" in splits and len(splits) != 1:
        raise ValueError("locked_test and non-locked gold rows must be imported separately")
    return rows


def _validate_id_coverage(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_ids: set[str],
    path: Path,
    id_field: str,
) -> dict[str, Mapping[str, str]]:
    values: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        value = str(row.get(id_field, "")).strip()
        if not value:
            raise ValueError(f"{path}:{row_number}: missing {id_field}")
        if value in values:
            raise ValueError(f"{path}:{row_number}: duplicate {id_field}={value!r}")
        values[value] = row
    actual_ids = set(values)
    unknown = sorted(actual_ids - expected_ids)
    missing = sorted(expected_ids - actual_ids)
    if unknown:
        raise ValueError(f"{path} contains unknown {id_field}s: {unknown[:5]}")
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} required {id_field}s: {missing[:5]}")
    return values


def _validate_public_rows(
    path: Path,
    *,
    expected_mapping: Mapping[str, Mapping[str, str]],
    require_complete: bool,
) -> tuple[dict[str, Mapping[str, str]], str | None]:
    fields, rows = _read_csv(path)
    _require_exact_header(path, fields, PUBLIC_FIELDS)
    by_id = _validate_id_coverage(
        rows,
        expected_ids=set(expected_mapping),
        path=path,
        id_field="review_id",
    )
    annotator_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        context = f"{path}:{row_number}"
        review_id = str(row["review_id"]).strip()
        expected = expected_mapping[review_id]
        if _validate_relative_image(row["image"], context=context) != str(expected["image"]):
            raise ValueError(f"{context}: image changed for review_id={review_id!r}")
        annotator_id = str(row["annotator_id"]).strip()
        if annotator_id:
            annotator_ids.add(annotator_id)
        if require_complete:
            for field in EDITABLE_FIELDS:
                value = str(row[field])
                if value != value.strip():
                    raise ValueError(f"{context}: {field} has leading/trailing whitespace")
            effect = str(row["effect_label"]).strip()
            contact = str(row["contact_label"]).strip()
            if effect not in EFFECT_LABELS:
                raise ValueError(f"{context}: invalid effect_label={effect!r}")
            if contact not in CONTACT_LABELS:
                raise ValueError(f"{context}: invalid contact_label={contact!r}")
            if effect == "ambiguous" and not str(row["ambiguous_reason"]).strip():
                raise ValueError(f"{context}: ambiguous requires ambiguous_reason")
            if effect != "ambiguous" and str(row["ambiguous_reason"]).strip():
                raise ValueError(f"{context}: ambiguous_reason must be blank for non-ambiguous labels")
            if not ANNOTATOR_ID_RE.fullmatch(annotator_id):
                raise ValueError(
                    f"{context}: annotator_id must use 2-64 letters, digits, '.', '_' or '-'"
                )
            if len(str(row["ambiguous_reason"])) > 256:
                raise ValueError(f"{context}: ambiguous_reason exceeds 256 characters")
            if len(str(row["notes"])) > 1000:
                raise ValueError(f"{context}: notes exceeds 1000 characters")
        elif any(str(row[field]).strip() for field in EDITABLE_FIELDS):
            raise ValueError(f"{context}: annotation template must be blank")
    if require_complete and len(annotator_ids) != 1:
        raise ValueError(f"{path}: annotator_id must be stable across every row")
    return by_id, next(iter(annotator_ids)) if len(annotator_ids) == 1 else None


def _validate_canonical_blank_templates(
    paths: Sequence[Path],
    *,
    mapping_rows: Sequence[Mapping[str, str]],
) -> None:
    by_sample = {str(row["sample_id"]): row for row in mapping_rows}
    combined: list[dict[str, str]] = []
    for path in paths:
        fields, rows = _read_csv(path)
        if tuple(fields) != CANONICAL_FIELDS:
            raise ValueError(
                f"{path} is neither a public task template nor a canonical blank template"
            )
        combined.extend(rows)
        for row_number, row in enumerate(rows, start=2):
            if any(str(row[field]).strip() for field in EDITABLE_FIELDS):
                raise ValueError(f"{path}:{row_number}: annotation template must be blank")
    indexed = _validate_id_coverage(
        combined,
        expected_ids=set(by_sample),
        path=paths[0],
        id_field="sample_id",
    )
    for sample_id, template in indexed.items():
        mapping = by_sample[sample_id]
        for field in FROZEN_FIELDS:
            left = _normalized_frozen(field, template[field], context=str(paths[0]))
            right = _normalized_frozen(field, mapping[field], context=str(paths[0]))
            if left != right:
                raise ValueError(f"canonical template changed frozen field {field} for {sample_id!r}")


def validate_templates(
    paths: Sequence[Path],
    *,
    expected_mapping: Mapping[str, Mapping[str, str]],
    mapping_rows: Sequence[Mapping[str, str]],
) -> str:
    if not paths:
        raise ValueError("at least one blank template is required")
    first_fields, _ = _read_csv(paths[0])
    if tuple(first_fields) == PUBLIC_FIELDS:
        if len(paths) != 1:
            raise ValueError("a public delivery template must be supplied as one CSV")
        _validate_public_rows(paths[0], expected_mapping=expected_mapping, require_complete=False)
        return "public_delivery"
    _validate_canonical_blank_templates(paths, mapping_rows=mapping_rows)
    return "sealed_canonical"


def validate_delivery_manifest(
    path: Path,
    *,
    expected_role: str,
    expected_tasks: int,
    template_paths: Sequence[Path],
    template_kind: str,
    expected_mapping_sha256: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in DELIVERY_SCHEMAS:
        raise ValueError(f"unsupported delivery manifest schema: {payload.get('schema')!r}")
    if payload.get("artifact_role") != expected_role:
        raise ValueError(f"delivery manifest role mismatch: expected {expected_role}")
    if payload.get("tasks") != expected_tasks:
        raise ValueError("delivery manifest task count differs from the sealed mapping")
    for field in (
        "contains_frozen_sample_mapping",
        "contains_other_annotator_task",
        "weak_or_model_labels_in_pack",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"delivery manifest violates blind annotation contract: {field}")
    binding = payload.get("admin_binding")
    if not isinstance(binding, dict) or binding.get("sealed_mapping_sha256") != expected_mapping_sha256:
        raise ValueError("delivery manifest is not bound to the supplied sealed review mapping")
    hash_field = "task_template_sha256" if payload.get("task_template_sha256") else "task_sha256"
    template_hash_verified = False
    if template_kind == "public_delivery":
        if payload.get(hash_field) != sha256_file(template_paths[0]):
            raise ValueError(f"delivery manifest {hash_field} does not bind the supplied template")
        template_hash_verified = True
    elif payload.get("task_template_sha256"):
        raise ValueError("v2 delivery manifest requires its public task.template.csv")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema": payload["schema"],
        "artifact_role": payload["artifact_role"],
        "admin_binding": binding,
        "template_hash_field": hash_field,
        "template_hash_verified": template_hash_verified,
    }


def _canonical_rows(
    mapping_rows: Sequence[Mapping[str, str]],
    submission_by_id: Mapping[str, Mapping[str, str]],
    *,
    dual_only: bool,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for mapping in mapping_rows:
        if dual_only and not _bool_value(
            mapping["dual_annotation"],
            field="dual_annotation",
            context="review mapping",
        ):
            continue
        public = submission_by_id[str(mapping["review_id"])]
        output.append(
            {
                **{field: str(mapping[field]) for field in FROZEN_FIELDS},
                **{field: str(public[field]).strip() for field in EDITABLE_FIELDS},
            }
        )
    return output


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CANONICAL_FIELDS} for row in rows)


def import_submissions(
    *,
    review_mapping: Path,
    template_a: Sequence[Path],
    submission_a: Path,
    output_dir: Path,
    submission_b: Path | None = None,
    template_b: Sequence[Path] = (),
    delivery_manifest_a: Path | None = None,
    delivery_manifest_b: Path | None = None,
) -> dict:
    mapping_rows = load_mapping(review_mapping)
    mapping_sha256 = sha256_file(review_mapping)
    mapping_by_review = {str(row["review_id"]): row for row in mapping_rows}
    dual_mapping = {
        review_id: row
        for review_id, row in mapping_by_review.items()
        if _bool_value(row["dual_annotation"], field="dual_annotation", context=str(review_mapping))
    }
    template_kind_a = validate_templates(
        template_a,
        expected_mapping=mapping_by_review,
        mapping_rows=mapping_rows,
    )
    submission_rows_a, annotator_a = _validate_public_rows(
        submission_a,
        expected_mapping=mapping_by_review,
        require_complete=True,
    )
    if annotator_a is None:  # guarded by _validate_public_rows; keeps the type explicit
        raise AssertionError("annotator A ID validation failed")

    if template_b and submission_b is None:
        raise ValueError("--template-b requires --submission-b")
    if delivery_manifest_b is not None and submission_b is None:
        raise ValueError("--delivery-manifest-b requires an annotator-B submission")
    submission_rows_b: dict[str, Mapping[str, str]] | None = None
    annotator_b: str | None = None
    template_kind_b: str | None = None
    if submission_b is not None:
        if not dual_mapping:
            raise ValueError("annotator B was supplied but the mapping has no dual rows")
        dual_rows = list(dual_mapping.values())
        if template_b:
            template_kind_b = validate_templates(
                template_b,
                expected_mapping=dual_mapping,
                mapping_rows=dual_rows,
            )
        submission_rows_b, annotator_b = _validate_public_rows(
            submission_b,
            expected_mapping=dual_mapping,
            require_complete=True,
        )
        if annotator_b is None:
            raise AssertionError("annotator B ID validation failed")
        if annotator_a.casefold() == annotator_b.casefold():
            raise ValueError("annotator A and B must use different annotator IDs")

    manifest_records: dict[str, dict] = {}
    if delivery_manifest_a is not None:
        manifest_records["annotator_a"] = validate_delivery_manifest(
            delivery_manifest_a,
            expected_role="annotator_a_all_selected",
            expected_tasks=len(mapping_rows),
            template_paths=template_a,
            template_kind=template_kind_a,
            expected_mapping_sha256=mapping_sha256,
        )
    if delivery_manifest_b is not None:
        if not template_b:
            manifest_payload = json.loads(delivery_manifest_b.read_text(encoding="utf-8"))
            inferred_template = delivery_manifest_b.parent / "task.template.csv"
            if manifest_payload.get("task_template_sha256") and inferred_template.is_file():
                template_b = (inferred_template,)
                template_kind_b = validate_templates(
                    template_b,
                    expected_mapping=dual_mapping,
                    mapping_rows=list(dual_mapping.values()),
                )
        manifest_records["annotator_b"] = validate_delivery_manifest(
            delivery_manifest_b,
            expected_role="annotator_b_dual_only",
            expected_tasks=len(dual_mapping),
            template_paths=template_b,
            template_kind=str(template_kind_b),
            expected_mapping_sha256=mapping_sha256,
        )
    if len(manifest_records) == 2:
        binding_a = manifest_records["annotator_a"].get("admin_binding")
        binding_b = manifest_records["annotator_b"].get("admin_binding")
        if not binding_a or binding_a != binding_b:
            raise ValueError("annotator A and B delivery manifests have different admin bindings")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    locked_import = any(str(row["gold_split"]) == "locked_test" for row in mapping_rows)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    published = False
    try:
        staging.mkdir(mode=0o700 if locked_import and os.name == "posix" else 0o755)
        if locked_import and os.name == "posix":
            staging.chmod(0o700)
        canonical_a = _canonical_rows(mapping_rows, submission_rows_a, dual_only=False)
        path_a = staging / "annotations_a_canonical.csv"
        _write_csv(path_a, canonical_a)
        output_files = {
            "annotations_a_canonical.csv": {
                "rows": len(canonical_a),
                "sha256": sha256_file(path_a),
            }
        }
        for split in sorted({row["gold_split"] for row in canonical_a}):
            split_rows_a = [row for row in canonical_a if row["gold_split"] == split]
            split_path_a = staging / f"annotations_a_{split}.csv"
            _write_csv(split_path_a, split_rows_a)
            output_files[split_path_a.name] = {
                "rows": len(split_rows_a),
                "sha256": sha256_file(split_path_a),
            }
        if submission_rows_b is not None:
            canonical_b = _canonical_rows(mapping_rows, submission_rows_b, dual_only=True)
            path_b = staging / "annotations_b_dual_canonical.csv"
            _write_csv(path_b, canonical_b)
            output_files[path_b.name] = {
                "rows": len(canonical_b),
                "sha256": sha256_file(path_b),
            }
            for split in sorted({row["gold_split"] for row in canonical_b}):
                split_rows_b = [row for row in canonical_b if row["gold_split"] == split]
                split_path_b = staging / f"annotations_b_dual_{split}.csv"
                _write_csv(split_path_b, split_rows_b)
                output_files[split_path_b.name] = {
                    "rows": len(split_rows_b),
                    "sha256": sha256_file(split_path_b),
                }
        split_counts = dict(sorted(Counter(str(row["gold_split"]) for row in mapping_rows).items()))
        result = {
            "schema": "fact-gold300-submission-import-v1",
            "passed": True,
            "locked_isolation": locked_import,
            "output_security": {
                "locked_private_directory_required": locked_import,
                "posix_mode": "0700" if locked_import and os.name == "posix" else None,
                "platform_enforced": locked_import and os.name == "posix",
            },
            "split_counts": split_counts,
            "samples_a": len(mapping_rows),
            "dual_samples": len(dual_mapping),
            "annotator_a_id": annotator_a,
            "annotator_b_id": annotator_b,
            "input_files": {
                "review_mapping": {
                    "path": str(review_mapping.resolve()),
                    "sha256": sha256_file(review_mapping),
                },
                "template_a": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in template_a
                ],
                "submission_a": {
                    "path": str(submission_a.resolve()),
                    "sha256": sha256_file(submission_a),
                },
                "template_b": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in template_b
                ],
                "submission_b": (
                    {"path": str(submission_b.resolve()), "sha256": sha256_file(submission_b)}
                    if submission_b is not None
                    else None
                ),
            },
            "template_kinds": {"annotator_a": template_kind_a, "annotator_b": template_kind_b},
            "delivery_manifests": manifest_records,
            "canonical_fields": list(CANONICAL_FIELDS),
            "output_files": output_files,
        }
        manifest_path = staging / "result_manifest.json"
        manifest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = sha256_file(manifest_path)
        (staging / "result_manifest.sha256").write_text(
            f"{manifest_sha256}  result_manifest.json\n",
            encoding="ascii",
        )
        staging.replace(output_dir)
        if locked_import and os.name == "posix" and stat.S_IMODE(output_dir.stat().st_mode) != 0o700:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise PermissionError("published locked annotation output is not private mode 0700")
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return {
        **result,
        "output_dir": str(output_dir),
        "result_manifest_sha256": manifest_sha256,
    }


def main() -> None:
    args = parse_args()
    result = import_submissions(
        review_mapping=args.review_mapping,
        template_a=args.template_a,
        submission_a=args.submission_a,
        submission_b=args.submission_b,
        template_b=args.template_b,
        delivery_manifest_a=args.delivery_manifest_a,
        delivery_manifest_b=args.delivery_manifest_b,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
