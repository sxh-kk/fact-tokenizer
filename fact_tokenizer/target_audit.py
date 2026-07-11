"""Strict evidence-chain validation for FACT v7 target visual audits.

The visual gate is not a self-attested score file.  A v2 gate is valid only
while the reviewed CSV, frozen audit pack, rendered PNGs, target cache,
transition inputs, capability manifest, builder provenance, and deterministic
selection contract still match the hashes that were reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .effect_manifest import read_manifest_jsonl
from .target_audit_selection import validate_audit_selection_contract


AUDIT_PACK_SCHEMA = "fact-target-visual-audit-pack-v2"
AUDIT_GATE_SCHEMA = "fact-target-visual-audit-gate-v2"
SELECTION_SCHEMA = "fact-target-audit-selection-v1"
REVIEW_COLUMNS = (
    "review_id",
    "sample_id",
    "image",
    "coverage_json",
    "time_direction_ok",
    "coordinate_orientation_ok",
    "mask_flow_alignment_ok",
    "notes",
)
REVIEW_DECISION_COLUMNS = (
    "time_direction_ok",
    "coordinate_orientation_ok",
    "mask_flow_alignment_ok",
)
IMMUTABLE_REVIEW_COLUMNS = ("review_id", "sample_id", "image", "coverage_json")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    size_bytes: int
    data: bytes | None = None

    def reference(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def canonical_json_sha256(value: Any) -> str:
    # Match EffectTargetCacheWriter's identity serialization exactly.
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_file(path: Path | str, *, keep_bytes: bool = False) -> FileSnapshot:
    """Read a file once and return the exact bytes/hash observed.

    Large NPY files are streamed.  ``fstat`` is checked before and after the
    read so an in-place size/mtime change cannot silently cross the snapshot.
    """

    requested = Path(path)
    if not requested.is_file():
        raise ValueError(f"audit evidence file is missing: {requested}")
    resolved = requested.resolve(strict=True)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if keep_bytes else None
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"audit evidence changed while it was read: {resolved}")
    return FileSnapshot(
        path=resolved,
        sha256=digest.hexdigest(),
        size_bytes=int(after.st_size),
        data=(b"".join(chunks) if chunks is not None else None),
    )


def require_sha256(value: object, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _snapshot_reference(reference: Mapping[str, Any], label: str, *, keep_bytes: bool = False) -> FileSnapshot:
    if set(reference) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} must contain exactly path, sha256, and size_bytes")
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    expected_hash = require_sha256(reference["sha256"], f"{label}.sha256")
    snapshot = snapshot_file(path, keep_bytes=keep_bytes)
    if snapshot.sha256 != expected_hash or snapshot.size_bytes != int(reference["size_bytes"]):
        raise ValueError(f"{label} no longer matches its frozen hash/size")
    return snapshot


def _json_from_snapshot(snapshot: FileSnapshot, label: str) -> Any:
    if snapshot.data is None:
        raise AssertionError("JSON snapshot must retain bytes")
    try:
        return json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _jsonl_from_snapshot(snapshot: FileSnapshot, label: str) -> list[dict[str, Any]]:
    if snapshot.data is None:
        raise AssertionError("JSONL snapshot must retain bytes")
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} has invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(value)
    return rows


def immutable_review_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {column: str(row.get(column, "")) for column in IMMUTABLE_REVIEW_COLUMNS}
        for row in rows
    ]


def immutable_review_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(immutable_review_rows(rows))


def immutable_review_row_hashes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    result = []
    for row in immutable_review_rows(rows):
        result.append(
            {
                "review_id": row["review_id"],
                "sample_id": row["sample_id"],
                "sha256": canonical_json_sha256(row),
            }
        )
    return result


def _parse_review_csv(snapshot: FileSnapshot) -> list[dict[str, str]]:
    if snapshot.data is None:
        raise AssertionError("review CSV snapshot must retain bytes")
    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("review CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
        raise ValueError(f"review CSV columns must be exactly {list(REVIEW_COLUMNS)}")
    rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("review CSV contains fields outside the frozen column contract")
    return [{key: str(value) for key, value in row.items()} for row in rows]


def _validate_target_config(config: Mapping[str, Any]) -> tuple[str, str]:
    config_payload = config.get("config")
    identity_payload = config.get("identity")
    if not isinstance(config_payload, Mapping) or not isinstance(identity_payload, Mapping):
        raise ValueError("target_config.json lacks config/identity objects")
    config_sha = canonical_json_sha256(config_payload)
    if config.get("config_sha256") != config_sha:
        raise ValueError("target_config.json config_sha256 is not derived from config")
    identity_sha = canonical_json_sha256(
        {"config_sha256": config_sha, "sources": identity_payload}
    )
    if config.get("identity_sha256") != identity_sha:
        raise ValueError("target_config.json identity_sha256 is not derived from config/identity")
    return config_sha, identity_sha


def _require_unique_ids(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"{label} has missing/duplicate sample_id at record {line_number}")
        result[sample_id] = row
    return result


def _validate_target_files(
    target_manifest: Sequence[Mapping[str, Any]],
    target_root: Path,
    selected_ids: Sequence[str],
    source_records: Mapping[str, Mapping[str, Any]],
    config_sha256: str,
    identity_sha256: str,
) -> None:
    records = _require_unique_ids(target_manifest, "target manifest")
    if set(records) != set(selected_ids) or len(records) != len(selected_ids):
        raise ValueError("audit target cache must contain exactly the frozen selected sample IDs")
    sample_root = (target_root / "samples").resolve()
    for sample_id, record in records.items():
        if (
            record.get("config_sha256") != config_sha256
            or record.get("identity_sha256") != identity_sha256
        ):
            raise ValueError(f"target record identity differs for {sample_id}")
        source = source_records[sample_id]
        metadata = record.get("metadata")
        expected_metadata = {
            key: source.get(key)
            for key in ("sample_key", "take_uid", "split", "row_index", "timestamp")
        }
        if not isinstance(metadata, Mapping) or {
            key: metadata.get(key) for key in expected_metadata
        } != expected_metadata:
            raise ValueError(f"target record metadata differs from source manifest for {sample_id}")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute():
            raise ValueError(f"target record path must be relative for {sample_id}")
        target_path = (target_root / relative).resolve()
        try:
            target_path.relative_to(sample_root)
        except ValueError as exc:
            raise ValueError(f"target record escapes samples/ for {sample_id}") from exc
        expected = require_sha256(record.get("sha256"), f"target record {sample_id} sha256")
        target_snapshot = snapshot_file(target_path, keep_bytes=True)
        if target_snapshot.sha256 != expected:
            raise ValueError(f"target NPZ hash mismatch for {sample_id}")
        assert target_snapshot.data is not None
        try:
            with np.load(io.BytesIO(target_snapshot.data), allow_pickle=False) as payload:
                keys = sorted(payload.files)
                if keys != sorted(str(value) for value in record.get("target_keys", [])):
                    raise ValueError(f"target NPZ keys differ from manifest for {sample_id}")
                for validity in ("depth_valid", "flow_3d_valid"):
                    if validity not in payload.files or bool(np.asarray(payload[validity]).item()):
                        raise ValueError(f"target NPZ violates {validity}=false for {sample_id}")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"target file is not a safe NPZ for {sample_id}") from exc


def _validate_pack_evidence(pack: Mapping[str, Any], review_dir: Path) -> dict[str, FileSnapshot]:
    if pack.get("schema") != AUDIT_PACK_SCHEMA:
        raise ValueError(f"audit pack must use {AUDIT_PACK_SCHEMA}")
    if int(pack.get("sample_count", -1)) != 50:
        raise ValueError("formal target audit pack requires exactly 50 samples")
    selected_ids = [str(value) for value in pack.get("selected_sample_ids", [])]
    if len(selected_ids) != 50 or len(set(selected_ids)) != 50:
        raise ValueError("audit pack selected_sample_ids must contain 50 unique IDs")
    artifacts = pack.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("audit pack lacks artifacts")

    snapshots: dict[str, FileSnapshot] = {}
    for name in (
        "source_manifest",
        "target_config",
        "target_manifest",
        "builder_provenance",
        "source_contract",
        "selection_contract",
        "selection_index",
    ):
        reference = artifacts.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"audit pack lacks artifacts.{name}")
        snapshots[name] = _snapshot_reference(reference, f"artifacts.{name}", keep_bytes=True)

    config = _json_from_snapshot(snapshots["target_config"], "target_config.json")
    if not isinstance(config, Mapping):
        raise ValueError("target_config.json must contain an object")
    config_sha, identity_sha = _validate_target_config(config)
    if pack.get("target_config_sha256") != config_sha:
        raise ValueError("audit pack target_config_sha256 differs from target config")
    if pack.get("target_identity_sha256") != identity_sha:
        raise ValueError("audit pack target_identity_sha256 differs from target config")

    identity = config["identity"]
    source_hashes = identity.get("source_hashes")
    input_contract = identity.get("input_transition_contract")
    if not isinstance(source_hashes, Mapping) or not isinstance(input_contract, Mapping):
        raise ValueError("target identity lacks formal source hashes/input transition contract")
    if source_hashes.get("manifest") != snapshots["source_manifest"].sha256:
        raise ValueError("source manifest hash differs from target identity")
    contract_files = input_contract.get("files")
    identity_views = source_hashes.get("views")
    if not isinstance(contract_files, Mapping) or not isinstance(identity_views, Mapping) or not identity_views:
        raise ValueError("target identity lacks input files/views")
    if (
        Path(str(input_contract.get("source_contract", ""))).resolve()
        != snapshots["source_contract"].path
        or input_contract.get("source_contract_sha256") != snapshots["source_contract"].sha256
        or input_contract.get("source_contract_schema") != "fact-npy-source-contract-v1"
    ):
        raise ValueError("source contract artifact differs from the target input identity")

    input_artifacts = artifacts.get("input")
    if not isinstance(input_artifacts, Mapping):
        raise ValueError("audit pack lacks artifacts.input")
    sample_ref = input_artifacts.get("sample_id")
    view_refs = input_artifacts.get("views")
    if not isinstance(sample_ref, Mapping) or not isinstance(view_refs, Mapping):
        raise ValueError("audit pack input evidence lacks sample_id/views")
    if set(view_refs) != set(identity_views):
        raise ValueError("audit pack must bind every and only target-identity view")
    snapshots["input.sample_id"] = _snapshot_reference(
        sample_ref, "artifacts.input.sample_id"
    )
    sample_contract = contract_files.get("sample_id")
    if not isinstance(sample_contract, Mapping) or sample_contract.get("sha256") != snapshots["input.sample_id"].sha256:
        raise ValueError("input sample_id.npy hash differs from transition contract")
    for view in sorted(identity_views):
        snapshots[f"input.views.{view}"] = _snapshot_reference(
            view_refs[view], f"artifacts.input.views.{view}"
        )
        actual = snapshots[f"input.views.{view}"].sha256
        contract_view = contract_files.get(view)
        if identity_views.get(view) != actual:
            raise ValueError(f"input {view}.npy hash differs from target identity")
        if not isinstance(contract_view, Mapping) or contract_view.get("sha256") != actual:
            raise ValueError(f"input {view}.npy hash differs from transition contract")
    source_contract = _json_from_snapshot(snapshots["source_contract"], "source contract")
    if not isinstance(source_contract, Mapping):
        raise ValueError("source contract must contain an object")
    source_contract_files = source_contract.get("files")
    input_root = snapshots["input.sample_id"].path.parent
    if (
        source_contract.get("schema") != "fact-npy-source-contract-v1"
        or Path(str(source_contract.get("source_dir", ""))).resolve() != input_root
        or not isinstance(source_contract_files, Mapping)
    ):
        raise ValueError("source contract content does not describe the frozen input directory")
    for name in ("sample_id", *sorted(identity_views)):
        source_entry = source_contract_files.get(name)
        snapshot = snapshots[f"input.views.{name}"] if name in identity_views else snapshots["input.sample_id"]
        if (
            snapshot.path.parent != input_root
            or not isinstance(source_entry, Mapping)
            or source_entry.get("sha256") != snapshot.sha256
        ):
            raise ValueError(f"source contract content differs for {name}.npy")

    provenance = _json_from_snapshot(snapshots["builder_provenance"], "builder_provenance.json")
    if not isinstance(provenance, Mapping):
        raise ValueError("builder_provenance.json must contain an object")
    provenance_identity = {
        key: value
        for key, value in provenance.items()
        if key not in {"selection_contract", "target_identity_sha256", "formal_release"}
    }
    if provenance_identity != identity or provenance.get("target_identity_sha256") != identity_sha:
        raise ValueError("builder provenance does not reproduce the target identity")

    selection_provenance = provenance.get("selection_contract")
    if not isinstance(selection_provenance, Mapping):
        raise ValueError("builder provenance lacks frozen selection provenance")
    if selection_provenance.get("selection_contract_sha256") != snapshots["selection_contract"].sha256:
        raise ValueError("selection contract hash differs from builder provenance")
    if selection_provenance.get("sample_index_sha256") != snapshots["selection_index"].sha256:
        raise ValueError("selection index hash differs from builder provenance")

    selection = _json_from_snapshot(snapshots["selection_contract"], "audit_selection.json")
    if not isinstance(selection, Mapping) or selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("selection contract has the wrong schema")
    if (
        int(selection.get("sample_count", -1)) != 50
        or int(selection.get("seed", -1)) != int(pack.get("seed", -2))
        or [str(value) for value in selection.get("sample_ids", [])] != selected_ids
        or selection.get("strata_counts") != pack.get("strata_counts")
        or selection.get("manifest_sha256") != snapshots["source_manifest"].sha256
        or selection.get("index_sha256") != snapshots["selection_index"].sha256
    ):
        raise ValueError("audit pack differs from the deterministic selection contract")

    # Do not trust a syntactically plausible contract: replay the one canonical
    # selector against the frozen manifest.  The returned hashes must also
    # match the byte snapshots already taken above, closing the read window.
    typed_source_records = read_manifest_jsonl(snapshots["source_manifest"].path)
    canonical_selection = validate_audit_selection_contract(
        typed_source_records,
        manifest_path=snapshots["source_manifest"].path,
        index_path=snapshots["selection_index"].path,
        contract_path=snapshots["selection_contract"].path,
    )
    if (
        canonical_selection.get("manifest_sha256") != snapshots["source_manifest"].sha256
        or canonical_selection.get("sample_index_sha256") != snapshots["selection_index"].sha256
        or canonical_selection.get("selection_contract_sha256") != snapshots["selection_contract"].sha256
        or canonical_selection.get("sample_ids") != selected_ids
        or canonical_selection.get("strata_counts") != pack.get("strata_counts")
    ):
        raise ValueError("canonical audit selection differs from the frozen evidence snapshots")

    source_rows = _jsonl_from_snapshot(snapshots["source_manifest"], "source manifest")
    source_by_id = _require_unique_ids(source_rows, "source manifest")
    if any(sample_id not in source_by_id for sample_id in selected_ids):
        raise ValueError("selection contains sample IDs absent from the source manifest")
    if any(source_by_id[sample_id].get("training_valid") is not True for sample_id in selected_ids):
        raise ValueError("visual audit selection contains a training-invalid sample")
    index_data = snapshots["selection_index"].data
    assert index_data is not None
    try:
        selected_indices = np.load(io.BytesIO(index_data), allow_pickle=False)
    except Exception as exc:
        raise ValueError("selection index is not a safe NPY") from exc
    if (
        selected_indices.ndim != 1
        or len(selected_indices) != 50
        or not np.issubdtype(selected_indices.dtype, np.integer)
        or len(set(int(value) for value in selected_indices)) != 50
        or (selected_indices < 0).any()
        or (selected_indices >= len(source_rows)).any()
    ):
        raise ValueError("selection index must contain 50 unique in-range integer rows")
    indexed_ids = [str(source_rows[int(index)].get("sample_id", "")) for index in selected_indices]
    if indexed_ids != selected_ids:
        raise ValueError("selection index rows do not reproduce selected_sample_ids")

    target_rows = _jsonl_from_snapshot(snapshots["target_manifest"], "target manifest")
    if int(config.get("samples", -1)) != len(target_rows):
        raise ValueError("target_config sample count differs from target manifest")
    if snapshots["target_config"].path.parent != snapshots["target_manifest"].path.parent:
        raise ValueError("target config and manifest must be in the same cache root")
    _validate_target_files(
        target_rows,
        snapshots["target_config"].path.parent,
        selected_ids,
        source_by_id,
        config_sha,
        identity_sha,
    )

    images = artifacts.get("images")
    if not isinstance(images, list) or len(images) != 50:
        raise ValueError("audit pack must bind exactly 50 rendered PNGs")
    image_root = (review_dir / "images").resolve()
    image_keys: list[tuple[str, str]] = []
    for index, entry in enumerate(images):
        if not isinstance(entry, Mapping) or set(entry) != {
            "review_id", "sample_id", "relative_path", "path", "sha256", "size_bytes"
        }:
            raise ValueError(f"image evidence {index} has an invalid schema")
        image_snapshot = _snapshot_reference(
            {key: entry[key] for key in ("path", "sha256", "size_bytes")},
            f"artifacts.images[{index}]",
            keep_bytes=True,
        )
        try:
            image_snapshot.path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError("audit PNG escapes the review images directory") from exc
        if not image_snapshot.data or not image_snapshot.data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("audit image is not a PNG")
        try:
            from PIL import Image

            with Image.open(io.BytesIO(image_snapshot.data)) as image:
                image.verify()
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("audit image has invalid dimensions")
        except Exception as exc:
            raise ValueError("audit image is not a decodable PNG") from exc
        expected_relative = image_snapshot.path.relative_to(review_dir.resolve()).as_posix()
        if entry["relative_path"] != expected_relative:
            raise ValueError("audit image relative_path is not canonical")
        image_keys.append((str(entry["review_id"]), str(entry["sample_id"])))
        snapshots[f"image.{index}"] = image_snapshot
    expected_image_keys = [(f"{index:02d}", sample_id) for index, sample_id in enumerate(selected_ids, start=1)]
    if image_keys != expected_image_keys:
        raise ValueError("audit image ordering/identity differs from selected_sample_ids")
    return snapshots


def validate_review_artifacts(review_path: Path | str, minimum_pass_fraction: float = 0.90) -> dict[str, Any]:
    """Validate a reviewed CSV and every frozen artifact it refers to."""

    minimum_pass_fraction = float(minimum_pass_fraction)
    if not math.isfinite(minimum_pass_fraction) or not 0.90 <= minimum_pass_fraction <= 1.0:
        raise ValueError(
            "formal target audit minimum pass fraction cannot be below 0.90, above 1.0, or non-finite"
        )
    review = snapshot_file(review_path, keep_bytes=True)
    pack_path = review.path.parent / "audit_pack.json"
    pack_snapshot = snapshot_file(pack_path, keep_bytes=True)
    pack = _json_from_snapshot(pack_snapshot, "audit_pack.json")
    if not isinstance(pack, Mapping):
        raise ValueError("audit_pack.json must contain an object")
    snapshots = _validate_pack_evidence(pack, review.path.parent)
    rows = _parse_review_csv(review)
    if len(rows) != 50:
        raise ValueError(f"formal target audit requires exactly 50 rows, found {len(rows)}")
    selected_ids = [str(value) for value in pack["selected_sample_ids"]]
    if [row["sample_id"] for row in rows] != selected_ids:
        raise ValueError("review rows differ from the frozen selected sample IDs")
    review_contract = pack.get("review_contract")
    if not isinstance(review_contract, Mapping):
        raise ValueError("audit pack lacks review_contract")
    if (
        review_contract.get("columns") != list(REVIEW_COLUMNS)
        or review_contract.get("immutable_columns") != list(IMMUTABLE_REVIEW_COLUMNS)
        or review_contract.get("immutable_rows_sha256") != immutable_review_rows_sha256(rows)
        or review_contract.get("immutable_row_hashes") != immutable_review_row_hashes(rows)
    ):
        raise ValueError("review CSV immutable fields differ from the frozen audit pack")
    image_entries = pack["artifacts"]["images"]
    for row, image_entry in zip(rows, image_entries):
        if row["review_id"] != image_entry["review_id"] or row["image"] != image_entry["relative_path"]:
            raise ValueError("review CSV image binding differs from the frozen audit pack")
        try:
            coverage = json.loads(row["coverage_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("review coverage_json is invalid") from exc
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("review coverage_json must contain per-view coverage")

    invalid_rows: list[int] = []
    passed = 0
    for row_number, row in enumerate(rows, start=2):
        values = [row[column].strip().lower() for column in REVIEW_DECISION_COLUMNS]
        if any(value not in {"yes", "no"} for value in values):
            invalid_rows.append(row_number)
        elif all(value == "yes" for value in values):
            passed += 1
    if invalid_rows:
        raise ValueError(f"audit rows need yes/no in all review fields: {invalid_rows[:5]}")
    fraction = passed / len(rows)
    if fraction < minimum_pass_fraction:
        raise RuntimeError(f"target audit gate failed: {fraction:.3f} < {minimum_pass_fraction:.3f}")
    image_refs = pack["artifacts"]["images"]
    report = {
        "schema": AUDIT_GATE_SCHEMA,
        "rows": len(rows),
        "fully_aligned": passed,
        "pass_fraction": fraction,
        "minimum_pass_fraction": minimum_pass_fraction,
        "passed": True,
        "target_identity_sha256": pack["target_identity_sha256"],
        "target_config_sha256": pack["target_config_sha256"],
        "strata_counts": pack["strata_counts"],
        "review_csv_sha256": review.sha256,
        "audit_pack_sha256": pack_snapshot.sha256,
        "evidence": {
            "review_csv": review.reference(),
            "audit_pack": pack_snapshot.reference(),
            "source_manifest": snapshots["source_manifest"].reference(),
            "target_config": snapshots["target_config"].reference(),
            "target_manifest": snapshots["target_manifest"].reference(),
            "builder_provenance": snapshots["builder_provenance"].reference(),
            "source_contract": snapshots["source_contract"].reference(),
            "selection_contract": snapshots["selection_contract"].reference(),
            "selection_index": snapshots["selection_index"].reference(),
            "input": {
                "sample_id": snapshots["input.sample_id"].reference(),
                "views": {
                    name.removeprefix("input.views."): value.reference()
                    for name, value in snapshots.items()
                    if name.startswith("input.views.")
                },
            },
            "images_sha256": canonical_json_sha256(image_refs),
        },
    }
    return report


def validate_visual_audit_gate(
    gate_path: Path | str,
    expected_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a v2 gate and revalidate the complete evidence chain."""

    gate_snapshot = snapshot_file(gate_path, keep_bytes=True)
    gate = _json_from_snapshot(gate_snapshot, "target_visual_audit_gate.json")
    if not isinstance(gate, Mapping) or gate.get("schema") != AUDIT_GATE_SCHEMA:
        raise ValueError(f"visual audit gate must use {AUDIT_GATE_SCHEMA}")
    evidence = gate.get("evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("review_csv"), Mapping):
        raise ValueError("visual audit gate lacks reviewed evidence")
    review_path = Path(str(evidence["review_csv"].get("path", "")))
    recomputed = validate_review_artifacts(
        review_path,
        float(gate.get("minimum_pass_fraction", 0.0)),
    )
    if dict(gate) != recomputed:
        raise ValueError("visual audit gate does not exactly reproduce its live evidence chain")
    if expected_identity_sha256 is not None and gate.get("target_identity_sha256") != expected_identity_sha256:
        raise ValueError("visual audit gate was produced from a different target identity")
    validated = dict(gate)
    # These values come from the one gate snapshot read above.  They are
    # intentionally not serialized inside the gate (which would be recursive)
    # and let consumers avoid reopening the gate after validation.
    validated["_gate_sha256"] = gate_snapshot.sha256
    validated["_gate_path"] = str(gate_snapshot.path)
    return validated
