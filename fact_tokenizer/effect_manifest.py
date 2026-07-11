"""Versioned manifests for v7 continuous-effect training data.

The manifest is deliberately metadata-only: video tensors stay in the existing
``.npy`` stores and are opened with ``mmap_mode="r"``.  Each record carries an
explicit validity bit for every optional target so missing supervision can
never be confused with a numeric zero target.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EFFECT_MANIFEST_SCHEMA_VERSION = "fact-effect-manifest-v1"


class EffectCapability(str, Enum):
    """Optional evidence or supervision available for one transition."""

    RGB_PAIRED = "rgb_paired"
    CAMERA_POSE = "camera_pose"
    FLOW_2D = "flow_2d"
    OBJECT_MASK = "object_mask"
    BODY_POSE = "body_pose"
    HAND_POSE = "hand_pose"
    ATOMIC_TEXT = "atomic_text"
    PHASE = "phase"
    TAKE_QUALITY = "take_quality"
    DEPTH = "depth"
    FLOW_3D = "flow_3d"


def _capability(value: EffectCapability | str) -> EffectCapability:
    if isinstance(value, EffectCapability):
        return value
    try:
        return EffectCapability(str(value))
    except ValueError as exc:
        known = ", ".join(capability.value for capability in EffectCapability)
        raise ValueError(f"Unknown effect capability {value!r}; expected one of: {known}") from exc


def normalize_capability_validity(
    values: Mapping[EffectCapability | str, bool] | Iterable[EffectCapability | str] | None,
) -> dict[EffectCapability, bool]:
    """Return a complete, typed capability validity table.

    A mapping supplies explicit booleans.  An iterable is shorthand for the set
    of valid capabilities.  Omitted capabilities are always represented as
    ``False`` in serialized records.
    """

    normalized = {capability: False for capability in EffectCapability}
    if values is None:
        return normalized
    if isinstance(values, Mapping):
        for key, valid in values.items():
            if not isinstance(valid, (bool, np.bool_)):
                raise TypeError(f"Capability validity for {key!r} must be boolean, got {type(valid).__name__}")
            normalized[_capability(key)] = bool(valid)
    else:
        for value in values:
            normalized[_capability(value)] = True
    return normalized


def validate_capability_validity(values: Mapping[EffectCapability | str, bool]) -> None:
    """Validate cross-capability invariants.

    This v7 campaign has neither depth nor dense 3D-flow assets.  Both bits are
    therefore fail-fast invariants, rather than merely optional capabilities.
    Camera/body/hand poses remain useful sparse signals but do not relax them.
    """

    normalized = normalize_capability_validity(values)
    if normalized[EffectCapability.DEPTH] or normalized[EffectCapability.FLOW_3D]:
        raise ValueError(
            "FACT v7 requires depth_valid=false and flow_3d_valid=false for every sample"
        )


@dataclass(frozen=True)
class DiagnosticCodelabel:
    """A quarantined label from a diagnostic or failed code-label experiment."""

    value: str
    source: str = "legacy_failed_codelabel"
    training_valid: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.value):
            raise ValueError("Diagnostic codelabel value must be non-empty")
        if not str(self.source):
            raise ValueError("Diagnostic codelabel source must be non-empty")
        if self.training_valid is not False:
            raise ValueError("Diagnostic codelabels are quarantined and must have training_valid=False")
        object.__setattr__(self, "value", str(self.value))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "training_valid": False,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiagnosticCodelabel":
        return cls(
            value=str(payload["value"]),
            source=str(payload.get("source", "legacy_failed_codelabel")),
            training_valid=payload.get("training_valid", False),
            metadata=dict(payload.get("metadata", {})),
        )


def make_effect_sample_key(source_dataset: str, split: str, sample_id: str) -> str:
    """Build the stable primary key used by manifests and annotation joins."""

    parts = (str(source_dataset).strip(), str(split).strip(), str(sample_id).strip())
    if any(not part for part in parts):
        raise ValueError("source_dataset, split, and sample_id must be non-empty")
    return "/".join(parts)


@dataclass(frozen=True)
class EffectSampleRecord:
    """One aligned paired transition and its available effect supervision."""

    sample_id: str
    take_uid: str
    split: str
    row_index: int
    source_dataset: str = "egoexo"
    timestamp: float | None = None
    capability_validity: Mapping[EffectCapability | str, bool] = field(default_factory=dict)
    training_valid: bool = True
    quality_bucket: str | None = None
    quality_weight: float = 1.0
    annotation_refs: Mapping[str, Any] = field(default_factory=dict)
    content_hashes: Mapping[str, str] = field(default_factory=dict)
    diagnostic_codelabel: DiagnosticCodelabel | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        take_uid = str(self.take_uid).strip()
        split = str(self.split).strip()
        source_dataset = str(self.source_dataset).strip()
        if not sample_id or not take_uid or not split or not source_dataset:
            raise ValueError("sample_id, take_uid, split, and source_dataset must be non-empty")
        if int(self.row_index) < 0:
            raise ValueError("row_index must be non-negative")
        timestamp = None if self.timestamp is None else float(self.timestamp)
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite when provided")
        quality_weight = float(self.quality_weight)
        if not math.isfinite(quality_weight) or quality_weight < 0.0:
            raise ValueError("quality_weight must be finite and non-negative")
        capabilities = normalize_capability_validity(self.capability_validity)
        validate_capability_validity(capabilities)
        diagnostic = self.diagnostic_codelabel
        if diagnostic is not None and not isinstance(diagnostic, DiagnosticCodelabel):
            diagnostic = DiagnosticCodelabel.from_dict(diagnostic)
        hashes = {str(key): str(value) for key, value in self.content_hashes.items()}
        if any(not key or not value for key, value in hashes.items()):
            raise ValueError("content_hashes keys and values must be non-empty")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value)
            for value in hashes.values()
        ):
            raise ValueError("content_hashes values must be 64-character SHA256 digests")
        hashes = {key: value.lower() for key, value in hashes.items()}

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "take_uid", take_uid)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_dataset", source_dataset)
        object.__setattr__(self, "row_index", int(self.row_index))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "capability_validity", capabilities)
        # Attaching a legacy codelabel quarantines the whole sample from the
        # optimizer, not only the label field.
        object.__setattr__(self, "training_valid", False if diagnostic is not None else bool(self.training_valid))
        object.__setattr__(self, "quality_bucket", None if self.quality_bucket is None else str(self.quality_bucket))
        object.__setattr__(self, "quality_weight", quality_weight)
        object.__setattr__(self, "annotation_refs", dict(self.annotation_refs))
        object.__setattr__(self, "content_hashes", hashes)
        object.__setattr__(self, "diagnostic_codelabel", diagnostic)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def sample_key(self) -> str:
        return make_effect_sample_key(self.source_dataset, self.split, self.sample_id)

    @property
    def capabilities(self) -> frozenset[EffectCapability]:
        return frozenset(capability for capability, valid in self.capability_validity.items() if valid)

    def has_capability(self, capability: EffectCapability | str) -> bool:
        return bool(self.capability_validity[_capability(capability)])

    def require_capability(self, capability: EffectCapability | str) -> None:
        normalized = _capability(capability)
        if not self.has_capability(normalized):
            raise ValueError(f"Sample {self.sample_key} does not have valid {normalized.value} supervision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EFFECT_MANIFEST_SCHEMA_VERSION,
            "sample_key": self.sample_key,
            "sample_id": self.sample_id,
            "take_uid": self.take_uid,
            "split": self.split,
            "row_index": self.row_index,
            "source_dataset": self.source_dataset,
            "timestamp": self.timestamp,
            "capability_validity": {
                capability.value: bool(self.capability_validity[capability]) for capability in EffectCapability
            },
            "training_valid": self.training_valid,
            "quality_bucket": self.quality_bucket,
            "quality_weight": self.quality_weight,
            "annotation_refs": dict(self.annotation_refs),
            "content_hashes": dict(self.content_hashes),
            "diagnostic_codelabel": (
                None if self.diagnostic_codelabel is None else self.diagnostic_codelabel.to_dict()
            ),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectSampleRecord":
        schema_version = payload.get("schema_version", EFFECT_MANIFEST_SCHEMA_VERSION)
        if schema_version != EFFECT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported effect manifest schema {schema_version!r}; expected {EFFECT_MANIFEST_SCHEMA_VERSION!r}"
            )
        diagnostic_payload = payload.get("diagnostic_codelabel")
        record = cls(
            sample_id=str(payload["sample_id"]),
            take_uid=str(payload["take_uid"]),
            split=str(payload["split"]),
            row_index=int(payload["row_index"]),
            source_dataset=str(payload.get("source_dataset", "egoexo")),
            timestamp=payload.get("timestamp"),
            capability_validity=payload.get("capability_validity", {}),
            training_valid=payload.get("training_valid", True),
            quality_bucket=payload.get("quality_bucket"),
            quality_weight=payload.get("quality_weight", 1.0),
            annotation_refs=dict(payload.get("annotation_refs", {})),
            content_hashes=dict(payload.get("content_hashes", {})),
            diagnostic_codelabel=(
                None if diagnostic_payload is None else DiagnosticCodelabel.from_dict(diagnostic_payload)
            ),
            provenance=dict(payload.get("provenance", {})),
        )
        supplied_key = payload.get("sample_key")
        if supplied_key is not None and str(supplied_key) != record.sample_key:
            raise ValueError(
                f"Manifest sample_key mismatch: supplied {supplied_key!r}, computed {record.sample_key!r}"
            )
        return record

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "EffectSampleRecord":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Effect manifest JSON record must be an object")
        return cls.from_dict(payload)


def _assert_unique_keys(records: Sequence[EffectSampleRecord]) -> None:
    counts = Counter(record.sample_key for record in records)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate effect manifest sample_key values: {preview}")


def write_manifest_jsonl(path: Path | str, records: Iterable[EffectSampleRecord]) -> Path:
    path = Path(path)
    materialized = list(records)
    _assert_unique_keys(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialized:
            handle.write(record.to_json() + "\n")
    return path


def read_manifest_jsonl(path: Path | str) -> list[EffectSampleRecord]:
    path = Path(path)
    records: list[EffectSampleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(EffectSampleRecord.from_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid effect manifest record at {path}:{line_number}: {exc}") from exc
    _assert_unique_keys(records)
    return records


def audit_effect_manifest(records: Sequence[EffectSampleRecord]) -> dict[str, Any]:
    _assert_unique_keys(records)
    capability_counts = {
        capability.value: sum(record.has_capability(capability) for record in records)
        for capability in EffectCapability
    }
    split_counts = Counter(record.split for record in records)
    take_counts = Counter(record.take_uid for record in records)
    diagnostic_count = sum(record.diagnostic_codelabel is not None for record in records)
    diagnostic_training_valid_count = sum(
        bool(record.diagnostic_codelabel and record.diagnostic_codelabel.training_valid) for record in records
    )
    timestamps = [record.timestamp for record in records if record.timestamp is not None]
    return {
        "schema_version": EFFECT_MANIFEST_SCHEMA_VERSION,
        "sample_count": len(records),
        "unique_key_count": len({record.sample_key for record in records}),
        "duplicate_key_count": 0,
        "take_count": len(take_counts),
        "split_counts": dict(sorted(split_counts.items())),
        "training_valid_count": sum(record.training_valid for record in records),
        "training_invalid_count": sum(not record.training_valid for record in records),
        "capability_valid_counts": capability_counts,
        "capability_invalid_counts": {
            capability.value: len(records) - capability_counts[capability.value]
            for capability in EffectCapability
        },
        "diagnostic_codelabel_count": diagnostic_count,
        "diagnostic_codelabel_training_valid_count": diagnostic_training_valid_count,
        "no_depth_guard": {
            "depth_valid_count": capability_counts[EffectCapability.DEPTH.value],
            "flow_3d_valid_count": capability_counts[EffectCapability.FLOW_3D.value],
            "flow_3d_without_depth_count": 0,
        },
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "samples_per_take": {
            "min": min(take_counts.values()) if take_counts else 0,
            "max": max(take_counts.values()) if take_counts else 0,
        },
    }


def _load_required_mmap(path: Path, expected_length: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if expected_length is not None and len(array) != expected_length:
        raise ValueError(f"{path} has length {len(array)}, expected {expected_length}")
    return array


def _metadata_string(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8").strip()
    return str(value).strip()


def build_records_from_npy_directory(
    input_dir: Path | str,
    *,
    split: str,
    source_dataset: str = "egoexo",
    view_keys: Sequence[str] = ("ego", "exo"),
    capability_arrays: Mapping[EffectCapability | str, Path | str] | None = None,
    diagnostic_codelabels: Mapping[str, DiagnosticCodelabel | Mapping[str, Any]] | None = None,
    content_hashes: Mapping[str, str] | None = None,
) -> tuple[list[EffectSampleRecord], dict[str, Any]]:
    """Build base records from aligned mmap-backed NPY arrays.

    ``capability_arrays`` are boolean NPY sidecars aligned by row.  The paired
    RGB capability is derived from the two validated view stores and cannot be
    overridden to false.  Optional diagnostic code labels are joined by
    ``sample_id`` and remain quarantined by :class:`DiagnosticCodelabel`.
    """

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if len(view_keys) != 2:
        raise ValueError("v7 paired effect manifests require exactly two view keys")
    if len({str(key) for key in view_keys}) != 2:
        raise ValueError("v7 paired effect manifests require two distinct view keys")
    views: dict[str, np.ndarray] = {}
    for key in view_keys:
        views[str(key)] = _load_required_mmap(input_dir / f"{key}.npy")
    for key, array in views.items():
        if array.ndim != 5:
            raise ValueError(f"View array {key!r} must be 5D (N,T,H,W,C or N,T,C,H,W), got {array.shape}")
    lengths = {len(array) for array in views.values()}
    if len(lengths) != 1:
        raise ValueError(f"Paired view arrays have different lengths: {sorted(lengths)}")
    num_rows = next(iter(lengths))

    take_uid = _load_required_mmap(input_dir / "take_uid.npy", num_rows)
    timestamp = _load_required_mmap(input_dir / "timestamp.npy", num_rows)
    sample_id = _load_required_mmap(input_dir / "sample_id.npy", num_rows)
    for name, array in (("take_uid", take_uid), ("timestamp", timestamp), ("sample_id", sample_id)):
        if array.ndim != 1:
            raise ValueError(f"Metadata array {name!r} must be one-dimensional, got {array.shape}")

    aligned_capabilities: dict[EffectCapability, np.ndarray] = {}
    for key, path in (capability_arrays or {}).items():
        capability = _capability(key)
        if capability is EffectCapability.RGB_PAIRED:
            raise ValueError("rgb_paired is derived from validated view arrays and must not be supplied as a sidecar")
        values = _load_required_mmap(Path(path), num_rows)
        if values.ndim != 1:
            raise ValueError(f"Capability sidecar {path} must be one-dimensional, got shape {values.shape}")
        if values.dtype.kind not in {"b", "i", "u"}:
            raise TypeError(f"Capability sidecar {path} must contain boolean/integer values, got {values.dtype}")
        aligned_capabilities[capability] = values

    diagnostic_map: dict[str, DiagnosticCodelabel] = {}
    for key, value in (diagnostic_codelabels or {}).items():
        diagnostic_map[str(key)] = (
            value if isinstance(value, DiagnosticCodelabel) else DiagnosticCodelabel.from_dict(value)
        )

    records: list[EffectSampleRecord] = []
    matched_diagnostics: set[str] = set()
    for row_index in range(num_rows):
        row_sample_id = _metadata_string(sample_id[row_index])
        row_take_uid = _metadata_string(take_uid[row_index])
        if not row_sample_id or not row_take_uid:
            raise ValueError(f"Empty sample_id or take_uid at row {row_index}")
        row_timestamp = float(timestamp[row_index])
        validity: dict[EffectCapability, bool] = {EffectCapability.RGB_PAIRED: True}
        validity.update(
            {capability: bool(values[row_index]) for capability, values in aligned_capabilities.items()}
        )
        diagnostic = diagnostic_map.get(row_sample_id)
        if diagnostic is not None:
            matched_diagnostics.add(row_sample_id)
        records.append(
            EffectSampleRecord(
                sample_id=row_sample_id,
                take_uid=row_take_uid,
                split=split,
                row_index=row_index,
                source_dataset=source_dataset,
                timestamp=row_timestamp,
                capability_validity=validity,
                diagnostic_codelabel=diagnostic,
                content_hashes=dict(content_hashes or {}),
                provenance={
                    "input_dir": str(input_dir),
                    "view_keys": [str(key) for key in view_keys],
                    "metadata_storage": "npy_mmap",
                },
            )
        )
    _assert_unique_keys(records)

    audit = audit_effect_manifest(records)
    audit.update(
        {
            "input_dir": str(input_dir),
            "source_dataset": str(source_dataset),
            "view_keys": [str(key) for key in view_keys],
            "npy_arrays": {
                **{
                    key: {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "memory_mapped": isinstance(array, np.memmap),
                    }
                    for key, array in views.items()
                },
                "take_uid": {
                    "shape": list(take_uid.shape),
                    "dtype": str(take_uid.dtype),
                    "memory_mapped": isinstance(take_uid, np.memmap),
                },
                "timestamp": {
                    "shape": list(timestamp.shape),
                    "dtype": str(timestamp.dtype),
                    "memory_mapped": isinstance(timestamp, np.memmap),
                },
                "sample_id": {
                    "shape": list(sample_id.shape),
                    "dtype": str(sample_id.dtype),
                    "memory_mapped": isinstance(sample_id, np.memmap),
                },
            },
            "capability_sidecars": {
                capability.value: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "memory_mapped": isinstance(values, np.memmap),
                }
                for capability, values in aligned_capabilities.items()
            },
            "diagnostic_codelabel_join": {
                "provided": len(diagnostic_map),
                "matched": len(matched_diagnostics),
                "unmatched": sorted(set(diagnostic_map) - matched_diagnostics),
                "training_valid_count": 0,
            },
        }
    )
    return records, audit
