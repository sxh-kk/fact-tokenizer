"""Canonical, reproducible selection for the FACT v7 50-row target audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .effect_manifest import EffectCapability, EffectSampleRecord


AUDIT_SELECTION_SCHEMA = "fact-target-audit-selection-v1"
AUDIT_SAMPLE_COUNT = 50
AUDIT_SELECTION_SEED = 20260711


def select_audit_indices(
    records: Sequence[EffectSampleRecord],
    *,
    sample_count: int = AUDIT_SAMPLE_COUNT,
    seed: int = AUDIT_SELECTION_SEED,
) -> list[int]:
    """Reproduce the frozen quality/pose/mask round-robin selector."""

    if sample_count != AUDIT_SAMPLE_COUNT or seed != AUDIT_SELECTION_SEED:
        raise ValueError(
            f"formal target audit selection is fixed to {AUDIT_SAMPLE_COUNT} rows "
            f"and seed {AUDIT_SELECTION_SEED}"
        )
    eligible = [(index, record) for index, record in enumerate(records) if record.training_valid]
    if len(eligible) < sample_count:
        raise ValueError(f"only {len(eligible)} training-valid rows, need {sample_count}")
    groups: dict[tuple[str, bool, bool], list[int]] = {}
    for index, record in eligible:
        key = (
            record.quality_bucket or "unlabeled",
            record.has_capability(EffectCapability.CAMERA_POSE),
            record.has_capability(EffectCapability.OBJECT_MASK),
        )
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    active = sorted(groups)
    selected: list[int] = []
    while len(selected) < sample_count and active:
        next_active = []
        for key in active:
            values = groups[key]
            if values and len(selected) < sample_count:
                selected.append(values.pop())
            if values:
                next_active.append(key)
        active = next_active
    return selected


def audit_strata_counts(records: Sequence[EffectSampleRecord], indices: Sequence[int]) -> dict[str, int]:
    strata: dict[str, int] = {}
    for index in indices:
        record = records[int(index)]
        key = "|".join(
            [
                record.quality_bucket or "unlabeled",
                f"pose={int(record.has_capability(EffectCapability.CAMERA_POSE))}",
                f"mask={int(record.has_capability(EffectCapability.OBJECT_MASK))}",
            ]
        )
        strata[key] = strata.get(key, 0) + 1
    return dict(sorted(strata.items()))


def build_audit_selection_report(
    records: Sequence[EffectSampleRecord],
    indices: Sequence[int],
    *,
    manifest_sha256: str,
    index_sha256: str,
) -> dict[str, Any]:
    selected = [records[int(index)] for index in indices]
    return {
        "schema": AUDIT_SELECTION_SCHEMA,
        "sample_count": AUDIT_SAMPLE_COUNT,
        "seed": AUDIT_SELECTION_SEED,
        "sample_ids": [record.sample_id for record in selected],
        "strata_counts": audit_strata_counts(records, indices),
        "manifest_sha256": manifest_sha256,
        "index_sha256": index_sha256,
    }


def validate_audit_selection_contract(
    records: Sequence[EffectSampleRecord],
    *,
    manifest_path: Path | str,
    index_path: Path | str,
    contract_path: Path | str,
) -> dict[str, Any]:
    """Recompute and verify every field of the canonical pre-gate selection."""

    manifest_path = Path(manifest_path)
    index_path = Path(index_path)
    contract_path = Path(contract_path)
    contract_bytes = contract_path.read_bytes()
    contract: Mapping[str, Any] = json.loads(contract_bytes.decode("utf-8"))
    indices = np.load(index_path, allow_pickle=False)
    if indices.ndim != 1 or indices.dtype != np.dtype(np.int64):
        raise ValueError("audit sample index must be a one-dimensional int64 array")
    if len(indices) != AUDIT_SAMPLE_COUNT or len(np.unique(indices)) != AUDIT_SAMPLE_COUNT:
        raise ValueError("audit sample index must contain 50 unique rows")
    recomputed = select_audit_indices(records)
    if not np.array_equal(indices, np.asarray(recomputed, dtype=np.int64)):
        raise ValueError("audit sample index differs from the canonical deterministic selection")
    if any(index < 0 or index >= len(records) for index in indices.tolist()):
        raise ValueError("audit sample index is out of manifest bounds")
    if any(not records[int(index)].training_valid for index in indices):
        raise ValueError("audit selection contains a representation-training-invalid row")
    expected = build_audit_selection_report(
        records,
        recomputed,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
    )
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"audit selection contract field {key!r} is not canonical")
    return {
        **expected,
        "selection_contract": str(contract_path.resolve()),
        "selection_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "sample_index": str(index_path.resolve()),
        "sample_index_sha256": expected["index_sha256"],
    }
