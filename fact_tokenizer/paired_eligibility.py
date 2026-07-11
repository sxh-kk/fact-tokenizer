"""Frozen eligibility and donor contracts for the FACT v7 paired controls.

The P3 donor index is defined over the *eligible allowlist order*, not over an
unfiltered source dataset.  P0, P2, P3 and P4 must therefore first select the
same ``sample_id`` allowlist; P3 may then consume ``donor_index`` directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PAIRED_ELIGIBILITY_SCHEMA = "fact-v7-paired-control-eligibility-v1"
PAIRED_FINAL_CONTROLS = ("P0", "P2", "P3", "P4")
_UNKNOWN_PHASES = {"", "-1", "nan", "none", "null", "unknown"}
_REQUIRED_ARRAYS = (
    "sample_id",
    "take_uid",
    "phase_label",
    "manifest_index",
    "donor_index",
    "donor_sample_id",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_value(record: object, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _phase_text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return "" if value is None else str(value).strip()


def _phase_is_known(value: str) -> bool:
    return value.casefold() not in _UNKNOWN_PHASES


def _unicode_array(values: Sequence[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{max(1, width)}")


def validate_paired_eligibility(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate the frozen allowlist and the P3 donor relation."""

    missing = [name for name in _REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"eligibility sidecar is missing arrays: {missing}")
    values = {name: np.asarray(arrays[name]) for name in _REQUIRED_ARRAYS}
    lengths = {len(value) for value in values.values() if value.ndim == 1}
    if any(value.ndim != 1 for value in values.values()) or len(lengths) != 1:
        raise ValueError("eligibility arrays must be aligned one-dimensional vectors")
    count = next(iter(lengths), 0)
    if count == 0:
        raise ValueError("paired-control eligibility cannot be empty")

    sample_id = values["sample_id"].astype(str)
    take_uid = values["take_uid"].astype(str)
    phase_label = values["phase_label"].astype(str)
    manifest_index = values["manifest_index"].astype(np.int64)
    donor_index = values["donor_index"].astype(np.int64)
    donor_sample_id = values["donor_sample_id"].astype(str)
    if any(not value.strip() for value in sample_id) or len(set(sample_id.tolist())) != count:
        raise ValueError("eligible sample IDs must be unique and non-empty")
    if any(not value.strip() for value in take_uid):
        raise ValueError("eligible take UIDs must be non-empty")
    if any(not _phase_is_known(value) for value in phase_label):
        raise ValueError("eligible phase labels must be known")
    if (manifest_index < 0).any() or len(set(manifest_index.tolist())) != count:
        raise ValueError("eligible manifest indices must be unique and non-negative")
    if (donor_index < 0).any() or (donor_index >= count).any():
        raise ValueError("P3 donor indices must index the eligible allowlist")
    anchors = np.arange(count, dtype=np.int64)
    if np.any(donor_index == anchors):
        raise ValueError("a P3 donor cannot be the anchor itself")
    if np.any(take_uid[donor_index] != take_uid):
        raise ValueError("every P3 donor must come from the same take")
    phase_keys = np.asarray([value.casefold() for value in phase_label])
    if np.any(phase_keys[donor_index] == phase_keys):
        raise ValueError("every P3 donor must have a different phase")
    if np.any(sample_id[donor_index] != donor_sample_id):
        raise ValueError("donor_sample_id does not match donor_index")


def build_paired_control_eligibility(
    records: Sequence[object],
    phase_labels: Sequence[Any] | Mapping[str, Any],
    *,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one common P0/P2/P3/P4 allowlist and a deterministic P3 donor.

    Non-training records and records without a known phase are excluded before
    donor search. A take is eligible only when it contains at least two known
    phases, which guarantees that every retained sample has a valid donor.
    """

    sample_ids = [str(_record_value(record, "sample_id", "")).strip() for record in records]
    take_uids = [str(_record_value(record, "take_uid", "")).strip() for record in records]
    if any(not value for value in sample_ids) or any(not value for value in take_uids):
        raise ValueError("every aligned manifest row needs sample_id and take_uid")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("aligned manifest sample IDs must be globally unique")
    if isinstance(phase_labels, Mapping):
        labels = [_phase_text(phase_labels.get(sample_id)) for sample_id in sample_ids]
    else:
        if len(phase_labels) != len(records):
            raise ValueError("aligned phase labels must have one value per manifest row")
        labels = [_phase_text(value) for value in phase_labels]

    training_valid = [bool(_record_value(record, "training_valid", True)) for record in records]
    timestamps = []
    for index, record in enumerate(records):
        value = _record_value(record, "timestamp", None)
        timestamps.append(float(index) if value is None else float(value))

    candidate_indices = [
        index
        for index, (valid, label) in enumerate(zip(training_valid, labels))
        if valid and _phase_is_known(label)
    ]
    by_take: dict[str, list[int]] = defaultdict(list)
    for index in candidate_indices:
        by_take[take_uids[index]].append(index)

    selected_donor: dict[int, int] = {}
    for index in candidate_indices:
        donors = [
            donor
            for donor in by_take[take_uids[index]]
            if labels[donor].casefold() != labels[index].casefold()
        ]
        if not donors:
            continue
        donors.sort(
            key=lambda donor: (
                -abs(timestamps[donor] - timestamps[index]),
                hashlib.sha256(
                    f"{seed}:{sample_ids[index]}:{sample_ids[donor]}".encode("utf-8")
                ).hexdigest(),
            )
        )
        selected_donor[index] = donors[0]

    eligible_manifest_indices = sorted(selected_donor)
    if not eligible_manifest_indices:
        raise ValueError("no sample has a same-take, different-phase P3 donor")
    eligible_position = {
        manifest_index: position
        for position, manifest_index in enumerate(eligible_manifest_indices)
    }
    if any(donor not in eligible_position for donor in selected_donor.values()):
        raise AssertionError("a selected P3 donor is not present in the common eligibility set")

    donor_indices = [eligible_position[selected_donor[index]] for index in eligible_manifest_indices]
    donor_sample_ids = [sample_ids[selected_donor[index]] for index in eligible_manifest_indices]
    arrays = {
        "sample_id": _unicode_array([sample_ids[index] for index in eligible_manifest_indices]),
        "take_uid": _unicode_array([take_uids[index] for index in eligible_manifest_indices]),
        "phase_label": _unicode_array([labels[index] for index in eligible_manifest_indices]),
        "manifest_index": np.asarray(eligible_manifest_indices, dtype=np.int64),
        "donor_index": np.asarray(donor_indices, dtype=np.int64),
        "donor_sample_id": _unicode_array(donor_sample_ids),
    }
    validate_paired_eligibility(arrays)

    no_donor = [
        index
        for index in candidate_indices
        if index not in selected_donor
    ]
    report = {
        "schema": PAIRED_ELIGIBILITY_SCHEMA,
        "seed": int(seed),
        "manifest_rows": len(records),
        "training_valid_rows": int(sum(training_valid)),
        "known_phase_training_rows": len(candidate_indices),
        "eligible_rows": len(eligible_manifest_indices),
        "eligible_takes": len({take_uids[index] for index in eligible_manifest_indices}),
        "dropped": {
            "nontraining": int(sum(not value for value in training_valid)),
            "missing_or_unknown_phase": int(
                sum(valid and not _phase_is_known(label) for valid, label in zip(training_valid, labels))
            ),
            "no_same_take_different_phase_donor": len(no_donor),
        },
        "eligible_phase_counts": dict(
            sorted(Counter(labels[index] for index in eligible_manifest_indices).items())
        ),
        "common_controls": list(PAIRED_FINAL_CONTROLS),
        "allowlist_order": "ascending_aligned_manifest_index",
        "donor_index_basis": "eligible_allowlist_order",
        "donor_contract": "same_take_and_different_known_phase",
        "donor_policy": "maximum_timestamp_distance_then_seeded_sample_id_hash",
    }
    return arrays, report


def load_paired_control_eligibility(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError("paired-control eligibility SHA256 mismatch")
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    validate_paired_eligibility(arrays)
    return arrays


def dataset_indices_for_eligibility(
    dataset_sample_ids: Sequence[Any],
    eligible_sample_ids: Sequence[Any],
) -> np.ndarray:
    """Return dataset indices ordered by the common frozen allowlist."""

    dataset = [_phase_text(value) for value in dataset_sample_ids]
    eligible = [_phase_text(value) for value in eligible_sample_ids]
    if len(set(dataset)) != len(dataset):
        raise ValueError("dataset sample IDs must be unique before applying eligibility")
    if len(set(eligible)) != len(eligible) or any(not value for value in eligible):
        raise ValueError("eligible sample IDs must be unique and non-empty")
    positions = {sample_id: index for index, sample_id in enumerate(dataset)}
    missing = [sample_id for sample_id in eligible if sample_id not in positions]
    if missing:
        raise ValueError(f"dataset is missing {len(missing)} eligible samples, including {missing[:3]}")
    return np.asarray([positions[sample_id] for sample_id in eligible], dtype=np.int64)


__all__ = [
    "PAIRED_ELIGIBILITY_SCHEMA",
    "PAIRED_FINAL_CONTROLS",
    "build_paired_control_eligibility",
    "dataset_indices_for_eligibility",
    "load_paired_control_eligibility",
    "sha256_file",
    "validate_paired_eligibility",
]
