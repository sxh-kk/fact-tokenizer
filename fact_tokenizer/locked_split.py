"""Construction and leakage audits for the fresh FACT short-take locked set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


LOCKED_SET_ID = "short73_t0p5_s1_8t_locked"


@dataclass(frozen=True)
class ShortLockedConfig:
    set_id: str = LOCKED_SET_ID
    expected_takes: int = 73
    samples_per_take: int = 8
    transition_seconds: float = 0.5
    stride_seconds: float = 1.0
    frame_rate: float = 30.0

    def __post_init__(self) -> None:
        if self.expected_takes <= 0 or self.samples_per_take <= 0:
            raise ValueError("expected_takes and samples_per_take must be positive")
        if self.transition_seconds <= 0 or self.stride_seconds <= 0 or self.frame_rate <= 0:
            raise ValueError("transition, stride and frame rate must be positive")


@dataclass(frozen=True)
class LockedSample:
    sample_id: str
    take_uid: str
    split: str
    row_index: int
    timestamp: float
    end_timestamp: float
    start_frame: int
    end_frame: int
    capture_uid: Optional[str]
    participant_uid: Optional[str]
    locked_set_id: str = LOCKED_SET_ID
    training_valid: bool = False
    model_selection_valid: bool = False
    final_inference_only: bool = True

    @property
    def frame_keys(self) -> tuple[str, ...]:
        return tuple(
            f"{self.take_uid}:{frame}" for frame in range(self.start_frame, self.end_frame + 1)
        )


def canonical_sample_id(take_uid: str, timestamp: float) -> str:
    return f"{take_uid}:{timestamp:.3f}"


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _time_bounds(row: Mapping[str, Any]) -> tuple[float, float]:
    start_raw = _first(row, ("task_start_sec", "start_sec", "start_time"))
    end_raw = _first(row, ("task_end_sec", "end_sec", "end_time", "duration_sec"))
    start = float(start_raw) if start_raw is not None else 0.0
    if end_raw is None:
        raise ValueError(f"take {row.get('take_uid')} has no end/duration field")
    end = float(end_raw)
    return start, end


def deduplicate_insufficient_takes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select one complete legacy ``insufficient_duration`` record per take."""

    selected: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        if str(row.get("reason", "")) != "insufficient_duration":
            continue
        take_uid = str(row.get("take_uid", "")).strip()
        if not take_uid:
            raise ValueError("insufficient_duration row is missing take_uid")
        if take_uid in selected and selected[take_uid] != row:
            # Prefer the row with more source metadata but reject conflicting timing.
            if _time_bounds(selected[take_uid]) != _time_bounds(row):
                raise ValueError(f"conflicting insufficient_duration rows for take {take_uid}")
            if len(row) > len(selected[take_uid]):
                selected[take_uid] = row
        else:
            selected[take_uid] = row
    return [selected[key] for key in sorted(selected)]


def build_short_locked_samples(
    rows: Iterable[Mapping[str, Any]],
    config: ShortLockedConfig = ShortLockedConfig(),
    require_expected_takes: bool = True,
) -> tuple[list[LockedSample], list[dict[str, Any]]]:
    """Build exactly eight fixed-stride transitions from each unused short take."""

    takes = deduplicate_insufficient_takes(rows)
    if require_expected_takes and len(takes) != config.expected_takes:
        raise ValueError(f"expected {config.expected_takes} short takes, found {len(takes)}")
    samples: list[LockedSample] = []
    selected_takes: list[dict[str, Any]] = []
    for row in takes:
        take_uid = str(row["take_uid"])
        start, end = _time_bounds(row)
        last_end = start + (config.samples_per_take - 1) * config.stride_seconds + config.transition_seconds
        if last_end > end + 1e-6:
            raise ValueError(
                f"take {take_uid} is too short for {config.samples_per_take} transitions: "
                f"need through {last_end:.3f}s, ends at {end:.3f}s"
            )
        capture_uid = _first(row, ("capture_uid", "capture_id", "capture_name"))
        participant_uid = _first(row, ("participant_uid", "participant_id", "university_video_id"))
        selected_row = dict(row)
        selected_row.update(
            {
                "locked_set_id": config.set_id,
                "locked": True,
                "samples_per_take": config.samples_per_take,
                "transition_seconds": config.transition_seconds,
                "stride_seconds": config.stride_seconds,
                "selected_timestamps": [start + index * config.stride_seconds for index in range(config.samples_per_take)],
            }
        )
        selected_takes.append(selected_row)
        for index in range(config.samples_per_take):
            timestamp = start + index * config.stride_seconds
            end_timestamp = timestamp + config.transition_seconds
            samples.append(
                LockedSample(
                    sample_id=canonical_sample_id(take_uid, timestamp),
                    take_uid=take_uid,
                    split="locked_test",
                    row_index=index,
                    timestamp=timestamp,
                    end_timestamp=end_timestamp,
                    start_frame=int(round(timestamp * config.frame_rate)),
                    end_frame=int(round(end_timestamp * config.frame_rate)),
                    capture_uid=capture_uid,
                    participant_uid=participant_uid,
                    locked_set_id=config.set_id,
                )
            )
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("locked-set construction produced duplicate sample IDs")
    return samples, selected_takes


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def _records_from_npy_dir(path: Path) -> list[dict[str, Any]]:
    take_path = path / "take_uid.npy"
    if not take_path.is_file():
        raise FileNotFoundError(take_path)
    takes = np.load(take_path, mmap_mode="r", allow_pickle=False).astype(str)
    sample_path = path / "sample_id.npy"
    timestamp_path = path / "timestamp.npy"
    samples = (
        np.load(sample_path, mmap_mode="r", allow_pickle=False).astype(str)
        if sample_path.is_file()
        else np.asarray(["" for _ in takes])
    )
    timestamps = (
        np.load(timestamp_path, mmap_mode="r", allow_pickle=False).astype(np.float64)
        if timestamp_path.is_file()
        else np.full(len(takes), np.nan)
    )
    if not (len(takes) == len(samples) == len(timestamps)):
        raise ValueError(f"metadata lengths differ in {path}")
    return [
        {"take_uid": takes[index], "sample_id": samples[index], "timestamp": timestamps[index]}
        for index in range(len(takes))
    ]


def load_reference_records(path: Path | str) -> list[dict[str, Any]]:
    """Load prior sample references from NPY directories, JSONL/JSON, CSV or NPY."""

    source = Path(path)
    if source.is_dir():
        return _records_from_npy_dir(source)
    if source.suffix.lower() == ".jsonl":
        return load_jsonl(source)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("samples", value.get("records", []))
        if not isinstance(value, list):
            raise ValueError(f"cannot find a record list in {source}")
        return [dict(row) for row in value]
    if source.suffix.lower() == ".csv":
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if source.suffix.lower() == ".npy":
        values = np.load(source, mmap_mode="r", allow_pickle=False).astype(str)
        return [{"sample_id": str(value)} for value in values]
    raise ValueError(f"unsupported reference format: {source}")


def _sample_ref(row: Mapping[str, Any], frame_rate: float) -> tuple[Optional[str], Optional[str], set[str]]:
    sample_id = _first(row, ("sample_id", "transition_id"))
    take_uid = _first(row, ("take_uid", "take_id"))
    timestamp_raw = _first(row, ("timestamp", "timestamp_sec", "start_sec"))
    frames: set[str] = set()
    if take_uid is not None and timestamp_raw is not None:
        timestamp = float(timestamp_raw)
        end_raw = _first(row, ("end_timestamp", "end_sec", "end_time"))
        duration_raw = _first(row, ("transition_seconds", "duration", "duration_sec"))
        end_timestamp = (
            float(end_raw)
            if end_raw is not None
            else timestamp + (float(duration_raw) if duration_raw is not None else 0.5)
        )
        start_frame = int(round(timestamp * frame_rate))
        end_frame = int(round(end_timestamp * frame_rate))
        frames.update(f"{take_uid}:{frame}" for frame in range(start_frame, end_frame + 1))
    if take_uid is not None and row.get("start_frame") not in (None, ""):
        start_frame = int(float(row["start_frame"]))
        end_frame = int(float(row.get("end_frame", start_frame)))
        frames.update(f"{take_uid}:{frame}" for frame in range(start_frame, end_frame + 1))
    for key in ("frame_index", "frame"):
        if take_uid is not None and row.get(key) not in (None, ""):
            frames.add(f"{take_uid}:{int(float(row[key]))}")
    return sample_id, take_uid, frames


def audit_locked_samples(
    samples: Sequence[LockedSample],
    reference_groups: Mapping[str, Iterable[Mapping[str, Any]]],
    participant_fresh_takes: Iterable[str] = (),
    frame_rate: float = 30.0,
) -> dict[str, Any]:
    """Audit take/sample/frame/capture/participant overlap against all supplied history."""

    candidate_samples = {sample.sample_id for sample in samples}
    candidate_takes = {sample.take_uid for sample in samples}
    candidate_frames = {frame for sample in samples for frame in sample.frame_keys}
    candidate_captures = {sample.capture_uid for sample in samples if sample.capture_uid}
    candidate_participants = {sample.participant_uid for sample in samples if sample.participant_uid}
    groups: dict[str, Any] = {}
    failures: list[str] = []
    for name, records_iter in sorted(reference_groups.items()):
        records = list(records_iter)
        reference_samples: set[str] = set()
        reference_takes: set[str] = set()
        reference_frames: set[str] = set()
        reference_captures: set[str] = set()
        reference_participants: set[str] = set()
        for row in records:
            sample_id, take_uid, frames = _sample_ref(row, frame_rate)
            if sample_id:
                reference_samples.add(sample_id)
            if take_uid:
                reference_takes.add(take_uid)
            reference_frames.update(frames)
            capture = _first(row, ("capture_uid", "capture_id", "capture_name"))
            participant = _first(row, ("participant_uid", "participant_id", "university_video_id"))
            if capture:
                reference_captures.add(capture)
            if participant:
                reference_participants.add(participant)
        overlaps = {
            "sample_ids": sorted(candidate_samples & reference_samples),
            "takes": sorted(candidate_takes & reference_takes),
            "frames": sorted(candidate_frames & reference_frames),
            "captures": sorted(candidate_captures & reference_captures),
            "participants": sorted(candidate_participants & reference_participants),
        }
        groups[name] = {"reference_records": len(records), "overlaps": overlaps}
        # Participant overlap is reported rather than forbidden: only the listed
        # participant-fresh takes are required to be fresh at participant level.
        for overlap_kind in ("sample_ids", "takes", "frames"):
            if overlaps[overlap_kind]:
                failures.append(f"{name} has {len(overlaps[overlap_kind])} overlapping {overlap_kind}")

    fresh_required = {str(value).strip() for value in participant_fresh_takes if str(value).strip()}
    missing_fresh = sorted(fresh_required - candidate_takes)
    if missing_fresh:
        failures.append(f"missing {len(missing_fresh)} required participant-fresh takes")
    fresh_participant_overlap: dict[str, list[str]] = {}
    for take_uid in sorted(fresh_required):
        participants = {sample.participant_uid for sample in samples if sample.take_uid == take_uid and sample.participant_uid}
        conflicting_groups = [
            name
            for name, group in groups.items()
            if participants & set(group["overlaps"]["participants"])
        ]
        if conflicting_groups:
            fresh_participant_overlap[take_uid] = conflicting_groups
    if fresh_participant_overlap:
        failures.append(f"{len(fresh_participant_overlap)} required takes are not participant-fresh")
    return {
        "candidate_samples": len(samples),
        "candidate_takes": len(candidate_takes),
        "candidate_captures": len(candidate_captures),
        "candidate_participants": len(candidate_participants),
        "reference_groups": groups,
        "participant_fresh_required": sorted(fresh_required),
        "participant_fresh_missing": missing_fresh,
        "participant_fresh_overlap": fresh_participant_overlap,
        "passed": not failures,
        "failures": failures,
    }


def perceptual_vectors(videos: np.ndarray, size: int = 16, block_size: int = 64) -> np.ndarray:
    """Build motion-aware low-resolution transition vectors for leakage screening.

    The appearance component catches copied frames while the independently
    normalized temporal-difference component prevents a shared static Exo
    background from dominating the score.  The joined vector is normalized a
    final time so an exact duplicate always has cosine similarity one, even
    when both transitions are completely static.
    """

    array = np.asarray(videos)
    if array.ndim != 5 or array.shape[1] < 2:
        raise ValueError("videos must be NxTxHxWxC or NxTxCxHxW")
    channels_last = array.shape[-1] in (1, 3)
    channels_first = array.shape[2] in (1, 3)
    if not channels_last and not channels_first:
        raise ValueError("cannot infer video channels")
    height = array.shape[2] if channels_last else array.shape[3]
    width = array.shape[3] if channels_last else array.shape[4]
    y_index = np.linspace(0, height - 1, size).round().astype(int)
    x_index = np.linspace(0, width - 1, size).round().astype(int)
    vectors = np.empty((len(array), 3 * size * size), dtype=np.float32)
    for start in range(0, len(array), block_size):
        block = np.asarray(array[start : start + block_size][:, [0, -1]], dtype=np.float32)
        if channels_last:
            gray = block[:, :, y_index[:, None], x_index[None, :], :].mean(axis=-1, dtype=np.float32)
        else:
            gray = block[:, :, :, y_index[:, None], x_index[None, :]].mean(axis=2, dtype=np.float32)
        appearance = gray.reshape(len(block), -1)
        appearance -= appearance.mean(axis=1, keepdims=True)
        appearance /= np.maximum(np.linalg.norm(appearance, axis=1, keepdims=True), 1e-8)
        motion = (gray[:, 1] - gray[:, 0]).reshape(len(block), -1)
        motion -= motion.mean(axis=1, keepdims=True)
        motion /= np.maximum(np.linalg.norm(motion, axis=1, keepdims=True), 1e-8)
        joined = np.concatenate((appearance, motion), axis=1)
        joined /= np.maximum(np.linalg.norm(joined, axis=1, keepdims=True), 1e-8)
        vectors[start : start + len(block)] = joined
    return vectors


def perceptual_nearest_neighbor_audit(
    candidate_videos: np.ndarray,
    reference_videos: np.ndarray,
    maximum_similarity: float = 0.995,
    block_size: int = 1024,
) -> dict[str, Any]:
    """Find cross-set nearest neighbors without materializing an NxM matrix."""

    candidate = perceptual_vectors(candidate_videos)
    reference = perceptual_vectors(reference_videos)
    maxima = np.full(len(candidate), -np.inf, dtype=np.float32)
    indices = np.full(len(candidate), -1, dtype=np.int64)
    for start in range(0, len(reference), block_size):
        block = reference[start : start + block_size]
        # At most 584x1024 float32 values are materialized; the vectorized
        # contraction is far faster than a Python candidate loop.
        # ``einsum(..., optimize=False)`` stays in NumPy's bounded C loop and
        # avoids platform BLAS/OpenMP conflicts in mixed Torch environments.
        scores = np.einsum("nd,md->nm", candidate, block, optimize=False)
        local_indices = np.argmax(scores, axis=1)
        local_maxima = scores[np.arange(len(candidate)), local_indices]
        improved = local_maxima > maxima
        maxima[improved] = local_maxima[improved]
        indices[improved] = start + local_indices[improved]
    violations = np.flatnonzero(maxima >= maximum_similarity)
    return {
        "metric": "appearance_plus_temporal_difference_cosine_v2",
        "candidate_count": len(candidate),
        "reference_count": len(reference),
        "maximum_similarity_allowed": maximum_similarity,
        "maximum_observed_similarity": float(maxima.max(initial=-np.inf)),
        "violations": [
            {"candidate_index": int(index), "reference_index": int(indices[index]), "similarity": float(maxima[index])}
            for index in violations
        ],
        "passed": not len(violations),
    }


def write_frozen_locked_manifest(
    output_dir: Path | str,
    samples: Sequence[LockedSample],
    selected_takes: Sequence[Mapping[str, Any]],
    config: ShortLockedConfig,
    audit: Mapping[str, Any],
    *,
    stage: str = "final",
) -> dict[str, Any]:
    """Atomically publish a locked manifest only after leakage audits pass."""

    if not audit.get("passed", False):
        raise ValueError(f"locked-set audit failed: {audit.get('failures', [])}")
    if stage not in {"provisional", "final"}:
        raise ValueError("locked freeze stage must be provisional or final")
    perceptual_status = audit.get("perceptual_nearest_neighbor", {}).get("status")
    if stage == "final" and perceptual_status != "complete":
        raise ValueError("final locked freeze requires a completed perceptual nearest-neighbor audit")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "samples.jsonl"
    selected_path = output / "selected_takes.jsonl"
    for path, records in ((manifest_path, [asdict(sample) for sample in samples]), (selected_path, selected_takes)):
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        temp.replace(path)
    sample_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    selected_sha256 = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    freeze = {
        "locked_set_id": config.set_id,
        "freeze_stage": stage,
        "config": asdict(config),
        "takes": len({sample.take_uid for sample in samples}),
        "samples": len(samples),
        "samples_sha256": sample_sha256,
        "selected_takes_sha256": selected_sha256,
        "training_valid": False,
        "model_selection_valid": False,
        "final_inference_only": True,
        "evaluation_allowed": stage == "final",
        "audit": dict(audit),
    }
    freeze_path = output / "freeze.json"
    temp = freeze_path.with_suffix(".tmp")
    temp.write_text(json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(freeze_path)
    return freeze


def validate_final_freeze_candidate_contract(
    freeze: Mapping[str, Any], sample_count: int
) -> dict[str, str]:
    """Return exact candidate-array hashes accepted by a final dual-view PNN audit."""

    if freeze.get("freeze_stage") != "final":
        return {}
    if (
        freeze.get("evaluation_allowed") is not True
        or freeze.get("final_inference_only") is not True
        or freeze.get("training_valid") is not False
        or freeze.get("model_selection_valid") is not False
        or freeze.get("audit", {}).get("passed") is not True
        or int(freeze.get("samples", -1)) != sample_count
        or int(freeze.get("takes", -1)) != 73
    ):
        raise ValueError("final short73 freeze lacks its locked evaluation/isolation contract")
    pnn = freeze["audit"].get("perceptual_nearest_neighbor", {})
    reports = pnn.get("reports", [])
    if pnn.get("status") != "complete" or not isinstance(reports, list) or not reports:
        raise ValueError("final short73 freeze lacks a completed PNN audit")
    candidates: dict[str, str] = {}
    for report in reports:
        view = str(report.get("view", ""))
        candidate_sha256 = str(report.get("candidate_sha256", ""))
        if (
            view not in {"ego", "exo"}
            or len(candidate_sha256) != 64
            or report.get("passed") is not True
            or int(report.get("candidate_count", -1)) != sample_count
            or report.get("violations") not in ([], None)
        ):
            raise ValueError("final short73 freeze contains an invalid PNN report")
        previous = candidates.setdefault(view, candidate_sha256)
        if previous != candidate_sha256:
            raise ValueError(f"PNN reports disagree on the {view} candidate SHA256")
    if set(candidates) != {"ego", "exo"}:
        raise ValueError("final short73 PNN audit must bind both ego and exo candidates")
    provisional = freeze["audit"].get("provisional_evidence", {})
    sample_id_sha256 = str(provisional.get("candidate_sample_ids_sha256", ""))
    if len(sample_id_sha256) != 64:
        raise ValueError("final short73 freeze does not bind the audited candidate sample IDs")
    candidates["sample_id"] = sample_id_sha256
    return candidates
