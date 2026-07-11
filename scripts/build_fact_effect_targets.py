#!/usr/bin/env python3
"""Build versioned DINO/RAFT/camera-compensated FACT v7 effect targets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import EffectCapability, read_manifest_jsonl  # noqa: E402
from fact_tokenizer.effect_targets import (  # noqa: E402
    EffectTargetCacheWriter,
    EffectTargetConfig,
    RAFTFlowEstimator,
    align_atomic_descriptions,
    align_phase_segments,
    dino_delta,
    deterministic_weak_semantic_labels,
    forward_backward_consistency,
    invert_extrinsic,
    propagate_relation_mask,
    relative_rotation_homography,
    require_supported_target,
    roi_feature_delta,
    rotation_compensated_flow_2d,
    scale_intrinsics,
    WEAK_VERB_MAP_VERSION,
)
from fact_tokenizer.model import DINOv2PatchFeatureExtractor, MockPatchFeatureExtractor  # noqa: E402
from fact_tokenizer.target_audit_selection import (  # noqa: E402
    validate_audit_selection_contract,
)
from fact_tokenizer.target_audit import validate_visual_audit_gate  # noqa: E402


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected VIEW=PATH")
    name, path = (part.strip() for part in value.split("=", 1))
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty VIEW=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--source-contract",
        type=Path,
        help="Hash-bound fact-npy-source-contract-v1; mandatory outside smoke mode.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["ego", "exo"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-index-npy",
        type=Path,
        help="Optional frozen manifest-row index vector for a stratified audit cache shard.",
    )
    parser.add_argument(
        "--sample-index-contract",
        type=Path,
        help="Hash-bound selection JSON; required with --sample-index-npy for formal targets.",
    )
    parser.add_argument(
        "--visual-audit-gate",
        type=Path,
        help="Required for a full formal cache; must be the passed 50-sample gate JSON.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-backend", choices=["dinov2", "mock"], default="dinov2")
    parser.add_argument("--dino-model", default="dinov2_vitb14_reg")
    parser.add_argument("--torch-home")
    parser.add_argument("--mock-feature-dim", type=int, default=64)
    parser.add_argument("--raft-weights", type=Path)
    parser.add_argument("--zero-flow-smoke", action="store_true")
    parser.add_argument("--allow-smoke-targets", action="store_true")
    parser.add_argument("--camera-sidecar", action="append", default=[], type=named_path, metavar="VIEW=DIR")
    parser.add_argument("--pose-convention", choices=["world_to_camera", "camera_to_world"], default="world_to_camera")
    parser.add_argument("--object-mask-npy", action="append", default=[], type=named_path, metavar="VIEW=PATH")
    parser.add_argument(
        "--object-mask-anchor-offset-npy",
        action="append",
        default=[],
        type=named_path,
        metavar="VIEW=PATH",
    )
    parser.add_argument(
        "--object-mask-anchor-image-npy",
        action="append",
        default=[],
        type=named_path,
        metavar="VIEW=PATH",
        help="RGB image at the actual Relations annotation frame, required whenever anchor offset is non-zero.",
    )
    parser.add_argument("--weak-index-jsonl", type=Path)
    parser.add_argument(
        "--weak-calibration-report",
        type=Path,
        help="Frozen 60-sample dev calibration report; required before weak targets become valid.",
    )
    parser.add_argument("--body-delta-npy", type=Path)
    parser.add_argument("--body-valid-npy", type=Path)
    parser.add_argument("--body-joint-valid-npy", type=Path)
    parser.add_argument("--hand-delta-npy", type=Path)
    parser.add_argument("--hand-valid-npy", type=Path)
    parser.add_argument("--hand-joint-valid-npy", type=Path)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target", action="append", default=[], help="Optional requested target names; 3D fails early.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mmap(path: Path, expected: int | None = None) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if expected is not None and len(array) != expected:
        raise ValueError(f"{path} has {len(array)} rows, expected {expected}")
    return array


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def npy_identity(path: Path) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


CORE_SOURCE_ARRAYS = ("ego", "exo", "sample_id", "take_uid", "timestamp")


def _sample_id_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(value + "\n" for value in sorted(values)).encode("utf-8")
    ).hexdigest()


def _source_snapshot(directory: Path) -> tuple[dict[str, np.ndarray], dict, np.ndarray, dict]:
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in CORE_SOURCE_ARRAYS
    }
    if len({len(array) for array in arrays.values()}) != 1:
        raise ValueError("source contract arrays are not row aligned")
    for view in ("ego", "exo"):
        value = arrays[view]
        if value.ndim != 5 or value.shape[1] != 2 or value.dtype != np.uint8:
            raise ValueError(f"source contract {view}.npy is not uint8 paired RGB")
    files = {
        name: npy_identity(directory / f"{name}.npy") for name in CORE_SOURCE_ARRAYS
    }
    frame_path = directory / "frame_index.npy"
    frame = np.load(frame_path, mmap_mode="r", allow_pickle=False)
    if (
        frame.ndim != 1
        or len(frame) != len(arrays["sample_id"])
        or not np.issubdtype(frame.dtype, np.integer)
        or (frame < 0).any()
    ):
        raise ValueError("source contract frame_index.npy is invalid")
    return arrays, files, frame, npy_identity(frame_path)


def _validate_materialization_report(
    directory: Path,
    files: Mapping[str, Any],
    frame_identity: Mapping[str, Any],
) -> tuple[Path, dict]:
    report_path = directory / "materialization_report.json"
    if not report_path.is_file():
        raise ValueError("formal target input requires materialization_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema")
        not in {
            "fact-npy-transition-rebuild-v1",
            "fact-npy-transition-subset-v1",
            "fact-short73-materialization-v3",
        }
        or report.get("color_space") != "RGB"
        or float(report.get("transition_seconds", -1.0)) != 0.5
        or report.get("endpoint_semantics") != ["t", "t+0.5s"]
        or float(report.get("frame_rate_hz", -1.0)) != 30.0
        or int(report.get("endpoint_offset_frames", -1)) != 15
        or report.get("files") != files
        or report.get("frame_index") != frame_identity
    ):
        raise ValueError("formal target input has an invalid transition materialization report")
    return report_path, report


def _validate_raw_semantic_evidence(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    directory: Path,
    arrays: Mapping[str, np.ndarray],
    files: Mapping[str, Any],
    frame_indices: np.ndarray,
    frame_identity: Mapping[str, Any],
) -> None:
    report_path = Path(str(evidence.get("producer_report", "")))
    if (
        not report_path.is_file()
        or sha256_file(report_path) != evidence.get("producer_report_sha256")
    ):
        raise ValueError("source contract producer report hash mismatch")
    producer_code = evidence.get("producer_code_sha256", {})
    if not isinstance(producer_code, Mapping) or not producer_code:
        raise ValueError("source contract producer code evidence is missing")
    for code_path_text, expected_hash in producer_code.items():
        code_path = Path(code_path_text)
        if not code_path.is_absolute():
            code_path = ROOT / code_path
        if not code_path.is_file() or sha256_file(code_path) != expected_hash:
            raise ValueError("source contract producer code hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audited_ids = [str(value) for value in contract.get("audited_sample_ids", [])]
    materialization_path = directory / "materialization_report.json"
    audit_code_path = ROOT / "scripts" / "audit_fact_npy_source_semantics.py"
    decoder_code_path = ROOT / "scripts" / "prepare_fact_egoexo_npz.py"
    video_map_path = Path(str(report.get("video_map_jsonl", "")))
    gold_manifest_path = Path(str(report.get("gold_manifest", "")))
    if (
        report.get("schema") != "fact-npy-source-semantic-audit-v1"
        or report.get("passed") is not True
        or Path(str(report.get("source_dir", ""))).resolve() != directory.resolve()
        or int(report.get("rows", -1)) != len(arrays["sample_id"])
        or report.get("files") != files
        or report.get("color_space") != "RGB"
        or float(report.get("transition_seconds", -1.0)) != 0.5
        or report.get("endpoint_semantics") != ["t", "t+0.5s"]
        or int(report.get("resize", -1)) != 224
        or report.get("frame_selection_mode") != "frozen_frame_index_sidecar"
        or float(report.get("frame_rate_hz", -1.0)) != 30.0
        or int(report.get("endpoint_offset_frames", -1)) != 15
        or float(report.get("maximum_t0_timestamp_distance_frames", 999.0)) > 0.501
        or Path(str(report.get("frame_index_npy", ""))).resolve()
        != (directory / "frame_index.npy").resolve()
        or report.get("frame_index") != frame_identity
        or Path(str(report.get("materialization_report", ""))).resolve()
        != materialization_path.resolve()
        or report.get("materialization_report_sha256") != sha256_file(materialization_path)
        or sorted(str(value) for value in report.get("audited_sample_ids", []))
        != sorted(audited_ids)
        or report.get("audited_sample_ids_sha256") != _sample_id_digest(audited_ids)
        or int(report.get("audited_samples", -1)) != len(audited_ids)
        or int(report.get("exact_view_pair_matches", -1)) != 2 * len(audited_ids)
        or report.get("audit_code_sha256") != sha256_file(audit_code_path)
        or report.get("audit_code_sha256") not in set(producer_code.values())
        or report.get("decoder_code_sha256") != sha256_file(decoder_code_path)
        or not video_map_path.is_file()
        or report.get("video_map_sha256") != sha256_file(video_map_path)
        or not gold_manifest_path.is_file()
        or report.get("gold_manifest_sha256") != sha256_file(gold_manifest_path)
        or not isinstance(report.get("included_splits"), list)
        or not report.get("included_splits")
    ):
        raise ValueError("source contract raw-video semantic evidence is invalid")
    frame_rows = report.get("audited_frame_rows", [])
    rows_text = "".join(
        f"{row.get('sample_id')}\t{row.get('take_uid')}\t{row.get('frame_index')}\n"
        for row in frame_rows
    )
    if (
        len(frame_rows) != len(audited_ids)
        or [str(row.get("sample_id")) for row in frame_rows] != sorted(audited_ids)
        or report.get("audited_frame_rows_sha256")
        != hashlib.sha256(rows_text.encode("utf-8")).hexdigest()
    ):
        raise ValueError("source contract raw-video frame mapping digest is invalid")
    lookup = {as_text(value): index for index, value in enumerate(arrays["sample_id"])}
    for row in frame_rows:
        index = lookup.get(str(row.get("sample_id")))
        if (
            index is None
            or str(row.get("take_uid")) != as_text(arrays["take_uid"][index])
            or int(row.get("frame_index", -1)) != int(frame_indices[index])
        ):
            raise ValueError("source contract raw-video frame mapping differs from the source")
    audited_takes = {str(row.get("take_uid")) for row in frame_rows}
    inventory = report.get("video_inventory", [])
    inventory_takes: set[str] = set()
    if not isinstance(inventory, list):
        raise ValueError("source contract raw-video inventory is missing")
    for item in inventory:
        if not isinstance(item, Mapping):
            raise ValueError("source contract raw-video inventory entry is invalid")
        take_uid = str(item.get("take_uid", ""))
        if not take_uid or take_uid in inventory_takes:
            raise ValueError("source contract raw-video inventory has a duplicate/missing take")
        inventory_takes.add(take_uid)
        for view in ("ego", "exo"):
            video_path = Path(str(item.get(f"{view}_path", "")))
            digest = str(item.get(f"{view}_sha256", ""))
            if not video_path.is_file() or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("source contract raw-video inventory path/hash is invalid")
    if inventory_takes != audited_takes:
        raise ValueError("source contract raw-video inventory does not cover the audited takes")


def _validate_source_contract_recursive(
    contract_path: Path,
    expected_dir: Path,
    *,
    seen: set[Path] | None = None,
) -> tuple[dict, dict[str, np.ndarray], dict, np.ndarray, dict]:
    resolved_contract = contract_path.resolve()
    seen = set() if seen is None else seen
    if resolved_contract in seen:
        raise ValueError("source contract parent cycle detected")
    seen.add(resolved_contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    directory = expected_dir.resolve()
    arrays, files, frame_indices, frame_identity = _source_snapshot(directory)
    _validate_materialization_report(directory, files, frame_identity)
    audited_ids = [str(value) for value in contract.get("audited_sample_ids", [])]
    if (
        contract.get("schema") != "fact-npy-source-contract-v1"
        or Path(str(contract.get("source_dir", ""))).resolve() != directory
        or int(contract.get("rows", -1)) != len(arrays["sample_id"])
        or contract.get("color_space") != "RGB"
        or float(contract.get("transition_seconds", -1.0)) != 0.5
        or contract.get("endpoint_semantics") != ["t", "t+0.5s"]
        or contract.get("files") != files
        or contract.get("frame_index") != frame_identity
        or not audited_ids
        or len(audited_ids) != len(set(audited_ids))
        or contract.get("audited_sample_ids_sha256") != _sample_id_digest(audited_ids)
    ):
        raise ValueError("fact-npy-source-contract-v1 does not bind the current source")
    source_ids = {as_text(value) for value in arrays["sample_id"]}
    if not set(audited_ids).issubset(source_ids):
        raise ValueError("source contract audited IDs are not a subset of the source")
    evidence = contract.get("producer_evidence", {})
    mode = evidence.get("mode")
    if mode == "semantic_audit_against_raw_videos":
        _validate_raw_semantic_evidence(
            evidence, contract, directory, arrays, files, frame_indices, frame_identity
        )
    elif mode == "exact_parent_row_subset":
        parent_path = Path(str(evidence.get("parent_contract", "")))
        row_index_path = Path(str(evidence.get("source_row_index", "")))
        if (
            not parent_path.is_file()
            or sha256_file(parent_path) != evidence.get("parent_contract_sha256")
            or not row_index_path.is_file()
            or sha256_file(row_index_path) != evidence.get("source_row_index_sha256")
        ):
            raise ValueError("source subset parent/index evidence hash mismatch")
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_dir = Path(str(parent_payload.get("source_dir", "")))
        parent, parent_arrays, _, parent_frames, _ = _validate_source_contract_recursive(
            parent_path, parent_dir, seen=seen
        )
        indices_raw = np.load(row_index_path, mmap_mode="r", allow_pickle=False)
        if (
            indices_raw.ndim != 1
            or not np.issubdtype(indices_raw.dtype, np.integer)
            or evidence.get("source_row_index_dtype") != str(indices_raw.dtype)
        ):
            raise ValueError("source subset row index must retain an integer dtype")
        indices = np.asarray(indices_raw, dtype=np.int64)
        if (
            len(indices) != len(arrays["sample_id"])
            or len(np.unique(indices)) != len(indices)
            or (indices < 0).any()
            or (indices >= len(parent_arrays["sample_id"])).any()
        ):
            raise ValueError("source subset row index is not a valid one-to-one mapping")
        for name in CORE_SOURCE_ARRAYS:
            if (
                arrays[name].dtype != parent_arrays[name].dtype
                or arrays[name].shape[1:] != parent_arrays[name].shape[1:]
            ):
                raise ValueError(f"source subset {name}.npy dtype/row shape differs from its parent")
            for start in range(0, len(indices), 64):
                selection = indices[start : start + 64]
                if not np.array_equal(
                    np.asarray(arrays[name][start : start + len(selection)]),
                    np.asarray(parent_arrays[name][selection]),
                ):
                    raise ValueError(f"source subset {name}.npy differs from its parent rows")
        if not np.array_equal(frame_indices, parent_frames[indices]):
            raise ValueError("source subset frame_index.npy differs from its parent rows")
        expected_audited = sorted(set(parent.get("audited_sample_ids", [])) & source_ids)
        if sorted(audited_ids) != expected_audited:
            raise ValueError("source subset audited IDs do not equal the inherited audited rows")
        if evidence.get("inherited_producer_evidence") != parent.get("producer_evidence"):
            raise ValueError("source subset inherited producer evidence was edited")
        report = json.loads((directory / "materialization_report.json").read_text(encoding="utf-8"))
        if report.get("schema") == "fact-npy-transition-subset-v1" and (
            Path(str(report.get("parent_dir", ""))).resolve() != parent_dir.resolve()
            or report.get("parent_report_sha256")
            != sha256_file(parent_dir / "materialization_report.json")
            or Path(str(report.get("row_index_npy", ""))).resolve() != row_index_path.resolve()
            or report.get("row_index_sha256") != sha256_file(row_index_path)
            or report.get("row_index_dtype", str(indices_raw.dtype)) != str(indices_raw.dtype)
        ):
            raise ValueError("subset materialization report does not bind its parent/index")
    else:
        raise ValueError("source contract has no supported producer evidence")
    seen.remove(resolved_contract)
    return contract, arrays, files, frame_indices, frame_identity


def _validate_short73_isolation(
    input_dir: Path,
    records: Sequence,
    report: Mapping[str, Any],
    files: Mapping[str, Any],
) -> None:
    if (
        report.get("freeze_stage") != "final"
        or report.get("evaluation_allowed") is not True
        or report.get("final_inference_only") is not True
        or int(report.get("samples", -1)) != 584
        or int(report.get("takes", -1)) != 73
    ):
        raise ValueError("short73 formal input is not a final inference-only freeze")
    isolation = report.get("locked_isolation", {})
    freeze_path = Path(str(isolation.get("source_freeze", "")))
    sample_path = Path(str(isolation.get("samples", "")))
    selected_path = Path(str(isolation.get("selected_takes", "")))
    manifest_path = Path(str(isolation.get("effect_manifest", "")))
    for path, key in (
        (freeze_path, "source_freeze_sha256"),
        (sample_path, "samples_sha256"),
        (selected_path, "selected_takes_sha256"),
        (manifest_path, "effect_manifest_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != isolation.get(key):
            raise ValueError(f"short73 locked isolation hash mismatch: {key}")
    if isolation.get("source_freeze_sha256") != report.get("source_freeze_sha256"):
        raise ValueError("short73 source freeze binding differs across the report")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("freeze_stage") != "final"
        or freeze.get("evaluation_allowed") is not True
        or freeze.get("final_inference_only") is not True
        or freeze.get("training_valid") is not False
        or freeze.get("model_selection_valid") is not False
        or freeze.get("audit", {}).get("passed") is not True
        or freeze.get("samples_sha256") != isolation.get("samples_sha256")
        or freeze.get("selected_takes_sha256") != isolation.get("selected_takes_sha256")
    ):
        raise ValueError("short73 source freeze no longer proves locked isolation")
    role = np.load(input_dir / "role.npy", mmap_mode="r", allow_pickle=False)
    training_valid = np.load(input_dir / "training_valid.npy", mmap_mode="r", allow_pickle=False)
    if (
        isolation.get("role") != npy_identity(input_dir / "role.npy")
        or isolation.get("training_valid") != npy_identity(input_dir / "training_valid.npy")
        or len(role) != len(records)
        or len(training_valid) != len(records)
        or any(as_text(value) != "locked_test" for value in role)
        or training_valid.dtype != np.bool_
        or bool(np.asarray(training_valid).any())
    ):
        raise ValueError("short73 role/training_valid sidecars violate locked isolation")
    if report.get("effect_manifest_sha256") != isolation.get("effect_manifest_sha256"):
        raise ValueError("short73 effect manifest is not consistently hash-bound")
    frozen_records = read_manifest_jsonl(manifest_path)
    if list(records) != frozen_records:
        raise ValueError("short73 supplied manifest differs from the freeze-bound effect manifest")
    for index, record in enumerate(records):
        if (
            record.row_index != index
            or record.split != "locked_test"
            or record.training_valid is not False
            or record.sample_id != as_text(np.load(input_dir / "sample_id.npy", mmap_mode="r", allow_pickle=False)[index])
        ):
            raise ValueError("short73 effect manifest contains a trainable or misaligned row")
    pnn = freeze.get("audit", {}).get("perceptual_nearest_neighbor", {})
    reports = pnn.get("reports", [])
    candidates: dict[str, str] = {}
    if pnn.get("status") != "complete" or not reports:
        raise ValueError("short73 final freeze lacks completed PNN evidence")
    for item in reports:
        view = str(item.get("view", ""))
        digest = str(item.get("candidate_sha256", ""))
        if (
            view not in {"ego", "exo"}
            or item.get("passed") is not True
            or int(item.get("candidate_count", -1)) != 584
            or item.get("violations") not in ([], None)
            or (view in candidates and candidates[view] != digest)
        ):
            raise ValueError("short73 PNN evidence is invalid")
        candidates[view] = digest
    candidate_ids = str(
        freeze.get("audit", {}).get("provisional_evidence", {}).get(
            "candidate_sample_ids_sha256", ""
        )
    )
    candidates["sample_id"] = candidate_ids
    if set(candidates) != {"ego", "exo", "sample_id"} or candidates != isolation.get(
        "pnn_candidate_sha256"
    ):
        raise ValueError("short73 PNN candidate identity differs from locked isolation")
    for name in ("ego", "exo", "sample_id"):
        if files[name]["sha256"] != candidates[name]:
            raise ValueError("short73 pixels/IDs differ from the PNN-audited candidate arrays")


def validate_formal_input_contract(
    input_dir: Path,
    records: Sequence,
    transition_seconds: float,
    source_contract: Path,
) -> dict:
    if transition_seconds != 0.5:
        raise ValueError("formal v7 targets require exactly 0.5-second transitions")
    if source_contract is None:
        raise ValueError("formal target input requires --source-contract")
    contract, arrays, files, frame_indices, frame_identity = _validate_source_contract_recursive(
        source_contract, input_dir
    )
    report_path, report = _validate_materialization_report(input_dir, files, frame_identity)
    if len(frame_indices) != len(records):
        raise ValueError("formal target frame_index.npy length differs from manifest")
    sample_ids = arrays["sample_id"]
    take_uids = arrays["take_uid"]
    timestamps = arrays["timestamp"]
    for index, record in enumerate(records):
        if (
            record.row_index != index
            or record.sample_id != as_text(sample_ids[index])
            or record.take_uid != as_text(take_uids[index])
            or abs(float(record.timestamp or 0.0) - float(timestamps[index])) > 1e-3
        ):
            raise ValueError(f"manifest/source row contract differs at index {index}")
    if report.get("schema") == "fact-short73-materialization-v3":
        _validate_short73_isolation(input_dir, records, report, files)
    return {
        "materialization_report": str(report_path.resolve()),
        "materialization_report_sha256": sha256_file(report_path),
        "schema": report["schema"],
        "files": files,
        "frame_index": frame_identity,
        "source_contract": str(source_contract.resolve()),
        "source_contract_sha256": sha256_file(source_contract),
        "source_contract_schema": contract["schema"],
        "producer_evidence": contract["producer_evidence"],
    }


def video_tensor(sample: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(sample).copy())
    if value.ndim != 4:
        raise ValueError(f"video sample must be TxHxWxC or TxCxHxW, got {tuple(value.shape)}")
    if value.shape[-1] in (1, 3):
        value = value.permute(0, 3, 1, 2)
    elif value.shape[1] not in (1, 3):
        raise ValueError(f"cannot infer video channels from {tuple(value.shape)}")
    value = value.float()
    if value.numel() and value.max() > 1.5:
        value /= 255.0
    value = F.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)
    return value.unsqueeze(0).to(device)


def source_image_size(sample: np.ndarray) -> tuple[int, int]:
    if sample.ndim != 4:
        raise ValueError("source video sample must be 4D")
    if sample.shape[-1] in (1, 3):
        return int(sample.shape[1]), int(sample.shape[2])
    if sample.shape[1] in (1, 3):
        return int(sample.shape[2]), int(sample.shape[3])
    raise ValueError(f"cannot infer source image size from {sample.shape}")


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tree_code_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dino_repository_provenance(torch_home: str | None) -> dict[str, Any]:
    hub_dir = Path(torch_home).expanduser() / "hub" if torch_home else Path(torch.hub.get_dir())
    repository = (hub_dir / "facebookresearch_dinov2_main").resolve()
    result: dict[str, Any] = {"resolved_repo": str(repository), "exists": repository.is_dir()}
    if not repository.is_dir():
        return result
    result["python_code_sha256"] = _tree_code_sha256(repository)
    if (repository / ".git").exists():
        for key, command in (
            ("git_head", ["git", "rev-parse", "HEAD"]),
            ("git_tree", ["git", "rev-parse", "HEAD^{tree}"]),
        ):
            completed = subprocess.run(
                command,
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            result[key] = completed.stdout.strip()
    return result


def deterministic_runtime_provenance(device: torch.device) -> dict[str, Any]:
    opencv_version = None
    for distribution in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
        try:
            opencv_version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": np.__version__,
        "opencv": opencv_version,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


class CameraSidecar:
    def __init__(self, directory: Path, count: int) -> None:
        self.directory = directory
        self.intrinsics = load_mmap(directory / "intrinsics.npy", count)
        self.world_to_camera = load_mmap(directory / "world_to_camera.npy", count)
        valid_path = directory / "valid.npy"
        self.valid = load_mmap(valid_path, count) if valid_path.is_file() else np.ones(count, dtype=bool)
        source_size_path = directory / "source_size.npy"
        self.source_size = load_mmap(source_size_path, count) if source_size_path.is_file() else None

    @staticmethod
    def _endpoint(array: np.ndarray, row: int, endpoint: int) -> np.ndarray:
        value = np.asarray(array[row])
        if value.ndim >= 3 and value.shape[0] == 2:
            return value[endpoint]
        return value

    def pair(self, row: int, convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        validity = np.asarray(self.valid[row])
        if not bool(validity.all()):
            return None
        k0 = self._endpoint(self.intrinsics, row, 0)
        k1 = self._endpoint(self.intrinsics, row, 1)
        e0 = self._endpoint(self.world_to_camera, row, 0)
        e1 = self._endpoint(self.world_to_camera, row, 1)
        if convention == "camera_to_world":
            e0, e1 = invert_extrinsic(e0), invert_extrinsic(e1)
        return k0, k1, e0, e1

    def image_size(self, row: int, fallback: tuple[int, int]) -> tuple[int, int]:
        if self.source_size is None:
            return fallback
        value = np.asarray(self.source_size[row]).reshape(-1)
        if len(value) != 2:
            raise ValueError(f"source_size row must contain height,width, got {value}")
        return int(value[0]), int(value[1])


def load_weak_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in index:
                raise ValueError(f"invalid/duplicate weak sample_id at {path}:{line_number}")
            index[sample_id] = row
    return index


def load_weak_calibration(path: Path | None, weak_index_path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if weak_index_path is None:
        raise ValueError("--weak-calibration-report requires --weak-index-jsonl")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "fact-weak-semantic-calibration-v1":
        raise ValueError("unsupported weak calibration report schema")
    if report.get("mapping_version") != WEAK_VERB_MAP_VERSION:
        raise ValueError("weak calibration mapping version differs from the target builder")
    if int(report.get("dev_sample_count", -1)) != 60:
        raise ValueError("weak calibration must bind exactly 60 frozen dev samples")
    if report.get("weak_index_sha256") != sha256_file(weak_index_path):
        raise ValueError("weak calibration report does not bind the supplied weak index")
    minimum = float(report.get("minimum_precision", 0.80))
    measured = float(report.get("minimum_measured_precision", -1.0))
    expected_gate = measured >= minimum
    if bool(report.get("gate_passed")) != expected_gate:
        raise ValueError("weak calibration gate flag is inconsistent with its measured precision")
    return report


def feature_extractor(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.feature_backend == "mock":
        if not args.allow_smoke_targets:
            raise ValueError("mock features require --allow-smoke-targets and must never be used for formal runs")
        torch.manual_seed(0)
        model = MockPatchFeatureExtractor(3, args.mock_feature_dim, 14)
        provenance = {"backend": "mock", "seed": 0, "formal_target": False}
    else:
        model = DINOv2PatchFeatureExtractor(args.dino_model, args.torch_home)
        provenance = {
            "backend": "dinov2",
            "model": args.dino_model,
            "torch_home": args.torch_home,
            "repository": dino_repository_provenance(args.torch_home),
            "formal_target": True,
        }
    return model.eval().to(device), provenance


def _mask_at(array: np.ndarray, row: int, endpoint: int = 0) -> np.ndarray:
    value = np.asarray(array[row])
    if value.ndim == 3 and value.shape[0] == 2:
        value = value[endpoint]
    if value.ndim != 2:
        raise ValueError(f"object mask must be NxHxW or Nx2xHxW, got row {value.shape}")
    return value.astype(bool)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=(size, size), mode="nearest")[0, 0].bool().numpy()


def main() -> None:
    args = parse_args()
    for target in args.target:
        require_supported_target(target)
    if args.zero_flow_smoke and not args.allow_smoke_targets:
        raise ValueError("--zero-flow-smoke requires --allow-smoke-targets")
    if not args.zero_flow_smoke and args.raft_weights is None:
        raise ValueError("formal target building requires --raft-weights")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    records = read_manifest_jsonl(args.manifest)
    count = len(records)
    if args.sample_index_contract is not None and args.sample_index_npy is None:
        raise ValueError("--sample-index-contract requires --sample-index-npy")
    manifest_indices: np.ndarray | range
    selection_provenance = None
    if args.sample_index_npy is not None:
        if args.start != 0 or args.limit is not None:
            raise ValueError("sample-index selection cannot be combined with --start/--limit")
        if not args.allow_smoke_targets and args.sample_index_contract is None:
            raise ValueError("formal sample-index targets require --sample-index-contract")
        loaded_indices = np.load(args.sample_index_npy, mmap_mode="r", allow_pickle=False)
        if loaded_indices.ndim != 1 or not np.issubdtype(loaded_indices.dtype, np.integer):
            raise ValueError("sample-index NPY must be a one-dimensional integer manifest-row vector")
        manifest_indices = np.asarray(loaded_indices, dtype=np.int64)
        if len(set(manifest_indices.tolist())) != len(manifest_indices):
            raise ValueError("sample-index NPY contains duplicates")
        if (manifest_indices < 0).any() or (manifest_indices >= count).any():
            raise ValueError("sample-index NPY contains an out-of-range manifest row")
        selection_provenance = {
            "sample_index_npy": str(args.sample_index_npy.resolve()),
            "sample_index_sha256": sha256_file(args.sample_index_npy),
            "rows": len(manifest_indices),
        }
        if args.sample_index_contract is not None:
            selection_provenance = validate_audit_selection_contract(
                records,
                manifest_path=args.manifest,
                index_path=args.sample_index_npy,
                contract_path=args.sample_index_contract,
            )
    else:
        stop = count if args.limit is None else min(count, args.start + args.limit)
        manifest_indices = range(args.start, stop)
    input_contract = (
        None
        if args.allow_smoke_targets
        else validate_formal_input_contract(
            args.input_dir, records, args.transition_seconds, args.source_contract
        )
    )
    arrays = {view: load_mmap(args.input_dir / f"{view}.npy", count) for view in args.views}
    cameras = {view: CameraSidecar(path, count) for view, path in args.camera_sidecar}
    masks = {view: load_mmap(path, count) for view, path in args.object_mask_npy}
    mask_offsets = {
        view: load_mmap(path, count) for view, path in args.object_mask_anchor_offset_npy
    }
    mask_anchor_images = {
        view: load_mmap(path, count) for view, path in args.object_mask_anchor_image_npy
    }
    unknown = (set(cameras) | set(masks) | set(mask_offsets) | set(mask_anchor_images)) - set(args.views)
    if unknown:
        raise ValueError(f"sidecars supplied for unknown views: {sorted(unknown)}")
    if set(mask_offsets) - set(masks) or set(mask_anchor_images) - set(masks):
        raise ValueError("Relations anchor sidecars require a matching --object-mask-npy view")
    if masks and not args.allow_smoke_targets and set(mask_offsets) != set(masks):
        raise ValueError("formal Relations targets require an explicit anchor-offset sidecar for every mask view")
    if not args.allow_smoke_targets:
        for view, offsets in mask_offsets.items():
            values = np.asarray(offsets)
            nearby_nonzero = (values != 0) & (np.abs(values.astype(np.int64)) <= 15)
            if nearby_nonzero.any() and view not in mask_anchor_images:
                raise ValueError(
                    f"formal Relations view {view!r} has nonzero anchor offsets but no anchor-image sidecar"
                )
    weak_index = load_weak_index(args.weak_index_jsonl)
    weak_calibration = load_weak_calibration(args.weak_calibration_report, args.weak_index_jsonl)
    weak_calibration_sha256 = (
        sha256_file(args.weak_calibration_report) if args.weak_calibration_report else None
    )
    sparse = {}
    for name, value_path, valid_path, joint_valid_path in (
        (
            "body_joint_3d_delta",
            args.body_delta_npy,
            args.body_valid_npy,
            args.body_joint_valid_npy,
        ),
        (
            "hand_joint_3d_delta",
            args.hand_delta_npy,
            args.hand_valid_npy,
            args.hand_joint_valid_npy,
        ),
    ):
        supplied = (value_path is not None, valid_path is not None, joint_valid_path is not None)
        if any(supplied) and not all(supplied):
            raise ValueError(f"{name} needs value, sample-valid, and joint-valid NPYs")
        if value_path is not None:
            sparse[name] = (
                load_mmap(value_path, count),
                load_mmap(valid_path, count),
                load_mmap(joint_valid_path, count),
            )
    config = EffectTargetConfig(
        transition_seconds=args.transition_seconds,
        output_height=args.output_size,
        output_width=args.output_size,
        target_cache_version=("fact-effect-target-v1-smoke" if args.allow_smoke_targets else "fact-effect-target-v1"),
    )
    dino, dino_provenance = feature_extractor(args, device)
    dino_provenance["state_sha256"] = module_state_sha256(dino)
    raft = None if args.zero_flow_smoke else RAFTFlowEstimator(args.raft_weights, device)
    build_provenance = {
        "dino": dino_provenance,
        "raft": (
            {"backend": "zero_flow_smoke", "formal_target": False}
            if args.zero_flow_smoke
            else {"path": str(args.raft_weights), "sha256": sha256_file(args.raft_weights), "formal_target": True}
        ),
        "pose_convention": args.pose_convention,
        "claim": "pose-conditioned camera-compensated 2D observed interaction transition representation",
        "depth_available": False,
        "flow_3d_available": False,
        "runtime": deterministic_runtime_provenance(device),
        "input_transition_contract": input_contract,
        "selection_contract": selection_provenance,
        "source_hashes": {
            "builder_code": sha256_file(Path(__file__)),
            "effect_targets_code": sha256_file(ROOT / "fact_tokenizer" / "effect_targets.py"),
            "model_code": sha256_file(ROOT / "fact_tokenizer" / "model.py"),
            "manifest": sha256_file(args.manifest),
            "views": {
                view: (
                    input_contract["files"][view]["sha256"]
                    if input_contract is not None
                    else sha256_file(args.input_dir / f"{view}.npy")
                )
                for view in args.views
            },
            "camera_sidecars": {
                view: {
                    str(path.name): sha256_file(path)
                    for path in sorted(directory.glob("*.npy"))
                }
                for view, directory in args.camera_sidecar
            },
            "object_masks": {
                view: sha256_file(path) for view, path in args.object_mask_npy
            },
            "object_mask_anchor_offsets": {
                view: sha256_file(path) for view, path in args.object_mask_anchor_offset_npy
            },
            "object_mask_anchor_images": {
                view: sha256_file(path) for view, path in args.object_mask_anchor_image_npy
            },
            "weak_index": (
                sha256_file(args.weak_index_jsonl) if args.weak_index_jsonl is not None else None
            ),
            "weak_calibration": (
                weak_calibration_sha256
            ),
            "body_delta": sha256_file(args.body_delta_npy) if args.body_delta_npy else None,
            "body_valid": sha256_file(args.body_valid_npy) if args.body_valid_npy else None,
            "body_joint_valid": (
                sha256_file(args.body_joint_valid_npy) if args.body_joint_valid_npy else None
            ),
            "hand_delta": sha256_file(args.hand_delta_npy) if args.hand_delta_npy else None,
            "hand_valid": sha256_file(args.hand_valid_npy) if args.hand_valid_npy else None,
            "hand_joint_valid": (
                sha256_file(args.hand_joint_valid_npy) if args.hand_joint_valid_npy else None
            ),
        },
    }
    cache_identity = {
        key: value for key, value in build_provenance.items() if key != "selection_contract"
    }
    normalized_identity = json.loads(json.dumps(cache_identity, sort_keys=True))
    expected_identity_sha256 = hashlib.sha256(
        json.dumps(
            {
                "config_sha256": config.fingerprint(),
                "sources": normalized_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    pre_gate_audit_cache = bool(
        selection_provenance
        and selection_provenance.get("schema") == "fact-target-audit-selection-v1"
    )
    formal_gate_required = not args.allow_smoke_targets and not pre_gate_audit_cache
    formal_release = None
    if formal_gate_required:
        if args.visual_audit_gate is None:
            raise ValueError("formal target cache or shard requires --visual-audit-gate")
        gate = validate_visual_audit_gate(
            args.visual_audit_gate,
            expected_identity_sha256,
        )
        formal_release = {
            "visual_audit_gate": gate["_gate_path"],
            "visual_audit_gate_sha256": gate["_gate_sha256"],
            "visual_audit_gate_schema": gate["schema"],
            "review_rows": int(gate["rows"]),
            "pass_fraction": float(gate["pass_fraction"]),
            "target_identity_sha256": gate["target_identity_sha256"],
            "review_csv_sha256": gate["review_csv_sha256"],
            "audit_pack_sha256": gate["audit_pack_sha256"],
        }
    writer = EffectTargetCacheWriter(
        args.output_dir,
        config,
        resume=args.resume,
        identity=cache_identity,
    )
    if writer.identity_sha256 != expected_identity_sha256:
        raise AssertionError("target cache identity calculation drifted")
    build_provenance["target_identity_sha256"] = writer.identity_sha256
    build_provenance["formal_release"] = formal_release
    (args.output_dir / "builder_provenance.json").write_text(
        json.dumps(build_provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for manifest_index in manifest_indices:
        manifest_index = int(manifest_index)
        record = records[manifest_index]
        if record.row_index != manifest_index:
            raise ValueError(
                f"manifest row {manifest_index} points to NPY row {record.row_index}; split manifests must remain aligned"
            )
        if writer.contains(record.sample_id):
            continue
        targets: dict[str, Any] = {"depth_valid": False, "flow_3d_valid": False}
        metadata: dict[str, Any] = {
            "sample_key": record.sample_key,
            "take_uid": record.take_uid,
            "timestamp": record.timestamp,
            "split": record.split,
            "row_index": record.row_index,
            "annotation_refs": dict(record.annotation_refs),
            "capability_validity": {key.value: bool(value) for key, value in record.capability_validity.items()},
        }
        weak = weak_index.get(record.sample_id, {})
        weak_gate = bool(weak_calibration and weak_calibration["gate_passed"])
        transition_start = float(record.timestamp or 0.0)
        transition_end = transition_start + config.transition_seconds
        aligned_atomic = align_atomic_descriptions(
            weak.get("atomic_descriptions", []),
            transition_start,
            transition_end,
            config.atomic_midpoint_tolerance_seconds,
        )
        aligned_phase = align_phase_segments(
            weak.get("phase_segments", []), transition_start, transition_end
        )
        weak_labels = deterministic_weak_semantic_labels(aligned_atomic, aligned_phase)
        metadata["weak_semantics"] = {
            "atomic": aligned_atomic,
            "phase_segments": aligned_phase,
            "deterministic_labels": weak_labels,
            "raw_source": weak.get("source"),
            "calibration_report_sha256": weak_calibration_sha256,
            "weak_loss_gate_passed": weak_gate,
        }
        for view, array in arrays.items():
            source_sample = np.asarray(array[record.row_index])
            frames = video_tensor(source_sample, args.output_size, device)
            with torch.inference_mode():
                features = dino(frames)
            delta = dino_delta(features[:, 0], features[:, -1]).cpu()
            targets[f"{view}_full_dino_delta"] = delta[0].numpy().astype(np.float32)
            targets[f"{view}_full_dino_delta_valid"] = True
            targets[f"{view}_phase"] = np.int64(weak_labels["phase"])
            targets[f"{view}_phase_valid"] = bool(weak_gate and weak_labels["phase_valid"])
            targets[f"{view}_contact"] = np.int64(weak_labels["contact"])
            targets[f"{view}_contact_valid"] = bool(weak_gate and weak_labels["contact_valid"])

            object_mask = None
            if view in masks:
                object_mask = _resize_mask(_mask_at(masks[view], record.row_index), args.output_size)
            targets[f"{view}_object_roi_dino_delta"] = np.zeros(delta.shape[-1], dtype=np.float32)
            targets[f"{view}_object_roi_dino_delta_valid"] = False
            targets[f"{view}_relations_mask_t0"] = np.zeros(
                (args.output_size, args.output_size), dtype=bool
            )
            targets[f"{view}_relations_mask_t1"] = np.zeros(
                (args.output_size, args.output_size), dtype=bool
            )
            targets[f"{view}_relations_mask_valid"] = False

            pose_pair = cameras[view].pair(record.row_index, args.pose_convention) if view in cameras else None
            pose_valid = pose_pair is not None and record.has_capability(EffectCapability.CAMERA_POSE)
            needs_endpoint_flow = pose_valid or object_mask is not None
            forward = backward = None
            if needs_endpoint_flow:
                if args.zero_flow_smoke:
                    forward = np.zeros((args.output_size, args.output_size, 2), dtype=np.float32)
                    backward = np.zeros_like(forward)
                else:
                    with torch.inference_mode():
                        forward = np.asarray(raft(frames[:, 0], frames[:, -1])[0])
                        backward = np.asarray(raft(frames[:, -1], frames[:, 0])[0])

            if object_mask is not None and object_mask.any():
                anchor_offset = int(np.asarray(mask_offsets[view][record.row_index]).item()) if view in mask_offsets else 0
                mask_t0 = mask_t1 = None
                mask_valid = False
                if anchor_offset == 0 and forward is not None and backward is not None:
                    propagated = propagate_relation_mask(
                        object_mask,
                        forward,
                        backward,
                        config.flow_fb_threshold_px,
                    )
                    mask_t0 = propagated["mask_t0"]
                    mask_t1 = propagated["mask_t1"]
                    mask_valid = bool(propagated["valid"])
                elif view in mask_anchor_images and not args.zero_flow_smoke:
                    anchor_sample = np.asarray(mask_anchor_images[view][record.row_index])
                    if anchor_sample.ndim == 3:
                        anchor_sample = anchor_sample[None]
                    anchor = video_tensor(anchor_sample, args.output_size, device)
                    with torch.inference_mode():
                        anchor_to_t0 = np.asarray(raft(anchor[:, 0], frames[:, 0])[0])
                        t0_to_anchor = np.asarray(raft(frames[:, 0], anchor[:, 0])[0])
                        anchor_to_t1 = np.asarray(raft(anchor[:, 0], frames[:, -1])[0])
                        t1_to_anchor = np.asarray(raft(frames[:, -1], anchor[:, 0])[0])
                    propagated_t0 = propagate_relation_mask(
                        object_mask,
                        anchor_to_t0,
                        t0_to_anchor,
                        config.flow_fb_threshold_px,
                    )
                    propagated_t1 = propagate_relation_mask(
                        object_mask,
                        anchor_to_t1,
                        t1_to_anchor,
                        config.flow_fb_threshold_px,
                    )
                    mask_t0 = propagated_t0["mask_t1"]
                    mask_t1 = propagated_t1["mask_t1"]
                    mask_valid = bool(propagated_t0["valid"] and propagated_t1["valid"])
                else:
                    metadata.setdefault("relations_invalid_reasons", {})[view] = (
                        "nonzero_anchor_offset_requires_anchor_image_and_real_raft"
                    )
                if mask_valid and mask_t0 is not None and mask_t1 is not None:
                    targets[f"{view}_relations_mask_t0"] = mask_t0
                    targets[f"{view}_relations_mask_t1"] = mask_t1
                    targets[f"{view}_relations_mask_valid"] = True
                    pooled, roi_valid = roi_feature_delta(
                        delta,
                        torch.from_numpy(np.asarray(mask_t0, dtype=bool))[None],
                    )
                    targets[f"{view}_object_roi_dino_delta"] = pooled[0].numpy().astype(np.float32)
                    targets[f"{view}_object_roi_dino_delta_valid"] = bool(roi_valid[0])

            if not pose_valid:
                targets[f"{view}_rotation_compensated_flow_2d"] = np.zeros(
                    (args.output_size, args.output_size, 2), dtype=np.float32
                )
                targets[f"{view}_rotation_compensated_flow_2d_valid"] = np.zeros(
                    (args.output_size, args.output_size), dtype=bool
                )
                continue
            assert forward is not None and backward is not None
            _, fb_valid = forward_backward_consistency(forward, backward, config.flow_fb_threshold_px)
            k0, k1, e0, e1 = pose_pair
            source_size = cameras[view].image_size(record.row_index, source_image_size(source_sample))
            k0 = scale_intrinsics(k0, source_size, (args.output_size, args.output_size))
            k1 = scale_intrinsics(k1, source_size, (args.output_size, args.output_size))
            rotational_h = relative_rotation_homography(k0, k1, e0, e1)
            propagated_background_mask = targets[f"{view}_relations_mask_t0"]
            background = (
                None
                if not bool(targets[f"{view}_relations_mask_valid"])
                else ~np.asarray(propagated_background_mask, dtype=bool)
            )
            compensated = rotation_compensated_flow_2d(
                forward,
                rotational_h,
                valid_mask=fb_valid,
                background_mask=background,
                ransac_threshold_px=config.background_ransac_threshold_px,
                prefer_opencv=not args.zero_flow_smoke,
            )
            geometry_valid = compensated["valid"]
            targets[f"{view}_rotation_compensated_flow_2d"] = compensated[
                "rotation_compensated_flow_2d"
            ]
            targets[f"{view}_rotation_compensated_flow_2d_valid"] = geometry_valid
            targets[f"{view}_rotational_homography"] = rotational_h
            targets[f"{view}_background_homography"] = compensated["background_homography"]
        for name, (values, sample_validity, joint_validity) in sparse.items():
            sample_valid = bool(np.asarray(sample_validity[record.row_index]).item())
            joint_valid = np.asarray(joint_validity[record.row_index], dtype=bool)
            delta_value = np.asarray(values[record.row_index]).copy()
            if delta_value.shape[:-1] != joint_valid.shape:
                raise ValueError(f"{name} joint validity shape does not match delta shape")
            delta_value[~joint_valid] = 0
            validity_stem = name.removesuffix("_delta")
            targets[f"ego_{name}"] = delta_value
            targets[f"ego_{validity_stem}_joint_valid"] = joint_valid
            targets[f"ego_{validity_stem}_valid"] = bool(sample_valid and joint_valid.any())
        writer.write(record.sample_id, targets, metadata)
    manifest_path = writer.finalize()
    if formal_release is not None:
        config_path = args.output_dir / "target_config.json"
        released_config = json.loads(config_path.read_text(encoding="utf-8"))
        released_config["formal_release"] = formal_release
        config_path.write_text(
            json.dumps(released_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote/resumed target cache: {manifest_path}")


if __name__ == "__main__":
    main()
