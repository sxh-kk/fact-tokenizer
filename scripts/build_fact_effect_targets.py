#!/usr/bin/env python3
"""Build versioned DINO/RAFT/camera-compensated FACT v7 effect targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


def validate_formal_input_contract(
    input_dir: Path,
    records: Sequence,
    transition_seconds: float,
) -> dict:
    if transition_seconds != 0.5:
        raise ValueError("formal v7 targets require exactly 0.5-second transitions")
    report_path = input_dir / "materialization_report.json"
    frame_index_path = input_dir / "frame_index.npy"
    if not report_path.is_file() or not frame_index_path.is_file():
        raise ValueError("formal target input requires materialization_report.json and frame_index.npy")
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
    ):
        raise ValueError("formal target input has an invalid transition materialization report")
    names = ("ego", "exo", "sample_id", "take_uid", "timestamp")
    files = {name: npy_identity(input_dir / f"{name}.npy") for name in names}
    if report.get("files") != files:
        raise ValueError("formal target input arrays no longer match the materialization report")
    frame_identity = npy_identity(frame_index_path)
    reported_frame = report.get("frame_index", {})
    if any(reported_frame.get(key) != value for key, value in frame_identity.items()):
        raise ValueError("formal target frame index no longer matches the materialization report")
    frame_indices = np.load(frame_index_path, mmap_mode="r", allow_pickle=False)
    if (
        frame_indices.ndim != 1
        or len(frame_indices) != len(records)
        or not np.issubdtype(frame_indices.dtype, np.integer)
        or (frame_indices < 0).any()
    ):
        raise ValueError("formal target frame_index.npy is invalid")
    sample_ids = np.load(input_dir / "sample_id.npy", mmap_mode="r", allow_pickle=False)
    take_uids = np.load(input_dir / "take_uid.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(input_dir / "timestamp.npy", mmap_mode="r", allow_pickle=False)
    for index, record in enumerate(records):
        if (
            record.row_index != index
            or record.sample_id != as_text(sample_ids[index])
            or record.take_uid != as_text(take_uids[index])
            or abs(float(record.timestamp or 0.0) - float(timestamps[index])) > 1e-3
        ):
            raise ValueError(f"manifest/source row contract differs at index {index}")
    return {
        "materialization_report": str(report_path.resolve()),
        "materialization_report_sha256": sha256_file(report_path),
        "schema": report["schema"],
        "files": files,
        "frame_index": frame_identity,
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
        provenance = {"backend": "dinov2", "model": args.dino_model, "torch_home": args.torch_home, "formal_target": True}
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
        if loaded_indices.ndim != 1:
            raise ValueError("sample-index NPY must be a one-dimensional manifest-row vector")
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
            selection = json.loads(args.sample_index_contract.read_text(encoding="utf-8"))
            selected_ids = [records[int(index)].sample_id for index in manifest_indices]
            if (
                selection.get("schema") != "fact-target-audit-selection-v1"
                or selection.get("index_sha256") != selection_provenance["sample_index_sha256"]
                or selection.get("manifest_sha256") != sha256_file(args.manifest)
                or int(selection.get("sample_count", -1)) != len(manifest_indices)
                or [str(value) for value in selection.get("sample_ids", [])] != selected_ids
            ):
                raise ValueError("sample-index contract does not bind this manifest selection")
            selection_provenance.update(
                {
                    "selection_contract": str(args.sample_index_contract.resolve()),
                    "selection_contract_sha256": sha256_file(args.sample_index_contract),
                }
            )
    else:
        stop = count if args.limit is None else min(count, args.start + args.limit)
        manifest_indices = range(args.start, stop)
    input_contract = (
        None
        if args.allow_smoke_targets
        else validate_formal_input_contract(args.input_dir, records, args.transition_seconds)
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
        "input_transition_contract": input_contract,
        "selection_contract": selection_provenance,
        "source_hashes": {
            "builder_code": sha256_file(Path(__file__)),
            "effect_targets_code": sha256_file(ROOT / "fact_tokenizer" / "effect_targets.py"),
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
    writer = EffectTargetCacheWriter(
        args.output_dir,
        config,
        resume=args.resume,
        identity=build_provenance,
    )
    full_formal_cache = (
        not args.allow_smoke_targets
        and args.sample_index_npy is None
        and args.start == 0
        and args.limit is None
    )
    formal_release = None
    if full_formal_cache:
        if args.visual_audit_gate is None:
            raise ValueError("full formal target cache requires --visual-audit-gate")
        gate = json.loads(args.visual_audit_gate.read_text(encoding="utf-8"))
        if (
            gate.get("passed") is not True
            or int(gate.get("rows", 0)) != 50
            or int(gate.get("fully_aligned", 0)) < 45
            or float(gate.get("minimum_pass_fraction", 0.0)) < 0.90
            or float(gate.get("pass_fraction", 0.0)) < 0.90
        ):
            raise ValueError("visual audit gate must contain 50 reviewed rows and passed=true")
        if gate.get("target_identity_sha256") != writer.identity_sha256:
            raise ValueError("visual audit gate was produced from a different target source/weight identity")
        formal_release = {
            "visual_audit_gate_sha256": sha256_file(args.visual_audit_gate),
            "review_rows": int(gate["rows"]),
            "pass_fraction": float(gate["pass_fraction"]),
            "target_identity_sha256": gate["target_identity_sha256"],
        }
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
