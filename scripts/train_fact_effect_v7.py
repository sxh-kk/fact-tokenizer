#!/usr/bin/env python3
"""Train preregistered C0-T2 or P0-P4 continuous FACT v7 experiments."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_data import EffectClipSpec, FACTEffectNPYDataset  # noqa: E402
from fact_tokenizer.effect_experiments import (  # noqa: E402
    BRIDGE_EXPERIMENTS,
    FINAL_CONTRACT,
    PAIRED_CONTROLS,
    SCREEN_CONTRACT,
    SMOKE_STEPS,
    EffectTargetCache,
    PairedControlDataset,
    make_experiment_model,
    validate_batch_contract,
)
from fact_tokenizer.effect_losses import FACTV7ObjectiveConfig, compute_fact_v7_objective  # noqa: E402
from fact_tokenizer.model import DINOv2PatchFeatureExtractor, MockPatchFeatureExtractor  # noqa: E402
from fact_tokenizer.paired_eligibility import load_paired_control_eligibility  # noqa: E402
from fact_tokenizer.target_audit import validate_visual_audit_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[*BRIDGE_EXPERIMENTS, *PAIRED_CONTROLS], required=True)
    parser.add_argument("--stage", choices=["smoke", "screen", "final"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quality-sampling", choices=["uniform", "weighted"], default="uniform")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wrong-phase-index", type=Path)
    parser.add_argument(
        "--paired-eligibility",
        type=Path,
        help="Frozen common P0/P1/P2/P3/P4 eligibility NPZ.",
    )
    parser.add_argument(
        "--excluded-takes",
        type=Path,
        help="Frozen newline/JSON list of gold probe-train takes excluded from representation learning.",
    )
    parser.add_argument(
        "--gold-freeze",
        type=Path,
        help="gold300_freeze.json binding the 47-take exclusion to the frozen 140-row split.",
    )
    parser.add_argument("--extra-ego-input-dir", type=Path)
    parser.add_argument("--extra-ego-manifest", type=Path)
    parser.add_argument("--extra-ego-target-cache", type=Path)
    parser.add_argument(
        "--exo-target-quality-provenance",
        type=Path,
        help="Frozen P1 report proving Exo was used only to build targets/quality metadata.",
    )
    parser.add_argument("--take-quality-report", type=Path)
    parser.add_argument("--camera-context", action="append", default=[], metavar="VIEW=PATH")
    parser.add_argument("--backbone", choices=["dinov2", "mock"], default="dinov2")
    parser.add_argument("--dino-model", default="dinov2_vitb14_reg")
    parser.add_argument("--torch-home")
    parser.add_argument("--backbone-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--semantic-dim", type=int, default=128)
    parser.add_argument("--private-dim", type=int, default=64)
    parser.add_argument("--enable-weak-semantics", action="store_true")
    parser.add_argument("--weak-calibration-report", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tree_python_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_dino_repository(provenance: Mapping[str, Any]) -> None:
    repository = Path(str(provenance.get("resolved_repo", "")))
    if provenance.get("exists") is not True or not repository.is_dir():
        raise ValueError("formal target cache does not bind an available DINO implementation")
    if provenance.get("python_code_sha256") != tree_python_sha256(repository):
        raise ValueError("formal target DINO repository code has changed")
    for field, ref in (("git_head", "HEAD"), ("git_tree", "HEAD^{tree}")):
        expected = provenance.get(field)
        if expected is None:
            continue
        actual = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != expected:
            raise ValueError(f"formal target DINO repository {field} has changed")


def validate_formal_target_cache(
    cache: EffectTargetCache,
    *,
    input_dir: Path,
    manifest: Path,
    view_names: tuple[str, ...],
    dino_model: str,
) -> str:
    config = cache.config
    if config.get("target_cache_version") != "fact-effect-target-v1":
        raise ValueError("formal training refuses smoke or unknown target-cache versions")
    identity = config.get("identity")
    if not isinstance(identity, dict) or config.get("identity_sha256") is None:
        raise ValueError("formal target cache lacks a frozen source/weight identity")
    dino = identity.get("dino", {})
    if (
        dino.get("backend") != "dinov2"
        or dino.get("model") != dino_model
        or dino.get("formal_target") is not True
        or len(str(dino.get("state_sha256", ""))) != 64
        or not isinstance(dino.get("repository"), dict)
    ):
        raise ValueError("formal target cache DINO identity differs from the training backbone contract")
    validate_dino_repository(dino["repository"])
    runtime = identity.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cudnn_deterministic") is not True
        or runtime.get("cudnn_benchmark") is not False
        or runtime.get("cudnn_allow_tf32") is not False
        or runtime.get("cuda_matmul_allow_tf32") is not False
        or runtime.get("cublas_workspace_config") not in {":4096:8", ":16:8"}
    ):
        raise ValueError("formal target cache lacks deterministic target-build provenance")
    sources = identity.get("source_hashes", {})
    expected_hashes = {
        "manifest": sha256_file(manifest),
        "builder_code": sha256_file(ROOT / "scripts" / "build_fact_effect_targets.py"),
        "effect_targets_code": sha256_file(ROOT / "fact_tokenizer" / "effect_targets.py"),
        "model_code": sha256_file(ROOT / "fact_tokenizer" / "model.py"),
    }
    for name, expected in expected_hashes.items():
        if sources.get(name) != expected:
            raise ValueError(f"formal target cache {name} hash differs from current training assets/code")
    view_hashes = sources.get("views", {})
    actual_file_hashes: dict[str, str] = {}
    for view in view_names:
        path = input_dir / f"{view}.npy"
        actual_file_hashes[view] = sha256_file(path)
        if view_hashes.get(view) != actual_file_hashes[view]:
            raise ValueError(f"formal target cache view {view!r} hash differs from training input")
    transition_contract = identity.get("input_transition_contract", {})
    report_path = input_dir / "materialization_report.json"
    frame_index_path = input_dir / "frame_index.npy"
    source_contract_path = Path(str(transition_contract.get("source_contract", "")))
    if (
        Path(transition_contract.get("materialization_report", "")).resolve()
        != report_path.resolve()
        or not report_path.is_file()
        or transition_contract.get("materialization_report_sha256") != sha256_file(report_path)
        or not frame_index_path.is_file()
        or transition_contract.get("source_contract_schema") != "fact-npy-source-contract-v1"
        or not source_contract_path.is_file()
        or transition_contract.get("source_contract_sha256") != sha256_file(source_contract_path)
    ):
        raise ValueError("formal target cache does not bind the active transition materialization")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("color_space") != "RGB"
        or float(report.get("transition_seconds", -1.0)) != 0.5
        or report.get("endpoint_semantics") != ["t", "t+0.5s"]
        or float(report.get("frame_rate_hz", -1.0)) != 30.0
        or int(report.get("endpoint_offset_frames", -1)) != 15
    ):
        raise ValueError("formal target cache transition materialization is not RGB t-to-t+0.5s")
    contract_files = transition_contract.get("files", {})
    for name in (*view_names, "sample_id", "take_uid", "timestamp"):
        path = input_dir / f"{name}.npy"
        digest = actual_file_hashes.get(name) or sha256_file(path)
        if contract_files.get(name, {}).get("sha256") != digest:
            raise ValueError(f"formal target transition contract differs for {name}.npy")
    frame_contract = transition_contract.get("frame_index", {})
    frame_indices = np.load(frame_index_path, mmap_mode="r", allow_pickle=False)
    if (
        frame_contract.get("sha256") != sha256_file(frame_index_path)
        or frame_contract.get("shape") != list(frame_indices.shape)
        or frame_contract.get("dtype") != str(frame_indices.dtype)
        or frame_indices.ndim != 1
        or not np.issubdtype(frame_indices.dtype, np.integer)
        or (frame_indices < 0).any()
    ):
        raise ValueError("formal target cache frame_index contract differs from training input")
    release = config.get("formal_release")
    if (
        not isinstance(release, dict)
        or release.get("visual_audit_gate_schema") != "fact-target-visual-audit-gate-v2"
        or int(release.get("review_rows", 0)) != 50
        or float(release.get("pass_fraction", 0.0)) < 0.90
        or release.get("target_identity_sha256") != config.get("identity_sha256")
        or len(str(release.get("review_csv_sha256", ""))) != 64
        or len(str(release.get("audit_pack_sha256", ""))) != 64
    ):
        raise ValueError("formal target cache has not been released by the 50-sample visual audit gate")
    gate_path = Path(str(release.get("visual_audit_gate", "")))
    gate = validate_visual_audit_gate(gate_path, str(config["identity_sha256"]))
    if (
        release.get("visual_audit_gate_sha256") != gate["_gate_sha256"]
        or str(gate_path.resolve()) != gate["_gate_path"]
        or int(gate.get("rows", 0)) != int(release["review_rows"])
        or float(gate.get("pass_fraction", 0.0)) != float(release["pass_fraction"])
        or gate.get("review_csv_sha256") != release["review_csv_sha256"]
        or gate.get("audit_pack_sha256") != release["audit_pack_sha256"]
    ):
        raise ValueError("formal target cache release differs from its live visual-audit evidence")
    return str(dino["state_sha256"])


def named_paths(entries: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--camera-context must use VIEW=PATH")
        view, path = entry.split("=", 1)
        if view in result:
            raise ValueError(f"duplicate camera context for {view}")
        result[view] = str(Path(path).resolve())
    return result


def load_excluded_takes(
    path: Path | None,
    *,
    formal: bool,
    gold_freeze: Path | None = None,
    training_manifest: Path | None = None,
) -> tuple[frozenset[str], dict[str, Any]]:
    if path is None:
        if formal:
            raise ValueError("formal screen/final runs require --excluded-takes with the 47 gold train takes")
        return frozenset(), {"path": None, "sha256": None, "take_count": 0}
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("take_uids", payload) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError("excluded-takes JSON must be a list or contain a take_uids list")
        raw = [str(value).strip() for value in values]
    else:
        raw = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(raw) != len(set(raw)):
        raise ValueError("excluded-takes contains duplicate take_uid values")
    takes = frozenset(raw)
    if formal and len(takes) != 47:
        raise ValueError(f"formal gold exclusion must contain exactly 47 takes, got {len(takes)}")
    fingerprint = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "take_count": len(takes),
    }
    if formal:
        if gold_freeze is None or training_manifest is None:
            raise ValueError("formal gold exclusion requires --gold-freeze and the training manifest")
        freeze = json.loads(gold_freeze.read_text(encoding="utf-8"))
        exclusion_contract = freeze.get("representation_train_excluded_takes", {})
        if (
            exclusion_contract.get("takes") != 47
            or exclusion_contract.get("sha256") != fingerprint["sha256"]
            or exclusion_contract.get("path") != path.name
        ):
            raise ValueError("excluded-takes file differs from the gold freeze contract")
        frozen_manifest = gold_freeze.parent / "gold300_frozen.jsonl"
        if sha256_file(frozen_manifest) != freeze.get("manifest_sha256"):
            raise ValueError("gold300 frozen manifest hash differs from gold freeze")
        frozen_probe_takes = set()
        frozen_probe_rows = 0
        with frozen_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("gold_split") == "probe_train":
                        frozen_probe_rows += 1
                        frozen_probe_takes.add(str(row["take_uid"]))
        if frozen_probe_rows != 140:
            raise ValueError(f"gold manifest must contain exactly 140 probe-train rows, got {frozen_probe_rows}")
        if frozen_probe_takes != set(takes):
            raise ValueError("excluded takes are not exactly the 47 frozen probe-train takes")
        manifest_takes = set()
        with training_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    manifest_takes.add(str(json.loads(line)["take_uid"]))
        missing = sorted(set(takes) - manifest_takes)
        if missing:
            raise ValueError(f"training manifest is missing excluded gold takes: {missing[:3]}")
        fingerprint.update(
            {
                "gold_freeze_sha256": sha256_file(gold_freeze),
                "gold_manifest_sha256": sha256_file(frozen_manifest),
            }
        )
    return takes, fingerprint


def require_control_assets(args: argparse.Namespace, *, formal: bool) -> None:
    p4_assets = (args.extra_ego_input_dir, args.extra_ego_manifest, args.extra_ego_target_cache)
    if any(value is not None for value in p4_assets) and not all(value is not None for value in p4_assets):
        raise ValueError("P4 extra-Ego control requires all three --extra-ego-* arguments")
    if args.experiment == "P4" and not all(value is not None for value in p4_assets):
        raise ValueError("P4 requires a separate extra-Ego input, manifest, and target cache")
    if args.experiment != "P4" and any(value is not None for value in p4_assets):
        raise ValueError("--extra-ego-* assets are valid only for P4")
    if args.experiment == "P1" and args.exo_target_quality_provenance is None:
        raise ValueError("P1 requires --exo-target-quality-provenance")
    if args.experiment != "P1" and args.exo_target_quality_provenance is not None:
        raise ValueError("--exo-target-quality-provenance is valid only for P1")
    if args.experiment == "P1":
        if args.take_quality_report is None or not args.take_quality_report.is_file():
            raise ValueError("P1 requires the active --take-quality-report")
        path = args.exo_target_quality_provenance
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "fact-p1-exo-target-quality-provenance-v1":
            raise ValueError("unsupported P1 Exo target/quality provenance schema")
        if payload.get("exo_model_input") is not False or payload.get("exo_optimizer_input") is not False:
            raise ValueError("P1 provenance must prove Exo is absent from model and optimizer inputs")
        roles = set(payload.get("exo_roles", []))
        if roles != {"target_build", "quality_stratification"}:
            raise ValueError("P1 Exo roles must be exactly target_build and quality_stratification")
        if not isinstance(payload.get("source_hashes"), dict) or not payload["source_hashes"]:
            raise ValueError("P1 provenance requires immutable Exo source hashes")


def recursive_collate(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    if isinstance(first, Mapping):
        keys = set(first)
        if any(set(item) != keys for item in items):
            raise ValueError("training batches require identical target keys; rebuild a canonical cache")
        return {key: recursive_collate([item[key] for item in items]) for key in sorted(keys)}
    if first is None:
        if any(item is not None for item in items):
            raise ValueError("mixed optional values in batch")
        return None
    if isinstance(first, (str, int, float, bool)):
        return items
    raise TypeError(f"cannot collate {type(first).__name__}")


class DeterministicDistributedWeightedSampler(Sampler[int]):
    """Replacement sampler with one deterministic global draw sharded by rank."""

    def __init__(
        self,
        weights: np.ndarray,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or not len(self.weights) or float(self.weights.sum()) <= 0:
            raise ValueError("quality sampling needs a non-empty vector with positive total weight")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def nuisance_inputs(model_inputs: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """Apply a calibrated +1px/-1px image-coordinate nuisance transform.

    The same transform is applied to every history frame, and the principal
    point in the flattened 3x3 intrinsics is updated coherently.  This avoids
    treating an uncalibrated wraparound RGB roll as a camera-pose perturbation.
    """

    result: dict[str, dict[str, torch.Tensor]] = {}
    for view, values in model_inputs.items():
        copied = dict(values)
        videos = values["videos"]
        # Content moves right by one pixel and up by one pixel; uncovered pixels
        # are zero rather than wrapped from the opposite image boundary.
        shifted = F.pad(videos, (1, 0, 0, 1))[..., 1 : videos.shape[-2] + 1, : videos.shape[-1]]
        copied["videos"] = shifted
        context = values.get("camera_context")
        if context is not None:
            if context.shape[-1] < 9:
                raise ValueError("camera nuisance transform requires flattened 3x3 intrinsics")
            updated = context.clone()
            updated[..., 2] += 1.0
            updated[..., 5] -= 1.0
            copied["camera_context"] = updated
        result[view] = copied
    return result


def objective_for_spec(name: str, weak_enabled: bool, weak_precision: float | None) -> FACTV7ObjectiveConfig:
    if name in {"C0", "C1", "C2"}:
        return FACTV7ObjectiveConfig(
            object_roi_dino_delta_weight=0,
            rotation_compensated_flow_2d_weight=0,
            cross_view_semantic_consistency_weight=0,
            pose_camera_nuisance_consistency_weight=0,
            weak_semantic_weight=0,
        )
    if name == "T1":
        return FACTV7ObjectiveConfig(
            object_roi_dino_delta_weight=0,
            weak_semantic_weight=0,
        )
    return FACTV7ObjectiveConfig(
        weak_semantics_enabled=weak_enabled,
        weak_semantic_dev_precision=weak_precision,
    )


def load_weak_gate(
    path: Path | None,
    *,
    enabled: bool,
    target_cache: EffectTargetCache,
    gold_freeze: Path | None,
) -> tuple[float | None, dict[str, Any] | None]:
    if not enabled:
        if path is not None:
            raise ValueError("--weak-calibration-report requires --enable-weak-semantics")
        return None, None
    if path is None:
        raise ValueError("weak semantics require a frozen --weak-calibration-report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if gold_freeze is None or report.get("gold_freeze_sha256") != sha256_file(gold_freeze):
        raise ValueError("weak calibration report is not bound to the active gold freeze")
    if report.get("schema") != "fact-weak-semantic-calibration-v1":
        raise ValueError("unsupported weak calibration report schema")
    if int(report.get("dev_sample_count", -1)) != 60 or not bool(report.get("gate_passed")):
        raise ValueError("weak semantic calibration has not passed on all 60 dev samples")
    measured = float(report.get("minimum_measured_precision", -1.0))
    if measured < 0.80:
        raise ValueError("weak semantic calibration precision is below 0.80")
    digest = sha256_file(path)
    cache_digest = (
        target_cache.config.get("identity", {})
        .get("source_hashes", {})
        .get("weak_calibration")
    )
    if cache_digest != digest:
        raise ValueError("target cache was not built with this frozen weak calibration report")
    return measured, {
        "path": str(path.resolve()),
        "sha256": digest,
        "mapping_version": report.get("mapping_version"),
        "dev_sample_count": 60,
        "minimum_measured_precision": measured,
    }


def load_control_eligibility(
    path: Path | None,
    *,
    experiment: str,
    formal: bool,
    manifest: Path,
    exclusion_sha256: str | None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None]:
    is_paired = experiment in PAIRED_CONTROLS
    if formal and is_paired and path is None:
        raise ValueError("formal paired controls require --paired-eligibility")
    if not is_paired and path is not None:
        raise ValueError("--paired-eligibility is valid only for P0-P4")
    if path is None:
        return None, None
    arrays = load_paired_control_eligibility(path)
    report_path = path.parent / "eligibility_report.json"
    if formal and not report_path.is_file():
        raise FileNotFoundError("formal paired eligibility requires sibling eligibility_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    if report:
        if report.get("manifest_sha256") != sha256_file(manifest):
            raise ValueError("paired eligibility was not frozen from the supplied training manifest")
        if report.get("excluded_takes_sha256") != exclusion_sha256:
            raise ValueError("paired eligibility was not frozen with the active gold take exclusion")
        expected = report.get("artifacts_sha256", {}).get(path.name)
        if expected != sha256_file(path):
            raise ValueError("paired eligibility artifact hash differs from its freeze report")
    fingerprint = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "eligible_samples": len(arrays["sample_id"]),
    }
    return arrays, fingerprint


def validate_p1_cache_binding(
    provenance_path: Path,
    quality_report_path: Path,
    target_cache_root: Path,
    target_cache: EffectTargetCache,
) -> None:
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_hashes = payload["source_hashes"]
    expected = {
        "target_config": sha256_file(target_cache_root / "target_config.json"),
        "target_manifest": sha256_file(target_cache_root / "target_manifest.jsonl"),
        "take_quality_report": sha256_file(quality_report_path),
    }
    for name, digest in expected.items():
        if source_hashes.get(name) != digest:
            raise ValueError(f"P1 provenance {name} hash differs from the active training assets")
    if payload.get("target_identity_sha256") != target_cache.config.get("identity_sha256"):
        raise ValueError("P1 provenance target identity differs from the active target cache")


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank


def main() -> None:
    args = parse_args()
    spec = BRIDGE_EXPERIMENTS.get(args.experiment) or PAIRED_CONTROLS[args.experiment]
    contract = {"screen": SCREEN_CONTRACT, "final": FINAL_CONTRACT}.get(args.stage)
    steps = args.steps or (SMOKE_STEPS if args.stage == "smoke" else contract.steps)
    if args.stage == "smoke":
        if steps > SMOKE_STEPS:
            raise ValueError(f"smoke runs are capped at {SMOKE_STEPS} steps")
    else:
        if steps != contract.steps or args.seed not in contract.seeds:
            raise ValueError(f"{args.stage} contract requires steps={contract.steps}, seeds={contract.seeds}")
        if args.backbone == "mock":
            raise ValueError("formal screen/final runs cannot use the mock backbone")
        if args.stage == "final" and args.experiment not in {"P0", "P2", "P3", "P4"}:
            raise ValueError("final contract is frozen to P0/P2/P3/P4")
    formal = args.stage in {"screen", "final"}
    require_control_assets(args, formal=formal)
    excluded_takes, exclusion_fingerprint = load_excluded_takes(
        args.excluded_takes,
        formal=formal,
        gold_freeze=args.gold_freeze,
        training_manifest=args.manifest,
    )
    if not formal and args.gold_freeze is not None:
        raise ValueError("--gold-freeze is required only for formal screen/final runs")
    eligibility, eligibility_fingerprint = load_control_eligibility(
        args.paired_eligibility,
        experiment=args.experiment,
        formal=formal,
        manifest=args.manifest,
        exclusion_sha256=exclusion_fingerprint["sha256"],
    )
    world_size, rank, local_rank = init_distributed()
    if args.stage != "smoke":
        validate_batch_contract(args.micro_batch, args.gradient_accumulation, world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    camera_contexts = named_paths(args.camera_context)
    if formal and spec.target_tier != "full_dino_delta" and set(camera_contexts) != set(spec.view_names):
        raise ValueError(
            "pose-conditioned formal runs require an explicit --camera-context for every active view"
        )
    base = FACTEffectNPYDataset(
        args.input_dir,
        EffectClipSpec(history_frames=1, future_frames=1, source_current_index=0, resize=224),
        view_keys=spec.view_names,
        manifest=args.manifest,
        role="train",
        drop_nontraining=True,
        camera_context_keys=camera_contexts,
        excluded_take_uids=excluded_takes,
        included_sample_ids=(eligibility["sample_id"].astype(str) if eligibility is not None else None),
    )
    if eligibility is not None and not np.array_equal(
        np.asarray(eligibility["sample_id"]).astype(str), np.asarray(base.sample_ids)
    ):
        raise ValueError("paired eligible allowlist order differs from the active dataset order")
    target_cache = EffectTargetCache(args.target_cache, verify_hashes=True)
    target_dino_state_sha256 = (
        validate_formal_target_cache(
            target_cache,
            input_dir=args.input_dir,
            manifest=args.manifest,
            view_names=spec.view_names,
            dino_model=args.dino_model,
        )
        if formal
        else None
    )
    if args.experiment == "P1":
        validate_p1_cache_binding(
            args.exo_target_quality_provenance,
            args.take_quality_report,
            args.target_cache,
            target_cache,
        )
    weak_precision, weak_gate_fingerprint = load_weak_gate(
        args.weak_calibration_report,
        enabled=args.enable_weak_semantics,
        target_cache=target_cache,
        gold_freeze=args.gold_freeze,
    )
    wrong_phase = None
    wrong_phase_labels = None
    if args.experiment == "P3":
        if eligibility is not None:
            if args.wrong_phase_index is not None:
                raise ValueError("P3 cannot combine frozen eligibility with a separate wrong-phase index")
            if not np.array_equal(np.asarray(eligibility["sample_id"]).astype(str), np.asarray(base.sample_ids)):
                raise ValueError("P3 eligible allowlist order differs from the active dataset order")
            wrong_phase = np.asarray(eligibility["donor_index"], dtype=np.int64)
            wrong_phase_labels = np.asarray(eligibility["phase_label"]).astype(str)
        else:
            if args.wrong_phase_index is None:
                raise ValueError("P3 requires --paired-eligibility or a smoke-only --wrong-phase-index")
            wrong_payload = np.load(args.wrong_phase_index, mmap_mode="r", allow_pickle=False)
            if isinstance(wrong_payload, np.lib.npyio.NpzFile):
                required = {"donor_index", "sample_id", "take_uid", "phase_label"}
                missing = required - set(wrong_payload.files)
                if missing:
                    raise ValueError(f"P3 frozen index is missing arrays: {sorted(missing)}")
                active_ids = np.asarray(base.sample_ids, dtype=str)
                active_takes = np.asarray(base.take_uids, dtype=str)
                if not np.array_equal(np.asarray(wrong_payload["sample_id"]).astype(str), active_ids):
                    raise ValueError("P3 frozen sample_id order does not match the quarantined training dataset")
                if not np.array_equal(np.asarray(wrong_payload["take_uid"]).astype(str), active_takes):
                    raise ValueError("P3 frozen take_uid order does not match the quarantined training dataset")
                wrong_phase = np.asarray(wrong_payload["donor_index"], dtype=np.int64)
                wrong_phase_labels = np.asarray(wrong_payload["phase_label"]).astype(str)
                wrong_payload.close()
            else:
                wrong_phase = wrong_payload
    dataset = PairedControlDataset(
        base,
        target_cache,
        spec,
        wrong_phase_index=wrong_phase,
        phase_labels=wrong_phase_labels,
    )
    if not len(dataset):
        raise ValueError("no training samples remain after diagnostic/gold quarantine")
    if formal:
        first_targets = dataset[0]["targets"]
        for view in spec.view_names:
            target = first_targets[view].get("full_dino_delta")
            if target is None or int(target.shape[-1]) != args.backbone_dim:
                raise ValueError(
                    f"formal {view} DINO target dimension does not match backbone_dim={args.backbone_dim}"
                )

    extra_dataset = None
    extra_target_cache = None
    extra_base = None
    if args.experiment == "P4":
        extra_base = FACTEffectNPYDataset(
            args.extra_ego_input_dir,
            EffectClipSpec(history_frames=1, future_frames=1, source_current_index=0, resize=224),
            view_keys=("ego",),
            manifest=args.extra_ego_manifest,
            role="train",
            drop_nontraining=True,
            excluded_take_uids=excluded_takes,
        )
        extra_target_cache = EffectTargetCache(args.extra_ego_target_cache, verify_hashes=True)
        if formal:
            extra_dino_sha256 = validate_formal_target_cache(
                extra_target_cache,
                input_dir=args.extra_ego_input_dir,
                manifest=args.extra_ego_manifest,
                view_names=("ego",),
                dino_model=args.dino_model,
            )
            if extra_dino_sha256 != target_dino_state_sha256:
                raise ValueError("P4 primary and extra target caches use different DINO states")
        extra_dataset = PairedControlDataset(extra_base, extra_target_cache, spec)
        if not len(extra_dataset):
            raise ValueError("P4 extra-Ego dataset is empty after quarantine")
        sample_overlap = sorted(set(base.sample_ids) & set(extra_base.sample_ids))
        if sample_overlap:
            raise ValueError(
                f"P4 extra Ego must have zero primary sample overlap; found {sample_overlap[:3]}"
            )
        if formal and len(extra_dataset) < len(dataset):
            raise ValueError(
                "formal P4 extra-Ego pool must contain at least as many samples as the primary pool"
            )
        if formal:
            extra_target = extra_dataset[0]["targets"]["ego"].get("full_dino_delta")
            if extra_target is None or int(extra_target.shape[-1]) != args.backbone_dim:
                raise ValueError("P4 extra-Ego DINO target dimension differs from the backbone")
    if args.quality_sampling == "weighted":
        sampler = DeterministicDistributedWeightedSampler(
            base.quality_weights,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) >= args.micro_batch,
        collate_fn=recursive_collate,
        persistent_workers=args.num_workers > 0,
    )
    extra_sampler = None
    extra_loader = None
    if extra_dataset is not None:
        if args.quality_sampling == "weighted":
            extra_sampler = DeterministicDistributedWeightedSampler(
                extra_base.quality_weights,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed + 10_000,
            )
        else:
            extra_sampler = (
                DistributedSampler(extra_dataset, shuffle=True, seed=args.seed + 10_000)
                if world_size > 1
                else None
            )
        extra_loader = DataLoader(
            extra_dataset,
            batch_size=args.micro_batch,
            sampler=extra_sampler,
            shuffle=extra_sampler is None,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=len(extra_dataset) >= args.micro_batch,
            collate_fn=recursive_collate,
            persistent_workers=args.num_workers > 0,
        )

    if args.backbone == "dinov2":
        backbone = DINOv2PatchFeatureExtractor(args.dino_model, args.torch_home)
        if formal and module_state_sha256(backbone) != target_dino_state_sha256:
            raise ValueError("training DINO state hash differs from the formal target cache")
    else:
        if args.stage != "smoke":
            raise ValueError("mock backbone is smoke-only")
        args.backbone_dim = min(args.backbone_dim, 64)
        backbone = MockPatchFeatureExtractor(3, args.backbone_dim, 14)
    first = dataset[0]
    context_dims = {
        view: int(first["model_inputs"][view].get("camera_context", torch.empty(0)).numel())
        for view in spec.view_names
    }
    nonzero_context_dims = {value for value in context_dims.values() if value}
    if len(nonzero_context_dims) > 1 or (nonzero_context_dims and any(value == 0 for value in context_dims.values())):
        raise ValueError(f"camera context dimensions differ/missing across views: {context_dims}")
    camera_context_dim = next(iter(nonzero_context_dims), 0)
    if extra_dataset is not None:
        extra_first = extra_dataset[0]
        extra_context_dim = int(
            extra_first["model_inputs"]["ego"].get("camera_context", torch.empty(0)).numel()
        )
        if extra_context_dim != camera_context_dim:
            raise ValueError(
                "P4 primary and extra-Ego camera context dimensions must match: "
                f"{camera_context_dim} != {extra_context_dim}"
            )
    if spec.target_tier != "full_dino_delta" and args.stage != "smoke" and camera_context_dim == 0:
        raise ValueError("pose-conditioned formal runs require current camera context for every active view")
    model = make_experiment_model(
        spec,
        backbone=backbone,
        backbone_dim=args.backbone_dim,
        hidden_dim=args.hidden_dim,
        semantic_dim=args.semantic_dim,
        private_dim=args.private_dim,
        camera_context_dim=camera_context_dim,
        current_index=0,
        vq_enabled=False,
        geometry_mode="image2d",
    ).to(device)
    if any("vq" in key.lower() or "codebook" in key.lower() for key in model.state_dict()):
        raise AssertionError("continuous v7 state dict unexpectedly contains VQ/codebook state")
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank]) if world_size > 1 else model
    objective_config = objective_for_spec(args.experiment, args.enable_weak_semantics, weak_precision)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_fingerprint = {
        "experiment": args.experiment,
        "stage": args.stage,
        "steps": steps,
        "seed": args.seed,
        "global_batch": args.micro_batch * args.gradient_accumulation * world_size,
        "world_size": world_size,
        "gradient_accumulation": args.gradient_accumulation,
        "quality_sampling": args.quality_sampling,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "step_unit": "optimizer_update",
        "manifest_sha256": sha256_file(args.manifest),
        "target_manifest_sha256": sha256_file(args.target_cache / "target_manifest.jsonl"),
        "target_config_sha256": target_cache.config["config_sha256"],
        "target_identity_sha256": target_cache.config.get("identity_sha256"),
        "gold_take_exclusion": {
            **exclusion_fingerprint,
            "excluded_sample_count": base.excluded_sample_count,
        },
        "paired_eligibility": eligibility_fingerprint,
        "camera_context_sha256": {
            view: sha256_file(Path(path)) for view, path in sorted(camera_contexts.items())
        },
        "wrong_phase_index_sha256": (
            sha256_file(args.wrong_phase_index) if args.wrong_phase_index is not None else None
        ),
        "p1_exo_target_quality_provenance_sha256": (
            sha256_file(args.exo_target_quality_provenance)
            if args.exo_target_quality_provenance is not None
            else None
        ),
        "take_quality_report_sha256": (
            sha256_file(args.take_quality_report) if args.take_quality_report is not None else None
        ),
        "p4_extra_ego": (
            {
                "manifest_sha256": sha256_file(args.extra_ego_manifest),
                "target_manifest_sha256": sha256_file(
                    args.extra_ego_target_cache / "target_manifest.jsonl"
                ),
                "target_config_sha256": extra_target_cache.config["config_sha256"],
                "sample_count": len(extra_dataset),
                "excluded_sample_count": extra_dataset.base.excluded_sample_count,
                "camera_context_sha256": (
                    sha256_file(args.extra_ego_input_dir / "ego_camera_context.npy")
                    if (args.extra_ego_input_dir / "ego_camera_context.npy").is_file()
                    else None
                ),
                "compute_contract": "one_primary_plus_one_extra_ego_forward_per_micro_batch",
            }
            if extra_dataset is not None
            else None
        ),
        "objective": vars(objective_config),
        "weak_calibration": weak_gate_fingerprint,
        "spec": vars(spec),
        "model": {
            "backbone": args.backbone,
            "dino_model": args.dino_model,
            "backbone_dim": args.backbone_dim,
            "hidden_dim": args.hidden_dim,
            "semantic_dim": args.semantic_dim,
            "private_dim": args.private_dim,
            "camera_context_dim": camera_context_dim,
            "view_names": list(spec.view_names),
        },
        "vq_enabled": False,
        "gold_backprop": False,
    }
    shared_control_contract = {
        key: run_fingerprint[key]
        for key in (
            "stage",
            "steps",
            "global_batch",
            "world_size",
            "gradient_accumulation",
            "step_unit",
            "learning_rate",
            "weight_decay",
            "quality_sampling",
            "manifest_sha256",
            "target_manifest_sha256",
            "target_config_sha256",
            "target_identity_sha256",
            "gold_take_exclusion",
            "paired_eligibility",
            "objective",
            "weak_calibration",
        )
    }
    shared_control_contract["model"] = {
        key: run_fingerprint["model"][key]
        for key in (
            "backbone",
            "dino_model",
            "backbone_dim",
            "hidden_dim",
            "semantic_dim",
            "private_dim",
            "camera_context_dim",
        )
    }
    run_fingerprint["shared_control_contract"] = shared_control_contract
    run_fingerprint["shared_control_contract_sha256"] = hashlib.sha256(
        json.dumps(shared_control_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_fingerprint = json.loads(json.dumps(run_fingerprint, sort_keys=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_fingerprint:
            raise ValueError("run directory already contains a different frozen configuration")
    else:
        config_path.write_text(json.dumps(run_fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint["run_fingerprint"] != run_fingerprint:
            raise ValueError("checkpoint run fingerprint differs from the frozen run config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        expected_cursor = start_step * args.gradient_accumulation
        if int(checkpoint.get("data_cursor_micro_batches", -1)) != expected_cursor:
            raise ValueError("checkpoint data cursor is missing or inconsistent with optimizer step")
    log_path = args.output_dir / "train_metrics.jsonl"

    def position_iterator(
        active_loader: DataLoader,
        active_sampler: Any,
        consumed_batches: int,
    ) -> tuple[Any, int]:
        batches_per_epoch = len(active_loader)
        if batches_per_epoch <= 0:
            raise ValueError("training DataLoader has no batches")
        if active_sampler is not None and hasattr(active_sampler, "set_epoch"):
            epoch, offset = divmod(consumed_batches, batches_per_epoch)
            active_sampler.set_epoch(epoch)
            active_iterator = iter(active_loader)
            for _ in range(offset):
                next(active_iterator)
            return active_iterator, epoch
        # Smoke-only single-process fallback: replay iterator creation so every
        # RandomSampler permutation is consumed exactly as in an uninterrupted run.
        active_iterator = iter(active_loader)
        epoch = 0
        for _ in range(consumed_batches):
            try:
                next(active_iterator)
            except StopIteration:
                epoch += 1
                active_iterator = iter(active_loader)
                next(active_iterator)
        return active_iterator, epoch

    consumed_at_resume = start_step * args.gradient_accumulation
    iterator, loader_epoch = position_iterator(loader, sampler, consumed_at_resume)
    if extra_loader is not None:
        extra_iterator, extra_loader_epoch = position_iterator(
            extra_loader,
            extra_sampler,
            consumed_at_resume,
        )
    else:
        extra_iterator, extra_loader_epoch = None, 0

    def next_batch(
        active_loader: DataLoader,
        active_iterator: Any,
        active_sampler: Any,
        epoch: int,
    ) -> tuple[Any, Any, int]:
        try:
            return next(active_iterator), active_iterator, epoch
        except StopIteration:
            epoch += 1
            if active_sampler is not None:
                active_sampler.set_epoch(epoch)
            active_iterator = iter(active_loader)
            return next(active_iterator), active_iterator, epoch

    def forward_losses(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        model_inputs = move_to_device(batch["model_inputs"], device)
        targets = move_to_device(batch["targets"], device)
        output = ddp_model(model_inputs)
        nuisance = None
        if objective_config.pose_camera_nuisance_consistency_weight > 0:
            nuisance = ddp_model(nuisance_inputs(model_inputs))
        return compute_fact_v7_objective(
            output,
            targets,
            nuisance_output=nuisance,
            config=objective_config,
        )

    for optimizer_step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for accumulation_index in range(args.gradient_accumulation):
            batch, iterator, loader_epoch = next_batch(loader, iterator, sampler, loader_epoch)
            synchronize = accumulation_index + 1 == args.gradient_accumulation
            sync_context = (
                nullcontext()
                if synchronize or not isinstance(ddp_model, DistributedDataParallel)
                else ddp_model.no_sync()
            )
            with sync_context:
                losses = forward_losses(batch)
                if extra_loader is not None:
                    extra_batch, extra_iterator, extra_loader_epoch = next_batch(
                        extra_loader,
                        extra_iterator,
                        extra_sampler,
                        extra_loader_epoch,
                    )
                    extra_losses = forward_losses(extra_batch)
                    losses = {
                        name: 0.5 * (losses[name] + extra_losses[name])
                        for name in losses
                    }
                (losses["total"] / args.gradient_accumulation).backward()
            for name, value in losses.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value.detach())
        optimizer.step()
        completed_step = optimizer_step + 1
        averaged = {
            name: value / args.gradient_accumulation for name, value in accumulated.items()
        }
        if rank == 0 and (completed_step % args.log_every == 0 or optimizer_step == start_step):
            row = {
                "step": completed_step,
                "step_unit": "optimizer_update",
                "data_cursor_micro_batches": completed_step * args.gradient_accumulation,
                "micro_batches_consumed": completed_step * args.gradient_accumulation,
                **averaged,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if rank == 0 and (completed_step % args.checkpoint_every == 0 or completed_step == steps):
            checkpoint = {
                "step": completed_step,
                "step_unit": "optimizer_update",
                "data_cursor_micro_batches": completed_step * args.gradient_accumulation,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "run_fingerprint": run_fingerprint,
            }
            temp = args.output_dir / "checkpoint.tmp.pt"
            torch.save(checkpoint, temp)
            temp.replace(args.output_dir / f"checkpoint_{completed_step:06d}.pt")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
