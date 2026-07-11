"""Preregistered bridge and paired-control contracts for FACT v7 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from fact_tokenizer.effect_model import BridgeContinuousEffectModel, ContinuousEffectModel
from fact_tokenizer.effect_targets import verify_target_cache


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    private_future_visible: bool
    cross_uses_private: bool
    vq_enabled: bool
    target_tier: str
    paired_control: Optional[str] = None
    view_names: tuple[str, ...] = ("ego", "exo")
    exo_role: str = "privileged_cross_view"


BRIDGE_EXPERIMENTS: dict[str, ExperimentSpec] = {
    "C0": ExperimentSpec("C0", True, True, False, "full_dino_delta"),
    "C1": ExperimentSpec("C1", False, True, False, "full_dino_delta"),
    "C2": ExperimentSpec("C2", False, False, False, "full_dino_delta"),
    "T1": ExperimentSpec("T1", False, False, False, "pose_conditioned_2d"),
    "T2": ExperimentSpec("T2", False, False, False, "pose_2d_roi_weak"),
}

PAIRED_CONTROLS: dict[str, ExperimentSpec] = {
    "P0": ExperimentSpec("P0", False, False, False, "pose_2d_roi_weak", "P0", ("ego",), "absent"),
    "P1": ExperimentSpec("P1", False, False, False, "pose_2d_roi_weak", "P1", ("ego",), "target_quality_only"),
    "P2": ExperimentSpec("P2", False, False, False, "pose_2d_roi_weak", "P2", ("ego", "exo"), "synchronized"),
    "P3": ExperimentSpec("P3", False, False, False, "pose_2d_roi_weak", "P3", ("ego", "exo"), "same_take_wrong_phase"),
    "P4": ExperimentSpec("P4", False, False, False, "pose_2d_roi_weak", "P4", ("ego",), "absent_compute_matched"),
}


@dataclass(frozen=True)
class RunContract:
    stage: str
    steps: int
    seeds: tuple[int, ...]
    global_batch: int = 32
    gpu_count: int = 2


SCREEN_CONTRACT = RunContract("screen", 5_000, (42, 43))
FINAL_CONTRACT = RunContract("final", 20_000, (42, 43, 44))
SMOKE_STEPS = 100


def validate_batch_contract(micro_batch: int, gradient_accumulation: int, world_size: int) -> None:
    if world_size != 2:
        raise ValueError("paired-value runs are preregistered with exactly 2 GPUs per run")
    effective = micro_batch * gradient_accumulation * world_size
    if effective != 32:
        raise ValueError(f"global batch must be 32, got {effective}")


def make_experiment_model(spec: ExperimentSpec, **model_kwargs: Any) -> ContinuousEffectModel:
    if spec.vq_enabled:
        raise ValueError("v7 Stage-1A experiment registry cannot enable VQ")
    model_kwargs = dict(model_kwargs)
    model_kwargs["view_names"] = spec.view_names
    if spec.name in {"C0", "C1"}:
        return BridgeContinuousEffectModel(
            private_future_visible=spec.private_future_visible,
            cross_uses_private=spec.cross_uses_private,
            **model_kwargs,
        )
    if spec.private_future_visible or spec.cross_uses_private:
        raise ValueError("shortcut flags are allowed only in the explicit C0/C1 bridge ablations")
    return ContinuousEffectModel(**model_kwargs)


def build_same_take_wrong_phase_index(
    take_uids: Sequence[str],
    phase_labels: Sequence[str | int],
    timestamps: Sequence[float],
    *,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically choose a same-take donor with a different known phase."""

    if not (len(take_uids) == len(phase_labels) == len(timestamps)):
        raise ValueError("take, phase and timestamp arrays must have equal length")
    by_take: dict[str, list[int]] = {}
    for index, take_uid in enumerate(take_uids):
        by_take.setdefault(str(take_uid), []).append(index)
    donors = np.full(len(take_uids), -1, dtype=np.int64)
    for index, (take_uid, phase) in enumerate(zip(take_uids, phase_labels)):
        candidates = [
            donor
            for donor in by_take[str(take_uid)]
            if str(phase_labels[donor]) != str(phase)
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda donor: (
                -abs(float(timestamps[donor]) - float(timestamps[index])),
                hashlib.sha256(f"{seed}:{index}:{donor}".encode()).hexdigest(),
            )
        )
        donors[index] = candidates[0]
    return donors


class EffectTargetCache:
    """Hash-verified, pickle-free reader for per-sample target NPZ files."""

    def __init__(self, root: Path | str, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        config_path = self.root / "target_config.json"
        manifest_path = self.root / "target_manifest.jsonl"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.records: dict[str, dict[str, Any]] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    sample_id = str(record["sample_id"])
                    if sample_id in self.records:
                        raise ValueError(f"duplicate target sample_id={sample_id}")
                    self.records[sample_id] = record
        if verify_hashes:
            errors = verify_target_cache(self.root)
            if errors:
                raise ValueError(f"target cache integrity failure: {errors[:3]}")

    def load(self, sample_id: str, view_names: Sequence[str]) -> dict[str, dict[str, torch.Tensor]]:
        if sample_id not in self.records:
            raise KeyError(f"target cache has no sample_id={sample_id}")
        path = self.root / self.records[sample_id]["path"]
        nested: dict[str, dict[str, torch.Tensor]] = {view: {} for view in view_names}
        with np.load(path, allow_pickle=False) as payload:
            validity_keys = [
                key
                for key in payload.files
                if key.lower().endswith("depth_valid") or key.lower().endswith("flow_3d_valid")
            ]
            if "depth_valid" not in validity_keys or "flow_3d_valid" not in validity_keys:
                raise ValueError("current v7 cache requires explicit depth_valid=false and flow_3d_valid=false")
            if any(bool(np.asarray(payload[key]).item()) for key in validity_keys):
                raise ValueError("current v7 cache must keep every depth_valid/flow_3d_valid flag false")
            for key in payload.files:
                for view in view_names:
                    prefix = f"{view}_"
                    if key.startswith(prefix):
                        value = np.asarray(payload[key]).copy()
                        nested[view][key[len(prefix) :]] = torch.from_numpy(value)
                        break
        return nested


class PairedControlDataset(Dataset):
    """Join mmap RGB and target cache while implementing P0-P4 view controls."""

    def __init__(
        self,
        base: Dataset,
        target_cache: EffectTargetCache,
        spec: ExperimentSpec,
        *,
        wrong_phase_index: Optional[Sequence[int]] = None,
        phase_labels: Optional[Sequence[str | int]] = None,
    ) -> None:
        if spec.paired_control is not None and spec.paired_control not in PAIRED_CONTROLS:
            raise ValueError("unknown paired-control experiment")
        self.base = base
        self.target_cache = target_cache
        self.spec = spec
        sample_ids = (
            tuple(base.sample_ids)
            if hasattr(base, "sample_ids")
            else tuple(str(base[index]["sample_id"]) for index in range(len(base)))
        )
        missing_targets = [sample_id for sample_id in sample_ids if sample_id not in target_cache.records]
        if missing_targets:
            raise ValueError(
                f"target cache is missing {len(missing_targets)} active samples, including {missing_targets[:3]}"
            )
        self.wrong_phase_index = None if wrong_phase_index is None else np.asarray(wrong_phase_index, dtype=np.int64)
        self.phase_labels = None if phase_labels is None else np.asarray(phase_labels).astype(str)
        if spec.paired_control == "P3":
            if self.wrong_phase_index is None or len(self.wrong_phase_index) != len(base):
                raise ValueError("P3 requires one same-take wrong-phase donor index per sample")
            if (self.wrong_phase_index < 0).any():
                raise ValueError("P3 cannot run until every sample has a valid different-phase donor")
            if (self.wrong_phase_index >= len(base)).any():
                raise ValueError("P3 donor index is outside the eligible base dataset")
            if self.phase_labels is not None and len(self.phase_labels) != len(base):
                raise ValueError("P3 phase label vector must align one-to-one with the base dataset")
            take_uids = (
                tuple(base.take_uids)
                if hasattr(base, "take_uids")
                else tuple(str(base[index]["take_uid"]) for index in range(len(base)))
            )
            for index, donor in enumerate(self.wrong_phase_index):
                if take_uids[index] != take_uids[int(donor)]:
                    raise ValueError("P3 donor must come from the same take as its anchor")
                if self.phase_labels is not None and self.phase_labels[index] == self.phase_labels[int(donor)]:
                    raise ValueError("P3 donor must have a different frozen phase label")

    def __len__(self) -> int:
        return len(self.base)

    def _model_view(self, item: Mapping[str, Any], view: str) -> dict[str, torch.Tensor]:
        value = {"videos": item["views"][view]}
        context = item.get("camera_contexts", {}).get(view)
        if context is not None:
            value["camera_context"] = context
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = self.base[index]
        sample_id = str(anchor["sample_id"])
        targets = self.target_cache.load(sample_id, self.spec.view_names)
        model_inputs = {view: self._model_view(anchor, view) for view in self.spec.view_names}
        donor_sample_id = None
        if self.spec.paired_control == "P3":
            donor = self.base[int(self.wrong_phase_index[index])]
            donor_sample_id = str(donor["sample_id"])
            model_inputs["exo"] = self._model_view(donor, "exo")
            targets["exo"] = self.target_cache.load(donor_sample_id, ("exo",))["exo"]
        return {
            "model_inputs": model_inputs,
            "targets": targets,
            "sample_id": sample_id,
            "donor_sample_id": donor_sample_id,
            "take_uid": anchor["take_uid"],
            "timestamp": anchor["timestamp"],
            "quality_bucket": anchor.get("quality_bucket", "unlabeled"),
            "quality_weight": anchor.get("quality_weight", torch.tensor(1.0)),
            "paired_control": self.spec.paired_control,
        }


__all__ = [
    "BRIDGE_EXPERIMENTS",
    "FINAL_CONTRACT",
    "PAIRED_CONTROLS",
    "SCREEN_CONTRACT",
    "SMOKE_STEPS",
    "EffectTargetCache",
    "ExperimentSpec",
    "PairedControlDataset",
    "RunContract",
    "build_same_take_wrong_phase_index",
    "make_experiment_model",
    "validate_batch_contract",
]
