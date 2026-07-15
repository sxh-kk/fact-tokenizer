from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from fact_tokenizer.effect_experiments import (
    BRIDGE_EXPERIMENTS,
    FINAL_CONTRACT,
    PAIRED_CONTROLS,
    SCREEN_CONTRACT,
    EffectTargetCache,
    PairedControlDataset,
    build_same_take_wrong_phase_index,
    make_experiment_model,
    validate_batch_contract,
)
from fact_tokenizer.effect_targets import EffectTargetCacheWriter, EffectTargetConfig
from fact_tokenizer.effect_manifest import EffectCapability, EffectSampleRecord, write_manifest_jsonl
from fact_tokenizer.paired_eligibility import (
    build_paired_control_eligibility,
    dataset_indices_for_eligibility,
    load_paired_control_eligibility,
)
from scripts.train_fact_effect_v7 import (
    DeterministicDistributedWeightedSampler,
    distributed_timeout_seconds,
    init_distributed,
    load_excluded_takes,
    nuisance_inputs,
    require_control_assets,
    validate_formal_target_cache,
)


def test_nccl_binds_local_device_before_process_group_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("FACT_DDP_TIMEOUT_SECONDS", "47")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda rank: events.append(("set_device", rank)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: events.append(("init_process_group", kwargs)),
    )

    assert init_distributed() == (2, 1, 1)
    assert events[0] == ("set_device", 1)
    assert events[1][0] == "init_process_group"
    kwargs = events[1][1]
    assert isinstance(kwargs, dict)
    assert kwargs["backend"] == "nccl"
    assert kwargs["timeout"].total_seconds() == 47
    assert distributed_timeout_seconds() == 47


def test_distributed_timeout_contract_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACT_DDP_TIMEOUT_SECONDS", "29")
    with pytest.raises(ValueError, match="between 30 and 3600"):
        distributed_timeout_seconds()


def test_preregistered_experiment_matrix_and_batch_contract() -> None:
    assert set(BRIDGE_EXPERIMENTS) == {"C0", "C1", "C2", "T1", "T2"}
    assert set(PAIRED_CONTROLS) == {"P0", "P1", "P2", "P3", "P4"}
    assert SCREEN_CONTRACT.steps == 5_000 and SCREEN_CONTRACT.seeds == (42, 43)
    assert FINAL_CONTRACT.steps == 20_000 and FINAL_CONTRACT.seeds == (42, 43, 44)
    validate_batch_contract(micro_batch=8, gradient_accumulation=2, world_size=2)
    with pytest.raises(ValueError, match="global batch"):
        validate_batch_contract(micro_batch=4, gradient_accumulation=2, world_size=2)
    with pytest.raises(ValueError, match="2 GPUs"):
        validate_batch_contract(micro_batch=16, gradient_accumulation=2, world_size=1)


def test_formal_gold_exclusion_and_control_assets_are_mandatory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="excluded-takes"):
        load_excluded_takes(None, formal=True)
    exclusion = tmp_path / "excluded.txt"
    exclusion.write_text("".join(f"take-{index}\n" for index in range(47)), encoding="utf-8")
    training_manifest = tmp_path / "train.jsonl"
    training_manifest.write_text(
        "".join(json.dumps({"take_uid": f"take-{index}"}) + "\n" for index in range(47)),
        encoding="utf-8",
    )
    gold_manifest = tmp_path / "gold300_frozen.jsonl"
    probe_rows = [
        {
            "sample_id": f"gold-{index}",
            "take_uid": f"take-{min(index // 3, 46)}",
            "gold_split": "probe_train",
        }
        for index in range(140)
    ]
    gold_manifest.write_text("".join(json.dumps(row) + "\n" for row in probe_rows), encoding="utf-8")
    gold_freeze = tmp_path / "gold300_freeze.json"
    gold_freeze.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(gold_manifest.read_bytes()).hexdigest(),
                "representation_train_excluded_takes": {
                    "path": exclusion.name,
                    "takes": 47,
                    "sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    takes, fingerprint = load_excluded_takes(
        exclusion,
        formal=True,
        gold_freeze=gold_freeze,
        training_manifest=training_manifest,
    )
    assert len(takes) == fingerprint["take_count"] == 47

    p4 = SimpleNamespace(
        experiment="P4",
        extra_ego_input_dir=None,
        extra_ego_manifest=None,
        extra_ego_target_cache=None,
        exo_target_quality_provenance=None,
    )
    with pytest.raises(ValueError, match="P4 requires"):
        require_control_assets(p4, formal=True)


def test_model_registry_keeps_shortcuts_only_in_c0_c1() -> None:
    c0 = make_experiment_model(BRIDGE_EXPERIMENTS["C0"], backbone_dim=4, hidden_dim=8, semantic_dim=3)
    c2 = make_experiment_model(BRIDGE_EXPERIMENTS["C2"], backbone_dim=4, hidden_dim=8, semantic_dim=3)
    assert c0.private_future_visible and c0.cross_uses_private
    assert not hasattr(c2, "private_future_visible")


def test_camera_nuisance_updates_rgb_and_intrinsics_coherently() -> None:
    video = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    context = torch.zeros(1, 18)
    transformed = nuisance_inputs({"ego": {"videos": video, "camera_context": context}})["ego"]
    assert transformed["videos"][0, 0, 0, 0, 0].item() == 0.0
    assert transformed["videos"][0, 0, 0, 0, 1].item() == video[0, 0, 0, 1, 0].item()
    assert transformed["camera_context"][0, 2].item() == 1.0
    assert transformed["camera_context"][0, 5].item() == -1.0


def test_quality_weighted_sampler_is_deterministic_and_rank_sharded() -> None:
    weights = np.asarray([1.0, 0.0, 3.0, 1.0])
    rank0 = DeterministicDistributedWeightedSampler(weights, num_replicas=2, rank=0, seed=7)
    rank1 = DeterministicDistributedWeightedSampler(weights, num_replicas=2, rank=1, seed=7)
    first = list(rank0)
    assert first == list(rank0)
    assert len(first) == len(list(rank1)) == 2
    assert 1 not in first and 1 not in list(rank1)
    rank0.set_epoch(1)
    assert len(list(rank0)) == 2


def test_wrong_phase_index_is_same_take_and_different_phase() -> None:
    takes = ["a", "a", "a", "b", "b"]
    phases = ["reach", "grasp", "release", "reach", "grasp"]
    timestamps = [0, 1, 2, 0, 1]
    donors = build_same_take_wrong_phase_index(takes, phases, timestamps)
    assert np.all(donors >= 0)
    for index, donor in enumerate(donors):
        assert takes[index] == takes[donor]
        assert phases[index] != phases[donor]


class TinyBase(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {
                "views": {
                    "ego": torch.full((2, 3, 4, 4), float(index)),
                    "exo": torch.full((2, 3, 4, 4), float(index + 10)),
                },
                "camera_contexts": {},
                "sample_id": f"s{index}",
                "take_uid": "take",
                "timestamp": torch.tensor(float(index)),
            }
            for index in range(2)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_cache(path: Path) -> EffectTargetCache:
    writer = EffectTargetCacheWriter(path, EffectTargetConfig())
    for index in range(2):
        targets = {"depth_valid": False, "flow_3d_valid": False}
        for view in ("ego", "exo"):
            targets.update(
                {
                    f"{view}_full_dino_delta": np.full((4, 3), index, dtype=np.float32),
                    f"{view}_full_dino_delta_valid": True,
                }
            )
        writer.write(f"s{index}", targets)
    writer.finalize()
    return EffectTargetCache(path)


def test_p3_replaces_only_exo_with_same_take_wrong_phase_donor(tmp_path: Path) -> None:
    dataset = PairedControlDataset(
        TinyBase(),
        make_cache(tmp_path),
        PAIRED_CONTROLS["P3"],
        wrong_phase_index=[1, 0],
        phase_labels=["reach", "release"],
    )
    sample = dataset[0]
    assert sample["donor_sample_id"] == "s1"
    assert sample["model_inputs"]["ego"]["videos"].unique().item() == 0
    assert sample["model_inputs"]["exo"]["videos"].unique().item() == 11
    assert sample["targets"]["ego"]["full_dino_delta"].unique().item() == 0
    assert sample["targets"]["exo"]["full_dino_delta"].unique().item() == 1


def test_p3_rejects_cross_take_or_same_phase_donors(tmp_path: Path) -> None:
    base = TinyBase()
    base.rows[1]["take_uid"] = "other"
    with pytest.raises(ValueError, match="same take"):
        PairedControlDataset(
            base,
            make_cache(tmp_path / "cross_take"),
            PAIRED_CONTROLS["P3"],
            wrong_phase_index=[1, 0],
            phase_labels=["reach", "release"],
        )

    base.rows[1]["take_uid"] = "take"
    with pytest.raises(ValueError, match="different frozen phase"):
        PairedControlDataset(
            base,
            make_cache(tmp_path / "same_phase"),
            PAIRED_CONTROLS["P3"],
            wrong_phase_index=[1, 0],
            phase_labels=["reach", "reach"],
        )


def test_target_cache_refuses_depth_or_3d_validity(tmp_path: Path) -> None:
    writer = EffectTargetCacheWriter(tmp_path, EffectTargetConfig())
    with pytest.raises(ValueError, match="must be false"):
        writer.write(
            "s",
            {
                "depth_valid": True,
                "flow_3d_valid": False,
                "ego_full_dino_delta": np.zeros((1, 2)),
            },
        )


def test_formal_trainer_refuses_smoke_target_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "smoke-cache"
    writer = EffectTargetCacheWriter(
        cache_dir,
        EffectTargetConfig(target_cache_version="fact-effect-target-v1-smoke"),
    )
    writer.write("s", {"depth_valid": False, "flow_3d_valid": False})
    writer.finalize()
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(
        manifest,
        [EffectSampleRecord(sample_id="s", take_uid="t", split="train", row_index=0)],
    )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    np.save(input_dir / "ego.npy", np.zeros((1, 2, 2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="refuses smoke"):
        validate_formal_target_cache(
            EffectTargetCache(cache_dir),
            input_dir=input_dir,
            manifest=manifest,
            view_names=("ego",),
            dino_model="dinov2_vitb14_reg",
        )


def test_training_script_one_step_continuous_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rng = np.random.default_rng(2)
    count = 4
    for view in ("ego", "exo"):
        np.save(input_dir / f"{view}.npy", rng.integers(0, 255, (count, 2, 28, 28, 3), dtype=np.uint8))
    np.save(input_dir / "sample_id.npy", np.asarray([f"s{index}" for index in range(count)]))
    np.save(input_dir / "take_uid.npy", np.asarray([f"t{index // 2}" for index in range(count)]))
    np.save(input_dir / "timestamp.npy", np.arange(count, dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(
        manifest,
        [
            EffectSampleRecord(
                sample_id=f"s{index}",
                take_uid=f"t{index // 2}",
                split="train",
                row_index=index,
                timestamp=float(index),
                capability_validity={EffectCapability.RGB_PAIRED: True},
            )
            for index in range(count)
        ],
    )
    cache_dir = tmp_path / "cache"
    writer = EffectTargetCacheWriter(cache_dir, EffectTargetConfig())
    for index in range(count):
        targets = {"depth_valid": False, "flow_3d_valid": False}
        for view in ("ego", "exo"):
            targets[f"{view}_full_dino_delta"] = np.zeros((256, 8), dtype=np.float32)
            targets[f"{view}_full_dino_delta_valid"] = True
        writer.write(f"s{index}", targets)
    writer.finalize()
    output = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_fact_effect_v7.py",
            "--experiment",
            "C2",
            "--stage",
            "smoke",
            "--input-dir",
            str(input_dir),
            "--manifest",
            str(manifest),
            "--target-cache",
            str(cache_dir),
            "--output-dir",
            str(output),
            "--seed",
            "42",
            "--steps",
            "1",
            "--micro-batch",
            "1",
            "--gradient-accumulation",
            "2",
            "--num-workers",
            "0",
            "--backbone",
            "mock",
            "--backbone-dim",
            "8",
            "--hidden-dim",
            "8",
            "--semantic-dim",
            "4",
            "--private-dim",
            "4",
            "--checkpoint-every",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = torch.load(output / "checkpoint_000001.pt", map_location="cpu")
    assert checkpoint["step"] == 1
    assert checkpoint["step_unit"] == "optimizer_update"
    assert checkpoint["data_cursor_micro_batches"] == 2
    metric = json.loads((output / "train_metrics.jsonl").read_text(encoding="utf-8").strip())
    assert metric["micro_batches_consumed"] == 2
    assert not any("vq" in key.lower() or "codebook" in key.lower() for key in checkpoint["model"])
    feature_dir = tmp_path / "features"
    subprocess.run(
        [
            sys.executable,
            "scripts/extract_fact_effect_features.py",
            "--checkpoint",
            str(output / "checkpoint_000001.pt"),
            "--input-dir",
            str(input_dir),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(feature_dir),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    extracted = np.load(feature_dir / "features.npy", allow_pickle=False)
    assert extracted.shape == (count, 4)
    metadata = json.loads((feature_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    assert metadata["encoder_frozen"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="torchrun gloo rendezvous is not reliable on Windows CI")
def test_training_script_two_rank_handles_inactive_branches(tmp_path: Path) -> None:
    """DDP must survive branches that are intentionally inactive in C2."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rng = np.random.default_rng(9)
    count = 4
    for view in ("ego", "exo"):
        np.save(input_dir / f"{view}.npy", rng.integers(0, 255, (count, 2, 28, 28, 3), dtype=np.uint8))
    np.save(input_dir / "sample_id.npy", np.asarray([f"s{index}" for index in range(count)]))
    np.save(input_dir / "take_uid.npy", np.asarray([f"t{index // 2}" for index in range(count)]))
    np.save(input_dir / "timestamp.npy", np.arange(count, dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(
        manifest,
        [
            EffectSampleRecord(
                sample_id=f"s{index}",
                take_uid=f"t{index // 2}",
                split="train",
                row_index=index,
                timestamp=float(index),
                capability_validity={EffectCapability.RGB_PAIRED: True},
            )
            for index in range(count)
        ],
    )
    cache_dir = tmp_path / "cache"
    writer = EffectTargetCacheWriter(cache_dir, EffectTargetConfig())
    for index in range(count):
        targets = {"depth_valid": False, "flow_3d_valid": False}
        for view in ("ego", "exo"):
            targets[f"{view}_full_dino_delta"] = np.zeros((256, 8), dtype=np.float32)
            targets[f"{view}_full_dino_delta_valid"] = True
        writer.write(f"s{index}", targets)
    writer.finalize()
    output = tmp_path / "ddp-run"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "scripts/train_fact_effect_v7.py",
            "--experiment",
            "C2",
            "--stage",
            "smoke",
            "--input-dir",
            str(input_dir),
            "--manifest",
            str(manifest),
            "--target-cache",
            str(cache_dir),
            "--output-dir",
            str(output),
            "--seed",
            "42",
            "--steps",
            "2",
            "--micro-batch",
            "1",
            "--gradient-accumulation",
            "1",
            "--num-workers",
            "0",
            "--backbone",
            "mock",
            "--backbone-dim",
            "8",
            "--hidden-dim",
            "8",
            "--semantic-dim",
            "4",
            "--private-dim",
            "4",
            "--checkpoint-every",
            "2",
            "--log-every",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = torch.load(output / "checkpoint_000002.pt", map_location="cpu")
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (output / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert checkpoint["step"] == 2
    assert [row["step"] for row in metrics] == [1, 2]
    assert config["world_size"] == 2
    assert config["ddp_find_unused_parameters"] is True


def test_paired_control_eligibility_keeps_only_same_take_different_phase_samples() -> None:
    records = [
        EffectSampleRecord("a0", "a", "train", 0, timestamp=0.0),
        EffectSampleRecord("a1", "a", "train", 1, timestamp=1.0),
        EffectSampleRecord("a2", "a", "train", 2, timestamp=2.0),
        EffectSampleRecord("b0", "b", "train", 3, timestamp=0.0),
        EffectSampleRecord("b1", "b", "train", 4, timestamp=1.0),
        EffectSampleRecord("c0", "c", "train", 5, timestamp=0.0),
        EffectSampleRecord("c1", "c", "train", 6, timestamp=1.0, training_valid=False),
        EffectSampleRecord("d0", "d", "train", 7, timestamp=0.0),
        EffectSampleRecord("d1", "d", "train", 8, timestamp=1.0),
    ]
    phases = ["reach", "grasp", "release", "reach", "reach", "reach", "grasp", "", "release"]
    arrays, report = build_paired_control_eligibility(records, phases, seed=42)

    assert arrays["sample_id"].tolist() == ["a0", "a1", "a2"]
    assert arrays["manifest_index"].tolist() == [0, 1, 2]
    assert report["eligible_rows"] == 3
    assert report["dropped"] == {
        "nontraining": 1,
        "missing_or_unknown_phase": 1,
        "no_same_take_different_phase_donor": 4,
    }
    donor = arrays["donor_index"]
    assert np.all(arrays["take_uid"][donor] == arrays["take_uid"])
    assert np.all(arrays["phase_label"][donor] != arrays["phase_label"])
    assert arrays["donor_sample_id"].tolist() == arrays["sample_id"][donor].tolist()


def test_freeze_p3_eligibility_cli_writes_common_allowlist_and_hashed_sidecars(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        EffectSampleRecord(
            sample_id=f"{take}{index}",
            take_uid=take,
            split="train",
            row_index=row,
            timestamp=float(index),
        )
        for row, (take, index) in enumerate((("a", 0), ("a", 1), ("b", 0), ("b", 1)))
    ]
    records.extend(
        EffectSampleRecord(
            sample_id=f"excluded-{index}",
            take_uid=f"gold-{index}",
            split="train",
            row_index=4 + index,
            timestamp=0.0,
        )
        for index in range(47)
    )
    write_manifest_jsonl(manifest, records)
    phases = tmp_path / "phase.npy"
    np.save(phases, np.asarray(["reach", "release", "reach", "grasp", *(["unknown"] * 47)]))
    exclusions = tmp_path / "excluded.txt"
    exclusions.write_text("".join(f"gold-{index}\n" for index in range(47)), encoding="utf-8")
    gold_manifest = tmp_path / "gold300_frozen.jsonl"
    gold_rows = [
        {
            "sample_id": f"gold-sample-{index}",
            "take_uid": f"gold-{min(index // 3, 46)}",
            "gold_split": "probe_train",
        }
        for index in range(140)
    ]
    gold_manifest.write_text("".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8")
    gold_freeze = tmp_path / "gold300_freeze.json"
    gold_freeze.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(gold_manifest.read_bytes()).hexdigest(),
                "representation_train_excluded_takes": {
                    "path": exclusions.name,
                    "takes": 47,
                    "sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "eligibility"
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_fact_p3_eligibility.py",
            "--manifest",
            str(manifest),
            "--phase-index",
            str(phases),
            "--output-dir",
            str(output),
            "--excluded-takes",
            str(exclusions),
            "--gold-freeze",
            str(gold_freeze),
            "--seed",
            "42",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "eligibility_report.json").read_text(encoding="utf-8"))
    assert report["eligible_rows"] == 4
    assert report["common_controls"] == ["P0", "P2", "P3", "P4"]
    assert report["usage"]["p3_donor_index_basis"] == "eligible_sample_id.npy order"
    npz_hash = report["artifacts_sha256"]["paired_control_eligibility.npz"]
    arrays = load_paired_control_eligibility(
        output / "paired_control_eligibility.npz",
        expected_sha256=npz_hash,
    )
    np.testing.assert_array_equal(np.load(output / "p3_donor_index.npy"), arrays["donor_index"])
    np.testing.assert_array_equal(np.load(output / "eligible_sample_id.npy").astype(str), arrays["sample_id"])

    dataset_ids = ["b1", "a0", "b0", "a1", "unused"]
    common_indices = dataset_indices_for_eligibility(dataset_ids, arrays["sample_id"])
    assert [dataset_ids[index] for index in common_indices] == arrays["sample_id"].tolist()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_paired_control_eligibility(
            output / "paired_control_eligibility.npz",
            expected_sha256="0" * 64,
        )
