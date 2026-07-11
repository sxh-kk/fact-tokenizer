from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from fact_tokenizer.effect_targets import (
    EffectTargetCacheWriter,
    EffectTargetConfig,
    align_atomic_descriptions,
    align_phase_segments,
    deterministic_weak_semantic_labels,
    dino_delta,
    forward_backward_consistency,
    homography_flow,
    nearest_timestamp_index,
    propagate_relation_mask,
    relative_rotation_homography,
    require_supported_target,
    roi_feature_delta,
    rotation_compensated_flow_2d,
    scale_intrinsics,
    verify_target_cache,
)
from fact_tokenizer.effect_manifest import EffectCapability, EffectSampleRecord, write_manifest_jsonl
from scripts.audit_fact_effect_targets import validate as validate_visual_audit


def test_scale_intrinsics_uses_independent_xy_scales() -> None:
    source = np.asarray([[1000.0, 0.0, 500.0], [0.0, 800.0, 250.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(source, source_size=(500, 1000), target_size=(200, 250))
    np.testing.assert_allclose(
        scaled,
        [[250.0, 0.0, 125.0], [0.0, 320.0, 100.0], [0.0, 0.0, 1.0]],
    )


def test_pose_alignment_accepts_one_frame_and_rejects_more() -> None:
    timestamps = np.asarray([0.0, 10 / 30, 20 / 30])
    assert nearest_timestamp_index(timestamps, 10 / 30 + 0.49 / 30, 1 / 30) == 1
    assert nearest_timestamp_index(timestamps, 10 / 30 + 1.1 / 30, 1 / 30) is None


def test_relative_rotation_homography_has_expected_direction() -> None:
    # A +90 degree world-to-camera z rotation maps the +x ray to +y.
    k = np.eye(3)
    r0 = np.eye(3)
    r1 = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    homography = relative_rotation_homography(k, k, r0, r1)
    point = homography @ np.asarray([1.0, 0.0, 1.0])
    np.testing.assert_allclose(point[:2] / point[2], [0.0, 1.0], atol=1e-7)


def test_static_background_rotation_and_translation_residual_is_zero() -> None:
    # The total image motion is one-pixel camera translation.  Identity camera
    # rotation plus background RANSAC must remove it from the residual target.
    flow = np.zeros((12, 12, 2), dtype=np.float32)
    flow[..., 0] = 1.0
    result = rotation_compensated_flow_2d(flow, np.eye(3), np.ones((12, 12), dtype=bool))
    assert result["valid"].sum() >= 100
    np.testing.assert_allclose(result["rotation_compensated_flow_2d"][result["valid"]], 0.0, atol=1e-4)


def test_homography_flow_marks_out_of_frame_pixels_invalid() -> None:
    matrix = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    flow, valid = homography_flow(matrix, 5, 6)
    np.testing.assert_allclose(flow[..., 0], 2.0)
    assert valid[:, :4].all()
    assert not valid[:, 4:].any()


def test_forward_backward_consistency_and_mask_round_trip() -> None:
    forward = np.zeros((8, 8, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = 1.0
    backward[..., 0] = -1.0
    error, valid = forward_backward_consistency(forward, backward, 0.1)
    assert np.max(error[valid]) == pytest.approx(0.0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    propagated = propagate_relation_mask(mask, forward, backward, 0.1)
    expected = np.zeros_like(mask)
    expected[2:5, 3:6] = True
    np.testing.assert_array_equal(propagated["mask_t1"], expected)
    assert propagated["valid"] is True


def test_dino_and_roi_delta_have_explicit_empty_mask_validity() -> None:
    before = torch.zeros(2, 4, 3)
    after = torch.ones_like(before)
    delta = dino_delta(before, after)
    masks = torch.zeros(2, 8, 8)
    masks[0, :4, :4] = 1
    pooled, valid = roi_feature_delta(delta, masks, patch_grid=(2, 2))
    assert valid.tolist() == [True, False]
    torch.testing.assert_close(pooled[0], torch.ones(3))
    torch.testing.assert_close(pooled[1], torch.zeros(3))


def test_weak_annotations_are_raw_and_deterministically_aligned() -> None:
    atomic = align_atomic_descriptions(
        [
            {"timestamp": 1.2, "description": "open drawer", "source": "human"},
            {"timestamp": 3.0, "description": "too far"},
        ],
        1.0,
        1.5,
    )
    assert [row["text"] for row in atomic] == ["open drawer"]
    assert atomic[0]["source"] == "human"
    phase = align_phase_segments(
        [{"start_sec": 0.0, "end_sec": 1.1, "label": "reach"}, {"start_sec": 1.1, "end_sec": 2.0, "label": "grasp"}],
        1.0,
        1.5,
    )
    assert [row["step"] for row in phase] == ["grasp", "reach"]
    weak = deterministic_weak_semantic_labels(atomic, phase)
    assert weak["phase_name"] == "state_change_or_manipulate"
    assert weak["contact_name"] == "stable"
    assert weak["phase_valid"] is True
    unmatched = deterministic_weak_semantic_labels([{"text": "something happens"}])
    assert unmatched["phase_name"] == "ambiguous"
    assert unmatched["phase_valid"] is False


def test_target_cache_is_versioned_hashed_and_rejects_3d(tmp_path: Path) -> None:
    config = EffectTargetConfig()
    writer = EffectTargetCacheWriter(tmp_path, config)
    writer.write(
        "take:1.000",
        {
            "dino_delta": np.ones((4, 3), dtype=np.float32),
            "dino_delta_valid": True,
            "rotation_compensated_flow_2d": np.zeros((4, 4, 2), dtype=np.float32),
            "rotation_compensated_flow_2d_valid": np.ones((4, 4), dtype=bool),
        },
    )
    manifest = writer.finalize()
    record = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert record["config_sha256"] == config.fingerprint()
    assert verify_target_cache(tmp_path) == []
    with pytest.raises(ValueError, match="requires depth"):
        require_supported_target("flow_3d")
    with pytest.raises(ValueError, match="requires depth"):
        writer.write("bad", {"flow_3d": np.zeros(3)})
    with pytest.raises(ValueError, match="requires depth"):
        writer.write("bad-view", {"ego_flow_3d": np.zeros(3)})
    with pytest.raises(ValueError, match="must be false"):
        writer.write("bad-valid", {"depth_valid": True})


def test_target_cache_resume_binds_source_and_weight_identity(tmp_path: Path) -> None:
    config = EffectTargetConfig()
    writer = EffectTargetCacheWriter(tmp_path, config, identity={"raft_sha256": "a" * 64})
    writer.write("s", {"depth_valid": False, "flow_3d_valid": False})
    writer.finalize()
    EffectTargetCacheWriter(
        tmp_path,
        config,
        resume=True,
        identity={"raft_sha256": "a" * 64},
    )
    with pytest.raises(ValueError, match="source/weight identity changed"):
        EffectTargetCacheWriter(
            tmp_path,
            config,
            resume=True,
            identity={"raft_sha256": "b" * 64},
        )


def test_weak_calibration_is_bound_to_all_60_dev_samples(tmp_path: Path) -> None:
    weak_path = tmp_path / "weak.jsonl"
    weak_rows = [
        {
            "sample_id": f"s{index}",
            "atomic_descriptions": [{"text": "open drawer"}],
            "phase_segments": [],
        }
        for index in range(60)
    ]
    weak_path.write_text("".join(json.dumps(row) + "\n" for row in weak_rows), encoding="utf-8")
    gold_path = tmp_path / "gold60.csv"
    fields = [
        "sample_id",
        "take_uid",
        "gold_split",
        "representation_training_valid",
        "effect_label",
        "contact_label",
        "ambiguous_reason",
    ]
    with gold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(60):
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "take_uid": f"t{index // 3}",
                    "gold_split": "calibration_dev",
                    "representation_training_valid": "false",
                    "effect_label": "state_change_or_manipulate",
                    "contact_label": "stable",
                    "ambiguous_reason": "",
                }
            )
    report_path = tmp_path / "calibration.json"
    frozen_manifest = tmp_path / "gold300_frozen.jsonl"
    frozen_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"s{index}",
                    "take_uid": f"t{index // 3}",
                    "gold_split": "calibration_dev",
                }
            )
            + "\n"
            for index in range(60)
        ),
        encoding="utf-8",
    )
    import hashlib

    gold_freeze = tmp_path / "gold300_freeze.json"
    gold_freeze.write_text(
        json.dumps({"manifest_sha256": hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_fact_weak_semantics.py",
            "--weak-index-jsonl",
            str(weak_path),
            "--gold60-csv",
            str(gold_path),
            "--output-report",
            str(report_path),
            "--gold-freeze",
            str(gold_freeze),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dev_sample_count"] == 60
    assert report["gate_passed"] is True
    assert report["minimum_measured_precision"] == 1.0


def test_disjoint_target_shards_merge_atomically_with_one_identity(tmp_path: Path) -> None:
    for index in range(2):
        writer = EffectTargetCacheWriter(
            tmp_path / f"shard{index}",
            EffectTargetConfig(),
            identity={"source": "same"},
        )
        writer.write(
            f"s{index}",
            {
                "depth_valid": False,
                "flow_3d_valid": False,
                "ego_full_dino_delta": np.zeros((1, 2), dtype=np.float32),
                "ego_full_dino_delta_valid": True,
            },
        )
        writer.finalize()
    expected_manifest = tmp_path / "expected.jsonl"
    write_manifest_jsonl(
        expected_manifest,
        [
            EffectSampleRecord(sample_id=f"s{index}", take_uid=f"t{index}", split="train", row_index=index)
            for index in range(2)
        ],
    )
    identity = json.loads((tmp_path / "shard0" / "target_config.json").read_text(encoding="utf-8"))[
        "identity_sha256"
    ]
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "rows": 50,
                "fully_aligned": 45,
                "minimum_pass_fraction": 0.90,
                "pass_fraction": 0.90,
                "target_identity_sha256": identity,
            }
        ),
        encoding="utf-8",
    )
    merged = tmp_path / "merged"
    subprocess.run(
        [
            sys.executable,
            "scripts/merge_fact_effect_target_shards.py",
            "--shard",
            str(tmp_path / "shard0"),
            "--shard",
            str(tmp_path / "shard1"),
            "--output-dir",
            str(merged),
            "--expected-manifest",
            str(expected_manifest),
            "--visual-audit-gate",
            str(gate),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert verify_target_cache(merged) == []
    assert json.loads((merged / "target_config.json").read_text(encoding="utf-8"))["samples"] == 2


def test_visual_audit_threshold_cannot_be_lowered_below_90_percent(tmp_path: Path) -> None:
    review = tmp_path / "target_visual_audit.csv"
    fields = [
        "sample_id",
        "time_direction_ok",
        "coordinate_orientation_ok",
        "mask_flow_alignment_ok",
    ]
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(50):
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "time_direction_ok": "yes",
                    "coordinate_orientation_ok": "yes",
                    "mask_flow_alignment_ok": "yes" if index < 45 else "no",
                }
            )
    (tmp_path / "audit_pack.json").write_text(
        json.dumps(
            {
                "selected_sample_ids": [f"s{index}" for index in range(50)],
                "target_identity_sha256": "a" * 64,
                "target_config_sha256": "b" * 64,
                "strata_counts": {"all": 50},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be below"):
        validate_visual_audit(review, 0.09)
    report = validate_visual_audit(review, 0.90)
    assert report["fully_aligned"] == 45 and report["passed"] is True


def test_target_builder_smoke_writes_dino_flow_roi_and_no_3d(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rng = np.random.default_rng(3)
    for view in ("ego", "exo"):
        np.save(input_dir / f"{view}.npy", rng.integers(0, 255, (1, 2, 28, 28, 3), dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(
        manifest,
        [
            EffectSampleRecord(
                sample_id="sample",
                take_uid="take",
                split="train",
                row_index=0,
                timestamp=0.0,
                capability_validity={EffectCapability.RGB_PAIRED: True, EffectCapability.CAMERA_POSE: True},
            )
        ],
    )
    mask = np.zeros((1, 28, 28), dtype=np.uint8)
    mask[:, 8:20, 8:20] = 1
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    camera_args = []
    for view in ("ego", "exo"):
        camera = tmp_path / f"camera_{view}"
        camera.mkdir()
        k = np.asarray([[20.0, 0, 14.0], [0, 20.0, 14.0], [0, 0, 1.0]])
        np.save(camera / "intrinsics.npy", np.tile(k, (1, 2, 1, 1)))
        np.save(camera / "world_to_camera.npy", np.tile(np.eye(4), (1, 2, 1, 1)))
        np.save(camera / "valid.npy", np.ones(1, dtype=bool))
        np.save(camera / "source_size.npy", np.asarray([[28, 28]], dtype=np.int32))
        camera_args.extend(["--camera-sidecar", f"{view}={camera}"])
    output = tmp_path / "targets"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_fact_effect_targets.py",
            "--input-dir",
            str(input_dir),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--feature-backend",
            "mock",
            "--mock-feature-dim",
            "8",
            "--zero-flow-smoke",
            "--allow-smoke-targets",
            "--object-mask-npy",
            f"ego={mask_path}",
            *camera_args,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert verify_target_cache(output) == []
    record = json.loads((output / "target_manifest.jsonl").read_text(encoding="utf-8"))
    with np.load(output / record["path"], allow_pickle=False) as targets:
        assert targets["ego_full_dino_delta"].shape == (256, 8)
        assert targets["ego_rotation_compensated_flow_2d"].shape == (224, 224, 2)
        assert targets["ego_object_roi_dino_delta_valid"].item()
        assert not targets["ego_phase_valid"].item()
        assert not targets["ego_contact_valid"].item()
        assert not targets["depth_valid"].item()
        assert not targets["flow_3d_valid"].item()
