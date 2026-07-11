from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fact_tokenizer.locked_split import (
    LOCKED_SET_ID,
    ShortLockedConfig,
    audit_locked_samples,
    build_short_locked_samples,
    perceptual_nearest_neighbor_audit,
    validate_final_freeze_candidate_contract,
    write_frozen_locked_manifest,
)


def short_rows() -> list[dict]:
    return [
        {
            "take_uid": f"short_{index}",
            "reason": "insufficient_duration",
            "task_start_sec": 1.0,
            "task_end_sec": 10.0,
            "participant_uid": f"fresh_p{index}",
            "capture_uid": f"fresh_c{index}",
            "ego_relative_path": f"ego_{index}.mp4",
            "exo_relative_path": f"exo_{index}.mp4",
        }
        for index in range(2)
    ]


def test_build_locked_samples_is_exact_deterministic_and_nontraining() -> None:
    config = ShortLockedConfig(expected_takes=2)
    samples, selected = build_short_locked_samples(reversed(short_rows()), config)
    assert len(samples) == 16
    assert len(selected) == 2
    assert [sample.take_uid for sample in samples[:8]] == ["short_0"] * 8
    assert [sample.timestamp for sample in samples[:8]] == list(np.arange(1.0, 9.0))
    assert samples[0].sample_id == "short_0:1.000"
    assert samples[0].end_frame - samples[0].start_frame == 15
    assert all(sample.locked_set_id == LOCKED_SET_ID for sample in samples)
    assert all(not sample.training_valid and not sample.model_selection_valid for sample in samples)
    assert all(sample.final_inference_only for sample in samples)


def test_short_take_that_cannot_hold_eight_transitions_fails() -> None:
    rows = short_rows()
    rows[0]["task_end_sec"] = 8.0
    with pytest.raises(ValueError, match="too short"):
        build_short_locked_samples(rows, ShortLockedConfig(expected_takes=2))


def test_take_sample_frame_capture_audit_blocks_publish(tmp_path: Path) -> None:
    config = ShortLockedConfig(expected_takes=2)
    samples, selected = build_short_locked_samples(short_rows(), config)
    audit = audit_locked_samples(
        samples,
        {
            "legacy_train": [{"take_uid": "old", "sample_id": "old:1.000", "timestamp": 1.0}],
            "diagnostic500": [{"take_uid": "short_0", "sample_id": "short_0:1.000", "timestamp": 1.0}],
        },
        participant_fresh_takes=["short_0", "short_1"],
    )
    assert not audit["passed"]
    assert audit["reference_groups"]["diagnostic500"]["overlaps"]["sample_ids"] == ["short_0:1.000"]
    assert audit["reference_groups"]["diagnostic500"]["overlaps"]["takes"] == ["short_0"]
    with pytest.raises(ValueError, match="audit failed"):
        write_frozen_locked_manifest(tmp_path, samples, selected, config, audit)


def test_freeze_writes_hash_stable_manifest(tmp_path: Path) -> None:
    config = ShortLockedConfig(expected_takes=2)
    samples, selected = build_short_locked_samples(short_rows(), config)
    audit = audit_locked_samples(
        samples,
        {"legacy": [{"take_uid": "old", "sample_id": "old:0.000", "capture_uid": "old_capture"}]},
        participant_fresh_takes=["short_0", "short_1"],
    )
    audit["perceptual_nearest_neighbor"] = {"status": "complete", "reports": []}
    freeze = write_frozen_locked_manifest(tmp_path, samples, selected, config, audit)
    assert freeze["takes"] == 2
    assert freeze["samples"] == 16
    assert len(freeze["samples_sha256"]) == 64
    records = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(record["final_inference_only"] for record in records)
    assert all(not record["training_valid"] for record in records)


def test_final_freeze_requires_perceptual_audit_but_provisional_does_not(tmp_path: Path) -> None:
    config = ShortLockedConfig(expected_takes=2)
    samples, selected = build_short_locked_samples(short_rows(), config)
    audit = audit_locked_samples(samples, {"legacy": [{"take_uid": "old"}]})
    audit["perceptual_nearest_neighbor"] = {"status": "deferred_to_final_freeze", "reports": []}
    with pytest.raises(ValueError, match="completed perceptual"):
        write_frozen_locked_manifest(tmp_path / "final", samples, selected, config, audit)
    provisional = write_frozen_locked_manifest(
        tmp_path / "provisional", samples, selected, config, audit, stage="provisional"
    )
    assert provisional["freeze_stage"] == "provisional"
    assert provisional["evaluation_allowed"] is False


def test_frame_audit_checks_full_transition_intervals() -> None:
    samples, _ = build_short_locked_samples(short_rows(), ShortLockedConfig(expected_takes=2))
    audit = audit_locked_samples(
        samples,
        {
            "history": [
                {
                    "take_uid": "short_0",
                    "sample_id": "different-id",
                    "timestamp": 1.25,
                    "end_timestamp": 1.30,
                }
            ]
        },
    )
    assert audit["reference_groups"]["history"]["overlaps"]["frames"]
    assert not audit["passed"]


def test_perceptual_nearest_neighbor_detects_exact_duplicates() -> None:
    rng = np.random.default_rng(4)
    candidates = rng.integers(0, 255, size=(2, 2, 20, 20, 3), dtype=np.uint8)
    references = rng.integers(0, 255, size=(3, 2, 20, 20, 3), dtype=np.uint8)
    references[1] = candidates[0]
    report = perceptual_nearest_neighbor_audit(candidates, references, maximum_similarity=0.999)
    assert not report["passed"]
    assert report["violations"][0]["candidate_index"] == 0
    assert report["violations"][0]["reference_index"] == 1
    assert report["metric"] == "appearance_plus_temporal_difference_cosine_v2"


def test_perceptual_nearest_neighbor_does_not_treat_static_exo_background_as_leakage() -> None:
    rng = np.random.default_rng(8)
    background = rng.integers(32, 224, size=(64, 64, 3), dtype=np.uint8)
    candidates = np.repeat(background[None, None], repeats=2, axis=1)
    references = np.repeat(background[None, None], repeats=2, axis=1)
    candidates = candidates.copy()
    references = references.copy()
    candidates[0, 1, 8:24, 8:24] = np.clip(
        candidates[0, 1, 8:24, 8:24].astype(np.int16) + 30, 0, 255
    ).astype(np.uint8)
    references[0, 1, 40:56, 40:56] = np.clip(
        references[0, 1, 40:56, 40:56].astype(np.int16) + 30, 0, 255
    ).astype(np.uint8)
    report = perceptual_nearest_neighbor_audit(candidates, references, maximum_similarity=0.995)
    assert report["passed"]


def test_short73_cli_bootstraps_repo_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/prepare_fact_short73_locked.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--stage" in completed.stdout


def test_short73_materializer_requires_exact_dual_view_pnn_candidate_contract() -> None:
    freeze = {
        "freeze_stage": "final",
        "evaluation_allowed": True,
        "final_inference_only": True,
        "training_valid": False,
        "model_selection_valid": False,
        "samples": 584,
        "takes": 73,
        "audit": {
            "passed": True,
            "provisional_evidence": {"candidate_sample_ids_sha256": "c" * 64},
            "perceptual_nearest_neighbor": {
                "status": "complete",
                "reports": [
                    {
                        "view": "ego",
                        "candidate_sha256": "a" * 64,
                        "candidate_count": 584,
                        "passed": True,
                        "violations": [],
                    },
                    {
                        "view": "exo",
                        "candidate_sha256": "b" * 64,
                        "candidate_count": 584,
                        "passed": True,
                        "violations": [],
                    },
                ],
            },
        },
    }
    assert validate_final_freeze_candidate_contract(freeze, 584) == {
        "ego": "a" * 64,
        "exo": "b" * 64,
        "sample_id": "c" * 64,
    }
    freeze["audit"]["perceptual_nearest_neighbor"]["reports"].append(
        {
            "view": "ego",
            "candidate_sha256": "d" * 64,
            "candidate_count": 584,
            "passed": True,
            "violations": [],
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        validate_final_freeze_candidate_contract(freeze, 584)
