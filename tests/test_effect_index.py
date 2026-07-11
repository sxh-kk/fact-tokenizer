from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fact_tokenizer.effect_index import (
    atomic_by_take,
    build_weak_sample_index,
    camera_pose_pair_from_egoexo_json,
    homogeneous_extrinsic,
    nearest_dynamic_extrinsic,
    phases_by_take,
    select_relation_mask_entries,
    decode_relation_union,
    nearest_pose_annotation,
    joint_delta,
)


def test_egoexo_camera_pose_dynamic_and_static_alignment() -> None:
    extrinsics = {
        str(frame): np.concatenate([np.eye(3), np.asarray([[frame / 30], [0], [0]])], axis=1).tolist()
        for frame in range(31)
    }
    payload = {
        "aria01": {"camera_intrinsics": np.eye(3).tolist(), "camera_extrinsics": extrinsics},
        "cam01": {
            "camera_intrinsics": (np.eye(3) * [100, 100, 1]).tolist(),
            "camera_extrinsics": np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1).tolist(),
        },
    }
    dynamic = camera_pose_pair_from_egoexo_json(payload, "aria01", 0.0, 0.5)
    assert dynamic is not None
    assert dynamic["aligned_frames"].tolist() == [0, 15]
    assert dynamic["world_to_camera"].shape == (2, 4, 4)
    assert dynamic["source_size"].tolist() == [512, 512]
    static = camera_pose_pair_from_egoexo_json(payload, "cam01", 0.0, 0.5)
    assert static is not None
    np.testing.assert_array_equal(static["world_to_camera"][0], static["world_to_camera"][1])
    assert static["source_size"].tolist() == [2160, 3840]


def test_pose_skew_is_at_most_one_frame() -> None:
    mapping = {"10": np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1).tolist()}
    assert nearest_dynamic_extrinsic(mapping, 10.9 / 30, max_skew_frames=1) is not None
    assert nearest_dynamic_extrinsic(mapping, 11.1 / 30, max_skew_frames=1) is None
    assert homogeneous_extrinsic(mapping["10"]).shape == (4, 4)


def test_atomic_and_phase_sources_remain_raw_until_dev_gate(tmp_path: Path) -> None:
    atomic_path = tmp_path / "atomic.json"
    atomic_path.write_text(
        json.dumps(
            {
                "annotations": {
                    "take": [
                        {
                            "annotation_uid": "a",
                            "rejected": False,
                            "descriptions": [{"timestamp": 1.2, "text": "opens box", "subject": "C"}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(
        json.dumps(
            {
                "annotations": {
                    "take": {
                        "segments": [
                            {"start_time": 1.0, "end_time": 2.0, "step_name": "Open", "step_id": 3}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    atomic = atomic_by_take(atomic_path, ["take"])
    phases = phases_by_take([phase_path], ["take"])
    rows = build_weak_sample_index(
        [{"sample_id": "take:1.000", "take_uid": "take", "timestamp": 1.0}],
        atomic,
        phases,
    )
    assert rows[0]["atomic_valid"] is True
    assert rows[0]["phase_valid"] is True
    assert rows[0]["atomic_descriptions"][0]["text"] == "opens box"
    assert rows[0]["phase_segments"][0]["step_id"] == 3
    assert rows[0]["verb_map_dev_precision"] is None
    assert rows[0]["weak_loss_enabled"] is False


def test_relations_path_selects_nearest_aria_object_frame_and_excludes_hands() -> None:
    relation = {
        "object_masks": {
            "stainless bowl_0": {
                "aria01_214-1": {
                    "annotation": {
                        "270": {"width": 4, "height": 4, "encodedMask": "bowl270"},
                        "300": {"width": 4, "height": 4, "encodedMask": "bowl300"},
                    }
                }
            },
            "left_hand_0": {
                "aria01_214-1": {
                    "annotation": {"270": {"width": 4, "height": 4, "encodedMask": "hand270"}}
                }
            },
            "other_0": {
                "cam01": {"annotation": {"270": {"width": 4, "height": 4, "encodedMask": "exo"}}}
            },
        }
    }
    entries, frame = select_relation_mask_entries(relation, "aria01", 272, max_frame_distance=5)
    assert frame == 270
    assert [entry["object_id"] for entry in entries] == ["stainless bowl_0"]

    def decoder(value: dict) -> np.ndarray:
        assert value["width"] == 4 and value["height"] == 4
        mask = np.zeros((4, 4), dtype=bool)
        if value["encodedMask"] == "bowl270":
            mask[1:3, 1:3] = True
        return mask

    union = decode_relation_union(entries, output_size=(8, 8), decoder=decoder)
    assert union.shape == (8, 8)
    assert union.sum() == 16


def test_sparse_pose_delta_uses_nearest_frames_and_explicit_joint_validity() -> None:
    annotations = {
        "9": [{"annotation3D": {"left-wrist": {"x": 1, "y": 2, "z": 3}}}],
        "24": [{"annotation3D": {"left-wrist": {"x": 2, "y": 4, "z": 6}}}],
    }
    first = nearest_pose_annotation(annotations, 10, max_skew_frames=1)
    second = nearest_pose_annotation(annotations, 25, max_skew_frames=1)
    assert first is not None and first[1] == 9
    assert second is not None and second[1] == 24
    delta, valid, usable = joint_delta(
        first[0], second[0], ("left-wrist", "right-wrist"), minimum_common_joints=1
    )
    np.testing.assert_allclose(delta[0], [1, 2, 3])
    assert valid.tolist() == [True, False]
    assert usable is True
