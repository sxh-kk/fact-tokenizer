from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fact_tokenizer.effect_data import EffectClipSpec, FACTEffectNPYDataset


def _write_effect_arrays(path: Path, *, views: tuple[str, ...] = ("ego",), samples: int = 3) -> None:
    path.mkdir()
    base = np.arange(samples * 5 * 4 * 6 * 3, dtype=np.uint32).reshape(samples, 5, 4, 6, 3)
    for offset, view in enumerate(views):
        np.save(path / f"{view}.npy", ((base + offset * 17) % 256).astype(np.uint8))
    np.save(path / "sample_id.npy", np.asarray([f"sample-{index}" for index in range(samples)]))
    np.save(path / "take_uid.npy", np.asarray(["take-b", "take-a", "take-b"][:samples]))
    np.save(path / "timestamp.npy", np.arange(samples, dtype=np.float32) + 0.25)


def _write_manifest(path: Path, roles: tuple[str, ...]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "role",
                "world_mask",
                "token_mask",
                "geometry_mask",
                "contact_mask",
            ],
        )
        writer.writeheader()
        for index, role in enumerate(roles):
            writer.writerow(
                {
                    "sample_id": f"sample-{index}",
                    "role": role,
                    "world_mask": "1",
                    "token_mask": str(index % 2),
                    "geometry_mask": "true",
                    "contact_mask": "false",
                }
            )
    return path


def test_one_view_is_mmap_backed_and_copies_only_requested_sample(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir)
    calls = []
    original_load = np.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    spec = EffectClipSpec(history_frames=2, future_frames=2, source_current_index=2)
    dataset = FACTEffectNPYDataset(input_dir, spec, view_keys=("ego",), role="probe")

    assert calls and all(call["mmap_mode"] == "r" for call in calls)
    assert all(call["allow_pickle"] is False for call in calls)
    assert isinstance(dataset._view_arrays["ego"], np.memmap)

    item = dataset[0]
    assert set(item) == {
        "views",
        "camera_contexts",
        "current_index",
        "sample_id",
        "take_uid",
        "take_index",
        "timestamp",
        "quality_bucket",
        "quality_weight",
        "capability_masks",
    }
    assert set(item["views"]) == {"ego"}
    assert item["camera_contexts"] == {}
    assert item["views"]["ego"].shape == (4, 3, 4, 6)
    assert item["views"]["ego"].dtype == torch.float32
    assert item["current_index"].item() == 1
    assert item["sample_id"] == "sample-0"
    assert item["take_uid"] == "take-b"
    assert item["take_index"].item() == 1
    assert item["timestamp"].item() == pytest.approx(0.25)
    assert item["quality_bucket"] == "unlabeled"
    assert item["quality_weight"].item() == pytest.approx(1.0)
    assert not any(mask.item() for mask in item["capability_masks"].values())

    original = dataset[0]["views"]["ego"][0, 0, 0, 0].item()
    item["views"]["ego"].fill_(0.0)
    assert dataset[0]["views"]["ego"][0, 0, 0, 0].item() == pytest.approx(original)


def test_two_views_manifest_join_and_capability_masks(tmp_path: Path) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir, views=("ego", "exo"))
    manifest = _write_manifest(tmp_path / "manifest.csv", ("train", "train", "train"))
    np.save(input_dir / "current_index.npy", np.asarray([2, 2, 2], dtype=np.int64))

    dataset = FACTEffectNPYDataset(
        input_dir,
        EffectClipSpec(history_frames=2, future_frames=1),
        manifest=manifest,
        role="train",
    )
    item = dataset[1]

    assert set(item["views"]) == {"ego", "exo"}
    assert all(view.shape == (3, 3, 4, 6) for view in item["views"].values())
    assert item["sample_id"] == "sample-1"
    assert item["take_uid"] == "take-a"
    assert item["take_index"].item() == 0
    assert {name: value.item() for name, value in item["capability_masks"].items()} == {
        "world": True,
        "token": True,
        "geometry": True,
        "contact": False,
    }


def test_camera_context_sidecars_remain_mmap_and_are_returned_per_view(tmp_path: Path) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir, views=("ego", "exo"))
    np.save(input_dir / "ego_camera_context.npy", np.arange(18, dtype=np.float32).reshape(3, 6))
    np.save(input_dir / "exo_context.npy", np.ones((3, 2, 3), dtype=np.float32))
    dataset = FACTEffectNPYDataset(
        input_dir,
        view_keys=("ego", "exo"),
        camera_context_keys={"ego": "ego_camera_context", "exo": "exo_context"},
        role="probe",
    )
    assert all(isinstance(array, np.memmap) for array in dataset._camera_context_arrays.values())
    item = dataset[1]
    assert item["camera_contexts"]["ego"].shape == (6,)
    assert item["camera_contexts"]["exo"].shape == (6,)
    assert item["camera_contexts"]["exo"].tolist() == [1.0] * 6


@pytest.mark.parametrize("forbidden_role", ["probe", "diagnostic_candidate"])
def test_train_role_rejects_probe_and_diagnostic_manifest_rows(
    tmp_path: Path, forbidden_role: str
) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir)
    manifest = _write_manifest(tmp_path / "manifest.csv", ("train", forbidden_role, "train"))

    with pytest.raises(ValueError, match="train role refuses.*probe/diagnostic"):
        FACTEffectNPYDataset(input_dir, manifest=manifest, role="train")

    probe_dataset = FACTEffectNPYDataset(input_dir, manifest=manifest, role="probe")
    assert len(probe_dataset) == 3


def test_manifest_join_is_strict_and_one_to_one(tmp_path: Path) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir)
    manifest = _write_manifest(tmp_path / "manifest.csv", ("train", "train"))

    with pytest.raises(ValueError, match="Manifest is missing 1 dataset rows"):
        FACTEffectNPYDataset(input_dir, manifest=manifest)


def test_versioned_manifest_capabilities_and_diagnostic_quarantine(tmp_path: Path) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir)
    rows = []
    capability_names = ("rgb_paired", "camera_pose", "flow_2d", "depth", "flow_3d")
    for index in range(3):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "take_uid": "take-b" if index != 1 else "take-a",
                "split": "train",
                "row_index": index,
                "training_valid": True,
                "capability_validity": {
                    name: name in {"rgb_paired", "camera_pose"} for name in capability_names
                },
                "diagnostic_codelabel": None,
            }
        )
    manifest = tmp_path / "effect.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    dataset = FACTEffectNPYDataset(input_dir, manifest=manifest)
    masks = dataset[0]["capability_masks"]
    assert set(masks) == set(capability_names)
    assert masks["rgb_paired"].item() is True
    assert masks["camera_pose"].item() is True
    assert masks["flow_3d"].item() is False

    rows[1]["diagnostic_codelabel"] = {
        "value": "17",
        "source": "legacy_probe",
        "training_valid": False,
    }
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="train role refuses.*probe/diagnostic"):
        FACTEffectNPYDataset(input_dir, manifest=manifest, role="train")

    filtered = FACTEffectNPYDataset(
        input_dir,
        manifest=manifest,
        role="train",
        drop_nontraining=True,
    )
    assert len(filtered) == 2
    assert {filtered[index]["sample_id"] for index in range(len(filtered))} == {"sample-0", "sample-2"}


def test_train_take_exclusion_is_take_level_and_reported(tmp_path: Path) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir)
    manifest = _write_manifest(tmp_path / "manifest.csv", ("train", "train", "train"))
    dataset = FACTEffectNPYDataset(
        input_dir,
        manifest=manifest,
        role="train",
        excluded_take_uids={"take-b"},
    )
    assert dataset.excluded_sample_count == 2
    assert dataset.sample_ids == ("sample-1",)
    assert dataset.take_uids == ("take-a",)

    allowlisted = FACTEffectNPYDataset(
        input_dir,
        manifest=manifest,
        role="train",
        included_sample_ids=("sample-2", "sample-0"),
    )
    # Dataset order remains the immutable source/manifest order.
    assert allowlisted.sample_ids == ("sample-0", "sample-2")
    with pytest.raises(ValueError, match="missing 1 included"):
        FACTEffectNPYDataset(
            input_dir,
            manifest=manifest,
            role="train",
            included_sample_ids=("not-present",),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "locked_test"),
        ("split", "heldout"),
        ("source_dataset", "assembly101"),
    ],
)
def test_main_trainer_refuses_locked_and_assembly_probe_rows(
    tmp_path: Path, field: str, value: str
) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir)
    rows = [
        {
            "sample_id": f"sample-{index}",
            "take_uid": "take-b" if index != 1 else "take-a",
            "split": "train",
            "source_dataset": "egoexo",
            "training_valid": True,
        }
        for index in range(3)
    ]
    rows[0][field] = value
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="train role refuses"):
        FACTEffectNPYDataset(input_dir, manifest=manifest, role="train")
    filtered = FACTEffectNPYDataset(
        input_dir,
        manifest=manifest,
        role="train",
        drop_nontraining=True,
    )
    assert len(filtered) == 2


def test_clip_bounds_and_view_count_are_validated(tmp_path: Path) -> None:
    input_dir = tmp_path / "effect"
    _write_effect_arrays(input_dir, views=("ego", "exo"))
    np.save(input_dir / "third.npy", np.zeros((3, 5, 4, 6, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="one or two views"):
        FACTEffectNPYDataset(input_dir, view_keys=("ego", "exo", "third"))

    dataset = FACTEffectNPYDataset(
        input_dir,
        EffectClipSpec(history_frames=3, future_frames=3, source_current_index=2),
        view_keys=("ego",),
        role="probe",
    )
    with pytest.raises(IndexError, match="outside the source sample"):
        _ = dataset[0]
