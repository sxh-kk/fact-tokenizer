from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def identity(directory: Path, name: str) -> dict:
    path = directory / f"{name}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def test_materialize_transition_subset_preserves_frozen_row_order(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    parent = tmp_path / "parent"
    parent.mkdir()
    ego = np.arange(4 * 2 * 4 * 4 * 3, dtype=np.uint8).reshape(4, 2, 4, 4, 3)
    np.save(parent / "ego.npy", ego)
    np.save(parent / "exo.npy", ego[::-1])
    np.save(parent / "sample_id.npy", np.asarray(["a", "b", "c", "d"]))
    np.save(parent / "take_uid.npy", np.asarray(["ta", "tb", "tc", "td"]))
    np.save(parent / "timestamp.npy", np.asarray([1, 2, 3, 4], dtype=np.float32))
    frame_index = np.asarray([30, 60, 90, 120], dtype=np.int64)
    np.save(parent / "frame_index.npy", frame_index)
    files = {
        name: identity(parent, name)
        for name in ("ego", "exo", "sample_id", "take_uid", "timestamp")
    }
    (parent / "materialization_report.json").write_text(
        json.dumps(
            {
                "schema": "fact-npy-transition-rebuild-v1",
                "color_space": "RGB",
                "transition_seconds": 0.5,
                "endpoint_semantics": ["t", "t+0.5s"],
                "frame_rate_hz": 30.0,
                "endpoint_offset_frames": 15,
                "frame_index": identity(parent, "frame_index"),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    indices = np.asarray([3, 1], dtype=np.int64)
    index_path = tmp_path / "index.npy"
    np.save(index_path, indices)
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in ("sample_id", "take_uid", "timestamp"):
        np.save(reference / f"{name}.npy", np.load(parent / f"{name}.npy")[indices])
    output = tmp_path / "subset"
    subprocess.run(
        [
            sys.executable,
            "scripts/materialize_fact_npy_subset.py",
            "--parent-dir",
            str(parent),
            "--row-index-npy",
            str(index_path),
            "--metadata-reference-dir",
            str(reference),
            "--output-dir",
            str(output),
        ],
        cwd=root,
        check=True,
        text=True,
    )
    assert np.array_equal(np.load(output / "ego.npy"), ego[indices])
    assert np.array_equal(np.load(output / "sample_id.npy"), np.asarray(["d", "b"]))
    assert np.array_equal(np.load(output / "frame_index.npy"), frame_index[indices])
    report = json.loads((output / "materialization_report.json").read_text(encoding="utf-8"))
    assert report["transition_seconds"] == 0.5
    assert report["rows"] == 2
    assert report["frame_index"] == identity(output, "frame_index")
    float_index_path = tmp_path / "float_index.npy"
    np.save(float_index_path, indices.astype(np.float64))
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_fact_npy_subset.py",
            "--parent-dir",
            str(parent),
            "--row-index-npy",
            str(float_index_path),
            "--metadata-reference-dir",
            str(reference),
            "--output-dir",
            str(tmp_path / "float_subset"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "integer array" in rejected.stderr


def test_materialize_transition_subset_rejects_tampered_frame_index(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    image = np.zeros((1, 2, 2, 2, 3), dtype=np.uint8)
    for name, value in {
        "ego": image,
        "exo": image,
        "sample_id": np.asarray(["take:0.000"]),
        "take_uid": np.asarray(["take"]),
        "timestamp": np.asarray([0.0], dtype=np.float32),
        "frame_index": np.asarray([0], dtype=np.int64),
    }.items():
        np.save(parent / f"{name}.npy", value)
    files = {
        name: identity(parent, name)
        for name in ("ego", "exo", "sample_id", "take_uid", "timestamp")
    }
    (parent / "materialization_report.json").write_text(
        json.dumps(
            {
                "schema": "fact-npy-transition-rebuild-v1",
                "color_space": "RGB",
                "transition_seconds": 0.5,
                "endpoint_semantics": ["t", "t+0.5s"],
                "frame_rate_hz": 30.0,
                "endpoint_offset_frames": 15,
                "files": files,
                "frame_index": identity(parent, "frame_index"),
            }
        ),
        encoding="utf-8",
    )
    np.save(parent / "frame_index.npy", np.asarray([1], dtype=np.int64))
    index_path = tmp_path / "index.npy"
    np.save(index_path, np.asarray([0], dtype=np.int64))
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in ("sample_id", "take_uid", "timestamp"):
        np.save(reference / f"{name}.npy", np.load(parent / f"{name}.npy"))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_fact_npy_subset.py",
            "--parent-dir",
            str(parent),
            "--row-index-npy",
            str(index_path),
            "--metadata-reference-dir",
            str(reference),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "frame_index" in result.stderr
