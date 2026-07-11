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
    report = json.loads((output / "materialization_report.json").read_text(encoding="utf-8"))
    assert report["transition_seconds"] == 0.5
    assert report["rows"] == 2
