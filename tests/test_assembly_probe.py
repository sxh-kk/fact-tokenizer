from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_assembly_npz_unpack_and_split_freeze_bind_input_hashes(tmp_path: Path) -> None:
    source = tmp_path / "assembly.npz"
    takes = np.asarray([f"take-{index // 2}" for index in range(20)])
    np.savez_compressed(
        source,
        ego=np.zeros((20, 2, 4, 4, 3), dtype=np.uint8),
        exo=np.ones((20, 2, 4, 4, 3), dtype=np.uint8),
        sample_id=np.asarray([f"s{index}" for index in range(20)]),
        take_uid=takes,
        timestamp=np.arange(20, dtype=np.float32),
        action_label=np.asarray(["a", "b"] * 10),
    )
    npy = tmp_path / "npy"
    subprocess.run(
        [
            sys.executable,
            "scripts/unpack_fact_assembly101.py",
            "--input-npz",
            str(source),
            "--output-dir",
            str(npy),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    unpack = json.loads((npy / "unpack_report.json").read_text(encoding="utf-8"))
    assert unpack["samples"] == 20
    frozen = tmp_path / "frozen"
    command = [
        sys.executable,
        "scripts/prepare_fact_assembly101_probe.py",
        "--input-dir",
        str(npy),
        "--output-dir",
        str(frozen),
    ]
    for index in range(8):
        command.extend(["--train-take", f"take-{index}"])
    for index in range(8, 10):
        command.extend(["--test-take", f"take-{index}"])
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads((frozen / "assembly101_prepare_report.json").read_text(encoding="utf-8"))
    assert report["split_file_sha256"] == hashlib.sha256(
        (frozen / "assembly101_split_frozen.json").read_bytes()
    ).hexdigest()
    assert report["input_array_sha256"]["action_label"] == hashlib.sha256(
        (npy / "action_label.npy").read_bytes()
    ).hexdigest()
