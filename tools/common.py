from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VIEW_KEYS = ("ego", "exo")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels_by_take(path: Path | str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row.get("take_uid") or row.get("take_id")): row
        for row in read_jsonl(path)
        if row.get("take_uid") or row.get("take_id")
    }


def load_npz_metadata(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        first_view = next((key for key in VIEW_KEYS if key in data), None)
        if first_view is None:
            first_view = next(key for key, value in data.items() if getattr(value, "ndim", 0) >= 1)
        num_rows = int(data[first_view].shape[0])
        take_uid = (
            np.asarray(data["take_uid"]).astype(str)
            if "take_uid" in data
            else np.asarray([str(index) for index in range(num_rows)])
        )
        timestamp = (
            np.asarray(data["timestamp"], dtype=np.float32)
            if "timestamp" in data
            else np.arange(num_rows, dtype=np.float32)
        )
        sample_id = (
            np.asarray(data["sample_id"]).astype(str)
            if "sample_id" in data
            else np.asarray([str(index) for index in range(num_rows)])
        )
    if len(take_uid) != num_rows or len(timestamp) != num_rows or len(sample_id) != num_rows:
        raise ValueError(f"Metadata arrays in {path} do not match view length {num_rows}")
    return {"take_uid": take_uid, "timestamp": timestamp, "sample_id": sample_id}


def group_indices_by_take(take_uid: Iterable[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, take in enumerate(take_uid):
        grouped[str(take)].append(index)
    return dict(grouped)


def evenly_spaced_indices(indices: list[int] | np.ndarray, count: int) -> list[int]:
    values = list(map(int, indices))
    if not values:
        return []
    if count <= 0 or len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(pos)] for pos in positions]


def ensure_parent(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path | str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = ensure_parent(path)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_score(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(max(0.0, min(1.0, value / scale)))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
