#!/usr/bin/env python3
"""Freeze a hash-bound RGB/t→t+0.5s contract for FACT paired NPY arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CORE_ARRAYS = ("ego", "exo", "sample_id", "take_uid", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--producer-code",
        action="append",
        type=Path,
        default=[],
        help="Code evidence for BGR→RGB conversion and the 0.5-second endpoint default.",
    )
    parser.add_argument("--producer-report", type=Path)
    parser.add_argument("--parent-contract", type=Path)
    parser.add_argument("--source-row-index", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in CORE_ARRAYS
    }
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"unaligned source arrays: {lengths}")
    for view in ("ego", "exo"):
        array = arrays[view]
        if array.ndim != 5 or array.shape[1] != 2 or array.dtype != np.uint8:
            raise ValueError(f"{view}.npy must be uint8 Nx2 endpoints, got {array.shape}/{array.dtype}")
    files = {
        name: {
            "sha256": sha256_file(directory / f"{name}.npy"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        for name, array in arrays.items()
    }
    return arrays, files


def validate_direct_evidence(
    code_paths: list[Path],
    report_path: Path | None,
    source_dir: Path,
    files: dict[str, dict],
) -> tuple[dict, list[str]]:
    if report_path is None:
        raise ValueError("direct source contract requires an empirical --producer-report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "fact-npy-source-semantic-audit-v1" or report.get("passed") is not True:
        raise ValueError("producer report is not a passed raw-video semantic audit")
    if Path(report.get("source_dir", "")).resolve() != source_dir.resolve() or report.get("files") != files:
        raise ValueError("producer report does not bind the current source arrays")
    if report.get("color_space") != "RGB":
        raise ValueError("producer report does not declare RGB")
    if float(report.get("transition_seconds", -1.0)) != 0.5:
        raise ValueError("producer report does not declare 0.5 seconds")
    if report.get("endpoint_semantics") != ["t", "t+0.5s"]:
        raise ValueError("producer report endpoint semantics mismatch")
    audited_ids = [str(value) for value in report.get("audited_sample_ids", [])]
    expected_ids_hash = hashlib.sha256("".join(value + "\n" for value in sorted(audited_ids)).encode()).hexdigest()
    if (
        not audited_ids
        or len(audited_ids) != len(set(audited_ids))
        or report.get("audited_sample_ids_sha256") != expected_ids_hash
        or int(report.get("audited_samples", -1)) != len(audited_ids)
        or int(report.get("exact_view_pair_matches", -1)) != 2 * len(audited_ids)
    ):
        raise ValueError("producer report has an invalid audited-sample identity contract")
    evidence: dict = {
        "mode": "semantic_audit_against_raw_videos",
        "producer_report": str(report_path),
        "producer_report_sha256": sha256_file(report_path),
        "producer_code_sha256": {str(path): sha256_file(path) for path in code_paths},
    }
    return evidence, sorted(audited_ids)


def validate_parent_subset(
    arrays: dict[str, np.ndarray], parent_contract_path: Path, row_index_path: Path
) -> tuple[dict, dict, list[str]]:
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))
    if parent_contract.get("schema") != "fact-npy-source-contract-v1":
        raise ValueError("parent source contract schema mismatch")
    if parent_contract.get("color_space") != "RGB" or parent_contract.get("transition_seconds") != 0.5:
        raise ValueError("parent source semantics mismatch")
    parent_dir = Path(parent_contract["source_dir"])
    parent_arrays, parent_files = inspect_source(parent_dir)
    if parent_files != parent_contract.get("files"):
        raise ValueError("parent contract no longer matches parent NPY files")
    indices = np.load(row_index_path, mmap_mode="r", allow_pickle=False)
    if indices.ndim != 1 or len(indices) != len(arrays["sample_id"]):
        raise ValueError("source_row_index must align one-to-one with the filtered source")
    indices = np.asarray(indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(parent_arrays["sample_id"])).any():
        raise ValueError("source_row_index contains duplicate or out-of-range rows")
    for name in CORE_ARRAYS:
        for start in range(0, len(indices), 64):
            selected = np.asarray(parent_arrays[name][indices[start : start + 64]])
            current = np.asarray(arrays[name][start : start + len(selected)])
            if not np.array_equal(current, selected):
                raise ValueError(f"filtered source {name}.npy differs from parent rows at {start}")
    evidence = {
        "mode": "exact_parent_row_subset",
        "parent_contract": str(parent_contract_path),
        "parent_contract_sha256": sha256_file(parent_contract_path),
        "source_row_index": str(row_index_path),
        "source_row_index_sha256": sha256_file(row_index_path),
    }
    current_ids = {str(value) for value in arrays["sample_id"]}
    audited_ids = sorted(current_ids & set(parent_contract.get("audited_sample_ids", [])))
    if not audited_ids:
        raise ValueError("parent source contract has no audited IDs in the filtered subset")
    return evidence, parent_contract, audited_ids


def main() -> None:
    args = parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite source contract: {args.output_json}")
    arrays, files = inspect_source(args.input_dir)
    if (args.parent_contract is None) != (args.source_row_index is None):
        raise ValueError("--parent-contract and --source-row-index must be supplied together")
    if args.parent_contract is not None:
        producer_evidence, parent, audited_sample_ids = validate_parent_subset(
            arrays,
            args.parent_contract,
            args.source_row_index,
        )
        producer_evidence["inherited_producer_evidence"] = parent.get("producer_evidence")
    else:
        producer_evidence, audited_sample_ids = validate_direct_evidence(
            args.producer_code,
            args.producer_report,
            args.input_dir,
            files,
        )
    contract = {
        "schema": "fact-npy-source-contract-v1",
        "source_dir": str(args.input_dir.resolve()),
        "rows": len(arrays["sample_id"]),
        "color_space": "RGB",
        "transition_seconds": 0.5,
        "endpoint_semantics": ["t", "t+0.5s"],
        "files": files,
        "audited_sample_ids": audited_sample_ids,
        "audited_sample_ids_sha256": hashlib.sha256(
            "".join(value + "\n" for value in audited_sample_ids).encode("utf-8")
        ).hexdigest(),
        "producer_evidence": producer_evidence,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temp.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.output_json)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
