#!/usr/bin/env python3
"""Build a v7 continuous-effect JSONL manifest from mmap-backed NPY data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import (  # noqa: E402
    DiagnosticCodelabel,
    EffectCapability,
    build_records_from_npy_directory,
    write_manifest_jsonl,
)


def parse_capability_sidecar(value: str) -> tuple[EffectCapability, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Capability sidecars must use CAPABILITY=PATH syntax")
    capability_name, path_value = value.split("=", 1)
    try:
        capability = EffectCapability(capability_name.strip())
    except ValueError as exc:
        choices = ", ".join(capability.value for capability in EffectCapability)
        raise argparse.ArgumentTypeError(
            f"Unknown capability {capability_name!r}; expected one of: {choices}"
        ) from exc
    path = Path(path_value.strip())
    if not path_value.strip():
        raise argparse.ArgumentTypeError("Capability sidecar path must be non-empty")
    return capability, path


def parse_content_hash(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Content hashes must use NAME=SHA256 syntax")
    name, digest = (part.strip() for part in value.split("=", 1))
    if not name or len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise argparse.ArgumentTypeError("Content hash must be NAME followed by a 64-character SHA256")
    return name, digest.lower()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing ego/exo and metadata NPY files.")
    parser.add_argument("--split", required=True, help="Split name stored in every primary key, e.g. train or heldout.")
    parser.add_argument("--source-dataset", default="egoexo", help="Stable dataset namespace used in primary keys.")
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument(
        "--capability-npy",
        action="append",
        default=[],
        type=parse_capability_sidecar,
        metavar="CAPABILITY=PATH",
        help="Aligned boolean NPY sidecar; may be repeated. Camera pose does not imply depth or 3D flow.",
    )
    parser.add_argument(
        "--diagnostic-codelabels",
        type=Path,
        default=None,
        help="Optional CSV or JSONL legacy code labels, joined by sample_id and always quarantined.",
    )
    parser.add_argument("--diagnostic-sample-id-column", default="sample_id")
    parser.add_argument("--diagnostic-label-column", default="codelabel")
    parser.add_argument("--diagnostic-source", default="legacy_failed_codelabel")
    parser.add_argument(
        "--content-hash",
        action="append",
        default=[],
        type=parse_content_hash,
        metavar="NAME=SHA256",
        help="Repeatable immutable source/annotation hash copied into every record.",
    )
    return parser.parse_args(argv)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                rows.append(payload)
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_diagnostic_codelabels(
    path: Path | None,
    *,
    sample_id_column: str,
    label_column: str,
    source: str,
) -> dict[str, DiagnosticCodelabel]:
    if path is None:
        return {}
    rows = _read_rows(path)
    labels: dict[str, DiagnosticCodelabel] = {}
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get(sample_id_column, "")).strip()
        value = str(row.get(label_column, "")).strip()
        if not sample_id or not value:
            raise ValueError(
                f"Diagnostic codelabel row {row_number} requires non-empty "
                f"{sample_id_column!r} and {label_column!r}"
            )
        if sample_id in labels:
            raise ValueError(f"Duplicate diagnostic codelabel sample_id={sample_id!r}")
        metadata = {
            key: item
            for key, item in row.items()
            if key not in {sample_id_column, label_column, "training_valid"} and item not in {None, ""}
        }
        labels[sample_id] = DiagnosticCodelabel(
            value=value,
            source=source,
            training_valid=False,
            metadata=metadata,
        )
    return labels


def _capability_mapping(entries: Sequence[tuple[EffectCapability, Path]]) -> Mapping[EffectCapability, Path]:
    mapping: dict[EffectCapability, Path] = {}
    for capability, path in entries:
        if capability in mapping:
            raise ValueError(f"Duplicate --capability-npy for {capability.value}")
        mapping[capability] = path
    return mapping


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    capability_arrays = _capability_mapping(args.capability_npy)
    diagnostics = load_diagnostic_codelabels(
        args.diagnostic_codelabels,
        sample_id_column=args.diagnostic_sample_id_column,
        label_column=args.diagnostic_label_column,
        source=args.diagnostic_source,
    )
    records, audit = build_records_from_npy_directory(
        args.input_dir,
        split=args.split,
        source_dataset=args.source_dataset,
        view_keys=args.view_keys,
        capability_arrays=capability_arrays,
        diagnostic_codelabels=diagnostics,
        content_hashes=dict(args.content_hash),
    )
    write_manifest_jsonl(args.output_jsonl, records)
    audit_path = args.audit_json or args.output_jsonl.with_name(f"{args.output_jsonl.stem}_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(records)} effect manifest records to {args.output_jsonl}")
    print(f"Wrote manifest audit to {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
