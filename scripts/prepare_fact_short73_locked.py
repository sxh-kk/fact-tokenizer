#!/usr/bin/env python3
"""Build, audit and freeze ``short73_t0p5_s1_8t_locked`` before annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.locked_split import (
    ShortLockedConfig,
    audit_locked_samples,
    build_short_locked_samples,
    load_jsonl,
    load_reference_records,
    perceptual_nearest_neighbor_audit,
    write_frozen_locked_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-jsonl", type=Path, required=True, help="Legacy 48-transition failure records.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["provisional", "final"],
        required=True,
        help="Provisional enables audit-only materialization; final requires perceptual NN evidence.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for old train, heldout, diagnostic and historical sample indexes.",
    )
    parser.add_argument(
        "--participant-fresh-takes",
        type=Path,
        help="Text/JSON/JSONL list of all takes that must also be participant-fresh.",
    )
    parser.add_argument(
        "--candidate-videos",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="VIEW=PATH",
        help="Final-stage candidate NPY; canonical freeze requires ego and exo.",
    )
    parser.add_argument(
        "--perceptual-reference-videos",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="VIEW=PATH",
        help="Old video NPY arrays; all are checked against candidate videos.",
    )
    parser.add_argument("--candidate-sample-ids", type=Path)
    parser.add_argument("--provisional-freeze", type=Path)
    parser.add_argument("--maximum-perceptual-similarity", type=float, default=0.995)
    parser.add_argument("--expected-takes", type=int, default=73)
    parser.add_argument("--takes-json", type=Path, help="EgoExo takes.json used to enrich capture/participant audits.")
    parser.add_argument("--allow-noncanonical-count", action="store_true", help="Test/debug only.")
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--reference must be NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"--reference must be NAME=PATH, got {value!r}")
    return name.strip(), Path(raw_path)


def load_take_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    if path.suffix.lower() == ".jsonl":
        return [str(row.get("take_uid", "")).strip() for row in load_jsonl(path) if row.get("take_uid")]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("take_uids", payload.get("takes", []))
        return [str(value.get("take_uid") if isinstance(value, dict) else value).strip() for value in payload]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def take_metadata(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("takes.json must contain a list")
    return {str(row["take_uid"]): row for row in payload}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enrich(rows: list[dict], metadata: dict[str, dict]) -> list[dict]:
    enriched = []
    for row in rows:
        source = metadata.get(str(row.get("take_uid", "")), {})
        value = dict(row)
        for key in ("participant_uid", "capture_uid", "physical_setting_uid"):
            if value.get(key) in (None, "") and source.get(key) is not None:
                value[key] = source[key]
        enriched.append(value)
    return enriched


def main() -> None:
    args = parse_args()
    config = ShortLockedConfig(expected_takes=args.expected_takes)
    metadata = take_metadata(args.takes_json)
    rows = enrich(load_jsonl(args.failed_jsonl), metadata)
    samples, selected_takes = build_short_locked_samples(
        rows,
        config=config,
        require_expected_takes=not args.allow_noncanonical_count,
    )
    references: dict[str, list[dict]] = {}
    for value in args.reference:
        name, path = parse_named_path(value)
        if name in references:
            raise ValueError(f"duplicate reference name: {name}")
        references[name] = enrich(load_reference_records(path), metadata)
    if not args.allow_noncanonical_count:
        if args.takes_json is None:
            raise ValueError("canonical short73 freeze requires --takes-json")
        required_reference_kinds = {
            "train": ("train",),
            "heldout": ("heldout", "val"),
            "diagnostic": ("diagnostic", "codelabel"),
            "history": ("history", "historical", "run"),
        }
        missing_kinds = [
            kind
            for kind, aliases in required_reference_kinds.items()
            if not any(any(alias in name.lower() for alias in aliases) for name in references)
        ]
        if missing_kinds:
            raise ValueError(
                "canonical short73 freeze requires train/heldout/diagnostic/history references; "
                f"missing {missing_kinds}"
            )
    participant_fresh = load_take_list(args.participant_fresh_takes)
    if not participant_fresh and metadata and references:
        reference_participants = {
            str(row.get("participant_uid"))
            for rows_in_group in references.values()
            for row in rows_in_group
            if row.get("participant_uid") not in (None, "")
        }
        participant_fresh = sorted(
            {
                str(row["take_uid"])
                for row in selected_takes
                if row.get("participant_uid") not in (None, "")
                and str(row["participant_uid"]) not in reference_participants
            }
        )
    audit = audit_locked_samples(
        samples,
        references,
        participant_fresh_takes=participant_fresh,
        frame_rate=config.frame_rate,
    )
    perceptual_reports = []
    candidate_paths = dict(args.candidate_videos)
    if len(candidate_paths) != len(args.candidate_videos):
        raise ValueError("duplicate candidate video view")
    if args.stage == "final":
        if not args.allow_noncanonical_count and set(candidate_paths) != {"ego", "exo"}:
            raise ValueError("canonical final short73 freeze requires ego and exo candidate videos")
        if args.candidate_sample_ids is None or args.provisional_freeze is None:
            raise ValueError("final short73 freeze requires candidate sample IDs and provisional freeze")
        provisional_freeze_path = (
            args.provisional_freeze / "freeze.json"
            if args.provisional_freeze.is_dir()
            else args.provisional_freeze
        )
        provisional = json.loads(provisional_freeze_path.read_text(encoding="utf-8"))
        if provisional.get("freeze_stage") != "provisional":
            raise ValueError("--provisional-freeze is not a provisional short73 freeze")
        provisional_samples_path = provisional_freeze_path.parent / "samples.jsonl"
        if sha256_file(provisional_samples_path) != provisional.get("samples_sha256"):
            raise ValueError("provisional sample manifest hash mismatch")
        provisional_ids = [row["sample_id"] for row in load_jsonl(provisional_samples_path)]
        candidate_ids = np.load(args.candidate_sample_ids, mmap_mode="r", allow_pickle=False).astype(str)
        if candidate_ids.tolist() != provisional_ids or provisional_ids != [sample.sample_id for sample in samples]:
            raise ValueError("candidate IDs do not match provisional and final short73 manifests exactly")
        audit["provisional_evidence"] = {
            "freeze_sha256": sha256_file(provisional_freeze_path),
            "samples_sha256": provisional["samples_sha256"],
            "candidate_sample_ids_sha256": sha256_file(args.candidate_sample_ids),
        }
    references_by_view: dict[str, list[Path]] = {}
    for view, path in args.perceptual_reference_videos:
        references_by_view.setdefault(view, []).append(path)
    if candidate_paths:
        for view, candidate_path in sorted(candidate_paths.items()):
            if view not in references_by_view:
                raise ValueError(f"candidate view {view!r} has no same-view perceptual references")
            candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
            if len(candidate) != len(samples):
                raise ValueError(
                    f"candidate view {view!r} has {len(candidate)} rows, expected {len(samples)}"
                )
            for path in references_by_view[view]:
                reference = np.load(path, mmap_mode="r", allow_pickle=False)
                report = perceptual_nearest_neighbor_audit(
                    candidate,
                    reference,
                    maximum_similarity=args.maximum_perceptual_similarity,
                )
                report.update(
                    {
                        "view": view,
                        "candidate_path": str(candidate_path),
                        "candidate_sha256": sha256_file(candidate_path),
                        "reference_path": str(path),
                        "reference_sha256": sha256_file(path),
                    }
                )
                perceptual_reports.append(report)
                if not report["passed"]:
                    audit["passed"] = False
                    audit["failures"].append(
                        f"perceptual nearest-neighbor audit failed for {view} against {path}"
                    )
    audit["perceptual_nearest_neighbor"] = {
        "status": "complete" if perceptual_reports else "deferred_to_final_freeze",
        "reports": perceptual_reports,
    }
    audit["participant_fresh_detected_count"] = len(participant_fresh)
    freeze = write_frozen_locked_manifest(
        args.output_dir,
        samples,
        selected_takes,
        config,
        audit,
        stage=args.stage,
    )
    print(json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
