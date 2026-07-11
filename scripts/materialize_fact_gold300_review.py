#!/usr/bin/env python3
"""Render the frozen gold300 Ego/Exo endpoints into a blind annotation pack."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import sys
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.gold_annotations import GOLD_SPLIT_COUNTS  # noqa: E402


FROZEN_FIELDS = (
    "sample_id",
    "take_uid",
    "gold_split",
    "timestamp",
    "source_dataset",
    "dual_annotation",
    "representation_training_valid",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name.strip(), Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--gold-freeze", type=Path)
    parser.add_argument(
        "--include-split",
        action="append",
        choices=sorted(GOLD_SPLIT_COUNTS),
        default=[],
        help="Render only these physically isolated splits. Canonical mode requires an explicit selection.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=DIR",
        help="NPY directory containing ego/exo/sample_id; repeat for train, heldout and short73.",
    )
    parser.add_argument(
        "--source-contract",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=JSON",
        help="Hash-bound RGB/t→t+0.5 source contract; required for every canonical source.",
    )
    parser.add_argument(
        "--annotation-template",
        action="append",
        type=Path,
        default=[],
        help="Frozen blank annotation CSV copied into the review pack.",
    )
    parser.add_argument("--annotation-guide", type=Path, required=True)
    parser.add_argument("--admin-output-dir", type=Path, required=True)
    parser.add_argument("--annotator-a-output-dir", type=Path, required=True)
    parser.add_argument("--annotator-b-output-dir", type=Path, required=True)
    parser.add_argument("--allow-noncanonical-count", action="store_true", help="Test/debug only.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_gold(rows: list[dict], canonical: bool) -> None:
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("gold manifest sample IDs must be non-empty and unique")
    missing_fields = [
        (index, field)
        for index, row in enumerate(rows)
        for field in FROZEN_FIELDS
        if field not in row
    ]
    if missing_fields:
        raise ValueError(f"gold manifest is missing frozen fields: {missing_fields[:5]}")
    invalid_splits = sorted({str(row["gold_split"]) for row in rows} - set(GOLD_SPLIT_COUNTS))
    if invalid_splits:
        raise ValueError(f"invalid gold_split values: {invalid_splits}")
    invalid_training = [
        row["sample_id"]
        for row in rows
        if str(row["representation_training_valid"]).strip().lower() not in {"false", "0"}
    ]
    if invalid_training:
        raise ValueError("gold review samples must all have representation_training_valid=false")
    if canonical:
        counts = Counter(str(row["gold_split"]) for row in rows)
        if dict(counts) != GOLD_SPLIT_COUNTS:
            raise ValueError(f"canonical gold review pack requires {GOLD_SPLIT_COUNTS}, got {dict(counts)}")


def validate_freeze(
    path: Path | None,
    manifest: Path,
    rows: list[dict],
    guide: Path | None,
    *,
    canonical: bool,
) -> dict | None:
    if path is None:
        if canonical:
            raise ValueError("canonical gold review pack requires --gold-freeze")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "fact-effect-gold-v1":
        raise ValueError("gold freeze schema mismatch")
    manifest_sha256 = sha256_file(manifest)
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("gold freeze does not bind the supplied gold manifest")
    counts = dict(sorted(Counter(str(row["gold_split"]) for row in rows).items()))
    if payload.get("sample_counts") != counts:
        raise ValueError("gold freeze sample counts do not match the supplied manifest")
    if payload.get("representation_training_valid") is not False:
        raise ValueError("gold freeze must prohibit representation training")
    if not isinstance(payload.get("seed"), int):
        raise ValueError("gold freeze seed is missing")
    if guide is not None:
        guide_record = payload.get("annotation_guide", {})
        if guide_record.get("path") != guide.name or guide_record.get("sha256") != sha256_file(guide):
            raise ValueError("annotation guide is not bound by the gold freeze")
    elif canonical:
        raise ValueError("canonical gold review pack requires a freeze-bound annotation guide")
    return payload


def load_sources(values: list[tuple[str, Path]]) -> tuple[dict[str, tuple[str, int]], dict[str, dict]]:
    if not values:
        raise ValueError("at least one --source is required")
    names = [name for name, _ in values]
    if len(names) != len(set(names)):
        raise ValueError("duplicate --source name")
    lookup: dict[str, tuple[str, int]] = {}
    sources: dict[str, dict] = {}
    for name, directory in values:
        arrays = {
            key: np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in ("ego", "exo", "sample_id", "take_uid", "timestamp")
        }
        lengths = {key: len(value) for key, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"unaligned source arrays for {name}: {lengths}")
        for view in ("ego", "exo"):
            array = arrays[view]
            if array.ndim != 5 or array.shape[1] != 2:
                raise ValueError(f"formal review source {name}/{view} must be Nx2 endpoints, got {array.shape}")
            if array.dtype != np.uint8:
                raise ValueError(f"formal review source {name}/{view} must be uint8, got {array.dtype}")
        if arrays["take_uid"].ndim != 1 or arrays["timestamp"].ndim != 1:
            raise ValueError(f"source metadata for {name} must be one-dimensional")
        frame_index_path = directory / "frame_index.npy"
        frame_indices = None
        frame_index_identity = None
        if frame_index_path.is_file():
            frame_indices = np.load(frame_index_path, mmap_mode="r", allow_pickle=False)
            if (
                frame_indices.ndim != 1
                or len(frame_indices) != lengths["sample_id"]
                or not np.issubdtype(frame_indices.dtype, np.integer)
                or (frame_indices < 0).any()
            ):
                raise ValueError(f"source frame_index.npy for {name} is invalid")
            frame_index_identity = {
                "sha256": sha256_file(frame_index_path),
                "shape": list(frame_indices.shape),
                "dtype": str(frame_indices.dtype),
            }
        for index, raw_id in enumerate(arrays["sample_id"]):
            sample_id = text(raw_id)
            if sample_id in lookup:
                other = lookup[sample_id][0]
                raise ValueError(f"sample_id {sample_id!r} appears in both {other!r} and {name!r}")
            lookup[sample_id] = (name, index)
        sources[name] = {
            "directory": directory,
            "arrays": arrays,
            "rows": lengths["sample_id"],
            "frame_indices": frame_indices,
            "frame_index": frame_index_identity,
            "files": {
                key: {
                    "sha256": sha256_file(directory / f"{key}.npy"),
                    "shape": list(arrays[key].shape),
                    "dtype": str(arrays[key].dtype),
                }
                for key in arrays
            },
        }
    return lookup, sources


def validate_source_contracts(
    values: list[tuple[str, Path]],
    sources: dict[str, dict],
    required_sample_ids: dict[str, set[str]],
    *,
    canonical: bool,
) -> dict[str, dict]:
    paths = dict(values)
    if len(paths) != len(values):
        raise ValueError("duplicate --source-contract name")
    if canonical and set(paths) != set(sources):
        raise ValueError("canonical review pack requires one source contract per source")
    if set(paths) - set(sources):
        raise ValueError("source contract has no matching --source")
    validated: dict[str, dict] = {}
    for name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = sources[name]
        if payload.get("schema") != "fact-npy-source-contract-v1":
            raise ValueError(f"source contract schema mismatch for {name}")
        if Path(payload.get("source_dir", "")).resolve() != source["directory"].resolve():
            raise ValueError(f"source contract directory mismatch for {name}")
        if payload.get("color_space") != "RGB":
            raise ValueError(f"source contract color space is not RGB for {name}")
        if float(payload.get("transition_seconds", -1.0)) != 0.5:
            raise ValueError(f"source contract transition is not 0.5 seconds for {name}")
        if payload.get("endpoint_semantics") != ["t", "t+0.5s"]:
            raise ValueError(f"source contract endpoint semantics mismatch for {name}")
        if int(payload.get("rows", -1)) != source["rows"] or payload.get("files") != source["files"]:
            raise ValueError(f"source contract files do not match current NPY content for {name}")
        if canonical and (
            source["frame_index"] is None
            or payload.get("frame_index") != source["frame_index"]
        ):
            raise ValueError(f"canonical source contract does not bind frame_index.npy for {name}")
        audited_ids = [str(value) for value in payload.get("audited_sample_ids", [])]
        audited_hash = hashlib.sha256(
            "".join(value + "\n" for value in sorted(audited_ids)).encode("utf-8")
        ).hexdigest()
        if (
            len(audited_ids) != len(set(audited_ids))
            or payload.get("audited_sample_ids_sha256") != audited_hash
            or not required_sample_ids.get(name, set()).issubset(set(audited_ids))
        ):
            raise ValueError(f"source contract does not audit every selected gold sample for {name}")
        evidence = payload.get("producer_evidence", {})
        if canonical and evidence.get("mode") != "semantic_audit_against_raw_videos":
            raise ValueError(f"canonical source contract lacks raw-video semantic evidence for {name}")
        if evidence.get("mode") == "semantic_audit_against_raw_videos":
            report_path = Path(evidence.get("producer_report", ""))
            if not report_path.is_file() or sha256_file(report_path) != evidence.get("producer_report_sha256"):
                raise ValueError(f"source semantic report hash mismatch for {name}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("schema") != "fact-npy-source-semantic-audit-v1"
                or report.get("passed") is not True
                or Path(report.get("source_dir", "")).resolve() != source["directory"].resolve()
                or report.get("files") != source["files"]
                or sorted(str(value) for value in report.get("audited_sample_ids", [])) != sorted(audited_ids)
                or report.get("color_space") != "RGB"
                or float(report.get("transition_seconds", -1.0)) != 0.5
                or report.get("frame_selection_mode") != "frozen_frame_index_sidecar"
                or float(report.get("frame_rate_hz", -1.0)) != 30.0
                or int(report.get("endpoint_offset_frames", -1)) != 15
                or report.get("frame_index") != source["frame_index"]
                or Path(report.get("frame_index_npy", "")).resolve()
                != (source["directory"] / "frame_index.npy").resolve()
                or int(report.get("exact_view_pair_matches", -1)) != 2 * len(audited_ids)
                or float(report.get("maximum_t0_timestamp_distance_frames", 999.0)) > 0.501
            ):
                raise ValueError(f"source semantic report content mismatch for {name}")
            materialization_report = source["directory"] / "materialization_report.json"
            if (
                not materialization_report.is_file()
                or Path(report.get("materialization_report", "")).resolve()
                != materialization_report.resolve()
                or report.get("materialization_report_sha256")
                != sha256_file(materialization_report)
            ):
                raise ValueError(f"source semantic materialization evidence mismatch for {name}")
            frame_rows = report.get("audited_frame_rows", [])
            frame_rows_text = "".join(
                f"{row.get('sample_id')}\t{row.get('take_uid')}\t{row.get('frame_index')}\n"
                for row in frame_rows
            )
            source_lookup = {
                text(value): index for index, value in enumerate(source["arrays"]["sample_id"])
            }
            if (
                len(frame_rows) != len(audited_ids)
                or [str(row.get("sample_id")) for row in frame_rows] != sorted(audited_ids)
                or report.get("audited_frame_rows_sha256")
                != hashlib.sha256(frame_rows_text.encode("utf-8")).hexdigest()
            ):
                raise ValueError(f"source semantic frame mapping mismatch for {name}")
            for row in frame_rows:
                sample_id = str(row["sample_id"])
                index = source_lookup.get(sample_id)
                if (
                    index is None
                    or str(row.get("take_uid")) != text(source["arrays"]["take_uid"][index])
                    or int(row.get("frame_index", -1))
                    != int(source["frame_indices"][index])
                ):
                    raise ValueError(f"source semantic frame row differs for {name}/{sample_id}")
        validated[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "producer_evidence": payload.get("producer_evidence"),
        }
    return validated


def rgb_frame(value: np.ndarray) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3:
        raise ValueError(f"expected a 3D frame, got {frame.shape}")
    if frame.shape[-1] in (1, 3):
        result = frame
    elif frame.shape[0] in (1, 3):
        result = np.transpose(frame, (1, 2, 0))
    else:
        raise ValueError(f"cannot infer frame channel axis from {frame.shape}")
    if result.shape[-1] == 1:
        result = np.repeat(result, 3, axis=-1)
    if result.dtype != np.uint8:
        raise ValueError(f"review images must be uint8 RGB, got {result.dtype}")
    return result


def aligned_endpoints(source: dict, source_index: int, row: dict) -> tuple[np.ndarray, np.ndarray]:
    sample_id = str(row["sample_id"])
    source_take = text(source["arrays"]["take_uid"][source_index])
    source_timestamp = float(source["arrays"]["timestamp"][source_index])
    if source_take != str(row["take_uid"]):
        raise ValueError(f"source take_uid mismatch for {sample_id}")
    if not np.isfinite(source_timestamp) or abs(source_timestamp - float(row["timestamp"])) > 1e-3:
        raise ValueError(f"source timestamp mismatch for {sample_id}")
    ego = np.asarray(source["arrays"]["ego"][source_index])
    exo = np.asarray(source["arrays"]["exo"][source_index])
    return ego, exo


def render_sheet(ego: np.ndarray, exo: np.ndarray, review_id: str) -> Image.Image:
    if ego.ndim != 4 or exo.ndim != 4 or len(ego) != 2 or len(exo) != 2:
        raise ValueError("each view must contain exactly the two t/t+0.5s endpoints")
    panels = [
        ("Ego t", rgb_frame(ego[0])),
        ("Ego t+0.5s", rgb_frame(ego[-1])),
        ("Exo t", rgb_frame(exo[0])),
        ("Exo t+0.5s", rgb_frame(exo[-1])),
    ]
    height = max(frame.shape[0] for _, frame in panels)
    width = max(frame.shape[1] for _, frame in panels)
    label_height = 22
    header_height = 44
    sheet = Image.new("RGB", (2 * width, header_height + 2 * (height + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), f"Review {review_id}", fill="white", font=font)
    for panel_index, (label, frame) in enumerate(panels):
        row, column = divmod(panel_index, 2)
        x = column * width
        y = header_height + row * (height + label_height)
        image = Image.fromarray(frame, mode="RGB")
        fitted = ImageOps.contain(image, (width, height), method=Image.Resampling.BILINEAR)
        panel = Image.new("RGB", (width, height), "black")
        panel.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
        sheet.paste(panel, (x, y))
        draw.rectangle((x, y + height, x + width, y + height + label_height), fill="black")
        draw.text((x + 6, y + height + 4), label, fill="white", font=font)
    return sheet


def copy_inputs(
    paths: list[Path], guide: Path | None, manifest: Path, freeze: Path | None, output: Path
) -> dict[str, str]:
    sealed = output / "sealed"
    destination = sealed / "blank_templates"
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    copied: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        target = destination / path.name
        if target.exists():
            raise ValueError(f"duplicate annotation input filename: {path.name}")
        shutil.copy2(path, target)
        copied[str(target.relative_to(output))] = sha256_file(target)
    provenance = sealed / "provenance"
    provenance.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in [manifest, *([freeze] if freeze else [])]:
        if not path.is_file():
            raise FileNotFoundError(path)
        target = provenance / path.name
        shutil.copy2(path, target)
        copied[str(target.relative_to(output))] = sha256_file(target)
    if guide is not None:
        if not guide.is_file():
            raise FileNotFoundError(guide)
        target = output / guide.name
        shutil.copy2(guide, target)
        copied[str(target.relative_to(output))] = sha256_file(target)
    if os.name == "posix":
        sealed.chmod(0o700)
        for path in sealed.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        for path in [sealed, *sealed.rglob("*")]:
            expected = 0o700 if path.is_dir() else 0o600
            if stat.S_IMODE(path.stat().st_mode) != expected:
                raise PermissionError(f"private review input has unsafe mode: {path}")
    return copied


def normalize_frozen(field: str, value: object) -> object:
    if field == "timestamp":
        return float(value)
    if field in {"dual_annotation", "representation_training_valid"}:
        return str(value).strip().lower() in {"true", "1"}
    return str(value)


def validate_templates(
    paths: list[Path],
    gold_rows: list[dict],
    *,
    canonical: bool,
    expected_splits: set[str],
    freeze: dict | None = None,
) -> None:
    if canonical and len(paths) != len(expected_splits):
        raise ValueError("canonical review pack requires one blank template for every included split")
    if not paths:
        return
    frozen_by_id = {str(row["sample_id"]): row for row in gold_rows}
    template_ids: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {*FROZEN_FIELDS, "effect_label", "contact_label", "annotator_id"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"annotation template {path} is missing columns: {sorted(missing)}")
            allowed = {
                *FROZEN_FIELDS,
                "effect_label",
                "contact_label",
                "ambiguous_reason",
                "annotator_id",
                "notes",
            }
            unexpected = set(reader.fieldnames or []) - allowed
            if unexpected:
                raise ValueError(
                    f"annotation template {path} has forbidden prediction/weak columns: {sorted(unexpected)}"
                )
            for row_number, row in enumerate(reader, start=2):
                sample_id = str(row["sample_id"])
                if sample_id not in frozen_by_id:
                    raise ValueError(f"template {path}:{row_number} has unknown sample_id {sample_id!r}")
                if sample_id in template_ids:
                    raise ValueError(f"sample_id {sample_id!r} is duplicated across annotation templates")
                template_ids.add(sample_id)
                frozen = frozen_by_id[sample_id]
                for field in FROZEN_FIELDS:
                    if normalize_frozen(field, row[field]) != normalize_frozen(field, frozen[field]):
                        raise ValueError(f"template {path}:{row_number} changed frozen field {field}")
                editable = ("effect_label", "contact_label", "ambiguous_reason", "annotator_id", "notes")
                if any(str(row.get(field, "")).strip() for field in editable):
                    raise ValueError(f"template {path}:{row_number} is not blind/blank")
    if template_ids != set(frozen_by_id):
        missing = sorted(set(frozen_by_id) - template_ids)
        raise ValueError(f"annotation templates do not cover the frozen manifest: {missing[:5]}")
    if freeze is not None:
        bound = freeze.get("annotation_templates", {})
        expected = {
            str(record["path"]): str(record["template_sha256"])
            for split, record in bound.items()
            if split in expected_splits
        }
        actual = {path.name: sha256_file(path) for path in paths}
        if actual != expected:
            raise ValueError("annotation templates are not exactly bound by the gold freeze")


def build_delivery_artifact(
    staging: Path,
    *,
    role: str,
    task_source: Path,
    image_source_root: Path,
    inventory: list[dict],
    guide: Path,
    binding: dict,
) -> dict:
    staging.mkdir(parents=False)
    images = staging / "images"
    images.mkdir()
    public_inventory = []
    for row in inventory:
        source = image_source_root / row["image"]
        target = images / Path(row["image"]).name
        shutil.copy2(source, target)
        public_inventory.append(
            {
                "review_id": row["review_id"],
                "image": f"images/{target.name}",
                "image_sha256": sha256_file(target),
            }
        )
    task_path = staging / "task.csv"
    shutil.copy2(task_source, task_path)
    guide_path = staging / guide.name
    shutil.copy2(guide, guide_path)
    inventory_path = staging / "image_inventory.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_inventory),
        encoding="utf-8",
    )
    report = {
        "schema": "fact-gold300-annotator-delivery-v1",
        "artifact_role": role,
        "tasks": len(public_inventory),
        "task_sha256": sha256_file(task_path),
        "image_inventory_sha256": sha256_file(inventory_path),
        "annotation_guide_sha256": sha256_file(guide_path),
        "admin_binding": binding,
        "contains_frozen_sample_mapping": False,
        "contains_other_annotator_task": False,
        "weak_or_model_labels_in_pack": False,
    }
    report_path = staging / "delivery_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "delivery_manifest_sha256": sha256_file(report_path)}


def main() -> None:
    args = parse_args()
    canonical = not args.allow_noncanonical_count
    all_rows = load_jsonl(args.gold_manifest)
    validate_gold(all_rows, canonical=canonical)
    freeze = validate_freeze(
        args.gold_freeze,
        args.gold_manifest,
        all_rows,
        args.annotation_guide,
        canonical=canonical,
    )
    included_splits = set(args.include_split)
    if canonical and not included_splits:
        raise ValueError("canonical review pack requires an explicit --include-split selection")
    if not included_splits:
        included_splits = {str(row["gold_split"]) for row in all_rows}
    if "locked_test" in included_splits and len(included_splits) != 1:
        raise ValueError("locked_test must be rendered into a physically separate review pack")
    rows = [row for row in all_rows if str(row["gold_split"]) in included_splits]
    if not rows:
        raise ValueError("the selected review split is empty")
    validate_templates(
        args.annotation_template,
        rows,
        canonical=canonical,
        expected_splits=included_splits,
        freeze=freeze,
    )
    if canonical and args.annotation_guide is None:
        raise ValueError("canonical gold review pack requires --annotation-guide")
    lookup, sources = load_sources(args.source)
    missing = sorted({str(row["sample_id"]) for row in rows} - set(lookup))
    if missing:
        raise ValueError(f"{len(missing)} gold sample IDs are missing from sources: {missing[:5]}")
    required_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sample_id = str(row["sample_id"])
        required_by_source[lookup[sample_id][0]].add(sample_id)
    source_contracts = validate_source_contracts(
        args.source_contract,
        sources,
        required_by_source,
        canonical=canonical,
    )
    outputs = {
        "admin": args.admin_output_dir,
        "annotator_a": args.annotator_a_output_dir,
        "annotator_b": args.annotator_b_output_dir,
    }
    resolved = {name: path.resolve() for name, path in outputs.items()}
    if len(set(resolved.values())) != 3:
        raise ValueError("admin and annotator output directories must be physically distinct")
    for left_name, left in resolved.items():
        for right_name, right in resolved.items():
            if left_name != right_name and (left in right.parents or right in left.parents):
                raise ValueError("admin and annotator output directories cannot be nested")
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing review artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    staging = args.admin_output_dir.parent / f".gold-review-work-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, mode=0o700)
    if os.name == "posix" and stat.S_IMODE(staging.stat().st_mode) != 0o700:
        raise PermissionError("gold review working staging is not private")
    artifact_staging: dict[str, Path] = {}
    published: list[Path] = []
    try:
        copied = copy_inputs(
            args.annotation_template,
            args.annotation_guide,
            args.gold_manifest,
            args.gold_freeze,
            staging,
        )
        snapshot_guide = staging / args.annotation_guide.name
        if freeze is not None and sha256_file(snapshot_guide) != freeze["annotation_guide"]["sha256"]:
            raise ValueError("private guide snapshot differs from the gold freeze")
        expected_templates = {
            record["path"]: record["template_sha256"]
            for split, record in (freeze or {}).get("annotation_templates", {}).items()
            if split in included_splits
        }
        snapshot_templates = {
            path.name: sha256_file(path)
            for path in (staging / "sealed" / "blank_templates").glob("*.csv")
        }
        if freeze is not None and snapshot_templates != expected_templates:
            raise ValueError("private template snapshots differ from the gold freeze")
        selected_digest = hashlib.sha256()
        nonce = secrets.token_bytes(32)
        image_inventory: list[dict] = []
        mapping_rows: list[dict] = []
        task_rows: list[dict] = []
        review_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=1):
            sample_id = str(row["sample_id"])
            source_name, source_index = lookup[sample_id]
            source = sources[source_name]
            ego, exo = aligned_endpoints(source, source_index, row)
            for field, array in (("ego", ego), ("exo", exo)):
                header = json.dumps(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                    },
                    sort_keys=True,
                ).encode("utf-8")
                selected_digest.update(len(header).to_bytes(8, "little") + header)
                selected_digest.update(array.nbytes.to_bytes(8, "little") + array.tobytes(order="C"))
            review_id = hmac.new(nonce, sample_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
            if review_id in review_ids:
                raise RuntimeError("opaque review ID collision")
            review_ids.add(review_id)
            relative = Path("images") / f"{review_id}.png"
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            render_sheet(ego, exo, review_id).save(target, format="PNG", optimize=False)
            image_sha256 = sha256_file(target)
            image_inventory.append(
                {
                    "review_id": review_id,
                    "sample_id": sample_id,
                    "source_name": source_name,
                    "source_index": source_index,
                    "ego_sha256": hashlib.sha256(ego.tobytes(order="C")).hexdigest(),
                    "exo_sha256": hashlib.sha256(exo.tobytes(order="C")).hexdigest(),
                    "image": relative.as_posix(),
                    "image_sha256": image_sha256,
                }
            )
            mapping_rows.append(
                {
                    "review_id": review_id,
                    "image": relative.as_posix(),
                    "source_name": source_name,
                    "source_index": source_index,
                    **{field: row[field] for field in FROZEN_FIELDS},
                }
            )
            task_rows.append(
                {
                    "review_id": review_id,
                    "image": relative.as_posix(),
                    "effect_label": "",
                    "contact_label": "",
                    "ambiguous_reason": "",
                    "annotator_id": "",
                    "notes": "",
                }
            )
        sealed = staging / "sealed"
        sealed.mkdir(parents=True, exist_ok=True)
        mapping_path = sealed / "review_mapping.csv"
        mapping_fields = ["review_id", "image", "source_name", "source_index", *FROZEN_FIELDS]
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=mapping_fields)
            writer.writeheader()
            writer.writerows(mapping_rows)
        inventory_path = sealed / "image_inventory.jsonl"
        inventory_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in image_inventory),
            encoding="utf-8",
        )
        tasks = staging / "tasks"
        tasks.mkdir(parents=True, exist_ok=True)
        task_fields = [
            "review_id",
            "image",
            "effect_label",
            "contact_label",
            "ambiguous_reason",
            "annotator_id",
            "notes",
        ]
        task_paths = {
            "annotator_a": tasks / "annotator_a.csv",
            "annotator_b_dual_only": tasks / "annotator_b_dual_only.csv",
        }
        for role, task_path in task_paths.items():
            selected_tasks = task_rows
            if role == "annotator_b_dual_only":
                selected_tasks = [
                    task
                    for task, mapping in zip(task_rows, mapping_rows)
                    if normalize_frozen("dual_annotation", mapping["dual_annotation"])
                ]
            with task_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=task_fields)
                writer.writeheader()
                writer.writerows(selected_tasks)
        binding = {
            "gold_manifest_sha256": sha256_file(args.gold_manifest),
            "selected_ego_exo_content_sha256": selected_digest.hexdigest(),
            "review_id_nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        }
        report = {
            "schema": "fact-gold300-review-admin-v1",
            "samples": len(rows),
            "included_splits": sorted(included_splits),
            "split_counts": dict(sorted(Counter(str(row["gold_split"]) for row in rows).items())),
            "gold_manifest": str(args.gold_manifest),
            "gold_manifest_sha256": sha256_file(args.gold_manifest),
            "gold_freeze": str(args.gold_freeze) if args.gold_freeze else None,
            "gold_freeze_sha256": sha256_file(args.gold_freeze) if args.gold_freeze else None,
            "gold_seed": freeze.get("seed") if freeze else None,
            "source_color_space_contract": "RGB as written by the FACT NPY materializer",
            "transition_seconds": 0.5,
            "source_endpoints": 2,
            "sources": {
                name: {
                    "directory": str(source["directory"]),
                    "rows": source["rows"],
                    "files": source["files"],
                }
                for name, source in sorted(sources.items())
            },
            "source_contracts": source_contracts,
            "render": {
                "format": "lossless_png",
                "resize": "aspect_preserving_contain_with_black_letterbox",
                "panels": ["ego_t", "ego_t_plus_0p5", "exo_t", "exo_t_plus_0p5"],
                "header_contains_only_opaque_review_id": True,
            },
            **binding,
            "sealed_mapping_sha256": sha256_file(mapping_path),
            "sealed_image_inventory_sha256": sha256_file(inventory_path),
            "task_sha256": {role: sha256_file(path) for role, path in task_paths.items()},
            "dual_annotation_tasks": sum(
                normalize_frozen("dual_annotation", row["dual_annotation"]) for row in rows
            ),
            "annotation_inputs_sha256": copied,
            "labels_in_review_images": False,
            "sample_identity_in_review_images": False,
            "weak_or_model_labels_in_pack": False,
            "sealed_mapping_required_for_merge": True,
        }
        dual_ids = {
            mapping["review_id"]
            for mapping in mapping_rows
            if normalize_frozen("dual_annotation", mapping["dual_annotation"])
        }
        inventories = {
            "annotator_a": image_inventory,
            "annotator_b": [row for row in image_inventory if row["review_id"] in dual_ids],
        }
        for name, target in outputs.items():
            artifact_staging[name] = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
        delivery_reports = {
            "annotator_a": build_delivery_artifact(
                artifact_staging["annotator_a"],
                role="annotator_a_all_selected",
                task_source=task_paths["annotator_a"],
                image_source_root=staging,
                inventory=inventories["annotator_a"],
                guide=snapshot_guide,
                binding=binding,
            ),
            "annotator_b": build_delivery_artifact(
                artifact_staging["annotator_b"],
                role="annotator_b_dual_only",
                task_source=task_paths["annotator_b_dual_only"],
                image_source_root=staging,
                inventory=inventories["annotator_b"],
                guide=snapshot_guide,
                binding=binding,
            ),
        }
        report["delivery_manifests"] = delivery_reports
        admin_staging = artifact_staging["admin"]
        admin_staging.mkdir(parents=False, mode=0o700)
        if os.name == "posix" and stat.S_IMODE(admin_staging.stat().st_mode) != 0o700:
            raise PermissionError("admin review staging is not private")
        shutil.copytree(staging / "sealed", admin_staging / "sealed")
        shutil.copy2(staging / args.annotation_guide.name, admin_staging / args.annotation_guide.name)
        report_path = admin_staging / "review_pack_admin.json"
        temp_report = report_path.with_suffix(".tmp")
        temp_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_report.replace(report_path)
        for name in ("annotator_a", "annotator_b", "admin"):
            artifact_staging[name].replace(outputs[name])
            published.append(outputs[name])
        if os.name == "posix" and stat.S_IMODE(outputs["admin"].stat().st_mode) != 0o700:
            raise PermissionError("published admin artifact is not private")
        shutil.rmtree(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        for path in artifact_staging.values():
            shutil.rmtree(path, ignore_errors=True)
        for path in published:
            shutil.rmtree(path, ignore_errors=True)
        raise
    print(
        json.dumps(
            {**report, "published_artifacts": {name: str(path) for name, path in outputs.items()}},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
