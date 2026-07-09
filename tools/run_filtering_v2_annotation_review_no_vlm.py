#!/usr/bin/env python3
"""Build filtering_v2 human annotation review CSV without VLM features.

This script prepares the first human-guided filtering batch:

1. extract fast metadata/data-side proxy features,
2. merge features with the filtering policy,
3. export annotation_batch_v2_review.csv for manual labeling,
4. optionally build Ego/Exo contact sheets for selected review takes.

It intentionally does not call VLM tools and does not train/apply a ranker.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-split-dir",
        type=Path,
        required=True,
        help="Directory containing train_by_take.npz / heldout_by_take.npz and matching *_labels.jsonl files.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/filtering_v2_annotation_review_no_vlm")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/filter_policy_v2.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--splits", nargs="+", choices=["train", "heldout"], default=["train", "heldout"])
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"], metavar=("EGO_KEY", "EXO_KEY"))
    parser.add_argument(
        "--feature-mode",
        choices=["metadata", "video"],
        default="metadata",
        help="metadata is fast and does not read video tensors; video extracts motion proxies from NPZ arrays.",
    )
    parser.add_argument("--max-review", type=int, default=150)
    parser.add_argument("--random-audit-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-contact-frames", type=int, default=12)
    parser.add_argument("--thumb-size", type=int, default=112)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--skip-contact-sheets", dest="skip_contact_sheets", action="store_true")
    parser.add_argument("--with-contact-sheets", dest="skip_contact_sheets", action="store_false")
    parser.add_argument(
        "--contact-sheets-before-review",
        action="store_true",
        help="Slow path: build contact sheets for all takes before selecting the active-review batch.",
    )
    parser.set_defaults(skip_contact_sheets=True)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(shlex.quote(str(part)) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def split_paths(base: Path, split: str) -> tuple[Path, Path]:
    return base / f"{split}_by_take.npz", base / f"{split}_labels.jsonl"


def combine_csv(paths: list[Path], out: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in seen:
                    seen.add(name)
                    fieldnames.append(name)
            rows.extend(reader)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_review_contact_paths(review_csv: Path, manifests: list[Path]) -> None:
    contact_paths: dict[str, str] = {}
    for manifest in manifests:
        if not manifest.exists():
            continue
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                take_uid = row.get("take_uid", "")
                if take_uid:
                    contact_paths[take_uid] = row.get("contact_sheet_path", "")

    with review_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "contact_sheet_path" not in fieldnames:
        fieldnames.append("contact_sheet_path")
    for row in rows:
        take_uid = row.get("take_uid", "")
        if take_uid in contact_paths:
            row["contact_sheet_path"] = contact_paths[take_uid]
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_contact_sheets(args: argparse.Namespace, split: str, take_list_csv: Path | None = None) -> Path:
    npz, labels = split_paths(args.base_split_dir, split)
    if not npz.exists():
        raise FileNotFoundError(npz)
    if not labels.exists():
        raise FileNotFoundError(labels)

    split_dir = args.out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = split_dir / "contact_sheets"
    command = [
        args.python,
        "tools/make_take_contact_sheets.py",
        "--split",
        npz,
        "--labels-jsonl",
        labels,
        "--out",
        contact_dir,
        "--view-keys",
        *args.view_keys,
        "--num-frames",
        args.num_contact_frames,
        "--thumb-size",
        args.thumb_size,
        "--image-format",
        args.image_format,
    ]
    if take_list_csv is not None:
        command.extend(["--take-list-csv", take_list_csv])
    run(command)
    return contact_dir / "contact_sheet_manifest.csv"

def run_split(args: argparse.Namespace, split: str) -> Path:
    npz, labels = split_paths(args.base_split_dir, split)
    if not npz.exists():
        raise FileNotFoundError(npz)
    if not labels.exists():
        raise FileNotFoundError(labels)

    split_dir = args.out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    contact_manifest: Path | None = None
    if not args.skip_contact_sheets and args.contact_sheets_before_review:
        contact_manifest = run_contact_sheets(args, split)

    base_scores = split_dir / "relevance_v0.csv"
    relevance_command = [
        args.python,
        "tools/extract_relevance_features.py",
        "--split",
        npz,
        "--split-name",
        split,
        "--labels-jsonl",
        labels,
        "--out",
        base_scores,
        "--view-keys",
        *args.view_keys,
    ]
    if args.feature_mode == "metadata":
        relevance_command.append("--metadata-only")
    if contact_manifest is not None:
        relevance_command.extend(["--contact-sheet-manifest", contact_manifest])
    run(relevance_command)

    merged = split_dir / "take_relevance_scores_v2.csv"
    merge_command = [
        args.python,
        "tools/merge_relevance_features.py",
        "--base",
        base_scores,
        "--policy",
        args.policy,
        "--out",
        merged,
    ]
    if args.feature_mode == "video":
        ego_features = split_dir / "ego_hand_object_proxy.csv"
        run(
            [
                args.python,
                "tools/extract_ego_hand_object_features.py",
                "--split",
                npz,
                "--split-name",
                split,
                "--out",
                ego_features,
                "--view-key",
                args.view_keys[0],
            ]
        )

        exo_features = split_dir / "exo_pose_phase_proxy.csv"
        run(
            [
                args.python,
                "tools/extract_exo_pose_phase_features.py",
                "--split",
                npz,
                "--split-name",
                split,
                "--out",
                exo_features,
                "--view-key",
                args.view_keys[1],
            ]
        )
        merge_command.extend(["--ego-hand-object", ego_features, "--exo-pose-phase", exo_features])
    run(merge_command)
    return merged


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    score_paths = [run_split(args, split) for split in args.splits]
    all_scores = args.out_dir / "take_relevance_scores_v2_all_no_vlm.csv"
    combine_csv(score_paths, all_scores)

    review_csv = args.out_dir / "annotation_batch_v2_review.csv"
    run(
        [
            args.python,
            "tools/export_active_review_csv.py",
            "--scores",
            all_scores,
            "--out",
            review_csv,
            "--max-review",
            args.max_review,
            "--random-audit-fraction",
            args.random_audit_fraction,
            "--seed",
            args.seed,
        ]
    )

    if not args.skip_contact_sheets and not args.contact_sheets_before_review:
        manifests = [run_contact_sheets(args, split, review_csv) for split in args.splits]
        update_review_contact_paths(review_csv, manifests)

    print("\nDone. Human annotation files:")
    print(f"  scores: {all_scores}")
    print(f"  review: {review_csv}")
    print("\nNext step: copy/fill the annotation columns and save as annotation_batch_v2_labeled.csv")


if __name__ == "__main__":
    main()
