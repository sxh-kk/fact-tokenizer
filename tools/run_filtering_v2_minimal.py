#!/usr/bin/env python3
"""Run the minimal filtering_v2 validation pipeline on an existing FACT split."""

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
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/filtering_v2_minimal")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/filter_policy_v2.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--materialize-npz", action="store_true", default=True)
    parser.add_argument("--skip-materialize-npz", dest="materialize_npz", action="store_false")
    parser.add_argument("--fact-main-only-dirname", default="fact_main")
    parser.add_argument("--fact-main-strict-dirname", default="fact_main_strict")
    parser.add_argument("--fact-main-balanced-dirname", default="fact_main_balanced")
    parser.add_argument("--fact-main-plus-loco-dirname", default="fact_main_plus_loco25")
    parser.add_argument("--num-transitions", type=int, default=48)
    parser.add_argument("--loco-transitions", type=int, default=12)
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    parser.add_argument(
        "--selection-score-mode",
        choices=["temporal", "motion"],
        default="temporal",
        help="Use temporal for the fast MVP path; motion reloads video tensors for motion-aware row selection.",
    )
    parser.add_argument(
        "--build-negative-index",
        action="store_true",
        help="Also build same-take hard negative indices. This is heavy and is off by default for MVP validation.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def split_paths(base: Path, split: str) -> tuple[Path, Path]:
    return base / f"{split}_by_take.npz", base / f"{split}_labels.jsonl"


def combine_csv(paths: list[Path], out: Path) -> None:
    rows = []
    fieldnames: list[str] = []
    seen = set()
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


def run_split_features(args: argparse.Namespace, split: str) -> Path:
    npz, labels = split_paths(args.base_split_dir, split)
    if not npz.exists():
        raise FileNotFoundError(npz)
    if not labels.exists():
        raise FileNotFoundError(labels)
    split_dir = args.out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    contact_manifest = None
    if not args.skip_contact_sheets:
        contact_dir = split_dir / "contact_sheets"
        run(
            [
                args.python,
                "tools/make_take_contact_sheets.py",
                "--split",
                str(npz),
                "--labels-jsonl",
                str(labels),
                "--out",
                str(contact_dir),
            ]
        )
        contact_manifest = contact_dir / "contact_sheet_manifest.csv"

    base_scores = split_dir / "relevance_v0.csv"
    command = [
        args.python,
        "tools/extract_relevance_features.py",
        "--split",
        str(npz),
        "--split-name",
        split,
        "--labels-jsonl",
        str(labels),
        "--out",
        str(base_scores),
    ]
    if contact_manifest is not None:
        command.extend(["--contact-sheet-manifest", str(contact_manifest)])
    run(command)

    ego_features = split_dir / "ego_hand_object_proxy.csv"
    run(
        [
            args.python,
            "tools/extract_ego_hand_object_features.py",
            "--split",
            str(npz),
            "--split-name",
            split,
            "--out",
            str(ego_features),
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
            str(npz),
            "--split-name",
            split,
            "--out",
            str(exo_features),
            "--view-key",
            args.view_keys[1],
        ]
    )

    merged = split_dir / "take_relevance_scores_v2.csv"
    run(
        [
            args.python,
            "tools/merge_relevance_features.py",
            "--base",
            str(base_scores),
            "--ego-hand-object",
            str(ego_features),
            "--exo-pose-phase",
            str(exo_features),
            "--policy",
            str(args.policy),
            "--out",
            str(merged),
        ]
    )
    return merged


def materialize_filtered_npz(args: argparse.Namespace, filtered_split: Path, include_buckets: list[str], out_dir: Path) -> None:
    selections: dict[str, Path] = {}
    for split in ("train", "heldout"):
        npz, _labels = split_paths(args.base_split_dir, split)
        selection = args.out_dir / f"transition_selection_{out_dir.name}_{split}.csv"
        command = [
            args.python,
            "tools/build_transition_selection.py",
            "--npz",
            str(npz),
            "--split-name",
            split,
            "--filtered-split",
            str(filtered_split),
            "--out",
            str(selection),
            "--num-transitions",
            str(args.num_transitions),
            "--include-buckets",
            *include_buckets,
            "--view-keys",
            *args.view_keys,
            "--score-mode",
            args.selection_score_mode,
        ]
        if "loco_aux" in include_buckets:
            command.extend(["--bucket-num-transitions", f"loco_aux={args.loco_transitions}"])
        run(command)
        selections[split] = selection

    train_npz, _ = split_paths(args.base_split_dir, "train")
    heldout_npz, _ = split_paths(args.base_split_dir, "heldout")
    run(
        [
            args.python,
            "tools/build_filtered_npz.py",
            "--train-npz",
            str(train_npz),
            "--heldout-npz",
            str(heldout_npz),
            "--train-selection",
            str(selections["train"]),
            "--heldout-selection",
            str(selections["heldout"]),
            "--out-dir",
            str(out_dir),
        ]
    )
    if args.build_negative_index:
        run([args.python, "tools/build_same_take_negative_index.py", "--npz", str(out_dir / "train_by_take.npz"), "--out", str(out_dir / "same_take_negative_index_train.npz")])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_paths = [run_split_features(args, split) for split in ("train", "heldout")]
    all_scores = args.out_dir / "take_relevance_scores_v2_all.csv"
    combine_csv(score_paths, all_scores)

    review_csv = args.out_dir / "annotation_batch_v2_review.csv"
    run([args.python, "tools/export_active_review_csv.py", "--scores", str(all_scores), "--out", str(review_csv)])

    filtered_split = args.out_dir / "filtered_split_v2.json"
    run([args.python, "tools/build_filtered_split.py", "--ranked", str(all_scores), "--policy", str(args.policy), "--out", str(filtered_split)])
    strict_split = args.out_dir / "filtered_split_v2_fact_main_strict.json"
    run(
        [
            args.python,
            "tools/build_filtered_split.py",
            "--ranked",
            str(all_scores),
            "--policy",
            str(args.policy),
            "--fact-main-mode",
            "strict",
            "--out",
            str(strict_split),
        ]
    )
    balanced_split = args.out_dir / "filtered_split_v2_fact_main_balanced.json"
    run(
        [
            args.python,
            "tools/build_filtered_split.py",
            "--ranked",
            str(all_scores),
            "--policy",
            str(args.policy),
            "--fact-main-mode",
            "balanced",
            "--out",
            str(balanced_split),
        ]
    )
    run([args.python, "tools/audit_filtered_split.py", "--filtered-split", str(filtered_split), "--out", str(args.out_dir / "audit_report_v2.md")])
    run([args.python, "tools/audit_filtered_split.py", "--filtered-split", str(strict_split), "--out", str(args.out_dir / "audit_report_v2_fact_main_strict.md")])
    run([args.python, "tools/audit_filtered_split.py", "--filtered-split", str(balanced_split), "--out", str(args.out_dir / "audit_report_v2_fact_main_balanced.md")])

    labels = [str(args.base_split_dir / "train_labels.jsonl"), str(args.base_split_dir / "heldout_labels.jsonl")]
    run(
        [
            args.python,
            "tools/export_filtered_selected_takes.py",
            "--filtered-split",
            str(filtered_split),
            "--labels-jsonl",
            *labels,
            "--include-buckets",
            "fact_main",
            "--out",
            str(args.out_dir / "selected_takes_filtering_v2_fact_main.jsonl"),
        ]
    )
    run(
        [
            args.python,
            "tools/export_filtered_selected_takes.py",
            "--filtered-split",
            str(strict_split),
            "--labels-jsonl",
            *labels,
            "--include-buckets",
            "fact_main",
            "--out",
            str(args.out_dir / "selected_takes_filtering_v2_fact_main_strict.jsonl"),
        ]
    )
    run(
        [
            args.python,
            "tools/export_filtered_selected_takes.py",
            "--filtered-split",
            str(balanced_split),
            "--labels-jsonl",
            *labels,
            "--include-buckets",
            "fact_main",
            "--out",
            str(args.out_dir / "selected_takes_filtering_v2_fact_main_balanced.jsonl"),
        ]
    )

    if args.materialize_npz:
        materialize_filtered_npz(args, filtered_split, ["fact_main"], args.out_dir / "filtered_npz" / args.fact_main_only_dirname)
        materialize_filtered_npz(args, strict_split, ["fact_main"], args.out_dir / "filtered_npz" / args.fact_main_strict_dirname)
        materialize_filtered_npz(args, balanced_split, ["fact_main"], args.out_dir / "filtered_npz" / args.fact_main_balanced_dirname)
        materialize_filtered_npz(
            args,
            filtered_split,
            ["fact_main", "loco_aux"],
            args.out_dir / "filtered_npz" / args.fact_main_plus_loco_dirname,
        )

    print("")
    print(f"Minimal filtering_v2 artifacts written under {args.out_dir}")
    print(f"Filtered split: {filtered_split}")
    print(f"Strict split: {strict_split}")
    print(f"Balanced split: {balanced_split}")
    print(f"Review CSV: {review_csv}")


if __name__ == "__main__":
    main()
