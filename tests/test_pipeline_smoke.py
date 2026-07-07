from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_tool(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def make_npz(path: Path, take_count: int = 2, per_take: int = 8) -> Path:
    rng = np.random.default_rng(123)
    n = take_count * per_take
    ego = rng.integers(0, 255, size=(n, 2, 16, 16, 3), dtype=np.uint8)
    exo = rng.integers(0, 255, size=(n, 2, 16, 16, 3), dtype=np.uint8)
    take_uid = np.asarray([f"take_{take}" for take in range(take_count) for _ in range(per_take)])
    timestamp = np.asarray([time for _ in range(take_count) for time in range(per_take)], dtype=np.float32)
    sample_id = np.asarray([f"{take_uid[i]}:{timestamp[i]:.3f}" for i in range(n)])
    np.savez_compressed(path, ego=ego, exo=exo, take_uid=take_uid, timestamp=timestamp, sample_id=sample_id)
    return path


def write_labels(path: Path, take_count: int = 2) -> Path:
    rows = []
    parents = ["Cooking", "Basketball"]
    for index in range(take_count):
        rows.append(
            {
                "take_uid": f"take_{index}",
                "take_name": f"dummy_{index}",
                "parent_task_name": parents[index % len(parents)],
                "task_name": f"task_{index}",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_extract_features_and_annotation_export(tmp_path: Path) -> None:
    npz = make_npz(tmp_path / "split.npz")
    labels = write_labels(tmp_path / "labels.jsonl")
    scores = tmp_path / "scores.csv"
    run_tool(
        tmp_path,
        "tools/extract_relevance_features.py",
        "--split",
        str(npz),
        "--split-name",
        "train",
        "--labels-jsonl",
        str(labels),
        "--out",
        str(scores),
    )
    with scores.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"take_uid", "relevance_score", "auto_bucket"}.issubset(rows[0])

    annotation = tmp_path / "annotation.csv"
    run_tool(
        tmp_path,
        "tools/export_annotation_csv.py",
        "--scores",
        str(scores),
        "--sample-size",
        "2",
        "--out",
        str(annotation),
    )
    with annotation.open(newline="", encoding="utf-8") as handle:
        annotation_rows = list(csv.DictReader(handle))
    assert len(annotation_rows) == 2
    assert "usable_for" in annotation_rows[0]

    full_annotation = tmp_path / "annotation_full.csv"
    run_tool(
        tmp_path,
        "tools/export_annotation_csv.py",
        "--scores",
        str(scores),
        "--sample-size",
        "500",
        "--strategy",
        "v1_full_if_small",
        "--out",
        str(full_annotation),
    )
    with full_annotation.open(newline="", encoding="utf-8") as handle:
        full_rows = list(csv.DictReader(handle))
    assert len(full_rows) == 2


def test_validate_annotations_rejects_bad_values_and_accepts_valid_rows(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.csv"
    fieldnames = [
        "take_uid",
        "take_relevance",
        "ego_hand_visibility",
        "exo_body_visibility",
        "object_interaction",
        "phase_diversity",
        "usable_for",
        "notes",
    ]
    with annotation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "take_uid": "take_0",
                "take_relevance": "A_interaction_rich",
                "ego_hand_visibility": "2",
                "exo_body_visibility": "2",
                "object_interaction": "2",
                "phase_diversity": "2",
                "usable_for": "tokenizer_main",
                "notes": "",
            }
        )
    run_tool(tmp_path, "tools/validate_annotations.py", "--annotations", str(annotation))

    bad = tmp_path / "bad_annotation.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "take_uid": "take_1",
                "take_relevance": "D_scene_only",
                "ego_hand_visibility": "0",
                "exo_body_visibility": "0",
                "object_interaction": "0",
                "phase_diversity": "0",
                "usable_for": "tokenizer_main",
                "notes": "",
            }
        )
    result = run_tool_allow_failure(tmp_path, "tools/validate_annotations.py", "--annotations", str(bad))
    assert result.returncode != 0
    assert "scene-only" in result.stderr


def test_annotation_review_pack_copies_friendly_images(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    server_root = "/server/filtering_v0"
    image_dir = local_root / "contact_sheets_train"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "abc-123.jpg"
    image_path.write_bytes(b"fake-jpeg")
    annotations = tmp_path / "annotations.csv"
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "auto_bucket",
                "contact_sheet_path",
                "ego_hand_visibility",
                "exo_body_visibility",
                "notes",
                "object_interaction",
                "parent_task_name",
                "phase_diversity",
                "relevance_score",
                "take_relevance",
                "take_uid",
                "task_name",
                "usable_for",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "auto_bucket": "F_uncertain",
                "contact_sheet_path": f"{server_root}/contact_sheets_train/abc-123.jpg",
                "parent_task_name": "Bike Repair",
                "task_name": "Tighten Wheel",
                "relevance_score": "0.7",
                "take_uid": "abc-123",
            }
        )
    out_dir = tmp_path / "review_pack"
    run_tool(
        tmp_path,
        "tools/build_annotation_review_pack.py",
        "--annotations",
        str(annotations),
        "--out-dir",
        str(out_dir),
        "--path-prefix-from",
        server_root,
        "--path-prefix-to",
        str(local_root),
        "--copy-images",
    )
    review_csv = out_dir / "annotation_review.csv"
    with review_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["review_id"] == "0001"
    assert rows[0]["review_image"].startswith("review_images/0001_train_Bike_Repair_Tighten_Wheel_abc_123")
    assert (out_dir / rows[0]["review_image"]).exists()
    assert (out_dir / "index.html").exists()
    assert (out_dir / "ANNOTATOR_GUIDE.zh-CN.md").exists()


def run_tool_allow_failure(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_filtered_split_has_separate_train_heldout_takes(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.csv"
    with ranked.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "take_uid",
            "split",
            "parent_task_name",
            "task_name",
            "relevance_score",
            "interaction_score",
            "loco_score",
            "scene_only_score",
            "temporal_diversity_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "take_uid": "train_take",
                "split": "train",
                "parent_task_name": "Cooking",
                "task_name": "dummy",
                "relevance_score": 0.8,
                "interaction_score": 0.8,
                "loco_score": 0.2,
                "scene_only_score": 0.1,
                "temporal_diversity_score": 0.5,
            }
        )
        writer.writerow(
            {
                "take_uid": "heldout_take",
                "split": "heldout",
                "parent_task_name": "Bike Repair",
                "task_name": "dummy",
                "relevance_score": 0.8,
                "interaction_score": 0.8,
                "loco_score": 0.2,
                "scene_only_score": 0.1,
                "temporal_diversity_score": 0.5,
            }
        )
    out = tmp_path / "filtered.json"
    run_tool(
        tmp_path,
        "tools/build_filtered_split.py",
        "--ranked",
        str(ranked),
        "--policy",
        str(ROOT / "configs/filter_policy_v0.yaml"),
        "--out",
        str(out),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    train_takes = {row["take_uid"] for bucket in payload["splits"]["train"].values() for row in bucket}
    heldout_takes = {row["take_uid"] for bucket in payload["splits"]["heldout"].values() for row in bucket}
    assert train_takes == {"train_take"}
    assert heldout_takes == {"heldout_take"}
    assert train_takes.isdisjoint(heldout_takes)


def test_v1_split_prefers_usable_for_probabilities(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked_v1.csv"
    with ranked.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "take_uid",
            "split",
            "parent_task_name",
            "task_name",
            "relevance_score",
            "interaction_score",
            "loco_score",
            "scene_only_score",
            "temporal_diversity_score",
            "prob_tokenizer_main",
            "prob_loco_aux",
            "prob_discard",
            "prob_diagnostic_candidate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "take_uid": "main_take",
                "split": "train",
                "parent_task_name": "Cooking",
                "task_name": "dummy",
                "relevance_score": 0.8,
                "interaction_score": 0.1,
                "loco_score": 0.1,
                "scene_only_score": 0.1,
                "temporal_diversity_score": 0.5,
                "prob_tokenizer_main": 0.9,
                "prob_loco_aux": 0.1,
                "prob_discard": 0.0,
                "prob_diagnostic_candidate": 0.0,
            }
        )
        writer.writerow(
            {
                "take_uid": "loco_take",
                "split": "train",
                "parent_task_name": "Basketball",
                "task_name": "dummy",
                "relevance_score": 0.5,
                "interaction_score": 0.1,
                "loco_score": 0.1,
                "scene_only_score": 0.2,
                "temporal_diversity_score": 0.5,
                "prob_tokenizer_main": 0.1,
                "prob_loco_aux": 0.8,
                "prob_discard": 0.0,
                "prob_diagnostic_candidate": 0.0,
            }
        )
        writer.writerow(
            {
                "take_uid": "blocked_take",
                "split": "train",
                "parent_task_name": "Music",
                "task_name": "dummy",
                "relevance_score": 0.9,
                "interaction_score": 0.9,
                "loco_score": 0.9,
                "scene_only_score": 0.1,
                "temporal_diversity_score": 0.5,
                "prob_tokenizer_main": 0.9,
                "prob_loco_aux": 0.0,
                "prob_discard": 0.8,
                "prob_diagnostic_candidate": 0.0,
            }
        )
    out = tmp_path / "filtered_v1.json"
    run_tool(
        tmp_path,
        "tools/build_filtered_split.py",
        "--ranked",
        str(ranked),
        "--policy",
        str(ROOT / "configs/filter_policy_v1.yaml"),
        "--out",
        str(out),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    train = payload["splits"]["train"]
    assert {row["take_uid"] for row in train["fact_main"]} == {"main_take"}
    assert {row["take_uid"] for row in train["loco_aux"]} == {"loco_take"}
    assert {row["take_uid"] for row in train["discard"]} == {"blocked_take"}


def test_usable_for_ranker_outputs_main_and_loco_probabilities(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    annotations = tmp_path / "annotations.csv"
    feature_rows = []
    label_rows = []
    classes = ["tokenizer_main", "loco_aux", "discard"]
    for index in range(18):
        label = classes[index % len(classes)]
        take_uid = f"take_{index}"
        feature_rows.append(
            {
                "take_uid": take_uid,
                "ego_motion_score": 0.8 if label == "tokenizer_main" else 0.2,
                "exo_body_motion_score": 0.8 if label == "loco_aux" else 0.2,
                "object_motion_proxy": 0.7 if label == "tokenizer_main" else 0.1,
                "temporal_diversity_score": 0.7,
                "metadata_interaction_prior": 0.8 if label == "tokenizer_main" else 0.2,
                "metadata_loco_prior": 0.8 if label == "loco_aux" else 0.2,
                "scene_only_score": 0.8 if label == "discard" else 0.1,
                "interaction_score": 0.8 if label == "tokenizer_main" else 0.2,
                "loco_score": 0.8 if label == "loco_aux" else 0.2,
                "fine_dexterous_score": 0.1,
                "relevance_score": 0.8 if label != "discard" else 0.1,
            }
        )
        label_rows.append({"take_uid": take_uid, "usable_for": label})
    with features.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["take_uid", "usable_for"])
        writer.writeheader()
        writer.writerows(label_rows)
    ranker = tmp_path / "ranker.pkl"
    ranked = tmp_path / "ranked.csv"
    run_tool(
        tmp_path,
        "tools/train_relevance_ranker.py",
        "--features",
        str(features),
        "--labels",
        str(annotations),
        "--label-column",
        "usable_for",
        "--out",
        str(ranker),
    )
    run_tool(tmp_path, "tools/apply_relevance_ranker.py", "--features", str(features), "--ranker", str(ranker), "--out", str(ranked))
    with ranked.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"prob_tokenizer_main", "prob_loco_aux"}.issubset(rows[0])


def test_same_take_negative_index_invariants(tmp_path: Path) -> None:
    npz = make_npz(tmp_path / "filtered.npz", take_count=2, per_take=8)
    out = tmp_path / "negative_index.npz"
    run_tool(
        tmp_path,
        "tools/build_same_take_negative_index.py",
        "--npz",
        str(npz),
        "--out",
        str(out),
        "--top-k",
        "3",
        "--min-gap",
        "2",
        "--max-gap",
        "5",
    )
    with np.load(npz, allow_pickle=False) as source, np.load(out, allow_pickle=False) as neg:
        take_uid = source["take_uid"].astype(str)
        timestamp = source["timestamp"]
        donors = neg["donor_index_topk"]
    valid = donors >= 0
    anchors = np.repeat(np.arange(donors.shape[0])[:, None], donors.shape[1], axis=1)
    assert valid.any()
    assert np.all(take_uid[anchors[valid]] == take_uid[donors[valid]])
    gap = np.abs(timestamp[anchors[valid]] - timestamp[donors[valid]])
    assert np.all((gap >= 2) & (gap <= 5))


def test_transition_selection_bucket_caps(tmp_path: Path) -> None:
    npz = make_npz(tmp_path / "split.npz", take_count=2, per_take=8)
    filtered = tmp_path / "filtered.json"
    filtered.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "fact_main": [{"take_uid": "take_0"}],
                        "loco_aux": [{"take_uid": "take_1"}],
                        "discard": [],
                    },
                    "heldout": {"fact_main": [], "loco_aux": [], "discard": []},
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "selection.csv"
    run_tool(
        tmp_path,
        "tools/build_transition_selection.py",
        "--npz",
        str(npz),
        "--split-name",
        "train",
        "--filtered-split",
        str(filtered),
        "--include-buckets",
        "fact_main",
        "loco_aux",
        "--num-transitions",
        "4",
        "--bucket-num-transitions",
        "loco_aux=2",
        "--out",
        str(out),
    )
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_by_bucket: dict[str, int] = {}
    for row in rows:
        if row["selected"] == "1":
            selected_by_bucket[row["bucket"]] = selected_by_bucket.get(row["bucket"], 0) + 1
    assert selected_by_bucket == {"fact_main": 4, "loco_aux": 2}
