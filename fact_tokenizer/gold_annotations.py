"""Deterministic construction and validation of the 300-sample FACT gold pack."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


EFFECT_LABELS = (
    "no_effect",
    "approach_align",
    "acquire_control",
    "state_change_or_manipulate",
    "transport_reposition",
    "release_complete",
    "recover_abort",
    "ambiguous",
)
CONTACT_LABELS = ("none", "onset", "stable", "release", "unknown")

GOLD_SPLIT_FILES = {
    "probe_train": "gold140_probe_train_annotations.csv",
    "calibration_dev": "gold60_calibration_dev_annotations.csv",
    "locked_test": "gold100_locked_test_annotations.csv",
}
GOLD_SPLIT_COUNTS = {
    "probe_train": 140,
    "calibration_dev": 60,
    "locked_test": 100,
}


def _normalized_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        take_uid = str(row.get("take_uid", "")).strip()
        if not sample_id or not take_uid:
            raise ValueError("every gold candidate needs sample_id and take_uid")
        record = dict(row)
        record["sample_id"] = sample_id
        record["take_uid"] = take_uid
        records.append(record)
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("gold candidate records contain duplicate sample IDs")
    return records


def select_take_balanced_gold(
    rows: Iterable[Mapping[str, Any]],
    *,
    take_count: int,
    sample_count: int,
    seed: int,
    excluded_sample_ids: Iterable[str] = (),
    required_take_uids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select exact counts while spreading samples uniformly over selected takes."""

    excluded = {str(value) for value in excluded_sample_ids}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _normalized_records(rows):
        if row["sample_id"] not in excluded:
            grouped[row["take_uid"]].append(row)
    required = {str(value).strip() for value in required_take_uids if str(value).strip()}
    missing_required = sorted(required - set(grouped))
    if missing_required:
        raise ValueError(f"required gold takes are missing: {missing_required[:5]}")
    if len(required) > take_count:
        raise ValueError("required take count exceeds requested take_count")
    base, remainder = divmod(sample_count, take_count)
    needed_counts = [base + (index < remainder) for index in range(take_count)]
    if base < 1:
        raise ValueError("sample_count must be at least take_count")
    eligible = [take for take, values in grouped.items() if len(values) >= base]
    if len(eligible) < take_count:
        raise ValueError(f"need {take_count} eligible takes, found {len(eligible)}")
    rng = random.Random(seed)
    required_order = sorted(required)
    remaining = sorted(set(eligible) - required)
    rng.shuffle(required_order)
    rng.shuffle(remaining)
    selected_takes = required_order + remaining[: take_count - len(required_order)]
    # Assign higher per-take counts only to takes that can support them.
    selected_takes.sort(key=lambda take: (len(grouped[take]) < base + 1, hashlib.sha256(f"{seed}:{take}".encode()).hexdigest()))
    selected: list[dict[str, Any]] = []
    for take_uid, needed in zip(selected_takes, needed_counts):
        candidates = sorted(grouped[take_uid], key=lambda row: row["sample_id"])
        if len(candidates) < needed:
            raise ValueError(f"take {take_uid} has {len(candidates)} samples, needs {needed}")
        take_rng = random.Random(f"{seed}:{take_uid}")
        take_rng.shuffle(candidates)
        selected.extend(sorted(candidates[:needed], key=lambda row: row["sample_id"]))
    if len(selected) != sample_count or len({row["take_uid"] for row in selected}) != take_count:
        raise AssertionError("gold selection did not satisfy requested cardinality")
    return selected


def build_gold_pack(
    train_rows: Iterable[Mapping[str, Any]],
    dev_rows: Iterable[Mapping[str, Any]],
    locked_rows: Iterable[Mapping[str, Any]],
    diagnostic_sample_ids: Iterable[str],
    *,
    seed: int = 20260711,
    required_locked_takes: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build the frozen 140/60/100 split and mark 60 dual-annotation samples."""

    excluded = {str(value) for value in diagnostic_sample_ids}
    groups = {
        "probe_train": select_take_balanced_gold(
            train_rows,
            take_count=47,
            sample_count=140,
            seed=seed,
            excluded_sample_ids=excluded,
        ),
        "calibration_dev": select_take_balanced_gold(
            dev_rows,
            take_count=20,
            sample_count=60,
            seed=seed + 1,
            excluded_sample_ids=excluded,
        ),
        "locked_test": select_take_balanced_gold(
            locked_rows,
            take_count=50,
            sample_count=100,
            seed=seed + 2,
            excluded_sample_ids=excluded,
            required_take_uids=required_locked_takes,
        ),
    }
    all_ids = [row["sample_id"] for values in groups.values() for row in values]
    if len(set(all_ids)) != 300:
        raise ValueError("gold splits overlap or do not total 300 unique sample IDs")
    pack: list[dict[str, Any]] = []
    for split, rows in groups.items():
        dual_ids = {
            row["sample_id"]
            for row in sorted(rows, key=lambda row: hashlib.sha256(f"dual:{seed}:{row['sample_id']}".encode()).hexdigest())[:20]
        }
        for row in rows:
            pack.append(
                {
                    "sample_id": row["sample_id"],
                    "take_uid": row["take_uid"],
                    "gold_split": split,
                    "timestamp": row.get("timestamp", ""),
                    "source_dataset": row.get("source_dataset", "egoexo"),
                    "dual_annotation": row["sample_id"] in dual_ids,
                    "representation_training_valid": False,
                    "effect_label": "",
                    "contact_label": "",
                    "ambiguous_reason": "",
                    "annotator_id": "",
                    "notes": "",
                }
            )
    split_order = {"probe_train": 0, "calibration_dev": 1, "locked_test": 2}
    return sorted(pack, key=lambda row: (split_order[row["gold_split"]], row["take_uid"], row["sample_id"]))


def write_gold_pack(output_dir: Path | str, rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "take_uid",
        "gold_split",
        "timestamp",
        "source_dataset",
        "dual_annotation",
        "representation_training_valid",
        "effect_label",
        "contact_label",
        "ambiguous_reason",
        "annotator_id",
        "notes",
    ]
    counts = Counter(str(row["gold_split"]) for row in rows)
    if dict(counts) != GOLD_SPLIT_COUNTS:
        raise ValueError(
            "gold pack must contain exactly 140 probe-train, 60 calibration-dev, "
            f"and 100 locked-test rows; got {dict(counts)}"
        )
    annotation_templates: dict[str, dict[str, Any]] = {}
    for split, filename in GOLD_SPLIT_FILES.items():
        csv_path = output / filename
        temp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
        split_rows = [row for row in rows if str(row["gold_split"]) == split]
        with temp_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in split_rows)
        temp_csv.replace(csv_path)
        annotation_templates[split] = {
            "path": filename,
            "rows": len(split_rows),
            "template_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }
    manifest_path = output / "gold300_frozen.jsonl"
    temp_manifest = manifest_path.with_suffix(".tmp")
    with temp_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            frozen = {key: row.get(key) for key in fields[:7]}
            handle.write(json.dumps(frozen, ensure_ascii=False, sort_keys=True) + "\n")
    temp_manifest.replace(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    take_counts = {
        split: len({row["take_uid"] for row in rows if row["gold_split"] == split})
        for split in counts
    }
    metadata = {
        "schema": "fact-effect-gold-v1",
        "seed": seed,
        "manifest_sha256": digest,
        "sample_counts": dict(sorted(counts.items())),
        "take_counts": dict(sorted(take_counts.items())),
        "dual_annotation_count": sum(bool(row["dual_annotation"]) for row in rows),
        "effect_labels": list(EFFECT_LABELS),
        "contact_labels": list(CONTACT_LABELS),
        "representation_training_valid": False,
        "annotation_templates": annotation_templates,
        "combined_annotation_csv_written": False,
    }
    (output / "gold300_freeze.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "ANNOTATION_GUIDE.zh-CN.md").write_text(
        "# FACT effect/contact 标注指南\n\n"
        "每条样本分别填写 `effect_label` 与 `contact_label`。effect 可选：\n\n- "
        + "\n- ".join(EFFECT_LABELS)
        + "\n\ncontact 可选：\n\n- "
        + "\n- ".join(CONTACT_LABELS)
        + "\n\n看不清或存在多个同等合理解释时使用 `ambiguous`，并填写 `ambiguous_reason`。"
        "不要从 weak text、旧 codelabel 或模型预测复制标签。\n",
        encoding="utf-8",
    )
    return metadata


def validate_gold_rows(rows: Sequence[Mapping[str, Any]], *, require_complete: bool = True) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            errors.append(f"row {row_number}: missing sample_id")
        elif sample_id in seen:
            errors.append(f"row {row_number}: duplicate sample_id={sample_id}")
        seen.add(sample_id)
        if str(row.get("representation_training_valid", "")).strip().lower() not in {"false", "0"}:
            errors.append(f"row {row_number}: gold must have representation_training_valid=false")
        if require_complete:
            effect = str(row.get("effect_label", "")).strip()
            contact = str(row.get("contact_label", "")).strip()
            if effect not in EFFECT_LABELS:
                errors.append(f"row {row_number}: invalid effect_label={effect!r}")
            if contact not in CONTACT_LABELS:
                errors.append(f"row {row_number}: invalid contact_label={contact!r}")
            if effect == "ambiguous" and not str(row.get("ambiguous_reason", "")).strip():
                errors.append(f"row {row_number}: ambiguous requires ambiguous_reason")
    return errors


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Cohen kappa needs equal non-empty label sequences")
    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(counts_a[label] * counts_b[label] for label in categories) / (len(labels_a) ** 2)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
