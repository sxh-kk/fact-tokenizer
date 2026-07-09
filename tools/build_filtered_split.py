#!/usr/bin/env python3
"""Build FACT-main/loco-aux/diagnostic/discard split JSON from relevance rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, to_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked", "--scores", dest="ranked", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fact-main-mode",
        choices=["policy", "strict", "balanced"],
        default="policy",
        help="Data-side candidate split mode. strict raises fact_main precision; balanced relaxes thresholds for coverage.",
    )
    parser.add_argument(
        "--prefer-human-labels",
        action="store_true",
        help="Use non-empty manual usable_for/take_relevance columns when present.",
    )
    return parser.parse_args()


def row_value(row: dict[str, str], names: list[str], default: float = 0.0) -> float:
    for name in names:
        if row.get(name, "") != "":
            return to_float(row.get(name), default)
    return default


def manual_bucket(row: dict[str, str]) -> str | None:
    usable = str(row.get("usable_for") or "").strip()
    if usable == "tokenizer_main":
        return "fact_main"
    if usable == "loco_aux":
        return "loco_aux"
    if usable == "diagnostic_candidate":
        return "diagnostic_candidate"
    if usable == "discard":
        return "discard"
    return None


def threshold_overrides(policy: dict[str, Any], mode: str) -> dict[str, float]:
    fact = policy["fact_main"]
    variants = policy.get("fact_main_variants", {})
    if mode == "strict":
        return {
            "min_tokenizer_main_prob": to_float(variants.get("strict", {}).get("min_tokenizer_main_prob"), to_float(fact["min_tokenizer_main_prob"]) + 0.08),
            "min_hand_object_contact": to_float(variants.get("strict", {}).get("min_hand_object_contact"), to_float(fact["min_hand_object_contact"]) + 0.10),
            "min_phase_diversity": to_float(variants.get("strict", {}).get("min_phase_diversity"), to_float(fact["min_phase_diversity"]) + 0.07),
            "max_scene_only": to_float(variants.get("strict", {}).get("max_scene_only"), max(0.0, to_float(fact["max_scene_only"]) - 0.08)),
            "max_discard_prob": to_float(variants.get("strict", {}).get("max_discard_prob"), max(0.0, to_float(fact["max_discard_prob"]) - 0.10)),
            "max_diagnostic_prob": to_float(variants.get("strict", {}).get("max_diagnostic_prob"), max(0.0, to_float(fact["max_diagnostic_prob"]) - 0.08)),
            "min_ego_hand_visibility": to_float(variants.get("strict", {}).get("min_ego_hand_visibility"), 0.30),
        }
    if mode == "balanced":
        return {
            "min_tokenizer_main_prob": to_float(variants.get("balanced", {}).get("min_tokenizer_main_prob"), max(0.0, to_float(fact["min_tokenizer_main_prob"]) - 0.05)),
            "min_hand_object_contact": to_float(variants.get("balanced", {}).get("min_hand_object_contact"), max(0.0, to_float(fact["min_hand_object_contact"]) - 0.07)),
            "min_phase_diversity": to_float(variants.get("balanced", {}).get("min_phase_diversity"), max(0.0, to_float(fact["min_phase_diversity"]) - 0.04)),
            "max_scene_only": to_float(variants.get("balanced", {}).get("max_scene_only"), min(1.0, to_float(fact["max_scene_only"]) + 0.05)),
            "max_discard_prob": to_float(variants.get("balanced", {}).get("max_discard_prob"), min(1.0, to_float(fact["max_discard_prob"]) + 0.05)),
            "max_diagnostic_prob": to_float(variants.get("balanced", {}).get("max_diagnostic_prob"), min(1.0, to_float(fact["max_diagnostic_prob"]) + 0.10)),
            "min_ego_hand_visibility": to_float(variants.get("balanced", {}).get("min_ego_hand_visibility"), 0.22),
        }
    return {
        "min_tokenizer_main_prob": to_float(fact["min_tokenizer_main_prob"]),
        "min_hand_object_contact": to_float(fact["min_hand_object_contact"]),
        "min_phase_diversity": to_float(fact["min_phase_diversity"]),
        "max_scene_only": to_float(fact["max_scene_only"]),
        "max_discard_prob": to_float(fact["max_discard_prob"]),
        "max_diagnostic_prob": to_float(fact["max_diagnostic_prob"]),
        "min_ego_hand_visibility": to_float(variants.get("policy", {}).get("min_ego_hand_visibility"), 0.0),
    }


def policy_bucket(row: dict[str, str], policy: dict[str, Any], fact_main_mode: str) -> str:
    auto = str(row.get("auto_usable_for") or row.get("ranker_bucket") or "").strip()
    main_prob = row_value(row, ["prob_tokenizer_main", "prob_a_interaction_rich", "prob_interaction_rich", "interaction_score"])
    loco_prob = row_value(row, ["prob_loco_aux", "prob_b_loco_body", "prob_loco_body", "loco_score"])
    discard_prob = row_value(row, ["prob_discard"])
    diag_prob = row_value(row, ["prob_diagnostic_candidate"])
    scene = row_value(row, ["scene_only_score"])
    phase = row_value(row, ["phase_diversity_score_v2", "temporal_diversity_score"])
    contact = row_value(row, ["hand_object_contact_score", "object_motion_proxy"])
    ego_hand = row_value(row, ["ego_hand_visibility_prob", "ego_hand_score"], 1.0)
    exo_body = row_value(row, ["exo_body_visibility_score", "exo_body_motion_score_v2", "exo_body_motion_score"])
    body_phase = row_value(row, ["body_phase_diversity_score", "phase_diversity_score_v2", "temporal_diversity_score"])

    if policy.get("version") == "filtering_v2":
        fact = threshold_overrides(policy, fact_main_mode)
        loco = policy["loco_aux"]
        diag = policy.get("diagnostic_candidate", {})
        if (
            (auto == "tokenizer_main" or main_prob >= fact["min_tokenizer_main_prob"])
            and contact >= fact["min_hand_object_contact"]
            and phase >= fact["min_phase_diversity"]
            and scene <= fact["max_scene_only"]
            and discard_prob <= fact["max_discard_prob"]
            and diag_prob <= fact["max_diagnostic_prob"]
            and ego_hand >= fact["min_ego_hand_visibility"]
        ):
            return "fact_main"
        if (
            (auto == "loco_aux" or loco_prob >= to_float(loco["min_loco_aux_prob"]))
            and exo_body >= to_float(loco["min_exo_body_visibility"])
            and body_phase >= to_float(loco["min_body_phase_diversity"])
            and scene <= to_float(loco["max_scene_only"])
            and discard_prob <= to_float(loco["max_discard_prob"])
        ):
            return "loco_aux"
        if auto == "diagnostic_candidate" or diag_prob >= to_float(diag.get("min_diagnostic_prob"), 0.52):
            return "diagnostic_candidate"
        return "discard"

    fact = policy["fact_main"]
    loco = policy["loco_aux"]
    for column, threshold in (fact.get("blocked_probability_columns") or {}).items():
        if row_value(row, [column]) >= to_float(threshold):
            break
    else:
        if (
            main_prob >= to_float(fact["min_interaction_prob"])
            and phase >= to_float(fact["min_temporal_diversity"])
            and scene <= to_float(fact["max_scene_only"])
        ):
            return "fact_main"
    for column, threshold in (loco.get("blocked_probability_columns") or {}).items():
        if row_value(row, [column]) >= to_float(threshold):
            return "discard"
    if (
        loco_prob >= to_float(loco["min_loco_prob"])
        and scene <= to_float(loco["max_scene_only"])
    ):
        return "loco_aux"
    return "discard"


def split_name(row: dict[str, str]) -> str:
    split = str(row.get("split") or "train")
    return split if split in {"train", "heldout"} else "train"


def item_from_row(row: dict[str, str], bucket: str) -> dict[str, Any]:
    return {
        "take_uid": str(row["take_uid"]),
        "parent_task_name": str(row.get("parent_task_name", "")),
        "task_name": str(row.get("task_name", "")),
        "take_name": str(row.get("take_name", "")),
        "bucket": bucket,
        "auto_take_relevance": str(row.get("auto_take_relevance", row.get("auto_bucket", ""))),
        "auto_usable_for": str(row.get("auto_usable_for", "")),
        "prob_tokenizer_main": row_value(row, ["prob_tokenizer_main", "interaction_score"]),
        "prob_loco_aux": row_value(row, ["prob_loco_aux", "loco_score"]),
        "prob_discard": row_value(row, ["prob_discard"]),
        "prob_diagnostic_candidate": row_value(row, ["prob_diagnostic_candidate"]),
        "scene_only_score": row_value(row, ["scene_only_score"]),
        "hand_object_contact_score": row_value(row, ["hand_object_contact_score", "object_motion_proxy"]),
        "phase_diversity_score_v2": row_value(row, ["phase_diversity_score_v2", "temporal_diversity_score"]),
        "exo_body_visibility_score": row_value(row, ["exo_body_visibility_score", "exo_body_motion_score"]),
        "auto_confidence": row_value(row, ["auto_confidence", "ranker_confidence", "relevance_score"]),
        "ranker_bucket": str(row.get("ranker_bucket", "")),
        "ranker_confidence": row_value(row, ["ranker_confidence"]),
        "needs_human_review": int(to_float(row.get("needs_human_review"), 0.0)),
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.ranked)
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    output = {
        "version": policy.get("version", "filtering_v2"),
        "policy": policy,
        "source": str(args.ranked),
        "fact_main_mode": args.fact_main_mode,
        "splits": {
            "train": {"fact_main": [], "loco_aux": [], "diagnostic_candidate": [], "discard": []},
            "heldout": {"fact_main": [], "loco_aux": [], "diagnostic_candidate": [], "discard": []},
        },
    }
    for row in rows:
        if not row.get("take_uid"):
            continue
        bucket = manual_bucket(row) if args.prefer_human_labels else None
        if bucket is None:
            bucket = policy_bucket(row, policy, args.fact_main_mode)
        split = split_name(row)
        output["splits"][split][bucket].append(item_from_row(row, bucket))

    for split in output["splits"].values():
        for bucket, items in split.items():
            items.sort(key=lambda item: (-float(item["auto_confidence"]), item["take_uid"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    for split, buckets in output["splits"].items():
        print(split, {bucket: len(items) for bucket, items in buckets.items()})
    print(f"Saved filtered split to {args.out}")


if __name__ == "__main__":
    main()
