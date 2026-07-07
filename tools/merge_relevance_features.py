#!/usr/bin/env python3
"""Merge filtering_v2 feature CSVs and produce automatic pre-labels."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, to_float, write_csv


V2_COLUMNS = [
    "take_uid",
    "split",
    "parent_task_name",
    "task_name",
    "take_name",
    "num_transitions",
    "ego_motion_score",
    "exo_body_motion_score",
    "object_motion_proxy",
    "temporal_diversity_score",
    "metadata_interaction_prior",
    "metadata_loco_prior",
    "scene_only_score",
    "interaction_score",
    "loco_score",
    "fine_dexterous_score",
    "relevance_score",
    "ego_hand_score",
    "ego_hand_visibility_prob",
    "object_presence_score",
    "object_motion_score",
    "hand_object_contact_score",
    "interacting_object_score",
    "exo_body_visibility_score",
    "exo_pose_confidence",
    "exo_body_motion_score_v2",
    "body_phase_diversity_score",
    "loco_motion_score",
    "phase_diversity_score_v2",
    "motion_state_change_score",
    "contact_state_change_score",
    "pose_state_change_score",
    "vlm_prob_tokenizer_main",
    "vlm_prob_loco_aux",
    "vlm_prob_discard",
    "vlm_prob_diagnostic_candidate",
    "vlm_take_relevance",
    "vlm_usable_for",
    "vlm_confidence",
    "vlm_reason",
    "prob_tokenizer_main",
    "prob_loco_aux",
    "prob_discard",
    "prob_diagnostic_candidate",
    "auto_take_relevance",
    "auto_usable_for",
    "auto_confidence",
    "auto_disagreement_score",
    "auto_review_priority",
    "needs_human_review",
    "review_reasons",
    "contact_sheet_path",
    "feature_source_ego_hand_object",
    "feature_source_exo_pose_phase",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="v0/v1 relevance score CSV")
    parser.add_argument("--ego-hand-object", type=Path, default=None)
    parser.add_argument("--exo-pose-phase", type=Path, default=None)
    parser.add_argument("--vlm", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/filter_policy_v2.yaml")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def load_optional(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row["take_uid"]: row for row in read_csv(path) if row.get("take_uid")}


def merge_row(base: dict[str, str], extras: list[dict[str, dict[str, str]]]) -> dict[str, Any]:
    row: dict[str, Any] = dict(base)
    take_uid = str(base["take_uid"])
    for table in extras:
        row.update({key: value for key, value in table.get(take_uid, {}).items() if key != "take_uid" and value != ""})
    return row


def weighted(base_prob: float, vlm_prob: float, vlm_conf: float) -> float:
    if vlm_conf <= 0.0:
        return clamp(base_prob)
    alpha = clamp(0.25 + 0.55 * vlm_conf)
    return clamp((1.0 - alpha) * base_prob + alpha * vlm_prob)


def infer_probabilities(row: dict[str, Any]) -> None:
    scene = to_float(row.get("scene_only_score"))
    interaction = to_float(row.get("interaction_score"))
    loco = to_float(row.get("loco_score"))
    fine = to_float(row.get("fine_dexterous_score"))
    temporal = to_float(row.get("temporal_diversity_score"))
    contact = to_float(row.get("hand_object_contact_score"), to_float(row.get("object_motion_proxy")))
    obj = to_float(row.get("interacting_object_score"), to_float(row.get("object_motion_score"), to_float(row.get("object_motion_proxy"))))
    exo_body = to_float(row.get("exo_body_visibility_score"), to_float(row.get("exo_body_motion_score")))
    loco_motion = to_float(row.get("loco_motion_score"), to_float(row.get("exo_body_motion_score_v2"), to_float(row.get("exo_body_motion_score"))))
    contact_change = to_float(row.get("contact_state_change_score"))
    pose_change = to_float(row.get("pose_state_change_score"))
    motion_change = to_float(row.get("motion_state_change_score"), temporal)
    phase = clamp(0.35 * temporal + 0.35 * contact_change + 0.30 * pose_change)
    row["phase_diversity_score_v2"] = round(phase, 6)
    row["motion_state_change_score"] = round(motion_change, 6)

    base_main = clamp(0.24 * interaction + 0.24 * contact + 0.18 * obj + 0.18 * phase + 0.16 * (1.0 - scene))
    base_loco = clamp(0.32 * loco + 0.28 * exo_body + 0.24 * loco_motion + 0.16 * max(phase, pose_change))
    base_discard = clamp(0.50 * scene + 0.25 * (1.0 - max(contact, exo_body)) + 0.15 * (1.0 - phase) + 0.10 * (1.0 - max(interaction, loco)))
    base_diag = clamp(0.45 * fine + 0.25 * abs(base_main - base_loco) + 0.30 * (1.0 - max(base_main, base_loco, base_discard)))

    vlm_conf = to_float(row.get("vlm_confidence"))
    main = weighted(base_main, to_float(row.get("vlm_prob_tokenizer_main")), vlm_conf)
    loco_prob = weighted(base_loco, to_float(row.get("vlm_prob_loco_aux")), vlm_conf)
    discard = weighted(base_discard, to_float(row.get("vlm_prob_discard")), vlm_conf)
    diag = weighted(base_diag, to_float(row.get("vlm_prob_diagnostic_candidate")), vlm_conf)
    # These are independent suitability scores, not a mutually exclusive
    # softmax. A take can be both interaction-rich and useful as loco_aux, and
    # policy thresholds decide the final bucket.
    row["prob_tokenizer_main"] = round(main, 6)
    row["prob_loco_aux"] = round(loco_prob, 6)
    row["prob_discard"] = round(discard, 6)
    row["prob_diagnostic_candidate"] = round(diag, 6)

    detector_main = base_main
    detector_loco = base_loco
    if vlm_conf > 0.0:
        disagreement = max(
            abs(to_float(row.get("vlm_prob_tokenizer_main")) - detector_main),
            abs(to_float(row.get("vlm_prob_loco_aux")) - detector_loco),
            abs(to_float(row.get("vlm_prob_discard")) - base_discard),
        )
    else:
        disagreement = max(abs(contact - interaction), abs(exo_body - loco), abs(scene - base_discard)) * 0.45
    row["auto_disagreement_score"] = round(clamp(disagreement), 6)


def classify(row: dict[str, Any], policy: dict[str, Any]) -> None:
    fact = policy["fact_main"]
    loco_policy = policy["loco_aux"]
    diag_policy = policy["diagnostic_candidate"]
    main_prob = to_float(row.get("prob_tokenizer_main"))
    loco_prob = to_float(row.get("prob_loco_aux"))
    discard_prob = to_float(row.get("prob_discard"))
    diag_prob = to_float(row.get("prob_diagnostic_candidate"))
    contact = to_float(row.get("hand_object_contact_score"))
    phase = to_float(row.get("phase_diversity_score_v2"))
    scene = to_float(row.get("scene_only_score"))
    exo_body = to_float(row.get("exo_body_visibility_score"))
    body_phase = to_float(row.get("body_phase_diversity_score"))
    fine = to_float(row.get("fine_dexterous_score"))
    vlm_main = to_float(row.get("vlm_prob_tokenizer_main"))

    strong_vlm_main = bool(fact.get("allow_vlm_override", True)) and vlm_main >= to_float(fact.get("vlm_strong_main_prob"), 1.0)
    is_fact = (
        main_prob >= to_float(fact["min_tokenizer_main_prob"])
        and (contact >= to_float(fact["min_hand_object_contact"]) or strong_vlm_main)
        and phase >= to_float(fact["min_phase_diversity"])
        and scene <= to_float(fact["max_scene_only"])
        and discard_prob <= to_float(fact["max_discard_prob"])
        and diag_prob <= to_float(fact["max_diagnostic_prob"])
    )
    is_loco = (
        loco_prob >= to_float(loco_policy["min_loco_aux_prob"])
        and exo_body >= to_float(loco_policy["min_exo_body_visibility"])
        and body_phase >= to_float(loco_policy["min_body_phase_diversity"])
        and scene <= to_float(loco_policy["max_scene_only"])
        and discard_prob <= to_float(loco_policy["max_discard_prob"])
    )
    is_diag = (
        diag_prob >= to_float(diag_policy["min_diagnostic_prob"])
        or fine >= to_float(diag_policy["fine_dexterous_score"])
        or to_float(row.get("auto_disagreement_score")) >= to_float(diag_policy["high_disagreement_score"])
    )

    if is_fact:
        usable = "tokenizer_main"
        relevance = "A_interaction_rich"
        confidence = main_prob
    elif is_loco:
        usable = "loco_aux"
        relevance = "B_loco_body"
        confidence = loco_prob
    elif is_diag:
        usable = "diagnostic_candidate"
        relevance = "E_fine_dexterous" if fine >= to_float(diag_policy["fine_dexterous_score"]) else "F_bad_or_unclear"
        confidence = max(diag_prob, to_float(row.get("auto_disagreement_score")))
    elif scene >= 0.55 or discard_prob >= 0.45:
        usable = "discard"
        relevance = "D_scene_only"
        confidence = discard_prob
    else:
        usable = "discard"
        relevance = "C_active_view_only"
        confidence = max(discard_prob, 1.0 - max(main_prob, loco_prob))

    row["auto_take_relevance"] = relevance
    row["auto_usable_for"] = usable
    row["auto_confidence"] = round(clamp(confidence), 6)


def add_review_flags(row: dict[str, Any], policy: dict[str, Any]) -> None:
    review = policy["active_review"]
    reasons = []
    confidence = to_float(row.get("auto_confidence"))
    disagreement = to_float(row.get("auto_disagreement_score"))
    if confidence < to_float(review["low_confidence_threshold"]):
        reasons.append("low_confidence")
    if disagreement >= to_float(review["high_disagreement_threshold"]):
        reasons.append("signal_disagreement")
    margin = min(
        abs(to_float(row.get("prob_tokenizer_main")) - to_float(policy["fact_main"]["min_tokenizer_main_prob"])),
        abs(to_float(row.get("prob_loco_aux")) - to_float(policy["loco_aux"]["min_loco_aux_prob"])),
    )
    if margin <= to_float(review["boundary_margin"]):
        reasons.append("near_policy_boundary")
    if row.get("auto_usable_for") == "tokenizer_main" and to_float(row.get("hand_object_contact_score")) < to_float(policy["fact_main"]["min_hand_object_contact"]):
        reasons.append("fact_main_low_contact")
    if row.get("auto_usable_for") == "discard" and max(to_float(row.get("hand_object_contact_score")), to_float(row.get("exo_body_visibility_score"))) >= 0.65:
        reasons.append("possible_false_drop")
    row["review_reasons"] = ";".join(reasons)
    row["needs_human_review"] = int(bool(reasons))
    priority = 0.35 * (1.0 - confidence) + 0.35 * disagreement + 0.20 * (1.0 - min(margin / 0.25, 1.0)) + 0.10 * len(reasons)
    row["auto_review_priority"] = round(clamp(priority), 6)


def main() -> None:
    args = parse_args()
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    base_rows = read_csv(args.base)
    extras = [load_optional(args.ego_hand_object), load_optional(args.exo_pose_phase), load_optional(args.vlm)]
    output = []
    for base in base_rows:
        row = merge_row(base, extras)
        infer_probabilities(row)
        classify(row, policy)
        add_review_flags(row, policy)
        output.append(row)
    fieldnames = [column for column in V2_COLUMNS if any(column in row for row in output)]
    extra_columns = sorted({key for row in output for key in row if key not in fieldnames})
    write_csv(args.out, output, fieldnames=fieldnames + extra_columns)
    counts: dict[str, int] = {}
    for row in output:
        key = str(row.get("auto_usable_for", ""))
        counts[key] = counts.get(key, 0) + 1
    print(f"Saved {len(output)} merged v2 rows to {args.out}; buckets={counts}")


if __name__ == "__main__":
    main()
