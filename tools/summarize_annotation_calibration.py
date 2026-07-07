#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual calibration labels and v1 split behavior.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--filtered-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def split_bucket_map(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for split, buckets in payload["splits"].items():
        for bucket, items in buckets.items():
            for item in items:
                rows.append({"take_uid": str(item["take_uid"]), "split": split, "v1_bucket": bucket})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_\n"
    table = frame.reset_index() if frame.index.name or list(frame.index) != list(range(len(frame))) else frame
    columns = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy().tolist()]
    widths = [len(column) for column in columns]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def top_examples(frame: pd.DataFrame, mask: pd.Series, columns: list[str], count: int = 12) -> pd.DataFrame:
    values = frame[mask].copy()
    if "ranker_confidence" in values:
        values = values.sort_values("ranker_confidence", ascending=False)
    return values[columns].head(count)


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.scores)
    ranked = pd.read_csv(args.ranked)
    annotations = pd.read_csv(args.annotations)
    split_map = split_bucket_map(args.filtered_split)
    merged = scores.merge(annotations, on="take_uid", how="inner", suffixes=("", "_label"))
    merged = merged.merge(ranked[[column for column in ranked.columns if column not in merged.columns or column == "take_uid"]], on="take_uid", how="left")
    merged = merged.merge(split_map[["take_uid", "v1_bucket"]], on="take_uid", how="left")
    merged["v1_bucket"] = merged["v1_bucket"].fillna("missing")

    lines = ["# filtering_v1 Calibration Report", ""]
    lines.extend(["## Label Coverage", ""])
    lines.append(markdown_table(annotations["usable_for"].value_counts().rename_axis("usable_for").reset_index(name="takes")))
    lines.extend(["", "## v0 Auto Bucket vs Manual Usable For", ""])
    lines.append(markdown_table(pd.crosstab(merged["auto_bucket"], merged["usable_for"])))
    lines.extend(["", "## v1 Bucket vs Manual Usable For", ""])
    lines.append(markdown_table(pd.crosstab(merged["v1_bucket"], merged["usable_for"])))
    lines.extend(["", "## Parent Task Retention", ""])
    parent = pd.crosstab(merged["parent_task_name"], merged["v1_bucket"])
    if not parent.empty:
        parent["total"] = parent.sum(axis=1)
    lines.append(markdown_table(parent))
    columns = [
        "take_uid",
        "parent_task_name",
        "task_name",
        "auto_bucket",
        "usable_for",
        "v1_bucket",
        "ranker_bucket",
        "ranker_confidence",
    ]
    columns = [column for column in columns if column in merged.columns]
    lines.extend(["", "## Likely False Keeps", ""])
    false_keeps = merged["v1_bucket"].isin(["fact_main", "loco_aux"]) & merged["usable_for"].isin(["discard", "diagnostic_candidate"])
    lines.append(markdown_table(top_examples(merged, false_keeps, columns)))
    lines.extend(["", "## Likely False Drops", ""])
    false_drops = (merged["v1_bucket"] == "discard") & merged["usable_for"].isin(["tokenizer_main", "loco_aux"])
    lines.append(markdown_table(top_examples(merged, false_drops, columns)))
    lines.extend(["", "## Notes", ""])
    lines.append("- Review false keeps before FACT training.")
    lines.append("- Review false drops before finalizing thresholds.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved calibration report to {args.out}")


if __name__ == "__main__":
    main()
