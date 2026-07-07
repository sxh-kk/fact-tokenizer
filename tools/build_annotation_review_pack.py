#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an annotator-friendly contact sheet review pack.")
    parser.add_argument("--annotations", type=Path, required=True, help="Annotation CSV exported by export_annotation_csv.py.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for review CSV, copied images, and HTML gallery.")
    parser.add_argument("--image-subdir", default="review_images")
    parser.add_argument(
        "--path-prefix-from",
        default="",
        help="Optional source path prefix to replace, e.g. /data_all/.../filtering_v0_500takes_20260624.",
    )
    parser.add_argument(
        "--path-prefix-to",
        default="",
        help="Optional local path prefix replacement, e.g. D:/egoexo_v1_annotation.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy contact sheets into the review pack.")
    return parser.parse_args()


def slugify(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "unknown")[:max_len]


def resolve_source(path_text: str, prefix_from: str, prefix_to: str) -> Path:
    source_text = path_text
    if prefix_from and prefix_to and source_text.startswith(prefix_from):
        source_text = prefix_to + source_text[len(prefix_from) :]
    return Path(source_text)


def infer_split(path_text: str) -> str:
    lower = path_text.lower()
    if "heldout" in lower:
        return "heldout"
    if "train" in lower:
        return "train"
    return "unknown"


def build_image_name(index: int, row: dict[str, str], source_path: Path) -> str:
    split = infer_split(row.get("contact_sheet_path", ""))
    parent = slugify(row.get("parent_task_name", ""))
    task = slugify(row.get("task_name", ""), max_len=56)
    take = slugify(row.get("take_uid", ""))[:8]
    suffix = source_path.suffix.lower() or ".jpg"
    return f"{index:04d}_{split}_{parent}_{task}_{take}{suffix}"


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        image = html.escape(row["review_image"])
        title = html.escape(f'{row["review_id"]} | {row.get("parent_task_name", "")} | {row.get("task_name", "")}')
        take_uid = html.escape(row.get("take_uid", ""))
        cards.append(
            f"""
<section class="card" id="{html.escape(row['review_id'])}">
  <h2>{title}</h2>
  <img src="{image}" alt="{title}">
  <dl>
    <dt>take_uid</dt><dd>{take_uid}</dd>
    <dt>auto_bucket</dt><dd>{html.escape(row.get('auto_bucket', ''))}</dd>
    <dt>relevance_score</dt><dd>{html.escape(row.get('relevance_score', ''))}</dd>
  </dl>
</section>
""".strip()
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Annotation Review Pack</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #1f2933; }}
    h1 {{ font-size: 24px; }}
    .card {{ background: white; border: 1px solid #d7dce0; border-radius: 6px; padding: 16px; margin: 0 0 20px; }}
    .card h2 {{ font-size: 16px; margin: 0 0 12px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #e2e8f0; }}
    dl {{ display: grid; grid-template-columns: 120px 1fr; gap: 4px 12px; font-size: 13px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; }}
  </style>
</head>
<body>
  <h1>Annotation Review Pack</h1>
  <p>Fill <code>annotation_review.csv</code> using <code>review_id</code> and <code>review_image</code>. See <a href="ANNOTATOR_GUIDE.zh-CN.md">ANNOTATOR_GUIDE.zh-CN.md</a> for the labeling rules.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_annotator_guide(path: Path) -> None:
    source = ROOT / "docs" / "annotator_quickstart.zh-CN.md"
    if source.exists():
        shutil.copy2(source, path)
        return
    path.write_text(
        "\n".join(
            [
                "# Ego-Exo v1 数据标注快速说明",
                "",
                "打开 annotation_review.csv 和 index.html。",
                "只填写 ego_hand_visibility, exo_body_visibility, object_interaction, phase_diversity, take_relevance, usable_for, notes。",
                "明显手物交互填 A_interaction_rich / tokenizer_main。",
                "明显身体移动但手物交互弱填 B_loco_body / loco_aux。",
                "纯场景、纯头动、看不清填 discard 或 diagnostic_candidate。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = read_csv(args.annotations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / args.image_subdir
    if args.copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for index, row in enumerate(rows, start=1):
        contact_sheet_path = row.get("contact_sheet_path", "")
        source = resolve_source(contact_sheet_path, args.path_prefix_from, args.path_prefix_to)
        image_name = build_image_name(index, row, source)
        relative_image = str(Path(args.image_subdir) / image_name).replace("\\", "/")
        if args.copy_images:
            target = image_dir / image_name
            if source.exists():
                shutil.copy2(source, target)
            else:
                missing.append(str(source))
        output_row = {
            "review_id": f"{index:04d}",
            "review_image": relative_image,
            "review_image_path": str((args.out_dir / relative_image).resolve()),
            **row,
        }
        output_rows.append(output_row)

    fieldnames = ["review_id", "review_image", "review_image_path", *rows[0].keys()] if rows else []
    review_csv = args.out_dir / "annotation_review.csv"
    write_csv(review_csv, output_rows, fieldnames=fieldnames)
    write_html(args.out_dir / "index.html", output_rows)
    write_annotator_guide(args.out_dir / "ANNOTATOR_GUIDE.zh-CN.md")
    if missing:
        missing_path = args.out_dir / "missing_images.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Missing {len(missing)} images; see {missing_path}")
    print(f"Saved {len(output_rows)} review rows to {review_csv}")
    print(f"Saved HTML gallery to {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
