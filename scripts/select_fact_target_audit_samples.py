#!/usr/bin/env python3
"""Freeze 50 manifest rows stratified by quality, camera pose, and object mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import read_manifest_jsonl  # noqa: E402
from fact_tokenizer.target_audit_selection import (  # noqa: E402
    AUDIT_SAMPLE_COUNT,
    AUDIT_SELECTION_SEED,
    build_audit_selection_report,
    select_audit_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=AUDIT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=AUDIT_SELECTION_SEED)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit selection: {args.output_dir}")
    records = read_manifest_jsonl(args.manifest)
    selected = select_audit_indices(records, sample_count=args.sample_count, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    index_path = args.output_dir / "audit_sample_index.npy"
    np.save(index_path, np.asarray(selected, dtype=np.int64))
    import hashlib

    report = build_audit_selection_report(
        records,
        selected,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
    )
    (args.output_dir / "audit_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
