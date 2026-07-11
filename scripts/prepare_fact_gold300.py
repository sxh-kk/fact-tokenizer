#!/usr/bin/env python3
"""Freeze the 300-sample EgoExo effect/contact gold annotation package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.gold_annotations import build_gold_pack, write_gold_pack
from fact_tokenizer.locked_split import load_reference_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--heldout-candidates", type=Path, required=True)
    parser.add_argument("--short73-candidates", type=Path, required=True)
    parser.add_argument("--diagnostic500", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--required-locked-takes", type=Path)
    return parser.parse_args()


def take_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("take_uids", payload.get("takes", []))
        return [str(row.get("take_uid") if isinstance(row, dict) else row) for row in payload]
    rows = load_reference_records(path)
    return [str(row.get("take_uid", "")) for row in rows if row.get("take_uid")]


def main() -> None:
    args = parse_args()
    diagnostics = load_reference_records(args.diagnostic500)
    diagnostic_ids = [str(row.get("sample_id", "")) for row in diagnostics if row.get("sample_id")]
    rows = build_gold_pack(
        load_reference_records(args.train_candidates),
        load_reference_records(args.heldout_candidates),
        load_reference_records(args.short73_candidates),
        diagnostic_ids,
        seed=args.seed,
        required_locked_takes=take_list(args.required_locked_takes),
    )
    metadata = write_gold_pack(args.output_dir, rows, args.seed)
    excluded_takes = sorted({row["take_uid"] for row in rows if row["gold_split"] == "probe_train"})
    excluded_path = args.output_dir / "representation_train_excluded_takes.txt"
    excluded_temp = excluded_path.with_suffix(excluded_path.suffix + ".tmp")
    excluded_temp.write_text(
        "\n".join(excluded_takes) + "\n",
        encoding="utf-8",
    )
    excluded_temp.replace(excluded_path)
    metadata["representation_train_excluded_takes"] = {
        "path": excluded_path.name,
        "takes": len(excluded_takes),
        "sha256": hashlib.sha256(excluded_path.read_bytes()).hexdigest(),
    }
    freeze_path = args.output_dir / "gold300_freeze.json"
    freeze_temp = freeze_path.with_suffix(freeze_path.suffix + ".tmp")
    freeze_temp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_temp.replace(freeze_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
