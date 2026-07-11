#!/usr/bin/env python3
"""Freeze proof that P1 uses Exo only for target building and quality stratification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--take-quality-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite P1 provenance: {args.output_json}")
    target_config_path = args.target_cache / "target_config.json"
    target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    view_hashes = target_config.get("identity", {}).get("source_hashes", {}).get("views", {})
    if "exo" not in view_hashes:
        raise ValueError("P1 target cache identity does not prove an Exo source was used")
    quality = json.loads(args.take_quality_report.read_text(encoding="utf-8"))
    if quality.get("schema") != "fact-take-quality-index-v1":
        raise ValueError("unsupported take-quality report schema")
    payload = {
        "schema": "fact-p1-exo-target-quality-provenance-v1",
        "exo_roles": ["target_build", "quality_stratification"],
        "exo_model_input": False,
        "exo_optimizer_input": False,
        "source_hashes": {
            "exo_rgb": view_hashes["exo"],
            "target_config": sha256_file(target_config_path),
            "target_manifest": sha256_file(args.target_cache / "target_manifest.jsonl"),
            "take_quality_report": sha256_file(args.take_quality_report),
        },
        "target_identity_sha256": target_config.get("identity_sha256"),
        "quality_usage": quality.get("quality_usage"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
