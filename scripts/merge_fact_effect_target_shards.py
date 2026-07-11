#!/usr/bin/env python3
"""Atomically merge disjoint, hash-verified FACT effect target-cache shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_targets import verify_target_cache
from fact_tokenizer.effect_manifest import read_manifest_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--visual-audit-gate", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merged cache: {args.output_dir}")
    if len({path.resolve() for path in args.shard}) != len(args.shard):
        raise ValueError("duplicate --shard paths are not allowed")
    configs = []
    all_records: dict[str, tuple[Path, dict]] = {}
    shard_reports = []
    for shard in args.shard:
        errors = verify_target_cache(shard)
        if errors:
            raise ValueError(f"target shard integrity failure for {shard}: {errors[:3]}")
        config_path = shard / "target_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs.append(config)
        count = 0
        with (shard / "target_manifest.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                if sample_id in all_records:
                    raise ValueError(f"sample_id {sample_id!r} occurs in multiple target shards")
                all_records[sample_id] = (shard, record)
                count += 1
        shard_reports.append(
            {
                "path": str(shard.resolve()),
                "config_sha256": sha256_file(config_path),
                "manifest_sha256": sha256_file(shard / "target_manifest.jsonl"),
                "samples": count,
            }
        )
    identity_pairs = {(config.get("config_sha256"), config.get("identity_sha256")) for config in configs}
    if len(identity_pairs) != 1:
        raise ValueError("target shards have different config or source/weight identities")
    expected_records = read_manifest_jsonl(args.expected_manifest)
    expected_ids = {record.sample_id for record in expected_records}
    actual_ids = set(all_records)
    if actual_ids != expected_ids:
        raise ValueError(
            "target shards do not exactly cover the expected manifest: "
            f"missing={sorted(expected_ids - actual_ids)[:3]}, extra={sorted(actual_ids - expected_ids)[:3]}"
        )
    gate = json.loads(args.visual_audit_gate.read_text(encoding="utf-8"))
    identity_sha256 = configs[0].get("identity_sha256")
    if (
        gate.get("passed") is not True
        or int(gate.get("rows", 0)) != 50
        or int(gate.get("fully_aligned", 0)) < 45
        or float(gate.get("minimum_pass_fraction", 0.0)) < 0.90
        or float(gate.get("pass_fraction", 0.0)) < 0.90
        or gate.get("target_identity_sha256") != identity_sha256
    ):
        raise ValueError("visual audit gate does not release these target shards")
    staging = args.output_dir.with_name(f".{args.output_dir.name}.staging-{uuid.uuid4().hex[:12]}")
    (staging / "samples").mkdir(parents=True)
    try:
        merged_records = []
        for sample_id in sorted(all_records):
            shard, record = all_records[sample_id]
            source = shard / record["path"]
            destination = staging / "samples" / source.name
            shutil.copy2(source, destination)
            if sha256_file(destination) != record["sha256"]:
                raise ValueError(f"copied target hash mismatch for {sample_id}")
            merged = dict(record)
            merged["path"] = f"samples/{destination.name}"
            merged_records.append(merged)
        with (staging / "target_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in merged_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        config = dict(configs[0])
        config["samples"] = len(merged_records)
        config["formal_release"] = {
            "visual_audit_gate_sha256": sha256_file(args.visual_audit_gate),
            "review_rows": 50,
            "pass_fraction": float(gate["pass_fraction"]),
            "target_identity_sha256": identity_sha256,
            "expected_manifest_sha256": sha256_file(args.expected_manifest),
        }
        (staging / "target_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema": "fact-effect-target-merge-v1",
            "samples": len(merged_records),
            "shards": shard_reports,
            "config_sha256": config["config_sha256"],
            "identity_sha256": config["identity_sha256"],
            "expected_manifest_sha256": sha256_file(args.expected_manifest),
            "visual_audit_gate_sha256": sha256_file(args.visual_audit_gate),
            "target_manifest_sha256": sha256_file(staging / "target_manifest.jsonl"),
        }
        (staging / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if verify_target_cache(staging):
            raise ValueError("merged target cache failed final integrity verification")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
