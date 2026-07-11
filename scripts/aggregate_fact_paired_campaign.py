#!/usr/bin/env python3
"""Require filtered, unfiltered, and fresh-short73 reports before a campaign GO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-report", type=Path, required=True)
    parser.add_argument("--unfiltered-report", type=Path, required=True)
    parser.add_argument("--fresh-short73-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "filtered": args.filtered_report,
        "unfiltered": args.unfiltered_report,
        "fresh_short73": args.fresh_short73_report,
    }
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    for name, report in reports.items():
        if report.get("schema") != "fact-v7-paired-value-gate-v1":
            raise ValueError(f"{name} report has an unsupported schema")
        if report.get("dataset_name") != name:
            raise ValueError(f"{name} report dataset identity mismatch")
        if len(str(report.get("domain_evidence_sha256", ""))) != 64:
            raise ValueError(f"{name} report lacks domain-bound evidence")
    evidence_hashes = [report["domain_evidence_sha256"] for report in reports.values()]
    if len(set(evidence_hashes)) != 3:
        raise ValueError("the three domains reuse identical prediction/leakage evidence")
    if reports["fresh_short73"].get("sample_ids_sha256") in {
        reports["filtered"].get("sample_ids_sha256"),
        reports["unfiltered"].get("sample_ids_sha256"),
    }:
        raise ValueError("fresh_short73 must use a distinct frozen sample set")
    if reports["filtered"].get("training_manifest_sha256") == reports["unfiltered"].get(
        "training_manifest_sha256"
    ):
        raise ValueError("filtered and unfiltered reports must bind different training manifests")
    reasons = [
        f"{name} domain did not meet the preregistered paired-value gate"
        for name, report in reports.items()
        if not report.get("decision", {}).get("go", False)
    ]
    result = {
        "schema": "fact-v7-paired-campaign-gate-v1",
        "decision": "GO" if not reasons else "NO_GO",
        "go": not reasons,
        "reasons": reasons,
        "domains": {
            name: {
                "samples": report.get("samples"),
                "takes": report.get("takes"),
                "decision": report.get("decision"),
                "report_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            }
            for name, report in reports.items()
        },
        "all_three_domains_required": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
