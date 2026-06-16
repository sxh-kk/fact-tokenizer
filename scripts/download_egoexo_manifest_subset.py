#!/usr/bin/env python3
"""Download a UID-filtered EgoExo4D manifest subset with boto3."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Iterable

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

from ego4d.internal.download.manifest import manifest_loads


MANIFEST_BUCKET = "ego4d-consortium-sharing"
MANIFEST_KEY = "egoexo-public/v2/downscaled_takes/448/manifest.json"


def make_s3_client():
    return boto3.client(
        "s3",
        config=Config(
            connect_timeout=60,
            read_timeout=300,
            retries={"mode": "standard", "max_attempts": 8},
            max_pool_connections=64,
        ),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    rest = uri[len("s3://") :]
    bucket, key = rest.split("/", 1)
    return bucket, key


def load_uids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def download_manifest(cache_path: Path, force: bool, transfer_config: TransferConfig | None) -> str:
    if cache_path.exists() and not force:
        return cache_path.read_text()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".part")
    s3 = make_s3_client()
    s3.download_file(MANIFEST_BUCKET, MANIFEST_KEY, str(tmp), Config=transfer_config)
    tmp.replace(cache_path)
    return cache_path.read_text()


def iter_targets(manifest_text: str, uids: set[str], out_dir: Path) -> list[tuple[str, str, int]]:
    targets = []
    for entry in manifest_loads(manifest_text):
        if entry.uid not in uids:
            continue
        for path in entry.paths:
            if path.file_type != "mp4":
                continue
            out_path = out_dir / path.relative_path
            if out_path.exists() and path.size is not None and out_path.stat().st_size == path.size:
                continue
            targets.append((path.source_path, str(out_path), int(path.size or 0)))
    return targets


def read_range_with_retry(
    bucket: str,
    key: str,
    start: int,
    end: int,
    retries: int,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            s3 = make_s3_client()
            obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
            return obj["Body"].read()
        except Exception as exc:  # transient S3 streams can break mid-read on this network
            last_exc = exc
            time.sleep(min(8.0, 0.5 * (attempt + 1)))
    assert last_exc is not None
    raise last_exc


def download_one_ranged(
    target: tuple[str, str, int],
    chunk_size: int,
    chunk_workers: int,
    chunk_retries: int,
) -> dict:
    source_uri, out_path, expected_size = target
    if expected_size <= 0:
        raise ValueError(f"Ranged mode needs a known file size for {out_path}")
    bucket, key = parse_s3_uri(source_uri)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    fd = os.open(tmp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, expected_size)

        def fetch_and_write(start: int) -> int:
            end = min(expected_size - 1, start + chunk_size - 1)
            data = read_range_with_retry(bucket, key, start, end, chunk_retries)
            if len(data) != end - start + 1:
                raise RuntimeError(f"Short range read for {out}: got {len(data)}, expected {end - start + 1}")
            os.pwrite(fd, data, start)
            return len(data)

        starts = range(0, expected_size, chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_workers) as pool:
            for _ in pool.map(fetch_and_write, starts):
                pass
    finally:
        os.close(fd)

    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {out}: got {tmp.stat().st_size}, expected {expected_size}")
    tmp.replace(out)
    return {"source": source_uri, "output": str(out), "bytes": expected_size}


def download_one_transfer(target: tuple[str, str, int], transfer_config: TransferConfig | None) -> dict:
    source_uri, out_path, expected_size = target
    bucket, key = parse_s3_uri(source_uri)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    s3 = make_s3_client()
    s3.download_file(bucket, key, str(tmp), Config=transfer_config)
    if expected_size and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {out}: got {tmp.stat().st_size}, expected {expected_size}")
    tmp.replace(out)
    return {"source": source_uri, "output": str(out), "bytes": expected_size or out.stat().st_size}


def download_one(
    target: tuple[str, str, int],
    transfer_config: TransferConfig | None,
    mode: str,
    chunk_size: int,
    chunk_workers: int,
    chunk_retries: int,
) -> dict:
    if mode == "ranged":
        return download_one_ranged(target, chunk_size, chunk_workers, chunk_retries)
    return download_one_transfer(target, transfer_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--manifest-cache", type=Path, default=Path("data/egoexo4d/fact_debug/downscaled_448_manifest.json"))
    parser.add_argument("--failed-jsonl", type=Path, default=Path("data/egoexo4d/fact_debug/download_failed.jsonl"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--transfer-concurrency", type=int, default=4)
    parser.add_argument("--download-mode", choices=["transfer", "ranged"], default="transfer")
    parser.add_argument("--chunk-size-mib", type=int, default=1)
    parser.add_argument("--chunk-workers", type=int, default=4)
    parser.add_argument("--chunk-retries", type=int, default=6)
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()

    uids = load_uids(args.uids)
    transfer_config = TransferConfig(max_concurrency=max(1, args.transfer_concurrency))
    manifest_text = download_manifest(args.manifest_cache, force=args.force_manifest, transfer_config=transfer_config)
    targets = iter_targets(manifest_text, uids, args.out_dir)
    total_bytes = sum(size for _, _, size in targets)
    print(f"uids={len(uids)} pending_files={len(targets)} pending_gib={total_bytes / (1024 ** 3):.3f}", flush=True)

    args.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    chunk_size = max(1, args.chunk_size_mib) * 1024 * 1024
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(
                download_one,
                target,
                transfer_config,
                args.download_mode,
                chunk_size,
                max(1, args.chunk_workers),
                max(1, args.chunk_retries),
            )
            for target in targets
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="downloads"):
            try:
                future.result()
            except Exception as exc:  # keep the batch moving and report at the end
                failures.append({"error": repr(exc)})

    with args.failed_jsonl.open("w") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"done failed={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
