#!/usr/bin/env python3
"""Fail-closed access to canonical generation artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
BUNDLE = Path(__file__).resolve().parents[1]
REGISTRY = BUNDLE / "canonical_generation_artifact_registry.json"
ALLOWLIST = BUNDLE / "canonical_hash_allowlist.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_jsonl(logical_id: str) -> tuple[list[dict], dict]:
    registry = json.loads(REGISTRY.read_text())
    allowlist = json.loads(ALLOWLIST.read_text())["allowed_sha256_by_logical_id"]
    if logical_id not in registry["artifacts"] or logical_id not in allowlist:
        raise RuntimeError(f"unregistered generation artifact: {logical_id}")
    entry = registry["artifacts"][logical_id]
    path = REPO / entry["path"]
    if not path.is_file():
        raise RuntimeError(f"canonical artifact is missing: {path}")
    if path.stat().st_size != entry["size_bytes"]:
        raise RuntimeError(f"size mismatch for {logical_id}: {path.stat().st_size} != {entry['size_bytes']}")
    observed = sha256(path)
    if observed != entry["sha256"] or observed != allowlist[logical_id]:
        raise RuntimeError(f"SHA-256 mismatch for {logical_id}: {observed}")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSON at {logical_id}:{line_number}") from exc
    if len(rows) != entry["records"]:
        raise RuntimeError(f"record-count mismatch for {logical_id}: {len(rows)} != {entry['records']}")
    return rows, entry
