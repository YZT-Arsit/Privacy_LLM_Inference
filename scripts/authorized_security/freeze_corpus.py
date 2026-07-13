#!/usr/bin/env python3
"""Validate attacker-view JSONL files and freeze an immutable corpus manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from view_contracts import View, ViewRecord


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_input(value: str) -> tuple[View, Path]:
    try:
        name, raw_path = value.split("=", 1)
        return View(name), Path(raw_path).resolve()
    except (ValueError, KeyError) as exc:
        raise argparse.ArgumentTypeError("input must be V0=/path/file.jsonl") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", type=parse_input, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    entries = []
    seen_paths: set[Path] = set()
    ids_by_view: dict[str, list[str]] = {}
    for view, path in args.input:
        if path in seen_paths:
            raise ValueError(f"duplicate corpus path: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        sample_ids = []
        with path.open() as f:
            for line_number, line in enumerate(f, 1):
                row = json.loads(line)
                record = ViewRecord.build(view, row)
                sample_id = str(record.get("sample_id"))
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
                sample_ids.append(sample_id)
        ids_by_view[view.value] = sample_ids
        entries.append({
            "view": view.value,
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "records": len(sample_ids),
            "sample_ids_sha256": hashlib.sha256(
                json.dumps(sample_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    shared = None
    if len(ids_by_view) > 1:
        sequences = list(ids_by_view.values())
        shared = all(ids == sequences[0] for ids in sequences[1:])
        if not shared:
            raise ValueError("view corpora do not have identical ordered sample IDs")
    manifest = {
        "schema": "authorized_security_immutable_corpus",
        "version": "1.0",
        "task": args.task,
        "split": args.split,
        "ordered_sample_ids_match": shared,
        "entries": sorted(entries, key=lambda x: x["view"]),
    }
    canonical = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != canonical:
        raise RuntimeError(f"refusing to overwrite frozen manifest: {out}")
    out.write_text(canonical)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    sidecar = out.with_suffix(out.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {out.name}\n")
    print(json.dumps({"manifest": str(out), "sha256": digest, "entries": len(entries)}))


if __name__ == "__main__":
    main()
