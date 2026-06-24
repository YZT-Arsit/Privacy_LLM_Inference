"""Robustly inspect a folded weight package's on-disk layout.

Schema-independent: scans the package directory for shard files
(``*.safetensors`` / ``*.pt``) and reports each shard's size + (for safetensors)
its tensor names/dtypes/shapes read straight from the file header -- so it works
even if the manifest has no top-level ``shards`` array (the manifest uses
``shard_index``). It then cross-checks the manifest (hash, declared shard list)
if one is present.

This also makes the float32 store precision visible (which is why a bf16 model
yields a ~2x folded package): the per-shard dtype is read from the safetensors
header.

Example::

    python scripts/inspect_folded_package.py \\
        --package-dir /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --output-json outputs/folded_package_inspection.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_GB = 1024 ** 3
_SHARD_EXTS = (".safetensors", ".pt")


def _safetensors_header(path: Path) -> dict | None:
    """Read the safetensors JSON header without loading tensor data."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]       # u64 little-endian
            header = json.loads(fh.read(n).decode("utf-8"))
        header.pop("__metadata__", None)
        return header
    except Exception:                                    # noqa: BLE001
        return None


def _scan_shards(package_dir: Path) -> list[dict]:
    shards = []
    for p in sorted(package_dir.iterdir()):
        if not (p.is_file() and p.suffix in _SHARD_EXTS):
            continue
        size = p.stat().st_size
        entry = {"name": p.name, "size_bytes": size,
                 "size_gb": round(size / _GB, 6), "format": p.suffix.lstrip(".")}
        hdr = _safetensors_header(p) if p.suffix == ".safetensors" else None
        if hdr is not None:
            entry["num_tensors"] = len(hdr)
            entry["tensors"] = sorted(hdr.keys())
            entry["dtypes"] = sorted({v.get("dtype", "?") for v in hdr.values()})
        shards.append(entry)
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--list-tensors", action="store_true",
                    help="print per-shard tensor names")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    pkg = Path(args.package_dir)
    if not pkg.is_dir():
        print("ERROR: not a directory: %s" % pkg, file=sys.stderr)
        return 2

    shards = _scan_shards(pkg)
    total_bytes = sum(s["size_bytes"] for s in shards)
    dtypes = sorted({d for s in shards for d in s.get("dtypes", [])})

    # cross-check manifest if present (schema-tolerant)
    manifest_info = {"present": False}
    man_path = pkg / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        # the shard list may be under shard_index (our schema), shards, or files
        declared = (man.get("shard_index") or man.get("shards")
                    or man.get("files") or [])
        manifest_info = {
            "present": True,
            "manifest_hash_field": man.get("manifest_hash"),
            "package_type": man.get("package_type"),
            "num_layers": man.get("num_layers"),
            "dtype": man.get("dtype"),
            "declared_shard_count": len(declared),
            "declared_shard_key": ("shard_index" if "shard_index" in man
                                   else "shards" if "shards" in man
                                   else "files" if "files" in man else None),
        }
        # recompute the canonical manifest hash via the library if importable
        try:
            from pllo.deployment import compute_manifest_hash, load_manifest
            manifest_info["manifest_hash_computed"] = compute_manifest_hash(
                load_manifest(pkg))
        except Exception as exc:                          # noqa: BLE001
            manifest_info["manifest_hash_computed_error"] = str(exc)

    rep = {
        "stage": "folded_package_inspection",
        "package_dir": str(pkg),
        "num_shards": len(shards),
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / _GB, 6),
        "store_dtypes": dtypes,
        "per_shard_size_gb": [s["size_gb"] for s in shards],
        "shards": shards,
        "manifest": manifest_info,
    }
    if not args.list_tensors:
        for s in rep["shards"]:
            s.pop("tensors", None)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("=== folded package inspection (%s) ===" % pkg)
    print("num_shards=%d total_size_gb=%s store_dtypes=%s"
          % (rep["num_shards"], rep["total_size_gb"], dtypes))
    if manifest_info["present"]:
        print("manifest: declared_shards=%s (key=%s) num_layers=%s dtype=%s"
              % (manifest_info.get("declared_shard_count"),
                 manifest_info.get("declared_shard_key"),
                 manifest_info.get("num_layers"), manifest_info.get("dtype")))
        if manifest_info.get("manifest_hash_computed"):
            print("manifest_hash=%s" % manifest_info["manifest_hash_computed"])
    for s in shards:
        line = "  %-22s %8.4f GB  fmt=%s" % (s["name"], s["size_gb"],
                                             s["format"])
        if "dtypes" in s:
            line += "  dtypes=%s  ntensors=%d" % (s["dtypes"], s["num_tensors"])
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
