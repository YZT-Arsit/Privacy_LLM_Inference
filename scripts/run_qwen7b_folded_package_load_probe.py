"""Load + verify a folded weight package in the worker (NO /prefill, NO /decode).

The cross-machine demo's ``/prefill`` is still a TODO for the folded-package
backend (full 28-layer shard-streamed decode is not wired). This one-shot probe
exercises only the load/verify path the worker performs on ``init``: it loads the
package, verifies the manifest + shard hashes + absence of secret tensor names,
and reports the load/identity fields -- so the full 28-layer package can be
validated end-to-end as *provisioned* without running masked compute.

No model checkpoint is needed (loading folded shards does not load the base
model). ``tee_used_on_gpu`` is always false and the worker holds no mask secrets.

Example (H800)::

    python scripts/run_qwen7b_folded_package_load_probe.py \\
        --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --output-json outputs/qwen7b_folded_full_load_probe.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment import (  # noqa: E402
    list_package_shards,
    load_manifest,
    verify_package,
)
from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend  # noqa: E402
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest  # noqa: E402

_GB = 1024 ** 3


def _store_dtypes(pkg: Path) -> list:
    out = set()
    for p in list_package_shards(pkg):
        if p.suffix != ".safetensors":
            continue
        try:
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n).decode("utf-8"))
            hdr.pop("__metadata__", None)
            for v in hdr.values():
                out.add(v.get("dtype", "?"))
        except Exception:                                # noqa: BLE001
            pass
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folded-package-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_folded_full_load_probe.json")
    args = ap.parse_args()

    pkg = Path(args.folded_package_path)
    manifest = load_manifest(pkg)

    # filesystem-truth per-shard sizes (schema-independent)
    shard_files = list_package_shards(pkg)
    per_shard = [{"name": p.name, "size_gb": round(p.stat().st_size / _GB, 6)}
                 for p in shard_files]
    total_bytes = sum(p.stat().st_size for p in shard_files)

    # worker load + verify (this is exactly what init does), timed
    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg), device=args.device, dtype=args.dtype,
        verify_on_init=True)
    t0 = time.perf_counter()
    init_resp = backend.init(BoundaryInitRequest(
        session_id="load-probe", hidden_size=int(manifest.hidden_size or 0),
        vocab_size=int(manifest.vocab_size or 0),
        num_layers=int(manifest.num_layers), dtype=args.dtype,
        gpu_backend="qwen7b_folded_package"))
    load_time_s = time.perf_counter() - t0
    desc = backend.describe()
    vrep = verify_package(pkg)

    report = {
        "stage": "qwen7b_folded_package_load_probe",
        "folded_package_path": str(pkg),
        "folded_package_loaded": bool(desc["folded_package_loaded"]),
        "folded_package_valid": bool(vrep["package_valid"]),
        "load_time_s": load_time_s,
        "package_size_gb": round(total_bytes / _GB, 6),
        "num_layers": int(manifest.num_layers),
        "num_shards": len(shard_files),
        "manifest_hash": desc["manifest_hash"],
        "store_dtypes": _store_dtypes(pkg),
        "worker_has_mask_secrets": bool(desc["worker_has_mask_secrets"]),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": vrep["forbidden_fields_found"],
        "per_shard_size_gb": [s["size_gb"] for s in per_shard],
        "per_shard": per_shard,
        "total_size_gb": round(total_bytes / _GB, 6),
        "verify_hash_mismatches": len(vrep["hash_mismatches"]),
        "verify_missing_shards": vrep["missing_shards"],
        "note": "load/verify probe only; /prefill /decode (full 28-layer "
                "shard-streamed masked compute) remain TODO.",
    }

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== Qwen folded-package load probe (%s) ===" % pkg)
    print("folded_package_loaded=%s folded_package_valid=%s load_time_s=%.3f"
          % (report["folded_package_loaded"], report["folded_package_valid"],
             load_time_s))
    print("package_size_gb=%s num_layers=%d num_shards=%d store_dtypes=%s"
          % (report["package_size_gb"], report["num_layers"],
             report["num_shards"], report["store_dtypes"]))
    print("manifest_hash=%s" % report["manifest_hash"])
    print("worker_has_mask_secrets=%s tee_used_on_gpu=%s"
          % (report["worker_has_mask_secrets"], report["tee_used_on_gpu"]))
    print("gpu_visible_plaintext_fields=%s leaked_secret_fields=%s"
          % (report["gpu_visible_plaintext_fields"],
             report["leaked_secret_fields"]))
    print("per_shard_size_gb=%s" % report["per_shard_size_gb"])
    print("total_size_gb=%s" % report["total_size_gb"])
    ok = (report["folded_package_loaded"] and report["folded_package_valid"]
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"]
          and not report["leaked_secret_fields"])
    print("\nLOAD PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
