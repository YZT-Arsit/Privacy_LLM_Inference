"""Preflight: compute the trusted-boundary runtime hash to bind into a TD Quote.

This is the *single source of truth* for the value that must go into the TD
Quote's ``report_data``. It computes exactly the runtime hash that
``run_tee_gpu_protocol_demo.py`` will later recompute and verify, using the same
shared recipe (``pllo.protocol.attestation``): SHA-512 over the trusted-boundary
manifest (source-file digests + public runtime identity).

Stable deployment workflow
--------------------------
1. Freeze the trusted-boundary code.
2. ``python scripts/write_tee_boundary_runtime_hash.py --boundary-backend process
   --gpu-backend mock [--expected-mr-td <mr_td>] --output outputs/runtime_hash.txt``
3. Bind the printed hash into the TD Quote ``report_data`` (your attestation
   client), obtain the signed attestation JWT, and assemble the evidence JSON.
4. ``python scripts/run_tee_gpu_protocol_demo.py ... --attestation-evidence
   evidence.json`` **with the identical** ``--boundary-backend / --gpu-backend /
   --expected-mr-td`` flags. The demo recomputes the same hash and verifies the
   binding (``runtime_hash_bound=True``).

Use the SAME flags in steps 2 and 4 -- ``expected_mr_td`` and the backends are
part of the runtime identity, so any change yields a different binding. Run this
*after* the code is frozen; editing any measured boundary file changes the hash.

numpy/stdlib only. No quote is generated here -- that is your attestation client.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.protocol.attestation import (  # noqa: E402
    DEFAULT_TRUSTED_BOUNDARY_PATHS,
    boundary_manifest_metadata,
    build_trusted_boundary_manifest,
    compute_runtime_hash_from_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boundary-backend", default="process",
                    choices=["process", "simulated"])
    ap.add_argument("--gpu-backend", default="mock", choices=["mock", "qwen7b"])
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--protocol-version", default="8.5")
    ap.add_argument(
        "--nonlinear-backend", default=None,
        help=("bind a nonlinear design into the runtime hash so design A and "
              "design B get different bindings (current|trusted_shortcut, "
              "aliases ok). Omit to preserve the legacy no-nonlinear hash."))
    ap.add_argument("--output", default=None,
                    help="write the runtime hash (hex) to this path")
    ap.add_argument("--manifest", default=None,
                    help="write the trusted-boundary manifest JSON to this path")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the hash on stdout")
    args = ap.parse_args()

    nonlinear_backend = None
    if args.nonlinear_backend is not None:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    metadata = boundary_manifest_metadata(
        args.boundary_backend, args.gpu_backend, args.expected_mr_td,
        protocol_version=args.protocol_version,
        nonlinear_backend=nonlinear_backend)
    manifest = build_trusted_boundary_manifest(metadata=metadata)
    runtime_hash = compute_runtime_hash_from_manifest(manifest)

    if args.manifest:
        p = Path(args.manifest)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, default=str),
                     encoding="utf-8")
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(runtime_hash + "\n", encoding="utf-8")

    # The hash is the only thing on stdout, so it is easy to capture/pipe.
    print(runtime_hash)
    if not args.quiet:
        measured = [e["path"] for e in manifest["files"]]
        print(f"# runtime_hash is the value to bind into TD Quote report_data",
              file=sys.stderr)
        print(f"# files_measured={len(measured)} "
              f"patterns={list(DEFAULT_TRUSTED_BOUNDARY_PATHS)}", file=sys.stderr)
        print(f"# metadata={metadata}", file=sys.stderr)
        if args.output:
            print(f"# wrote hash -> {args.output}", file=sys.stderr)
        if args.manifest:
            print(f"# wrote manifest -> {args.manifest}", file=sys.stderr)
        print("# verify later: run_tee_gpu_protocol_demo.py with the SAME "
              "--boundary-backend/--gpu-backend/--expected-mr-td + "
              "--attestation-evidence", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
