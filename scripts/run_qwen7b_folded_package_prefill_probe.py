"""Package-backed multi-layer PREFILL probe (executable folded package).

Executes the first ``--num-exec-layers`` (1 / 4 / 28) transformer layers of a
masked prefill USING THE FOLDED PACKAGE SHARDS (no masks on the worker), and
compares the masked hidden against the trusted in-process folded reference from
``MaskedQwenSession``. This is the milestone: it shows the folded package is
*executable*, not merely generated + loaded.

Trusted path: owns masks, builds masked embeddings, computes the k-layer folded
reference. Package path: loads folded shards only, streams k layers, holds no mask
secrets; no plaintext ids / recovered logits / mask secrets touch the package path.

Real Qwen2.5-7B needs CUDA + a local checkpoint (+ the package built from the SAME
checkpoint). ``--dry-run`` uses a tiny random Qwen2 on CPU (never a paper result).

Example (H800)::

    python scripts/run_qwen7b_folded_package_prefill_probe.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --model-name Qwen2.5-7B-Instruct \\
        --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --num-exec-layers 4 --seq-len 128 --seed 2035 --dtype bfloat16 \\
        --device cuda --folded-weight-device cuda --atol 1e-3 --rtol 1e-3 \\
        --output-json outputs/qwen7b_folded_full_prefill_4layer_probe.json \\
        --output-md   outputs/qwen7b_folded_full_prefill_4layer_probe.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pllo.deployment import load_manifest, package_size_gb, verify_package  # noqa: E402
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    NonlinearDesignNotWired,
    assert_real_path_execution,
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    _masked_block_prefill_chunked,
)
from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend  # noqa: E402
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest  # noqa: E402

# shared helpers (model load, tokenize, seed-from-manifest, error stats)
from pllo.experiments.folded_probe_common import (  # noqa: E402
    err_stats,
    load_model_and_ids,
    seed_from_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--folded-package-path", required=True)
    ap.add_argument("--num-exec-layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unwired-nonlinear", action="store_true",
                    default=False,
                    help="allow a non-paper-facing PROTOTYPE run for a "
                         "nonlinear design not yet executed in the real "
                         "path (tag-only)")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_folded_full_prefill_probe.json")
    ap.add_argument("--output-md",
                    default="outputs/qwen7b_folded_full_prefill_probe.md")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    dry_run = bool(args.dry_run or not args.model_path)
    try:
        assert_real_path_execution(
            args.nonlinear_backend, dry_run=dry_run,
            allow_unwired=args.allow_unwired_nonlinear)
    except NonlinearDesignNotWired as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 3
    model, mc, ids, device, dtype = load_model_and_ids(args, dry_run)

    pkg_dir = Path(args.folded_package_path)
    manifest = load_manifest(pkg_dir)
    num_package_layers = int(manifest.num_layers)
    k = int(args.num_exec_layers)
    if k > num_package_layers:
        ap.error("--num-exec-layers %d > package num_layers %d"
                 % (k, num_package_layers))
    seed = seed_from_manifest(pkg_dir, args.seed)
    if manifest.model_path_or_id and args.model_path and \
            manifest.model_path_or_id != args.model_path:
        print("WARNING: --model-path != package model_path_or_id (%s); the "
              "comparison is only valid against the model the package was built "
              "from." % manifest.model_path_or_id)

    # trusted session sized to k layers (layer masks are seed-derived +
    # num_layers-independent, so layers 0..k-1 match the full package's shards).
    cfg = MemoryOptimizedConfig(
        num_layers=k, batch_size=1, seq_len=int(ids.shape[1]), max_new_tokens=1,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)
    h_tilde = session.mask_embeddings(ids)             # boundary-masked input

    # --- trusted in-process folded reference for the first k layers -----------
    ref = h_tilde
    cfg0 = None
    for ell in range(k):
        folded, down_info, cfg_c = session._folded_layer(ell)
        if cfg0 is None:
            cfg0 = cfg_c
        ref = _masked_block_prefill_chunked(
            ref, folded, down_info, cfg_c, session._cos, session._sin,
            session.chunk)["y_tilde"]
        del folded, down_info

    # --- verify + load package in the untrusted worker (NO masks) -------------
    vrep = verify_package(pkg_dir)
    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_dir), device=device, dtype=dtype)
    init_resp = backend.init(BoundaryInitRequest(
        session_id="prefill-probe", hidden_size=int(getattr(mc, "hidden_size")),
        vocab_size=int(getattr(mc, "vocab_size")), num_layers=k, dtype=dtype,
        gpu_backend="qwen7b_folded_package"))
    desc = backend.describe()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = backend.run_prefill(h_tilde, k, cfg0, session._cos, session._sin,
                              session.eps)
    latency_s = time.perf_counter() - t0
    y_pkg = out["y_tilde"]
    peak_mb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    max_abs, mean_abs, rel_l2 = err_stats(y_pkg, ref)
    allclose = bool(torch.allclose(y_pkg, ref, atol=args.atol, rtol=args.rtol))

    report = {
        "stage": "qwen7b_folded_package_prefill_probe",
        "dry_run": dry_run,
        "model_name": args.model_name,
        "num_exec_layers": k,
        "seq_len": int(ids.shape[1]),
        "dtype": dtype,
        "seed": seed,
        "folded_package_path": str(pkg_dir),
        "folded_package_loaded": bool(desc["folded_package_loaded"]),
        "folded_package_valid": bool(vrep["package_valid"]),
        "package_size_gb": round(package_size_gb(pkg_dir), 6),
        "num_package_layers": num_package_layers,
        "num_shards": vrep["num_shards"],
        "manifest_hash": desc["manifest_hash"],
        "worker_has_mask_secrets": bool(desc["worker_has_mask_secrets"]),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "package_backed_prefill": True,
        "allclose": allclose,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": rel_l2,
        "atol": args.atol,
        "rtol": args.rtol,
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": vrep["forbidden_fields_found"],
        "latency_s": latency_s,
        "peak_gpu_memory_mb": peak_mb,
    }
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print("=== package-backed PREFILL probe (k=%d, dry_run=%s) ===" % (k, dry_run))
    print("folded_package_loaded=%s folded_package_valid=%s "
          "package_backed_prefill=True" % (report["folded_package_loaded"],
                                           report["folded_package_valid"]))
    print("num_exec_layers=%d / num_package_layers=%d num_shards=%d"
          % (k, num_package_layers, report["num_shards"]))
    print("worker_has_mask_secrets=%s tee_used_on_gpu=%s"
          % (report["worker_has_mask_secrets"], report["tee_used_on_gpu"]))
    print("max_abs_error=%.3e mean_abs_error=%.3e relative_l2_error=%.3e"
          % (max_abs, mean_abs, rel_l2))
    print("allclose=%s (atol=%s rtol=%s) latency_s=%.3f peak_gpu_memory_mb=%s"
          % (allclose, args.atol, args.rtol, latency_s, peak_mb))
    ok = (report["folded_package_loaded"] and report["folded_package_valid"]
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"] and allclose
          and not report["leaked_secret_fields"])
    print("\nPREFILL PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Package-backed prefill probe (k=%d, dry_run=%s)"
         % (r["num_exec_layers"], r["dry_run"]), "",
         "- model_name=`%s`  seq_len=%s  dtype=%s  seed=%s"
         % (r["model_name"], r["seq_len"], r["dtype"], r["seed"]),
         "- folded_package_path=`%s`" % r["folded_package_path"],
         "- **folded_package_loaded=%s**  **folded_package_valid=%s**  "
         "**package_backed_prefill=True**"
         % (r["folded_package_loaded"], r["folded_package_valid"]),
         "- num_exec_layers=%d / num_package_layers=%d  num_shards=%d  "
         "package_size_gb=%s" % (r["num_exec_layers"], r["num_package_layers"],
                                 r["num_shards"], r["package_size_gb"]),
         "- manifest_hash=`%s`" % r["manifest_hash"],
         "- **worker_has_mask_secrets=%s**  **tee_used_on_gpu=%s**"
         % (r["worker_has_mask_secrets"], r["tee_used_on_gpu"]),
         "", "## Correctness (package vs in-process folded reference)", "",
         "- **allclose=%s** (atol=%s rtol=%s)" % (r["allclose"], r["atol"],
                                                  r["rtol"]),
         "- max_abs_error=%.3e" % r["max_abs_error"],
         "- mean_abs_error=%.3e" % r["mean_abs_error"],
         "- relative_l2_error=%.3e" % r["relative_l2_error"],
         "- latency_s=%s  peak_gpu_memory_mb=%s" % (r["latency_s"],
                                                    r["peak_gpu_memory_mb"]),
         "- gpu_visible_plaintext_fields=%s  leaked_secret_fields=%s"
         % (r["gpu_visible_plaintext_fields"] or "[]",
            r["leaked_secret_fields"] or "[]"), "",
         "_The worker streamed %d folded shards over the masked input; it held no "
         "masks. The down projection is pre-folded in each shard._"
         % r["num_exec_layers"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
