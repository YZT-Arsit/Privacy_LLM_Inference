"""Package-backed one-step LOGITS probe (full prefill + folded head).

Runs a package-backed full prefill over ALL package layers, applies the package's
folded head to produce MASKED logits (no masks on the worker), recovers the logits
in the trusted path, and verifies next-token / top-k agreement + logits error
against the trusted in-process folded reference (``MaskedQwenSession``).

Trusted path owns the masks + recovery; the package path only executes the folded
layers + folded head over masked tensors. No plaintext ids / recovered logits /
mask secrets touch the package path.

``--dry-run`` uses a tiny random Qwen2 on CPU (never a paper result).

Example (H800)::

    python scripts/run_qwen7b_folded_package_onestep_logits_probe.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --model-name Qwen2.5-7B-Instruct \\
        --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --seq-len 128 --seed 2035 --dtype bfloat16 --device cuda \\
        --folded-weight-device cuda --topk 5 \\
        --output-json outputs/qwen7b_folded_full_onestep_logits_probe.json \\
        --output-md   outputs/qwen7b_folded_full_onestep_logits_probe.md
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
from pllo.experiments.folded_probe_common import (  # noqa: E402
    err_stats,
    load_model_and_ids,
    seed_from_manifest,
    topk_overlap,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    NonlinearDesignNotWired,
    assert_real_path_execution,
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    _cfg_to,
)
from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend  # noqa: E402
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--folded-package-path", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unwired-nonlinear", action="store_true",
                    default=False,
                    help="allow a non-paper-facing PROTOTYPE run for a "
                         "nonlinear design not yet executed in the real "
                         "path (tag-only)")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_folded_full_onestep_logits_probe.json")
    ap.add_argument("--output-md",
                    default="outputs/qwen7b_folded_full_onestep_logits_probe.md")
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
    n = int(manifest.num_layers)
    seed = seed_from_manifest(pkg_dir, args.seed)
    if manifest.model_path_or_id and args.model_path and \
            manifest.model_path_or_id != args.model_path:
        print("WARNING: --model-path != package model_path_or_id (%s)."
              % manifest.model_path_or_id)

    cfg = MemoryOptimizedConfig(
        num_layers=n, batch_size=1, seq_len=int(ids.shape[1]), max_new_tokens=1,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)
    h_tilde = session.mask_embeddings(ids)
    cfg0 = _cfg_to(session.layer_configs[0], session.compute_device)

    # --- trusted in-process folded reference (all n layers + head) ------------
    ref_out = session.worker_prefill(h_tilde)
    ref_logits_tilde_last = ref_out["logits_tilde"][:, -1, :]
    recovered_ref = session.recover(ref_logits_tilde_last)        # trusted recover

    # --- verify + load package; package-backed prefill + folded head ----------
    vrep = verify_package(pkg_dir)
    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_dir), device=device, dtype=dtype,
        nonlinear_backend=args.nonlinear_backend)
    init_resp = backend.init(BoundaryInitRequest(
        session_id="onestep-probe", hidden_size=int(getattr(mc, "hidden_size")),
        vocab_size=int(getattr(mc, "vocab_size")), num_layers=n, dtype=dtype,
        gpu_backend="qwen7b_folded_package"))
    desc = backend.describe()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pre = backend.run_prefill(h_tilde, n, cfg0, session._cos, session._sin,
                              session.eps)
    pkg_logits_tilde = backend.run_head(pre["y_tilde"], session.eps)  # masked
    latency_s = time.perf_counter() - t0
    peak_mb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    recovered_pkg = session.recover(pkg_logits_tilde[:, -1, :])   # trusted recover

    lmax, lmean, ll2 = err_stats(recovered_pkg, recovered_ref)
    next_ref = int(recovered_ref.argmax(-1).item())
    next_pkg = int(recovered_pkg.argmax(-1).item())
    top1_match = bool(next_ref == next_pkg)
    ov = topk_overlap(recovered_pkg, recovered_ref, args.topk)
    allclose = bool(torch.allclose(recovered_pkg, recovered_ref, atol=args.atol,
                                   rtol=args.rtol))

    report = {
        "stage": "qwen7b_folded_package_onestep_logits_probe",
        "dry_run": dry_run, "model_name": args.model_name,
        "num_exec_layers": n, "num_package_layers": n,
        "seq_len": int(ids.shape[1]), "dtype": dtype, "seed": seed,
        "folded_package_path": str(pkg_dir),
        "folded_package_loaded": bool(desc["folded_package_loaded"]),
        "folded_package_valid": bool(vrep["package_valid"]),
        "package_size_gb": round(package_size_gb(pkg_dir), 6),
        "num_shards": vrep["num_shards"], "manifest_hash": desc["manifest_hash"],
        "worker_has_mask_secrets": bool(desc["worker_has_mask_secrets"]),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "package_backed_prefill": True, "package_backed_head": True,
        "logits_max_abs_error": lmax, "logits_mean_abs_error": lmean,
        "logits_relative_l2_error": ll2,
        "top1_match": top1_match, "topk": args.topk, "topk_overlap": ov,
        "next_token_match": top1_match,
        "next_token_ref": next_ref, "next_token_pkg": next_pkg,
        "allclose": allclose, "atol": args.atol, "rtol": args.rtol,
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": vrep["forbidden_fields_found"],
        "latency_s": latency_s, "peak_gpu_memory_mb": peak_mb,
    }
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))
    # OVERRIDE the capability stamp with MEASURED execution counters from the
    # worker (proves design B genuinely lifted the activation onto the GPU).
    report.update(backend.nonlinear_execution_evidence())

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print("=== package-backed ONE-STEP LOGITS probe (n=%d, dry_run=%s) ==="
          % (n, dry_run))
    print("folded_package_loaded=%s folded_package_valid=%s"
          % (report["folded_package_loaded"], report["folded_package_valid"]))
    print("worker_has_mask_secrets=%s tee_used_on_gpu=%s"
          % (report["worker_has_mask_secrets"], report["tee_used_on_gpu"]))
    print("logits_max_abs_error=%.3e logits_mean_abs_error=%.3e "
          "logits_relative_l2_error=%.3e" % (lmax, lmean, ll2))
    print("top1_match=%s next_token_match=%s topk_overlap=%.4f (k=%d)"
          % (top1_match, top1_match, ov, args.topk))
    print("allclose=%s latency_s=%.3f peak_gpu_memory_mb=%s"
          % (allclose, latency_s, peak_mb))
    ok = (report["folded_package_loaded"] and report["folded_package_valid"]
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"] and top1_match
          and not report["leaked_secret_fields"])
    print("\nONESTEP LOGITS PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Package-backed one-step logits probe (n=%d, dry_run=%s)"
         % (r["num_exec_layers"], r["dry_run"]), "",
         "- model_name=`%s`  seq_len=%s  dtype=%s  seed=%s"
         % (r["model_name"], r["seq_len"], r["dtype"], r["seed"]),
         "- folded_package_path=`%s`  num_shards=%s  package_size_gb=%s"
         % (r["folded_package_path"], r["num_shards"], r["package_size_gb"]),
         "- manifest_hash=`%s`" % r["manifest_hash"],
         "- **folded_package_loaded=%s** **folded_package_valid=%s** "
         "**package_backed_prefill=True** **package_backed_head=True**"
         % (r["folded_package_loaded"], r["folded_package_valid"]),
         "- **worker_has_mask_secrets=%s** **tee_used_on_gpu=%s**"
         % (r["worker_has_mask_secrets"], r["tee_used_on_gpu"]),
         "", "## Recovered-logits agreement (package vs in-process reference)", "",
         "- logits_max_abs_error=%.3e" % r["logits_max_abs_error"],
         "- logits_mean_abs_error=%.3e" % r["logits_mean_abs_error"],
         "- logits_relative_l2_error=%.3e" % r["logits_relative_l2_error"],
         "- **top1_match=%s**  **next_token_match=%s**  topk_overlap=%.4f (k=%d)"
         % (r["top1_match"], r["next_token_match"], r["topk_overlap"], r["topk"]),
         "- allclose=%s  latency_s=%s  peak_gpu_memory_mb=%s"
         % (r["allclose"], r["latency_s"], r["peak_gpu_memory_mb"]),
         "- gpu_visible_plaintext_fields=%s  leaked_secret_fields=%s"
         % (r["gpu_visible_plaintext_fields"] or "[]",
            r["leaked_secret_fields"] or "[]")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
