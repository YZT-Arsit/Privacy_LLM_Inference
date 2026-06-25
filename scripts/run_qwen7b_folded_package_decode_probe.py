"""Package-backed short DECODE probe (prefill + KV + max_new_tokens steps).

Runs a package-backed full prefill, then ``--max-new-tokens`` greedy decode steps
USING THE FOLDED PACKAGE SHARDS + a masked KV cache (no masks on the worker), and
checks the generated token ids match the trusted in-process folded reference
(``MaskedQwenSession`` worker_prefill/worker_decode). The boundary owns masking /
recovery / sampling; the package path only executes folded layers + head over
masked tensors.

``--dry-run`` uses a tiny random Qwen2 on CPU (never a paper result).

Example (H800)::

    python scripts/run_qwen7b_folded_package_decode_probe.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --model-name Qwen2.5-7B-Instruct \\
        --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --max-new-tokens 4 --seq-len 128 --seed 2035 --dtype bfloat16 \\
        --device cuda --folded-weight-device cuda \\
        --output-json outputs/qwen7b_folded_full_decode_probe.json \\
        --output-md   outputs/qwen7b_folded_full_decode_probe.md
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
    load_model_and_ids,
    seed_from_manifest,
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


def _greedy(rec):
    return int(rec.argmax(-1).item())


def _trusted_reference_tokens(session, h_tilde, n_new, seq_len):
    """Trusted masked greedy decode via the session (worker_prefill/decode)."""
    toks = []
    out = session.worker_prefill(h_tilde)
    rec = session.recover(out["logits_tilde"][:, -1, :])
    tok = _greedy(rec)
    toks.append(tok)
    kv, position = out["kv"], seq_len
    for _ in range(n_new - 1):
        x = session.mask_token_embedding(torch.tensor([tok]))
        out = session.worker_decode(x, kv, position)
        kv = out["kv"]
        rec = session.recover(out["logits_tilde"][:, -1, :])
        tok = _greedy(rec)
        toks.append(tok)
        position += 1
    return toks


def _package_tokens(backend, session, h_tilde, n_new, seq_len, n_layers, cfg0):
    """Package-backed masked greedy decode (folded shards + masked KV)."""
    toks = []
    pre = backend.run_prefill(h_tilde, n_layers, cfg0, session._cos, session._sin,
                              session.eps)
    rec = session.recover(backend.run_head(pre["y_tilde"], session.eps)[:, -1, :])
    tok = _greedy(rec)
    toks.append(tok)
    position = seq_len
    for _ in range(n_new - 1):
        x = session.mask_token_embedding(torch.tensor([tok]))
        dec = backend.run_decode(x, position, cfg0, session._cos, session._sin,
                                 session.eps, num_exec_layers=n_layers)
        rec = session.recover(
            backend.run_head(dec["y_tilde"], session.eps)[:, -1, :])
        tok = _greedy(rec)
        toks.append(tok)
        position += 1
    return toks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--folded-package-path", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unwired-nonlinear", action="store_true",
                    default=False,
                    help="allow a non-paper-facing PROTOTYPE run for a "
                         "nonlinear design not yet executed in the real "
                         "path (tag-only)")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_folded_full_decode_probe.json")
    ap.add_argument("--output-md",
                    default="outputs/qwen7b_folded_full_decode_probe.md")
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
    n_new = int(args.max_new_tokens)

    pkg_dir = Path(args.folded_package_path)
    manifest = load_manifest(pkg_dir)
    n = int(manifest.num_layers)
    seed = seed_from_manifest(pkg_dir, args.seed)
    if manifest.model_path_or_id and args.model_path and \
            manifest.model_path_or_id != args.model_path:
        print("WARNING: --model-path != package model_path_or_id (%s)."
              % manifest.model_path_or_id)

    seq_len = int(ids.shape[1])
    cfg = MemoryOptimizedConfig(
        num_layers=n, batch_size=1, seq_len=seq_len, max_new_tokens=n_new,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)
    h_tilde = session.mask_embeddings(ids)
    cfg0 = _cfg_to(session.layer_configs[0], session.compute_device)

    ref_tokens = _trusted_reference_tokens(session, h_tilde, n_new, seq_len)

    vrep = verify_package(pkg_dir)
    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_dir), device=device, dtype=dtype,
        nonlinear_backend=args.nonlinear_backend)
    init_resp = backend.init(BoundaryInitRequest(
        session_id="decode-probe", hidden_size=int(getattr(mc, "hidden_size")),
        vocab_size=int(getattr(mc, "vocab_size")), num_layers=n, dtype=dtype,
        gpu_backend="qwen7b_folded_package"))
    desc = backend.describe()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pkg_tokens = _package_tokens(backend, session, h_tilde, n_new, seq_len, n,
                                 cfg0)
    latency_s = time.perf_counter() - t0
    peak_mb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    matches = sum(1 for a, b in zip(pkg_tokens, ref_tokens) if a == b)
    token_match_rate = matches / max(1, len(ref_tokens))
    tokens_exact_match = bool(pkg_tokens == ref_tokens)

    report = {
        "stage": "qwen7b_folded_package_decode_probe",
        "dry_run": dry_run, "model_name": args.model_name,
        "num_exec_layers": n, "num_package_layers": n,
        "max_new_tokens": n_new, "seq_len": seq_len, "dtype": dtype, "seed": seed,
        "folded_package_path": str(pkg_dir),
        "folded_package_loaded": bool(desc["folded_package_loaded"]),
        "folded_package_valid": bool(vrep["package_valid"]),
        "package_size_gb": round(package_size_gb(pkg_dir), 6),
        "num_shards": vrep["num_shards"], "manifest_hash": desc["manifest_hash"],
        "worker_has_mask_secrets": bool(desc["worker_has_mask_secrets"]),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "package_backed_prefill": True, "package_backed_decode": True,
        "reference_token_ids": ref_tokens, "package_token_ids": pkg_tokens,
        "tokens_exact_match": tokens_exact_match,
        "token_match_rate": token_match_rate,
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

    print("=== package-backed DECODE probe (n=%d, new=%d, dry_run=%s) ==="
          % (n, n_new, dry_run))
    print("folded_package_loaded=%s folded_package_valid=%s package_backed_decode"
          "=True" % (report["folded_package_loaded"],
                     report["folded_package_valid"]))
    print("worker_has_mask_secrets=%s tee_used_on_gpu=%s"
          % (report["worker_has_mask_secrets"], report["tee_used_on_gpu"]))
    print("reference_token_ids=%s" % ref_tokens)
    print("package_token_ids  =%s" % pkg_tokens)
    print("tokens_exact_match=%s token_match_rate=%.4f latency_s=%.3f "
          "peak_gpu_memory_mb=%s" % (tokens_exact_match, token_match_rate,
                                     latency_s, peak_mb))
    ok = (report["folded_package_loaded"] and report["folded_package_valid"]
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"] and tokens_exact_match
          and not report["leaked_secret_fields"])
    print("\nDECODE PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Package-backed decode probe (n=%d, new=%d, dry_run=%s)"
         % (r["num_exec_layers"], r["max_new_tokens"], r["dry_run"]), "",
         "- model_name=`%s`  seq_len=%s  dtype=%s  seed=%s"
         % (r["model_name"], r["seq_len"], r["dtype"], r["seed"]),
         "- folded_package_path=`%s`  num_shards=%s  package_size_gb=%s"
         % (r["folded_package_path"], r["num_shards"], r["package_size_gb"]),
         "- manifest_hash=`%s`" % r["manifest_hash"],
         "- **folded_package_loaded=%s** **folded_package_valid=%s** "
         "**package_backed_decode=True**"
         % (r["folded_package_loaded"], r["folded_package_valid"]),
         "- **worker_has_mask_secrets=%s** **tee_used_on_gpu=%s**"
         % (r["worker_has_mask_secrets"], r["tee_used_on_gpu"]),
         "", "## Generated tokens (package vs in-process reference)", "",
         "- reference_token_ids=%s" % r["reference_token_ids"],
         "- package_token_ids=%s" % r["package_token_ids"],
         "- **tokens_exact_match=%s**  token_match_rate=%.4f"
         % (r["tokens_exact_match"], r["token_match_rate"]),
         "- latency_s=%s  peak_gpu_memory_mb=%s"
         % (r["latency_s"], r["peak_gpu_memory_mb"]),
         "- gpu_visible_plaintext_fields=%s  leaked_secret_fields=%s"
         % (r["gpu_visible_plaintext_fields"] or "[]",
            r["leaked_secret_fields"] or "[]")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
