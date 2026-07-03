#!/usr/bin/env python3
"""Generation-task evaluation for `current` vs `trusted_shortcut`
(Amulet-style ``amulet_migrated`` nonlinear migration) on the folded-remote
Qwen2.5-7B worker.

Connects to an ALREADY-RUNNING GPU worker (the nonlinear backend is fixed at
worker launch, ``run_gpu_worker_server.py --nonlinear-backend ...``), drives
greedy generation over a prompt set, and emits a full audit:

    config.json  manifest_audit.json  dimension_audit.json
    nonlinear_execution_evidence.json  generations.jsonl
    comparison_current_vs_trusted_shortcut.json  metrics.csv
    per_layer_dimensions.csv  report.md

It VERIFIES (no silent fallback) that the running worker actually serves the
requested backend, and for ``trusted_shortcut`` that genuine ``amulet_migrated``
lift evidence is present -- otherwise it fails loudly.

Example (current smoke)::

    python scripts/run_generation_backend_eval.py \
      --backend current \
      --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
      --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024_pad \
      --embedding-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current_cuda \
      --gpu-worker-url http://127.0.0.1:18082 \
      --prompt-file data/eval/generation_smoke_prompts.jsonl \
      --max-new-tokens 64 --temperature 0 --do-sample false \
      --expected-nonlinear-backend current \
      --output-dir outputs/generation_backend_eval/current_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# repo import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pllo.benchmarks import generation_backend_eval as gbe   # noqa: E402
from pllo.benchmarks import ifeval_scoring                    # noqa: E402


# --------------------------------------------------------------------------- #
# prompt loading
# --------------------------------------------------------------------------- #
def load_prompts(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = str(d.get("id", d.get("key", len(rows))))
            rows.append({
                "prompt_id": pid,
                "prompt": d.get("prompt"),
                "meta": d.get("meta"),
                "category": d.get("category") or d.get("dataset"),
            })
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


# --------------------------------------------------------------------------- #
# default (real) providers -- lazily imported so the pipeline stays testable
# with stubs and without torch/transformers.
# --------------------------------------------------------------------------- #
def _default_config_loader(model_path: str) -> Dict[str, Any]:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    return {
        "hidden_size": H,
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": n_heads,
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads",
                                           n_heads)),
        "head_dim": int(getattr(cfg, "head_dim", H // n_heads)),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(getattr(cfg, "max_position_embeddings",
                                               0)),
        "_name_or_path": getattr(cfg, "_name_or_path", model_path),
    }


def _default_manifest_loader(package_dir: Optional[str]) -> Dict[str, Any]:
    if not package_dir:
        return {}
    from dataclasses import asdict, is_dataclass
    from pllo.deployment.folded_package_manifest import load_manifest
    try:
        man = load_manifest(package_dir)
    except Exception as e:                                    # noqa: BLE001
        return {"manifest_load_error": str(e)}
    d = asdict(man) if is_dataclass(man) else dict(getattr(man, "__dict__", {}))
    d["num_shards"] = getattr(man, "num_shards", None)
    try:
        from pllo.deployment.folded_package import package_size_gb
        d["package_size_gb"] = round(package_size_gb(package_dir), 3)
    except Exception:                                        # noqa: BLE001
        pass
    return d


def _default_verify(package_dir: Optional[str]) -> Dict[str, Any]:
    if not package_dir:
        return {}
    try:
        from pllo.deployment.folded_package import verify_package
        return verify_package(package_dir)
    except Exception as e:                                    # noqa: BLE001
        return {"verify_error": str(e)}


def _default_health(url: str) -> Dict[str, Any]:
    from pllo.protocol.remote import RemoteGpuWorker
    w = RemoteGpuWorker(url, gpu_backend="qwen7b_folded_package")
    try:
        return w.health() or {}
    finally:
        try:
            w.close()
        except Exception:                                    # noqa: BLE001
            pass


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def _default_predictor_factory(args, resolved) -> Any:
    from pllo.benchmarks.real_predictors import build_predictor
    # Real TDX: `tdx_attested_remote` enforces a genuine TD Quote (requires
    # --attestation-evidence-json). `folded_remote` runs the same folded compute
    # without the attestation gate.
    worker_backend = getattr(args, "worker_backend", "folded_remote")
    attest = _load_json(getattr(args, "attestation_evidence_json", None))
    return build_predictor(
        worker_backend,
        model_path=args.model_path,
        model_name="qwen",
        gpu_worker_url=args.gpu_worker_url,
        embedding_path=args.embedding_path,
        attestation_evidence=attest,
        expected_mr_td=getattr(args, "expected_mr_td", None),
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        audit=args.audit_nonlinear,
        nonlinear_backend=resolved["nonlinear_backend"],
        stop_on_eos=True,
        use_chat_template=args.use_chat_template,
        worker_persistent_conn=bool(getattr(args, "persistent_conn", True)),
        precompute_masked_embed=bool(getattr(args, "precompute_embed", True)),
    )


# --------------------------------------------------------------------------- #
# orchestration (factory-injectable for tests)
# --------------------------------------------------------------------------- #
def run_eval(
    args,
    *,
    predictor_factory: Optional[Callable] = None,
    config_loader: Optional[Callable] = None,
    manifest_loader: Optional[Callable] = None,
    verify_fn: Optional[Callable] = None,
    health_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    predictor_factory = predictor_factory or _default_predictor_factory
    config_loader = config_loader or _default_config_loader
    manifest_loader = manifest_loader or _default_manifest_loader
    verify_fn = verify_fn or _default_verify
    health_fn = health_fn or _default_health

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    exit_code = 0

    resolved = gbe.resolve_backend(args.backend)

    # config.json
    config_blob = {
        "args": {k: v for k, v in vars(args).items()},
        "resolved_backend": resolved,
        "timestamp_note": "wall-clock stamped by the caller/runbook",
    }
    gbe.write_json(out_dir / "config.json", config_blob)

    # manifest + verify
    manifest = manifest_loader(args.folded_package_path)
    verify = verify_fn(args.folded_package_path)
    prompts = load_prompts(args.prompt_file, args.limit)

    # health (pre-run): backend verification + pad coverage
    health = {}
    try:
        health = health_fn(args.gpu_worker_url)
    except Exception as e:                                    # noqa: BLE001
        errors.append(f"health poll failed: {e}")
    backend_verify = gbe.verify_backend_execution(
        health, args.expected_nonlinear_backend or args.backend,
        args.expected_op_backend)
    if not backend_verify["backend_verification_passed"]:
        exit_code = max(exit_code, 2)
        errors.append("backend verification failed (possible silent fallback)")

    # model config + package seq_len for dimension audit + seq-len gate
    config = config_loader(args.model_path) if args.model_path else {}
    exec_meta = {"seq_len": manifest.get("seq_len") or args.seq_len,
                 "mask_family": manifest.get("compatible_mask_family"),
                 "model_name": manifest.get("model_name")}
    pkg_seq_len = manifest.get("seq_len") or args.seq_len

    # seq-len compatibility gate (uses the longest prompt token estimate)
    longest = max((len((p.get("prompt") or "").split()) for p in prompts),
                  default=0)
    sl_ok, sl_note = gbe.seq_len_compatible(
        pkg_seq_len, longest, args.max_new_tokens)
    if not sl_ok:
        exit_code = max(exit_code, 5)
        errors.append(f"seq_len incompatible: {sl_note}")

    manifest_audit = {
        "manifest": manifest,
        "verify": verify,
        "pad_status": {k: health.get(k) for k in (
            "linear_pad_coverage", "base_linear_pad_all_modules_covered")},
        "backend_verification": backend_verify,
    }
    gbe.write_json(out_dir / "manifest_audit.json", manifest_audit)

    # --------------------------------------------------------------------- #
    # run generation
    # --------------------------------------------------------------------- #
    records: List[Dict[str, Any]] = []
    measured: Dict[str, Any] = {}
    predictor = None
    stats: Dict[str, Any] = {}
    rep_prompt_len = longest or 1

    if sl_ok and args.model_path:
        try:
            predictor = predictor_factory(args, resolved)
        except Exception as e:                                # noqa: BLE001
            errors.append(f"predictor build failed: {e}")
            exit_code = max(exit_code, 4)

    if predictor is not None:
        if getattr(args, "profile", True) and hasattr(
                predictor, "enable_decode_profiling"):
            try:
                predictor.enable_decode_profiling(True, request_worker_timing=(
                    not args.no_worker_timing))
            except Exception:                                # noqa: BLE001
                pass
        for i, p in enumerate(prompts):
            if i == 0 and hasattr(predictor, "enable_logits_capture"):
                try:
                    predictor.enable_logits_capture(2)
                except Exception:                            # noqa: BLE001
                    pass
            rec = {"prompt_id": p["prompt_id"], "prompt": p["prompt"],
                   "backend": resolved["nonlinear_backend"],
                   "category": p.get("category")}
            t0 = time.perf_counter()
            try:
                g = predictor.generate(p["prompt"])
                dt = time.perf_counter() - t0
                ids = g.get("token_ids") or []
                rec.update({
                    "generated_text": g.get("text"),
                    "generated_token_ids": ids,
                    "output_length": len(ids),
                    "finish_reason": g.get("finish_reason"),
                    "latency_sec": round(dt, 4),
                    "tokens_per_sec": round(len(ids) / dt, 3) if dt > 0 else None,
                    "error": None,
                    "meta": p.get("meta"),
                })
                if i == 0:
                    tokc = g.get("prompt_token_count") or g.get(
                        "formatted_prompt_token_count")
                    if tokc:
                        rep_prompt_len = int(tokc)
                        measured["input_ids"] = [1, int(tokc)]
                        measured["embedding_output"] = [
                            1, int(tokc), config.get("hidden_size")]
                    try:
                        cap = (predictor.captured_logits()
                               if hasattr(predictor, "captured_logits") else [])
                        if cap:
                            v = len(cap[0]) if hasattr(cap[0], "__len__") else \
                                config.get("vocab_size")
                            measured["recovered_logits"] = [1, int(v)]
                            measured["logits"] = [1, 1, int(v)]
                    except Exception:                        # noqa: BLE001
                        pass
            except Exception as e:                            # noqa: BLE001
                rec.update({"generated_text": None, "generated_token_ids": [],
                            "output_length": 0, "finish_reason": None,
                            "latency_sec": round(time.perf_counter() - t0, 4),
                            "error": str(e)})
                errors.append(f"{p['prompt_id']}: {e}")
            records.append(rec)
        try:
            stats = predictor.stats() if hasattr(predictor, "stats") else {}
        except Exception as e:                                # noqa: BLE001
            errors.append(f"stats() failed: {e}")
        try:
            if hasattr(predictor, "close"):
                predictor.close()
        except Exception:                                    # noqa: BLE001
            pass
    else:
        errors.append("no predictor -- generation skipped "
                      "(missing model-path or seq_len gate)")

    # real-TDX attestation verdict (populated by tdx_attested_remote via stats())
    attestation = {k: stats.get(k) for k in (
        "boundary_attested", "boundary_tee_type", "mr_td", "runtime_hash",
        "expected_runtime_hash", "runtime_hash_bound",
        "binding_mismatch_reason", "attestation_nonlinear_backend",
        "evidence_report_data") if k in stats}
    attestation["worker_backend"] = getattr(args, "worker_backend",
                                            "folded_remote")
    attestation["attestation_requested"] = bool(
        getattr(args, "attestation_evidence_json", None))
    if attestation.get("attestation_requested") or attestation.get(
            "boundary_attested") is not None:
        gbe.write_json(out_dir / "attestation.json", attestation)
    if getattr(args, "require_tdx", False):
        if attestation.get("boundary_attested") is not True:
            exit_code = max(exit_code, 6)
            errors.append(
                "real TDX required (--require-tdx) but boundary_attested="
                f"{attestation.get('boundary_attested')} "
                f"(reason: {attestation.get('binding_mismatch_reason')})")

    # health (post-run): evidence now populated
    post_health = {}
    try:
        post_health = health_fn(args.gpu_worker_url)
    except Exception:                                        # noqa: BLE001
        post_health = health
    evidence = (post_health.get("nonlinear_execution_evidence")
                or {k: stats.get(k) for k in stats
                    if k.startswith("lifted") or k.startswith("amulet")
                    or k.startswith("nonlinear") or k.startswith("trusted")
                    or k == "lift_k" or k == "migrated_ops_by_type"})
    nonlinear_summary = gbe.summarize_nonlinear_evidence(
        evidence, args.expected_nonlinear_backend or args.backend,
        args.expected_op_backend)
    gbe.write_json(out_dir / "nonlinear_execution_evidence.json",
                   nonlinear_summary)
    if (args.audit_nonlinear and
            nonlinear_summary.get("nonlinear_execution_evidence_missing")
            and resolved["nonlinear_backend"] == "trusted_shortcut"):
        if args.allow_missing_evidence:
            errors.append("WARNING: trusted_shortcut nonlinear evidence "
                          "missing (allowed by --allow-missing-evidence)")
        else:
            exit_code = max(exit_code, 3)
            errors.append("trusted_shortcut nonlinear execution evidence "
                          "missing")

    # dimension audit
    dimension_audit = {}
    if config:
        dimension_audit = gbe.build_dimension_audit(
            config, exec_meta=exec_meta, manifest=manifest,
            prompt_len=rep_prompt_len, max_new_tokens=args.max_new_tokens,
            batch_size=1, dtype=args.dtype, device=args.device,
            nonlinear_backend=resolved["nonlinear_backend"],
            lift_k=evidence.get("lift_k") if evidence else None,
            measured=measured, representative_decode_steps=(1, 8))
        gbe.write_json(out_dir / "dimension_audit.json", dimension_audit)
        gbe.write_per_layer_csv(out_dir / "per_layer_dimensions.csv",
                                gbe.dimension_audit_to_rows(dimension_audit))

    # generations + metrics
    gbe.write_generations_jsonl(out_dir / "generations.jsonl", records)
    metric_rows = [{"prompt_id": r["prompt_id"],
                    "backend": r["backend"],
                    "output_length": r.get("output_length"),
                    "finish_reason": r.get("finish_reason"),
                    "latency_sec": r.get("latency_sec"),
                    "tokens_per_sec": r.get("tokens_per_sec"),
                    "error": r.get("error")} for r in records]
    gbe.write_metrics_csv(out_dir / "metrics.csv", metric_rows)

    # quality + performance
    quality = gbe.generation_quality(records)
    total_tokens = sum(r.get("output_length") or 0 for r in records)
    total_latency = sum(r.get("latency_sec") or 0 for r in records)
    lats = [r["latency_sec"] for r in records if r.get("latency_sec")]
    perf = {
        "num_prompts": len(records),
        "total_tokens": total_tokens,
        "total_latency_sec": round(total_latency, 4),
        "tokens_per_sec": round(total_tokens / total_latency, 3)
        if total_latency else None,
        "mean_latency_sec": round(sum(lats) / len(lats), 4) if lats else None,
        "mean_first_token_latency_sec": (stats.get("decode_profile", {})
                                         or {}).get("first_token_latency_s"),
        "peak_gpu_memory_mb": post_health.get("peak_gpu_memory_mb"),
        "resident_cache_active": stats.get("resident_cache_active"),
        "boundary_calls": stats.get("boundary_calls"),
        "gpu_calls": stats.get("gpu_calls"),
        "trusted_bytes": stats.get("trusted_bytes"),
        "gpu_bytes": stats.get("gpu_bytes"),
        "decode_bottleneck_stage": stats.get("decode_bottleneck_stage"),
        "finish_reason": stats.get("finish_reason"),
    }

    # IFEval scoring (only if prompts carry instruction meta)
    ifeval_agg = None
    if any(p.get("meta", {}) and p["meta"].get("instruction_id_list")
           for p in prompts):
        scored = []
        by_id = {r["prompt_id"]: r for r in records}
        for p in prompts:
            meta = p.get("meta") or {}
            if not meta.get("instruction_id_list"):
                continue
            resp = (by_id.get(p["prompt_id"], {}) or {}).get(
                "generated_text") or ""
            scored.append(ifeval_scoring.score_response(meta, resp))
        if scored:
            ifeval_agg = ifeval_scoring.aggregate_ifeval(scored)
            gbe.write_json(out_dir / "ifeval_scores.json",
                           {"aggregate": ifeval_agg})

    # comparison against a baseline run dir (current vs trusted_shortcut)
    comparison = None
    if args.compare_baseline_dir:
        base = Path(args.compare_baseline_dir) / "generations.jsonl"
        if base.exists():
            base_recs = [json.loads(x) for x in open(base) if x.strip()]
            comparison = gbe.compare_backends(
                base_recs, records,
                label_a="baseline", label_b=resolved["nonlinear_backend"])
            gbe.write_json(
                out_dir / "comparison_current_vs_trusted_shortcut.json",
                comparison)
        else:
            errors.append(f"compare baseline not found: {base}")

    # report.md
    gbe.write_report_md(
        out_dir / "report.md",
        config=config, backend=resolved["nonlinear_backend"],
        backend_verify=backend_verify, nonlinear_summary=nonlinear_summary,
        quality=quality, perf=perf, dimension_audit=dimension_audit,
        generations=records, comparison=comparison, ifeval=ifeval_agg,
        seq_len_note=sl_note, attestation=attestation, errors=errors)

    summary = {
        "backend": resolved["nonlinear_backend"],
        "op_backend": resolved["op_backend"],
        "exit_code": exit_code,
        "worker_backend": attestation.get("worker_backend"),
        "boundary_attested": attestation.get("boundary_attested"),
        "backend_verification_passed":
            backend_verify["backend_verification_passed"],
        "nonlinear_execution_evidence_missing":
            nonlinear_summary.get("nonlinear_execution_evidence_missing"),
        "num_prompts": len(records),
        "num_errors": len([r for r in records if r.get("error")]),
        "output_dir": str(out_dir),
        "errors": errors,
    }
    gbe.write_json(out_dir / "run_summary.json", summary)
    return summary


def _bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "on")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True,
                    choices=list(gbe.SUPPORTED_BACKENDS),
                    help="current | trusted_shortcut")
    ap.add_argument("--model-path")
    ap.add_argument("--tokenizer-path")
    ap.add_argument("--folded-package-path")
    ap.add_argument("--embedding-path")
    ap.add_argument("--gpu-worker-url", default="http://127.0.0.1:18082")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--do-sample", type=_bool, default=False)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-chat-template", type=_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--audit-dimensions", type=_bool, default=True)
    ap.add_argument("--audit-nonlinear", type=_bool, default=True)
    ap.add_argument("--expected-nonlinear-backend")
    ap.add_argument("--expected-op-backend")
    ap.add_argument("--compare-baseline-dir")
    # --- real TDX attestation ---
    ap.add_argument("--worker-backend", default="folded_remote",
                    choices=["folded_remote", "tdx_attested_remote"],
                    help="tdx_attested_remote enforces a real TD Quote")
    ap.add_argument("--attestation-evidence-json",
                    help="real TDX quote evidence JSON (from the TDX VM via "
                    "generate_alibaba_tdx_quote_evidence.py)")
    ap.add_argument("--expected-mr-td")
    ap.add_argument("--require-tdx", action="store_true",
                    help="fail (exit 6) unless boundary_attested is True")
    ap.add_argument("--allow-missing-evidence", action="store_true")
    ap.add_argument("--no-worker-timing", action="store_true")
    ap.add_argument("--profile", type=_bool, default=True)
    # precision-neutral, bit-identical latency features (client-side only; do NOT
    # touch the attested boundary code, so the runtime hash / TDX quotes are
    # unaffected). keep-alive reuses ONE TCP connection across decode steps
    # (saves a full RTT per token over a real TEE<->GPU tunnel); precompute-embed
    # folds E @ N_0 once so the per-token input embed is a lookup, not a matmul.
    ap.add_argument("--persistent-conn", type=_bool, default=True,
                    help="reuse one keep-alive TCP connection across tokens")
    ap.add_argument("--precompute-embed", type=_bool, default=True,
                    help="precompute the masked embedding table (E @ N_0)")
    ap.add_argument("--output-dir", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.do_sample or args.temperature not in (0.0, 0):
        print("[warn] this evaluation is greedy-only; ignoring "
              "do_sample/temperature for decoding", file=sys.stderr)
    summary = run_eval(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
