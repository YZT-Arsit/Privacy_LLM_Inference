"""IFEval-style open-ended generation runner with optional precomputed schedule.

Reads a ``{id, prompt}`` JSONL, runs deterministic greedy generation with one
backend, and (optionally) PRECOMPUTES a per-step obfuscation schedule in a
warm-up phase that the decode then CONSUMES one fresh slot per generated token.

Backends:
* ``plaintext_local`` -- real Qwen checkpoint (greedy, attention_mask passed).
* ``folded_remote``   -- trusted lite boundary + remote folded GPU worker (the
  per-step schedule is attached to the predictor and consumed during decode; no
  schedule secret ever crosses to the GPU -- the remote client audits this).

Local development: with no model/worker this falls back to a deterministic stub
(``dry_run=True, paper_ready=False``); ``--require-real`` forbids the stub. No
downloads, no hardcoded server paths (all paths are CLI args), mock-testable.

Honesty: until the runtime actually moves the online remask/pad/inverse off the
critical path, the report carries ``schedule_used_for_metadata_only=true`` and
``online_remask_still_performed=true`` -- the schedule today reserves + consumes
the per-step trusted material and audits non-leakage; it does not yet fabricate a
speed-up. See ``schedule_precompute_latency_s`` vs ``online_generation_latency_s``.

# ---- server example only (run on the GPU server; NOT executed locally) ----
# cd /root/privacy_llm_obfuscation
# export PYTHONPATH=/root/privacy_llm_obfuscation/src:$PYTHONPATH
# python scripts/run_ifeval_generation.py \
#   --input-jsonl /root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ifeval_prompts.jsonl \
#   --backend folded_remote --model-name Qwen2.5-7B-Instruct \
#   --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
#   --gpu-worker-url http://127.0.0.1:18082 \
#   --embedding-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current \
#   --seq-len 1024 --max-new-tokens 256 --dtype bfloat16 --device cuda --audit \
#   --nonlinear-backend current --use-chat-template \
#   --precompute-obfuscation-schedule --schedule-max-steps 1024 --schedule-seed 2035 \
#   --report-schedule-stats --max-examples 1 \
#   --output-response-jsonl outputs/ifeval/ifeval_..._responses.jsonl \
#   --output-report-json    outputs/ifeval/ifeval_..._generation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_metrics import stub_generate  # noqa: E402
from pllo.benchmarks.real_predictors import (  # noqa: E402
    RealBackendUnavailable,
    build_predictor,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    normalize_nonlinear_backend,
)
from pllo.runtime.obfuscation_schedule import (  # noqa: E402
    PrecomputedMaskSchedule,
    audit_schedule_trusted_only,
    default_schedule_report_fields,
    schedule_report_fields,
)

_BACKENDS = ("plaintext_local", "folded_remote")


def _load_prompts(path, max_examples):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, ln in enumerate(fh):
            ln = ln.strip()
            if not ln:
                continue
            ex = json.loads(ln)
            ex.setdefault("id", "ex-%d" % i)
            if not str(ex.get("prompt", "")).strip():
                raise ValueError("example %r missing 'prompt'" % ex.get("id"))
            out.append(ex)
    if max_examples and max_examples > 0:
        out = out[:max_examples]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--backend", required=True, choices=list(_BACKENDS))
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit", action="store_true", default=False)
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--use-chat-template", action="store_true", default=False)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--require-real", action="store_true", default=False)
    ap.add_argument("--mock-runtime", action="store_true", default=False,
                    help="force the deterministic stub (local testing)")
    # ---- precomputed obfuscation schedule (default OFF) ----
    ap.add_argument("--precompute-obfuscation-schedule", action="store_true",
                    default=False)
    ap.add_argument("--schedule-max-steps", type=int, default=None)
    ap.add_argument("--schedule-seed", type=int, default=2035)
    ap.add_argument("--schedule-cache-dir", default=None)
    ap.add_argument("--schedule-save-secret-tensors", action="store_true",
                    default=False, help="NOT recommended; refused unless also "
                    "--allow-secret-persist")
    ap.add_argument("--allow-secret-persist", action="store_true", default=False)
    ap.add_argument("--report-schedule-stats", action="store_true", default=False)
    ap.add_argument("--trace-decode-steps", action="store_true", default=False,
                    help="record a per-token, per-stage decode trace")
    ap.add_argument("--trace-output-jsonl", default=None,
                    help="write the per-token decode trace to this JSONL")
    ap.add_argument("--output-response-jsonl", required=True)
    ap.add_argument("--output-report-json", required=True)
    args = ap.parse_args()

    nb = normalize_nonlinear_backend(args.nonlinear_backend or "current")
    examples = _load_prompts(args.input_jsonl, args.max_examples)
    if not examples:
        print("ERROR: no prompts in %s" % args.input_jsonl, file=sys.stderr)
        return 3

    # build the real predictor or fall back to the stub (mock/local)
    if args.backend == "plaintext_local":
        have_real = bool(args.model_path) and not args.mock_runtime
    else:
        have_real = bool(args.model_path and args.gpu_worker_url
                         and args.embedding_path) and not args.mock_runtime
    predictor = None
    if have_real or args.require_real:
        try:
            predictor = build_predictor(
                args.backend, model_path=args.model_path,
                model_name=args.model_name, gpu_worker_url=args.gpu_worker_url,
                embedding_path=args.embedding_path, seq_len=args.seq_len,
                max_new_tokens=args.max_new_tokens, dtype=args.dtype,
                device=args.device, audit=args.audit, nonlinear_backend=nb)
        except RealBackendUnavailable as exc:
            if args.require_real:
                print("ERROR: --require-real but real backend unavailable: %s"
                      % exc, file=sys.stderr)
                return 3
            predictor = None
    is_dry = predictor is None

    enabled = bool(args.precompute_obfuscation_schedule)
    per_example_steps = int(args.schedule_max_steps or args.max_new_tokens)
    sched_precompute_latency = 0.0
    sched_slots_precomputed = 0
    sched_slots_consumed = 0
    last_schedule = None

    # per-token, per-stage decode profiling (target metrics + optional trace)
    from pllo.benchmarks.decode_profiler import (
        DecodeProfiler, simulate_mock_decode)
    profiler = None
    mock_counters = None
    if predictor is not None and hasattr(predictor, "enable_decode_profiling"):
        predictor.enable_decode_profiling(True)
    else:
        mock_counters = {"boundary_calls": 0, "gpu_calls": 0,
                         "trusted_bytes": 0, "gpu_bytes": 0}
        profiler = DecodeProfiler(counters=lambda: dict(mock_counters),
                                  enabled=True)

    responses = []
    online_t0 = time.perf_counter()
    try:
        for idx, ex in enumerate(examples):
            schedule = None
            if enabled:
                tprep = time.perf_counter()
                schedule = PrecomputedMaskSchedule.precompute(
                    max_steps=per_example_steps, hidden_size=1,  # hidden unknown locally; metadata-only sizing
                    seed=int(args.schedule_seed) + idx, seq_len=args.seq_len,
                    max_new_tokens=args.max_new_tokens, dtype="float32",
                    device=args.device, mask_family="pairwise_complex_scaling",
                    nonlinear_backend=nb, with_secret_tensors=(not is_dry),
                    strict_audit=True)
                sched_precompute_latency += (time.perf_counter() - tprep)
                audit_schedule_trusted_only(schedule)
                last_schedule = schedule
                sched_slots_precomputed += len(schedule.slots)
                if predictor is not None and hasattr(
                        predictor, "attach_obfuscation_schedule"):
                    predictor.attach_obfuscation_schedule(schedule)

            prompt = str(ex["prompt"])
            if predictor is not None:
                g = predictor.generate(prompt)
                text = g.get("text", "")
                toks = g.get("token_ids") or []
            else:
                g = stub_generate(ex, args.max_new_tokens)
                text, toks = g["text"], g["token_ids"]
                # mock path: synthetic profiled decode that consumes one fresh
                # slot per generated token (no model/GPU); honest mock timings.
                def _on_step(kind, step, phase, _sched=schedule):
                    if kind == "schedule" and _sched is not None:
                        try:
                            _sched.consume(step)
                        except Exception:            # noqa: BLE001
                            pass
                simulate_mock_decode(profiler, mock_counters,
                                     n_tokens=max(1, len(toks)), hidden_size=64,
                                     on_step=_on_step)
            if schedule is not None:
                sched_slots_consumed += schedule.slots_consumed()
            responses.append({"id": ex.get("id"), "prompt": prompt,
                              "response": text, "num_tokens": len(toks)})
    finally:
        if predictor is not None and hasattr(predictor, "close"):
            try:
                predictor.close()
            except Exception:                                # noqa: BLE001
                pass
    online_latency = time.perf_counter() - online_t0

    # aggregate per-token decode profile (predictor's profiler on the real path)
    prof = (predictor.decode_profiler() if (predictor is not None and hasattr(
        predictor, "decode_profiler")) else profiler)
    total_tokens0 = sum(r["num_tokens"] for r in responses) or None
    decode_metrics = {}
    if prof is not None:
        decode_metrics = prof.aggregate(
            generated_tokens=total_tokens0,
            schedule_precompute_latency_s=round(sched_precompute_latency, 6)
            if enabled else None)
        if args.trace_decode_steps and args.trace_output_jsonl:
            prof.write_trace(args.trace_output_jsonl)

    stats = {}
    if predictor is not None and hasattr(predictor, "stats"):
        try:
            stats = predictor.stats() or {}
        except Exception:                                    # noqa: BLE001
            stats = {}

    total_tokens = sum(r["num_tokens"] for r in responses) or None
    boundary_calls = stats.get("boundary_calls")

    if enabled and last_schedule is not None:
        sfields = schedule_report_fields(
            last_schedule, enabled=True,
            online_generation_latency_s=online_latency,
            boundary_calls=boundary_calls, generated_tokens=total_tokens,
            trusted_bytes=stats.get("trusted_bytes"),
            gpu_bytes=stats.get("gpu_bytes"),
            schedule_used_for_metadata_only=True,
            online_remask_still_performed=True)
        # override per-schedule counts with run totals (one schedule per example)
        sfields["schedule_max_steps"] = per_example_steps
        sfields["schedule_slots_precomputed"] = sched_slots_precomputed
        sfields["schedule_slots_consumed"] = sched_slots_consumed
        sfields["schedule_precompute_latency_s"] = round(
            sched_precompute_latency, 6)
        sfields["latency_s_total_including_precompute"] = round(
            online_latency + sched_precompute_latency, 6)
    else:
        sfields = default_schedule_report_fields()
        sfields["online_generation_latency_s"] = round(online_latency, 6)
        sfields["latency_s_online_only"] = round(online_latency, 6)
        sfields["latency_s_total_including_precompute"] = round(
            online_latency, 6)
        sfields["boundary_calls"] = boundary_calls
        sfields["trusted_bytes"] = stats.get("trusted_bytes")
        sfields["gpu_bytes"] = stats.get("gpu_bytes")
        if boundary_calls is not None and total_tokens:
            sfields["boundary_calls_per_generated_token"] = round(
                boundary_calls / total_tokens, 6)

    if enabled and last_schedule is not None and args.schedule_cache_dir:
        last_schedule.to_disk(
            Path(args.schedule_cache_dir) / "obfuscation_schedule.json",
            save_secret_tensors=args.schedule_save_secret_tensors,
            allow_secret_persist=args.allow_secret_persist)

    report = {
        "stage": "ifeval_generation",
        "backend": args.backend,
        "nonlinear_backend": nb,
        "model_name": args.model_name,
        "decoding": "greedy",
        "use_chat_template": bool(args.use_chat_template),
        "num_examples": len(examples),
        "seq_len": int(args.seq_len),
        "max_new_tokens": int(args.max_new_tokens),
        "audit_passed": stats.get("audit_passed"),
        "tee_used_on_gpu": stats.get("tee_used_on_gpu"),
        "worker_has_mask_secrets": stats.get("worker_has_mask_secrets"),
        "worker_has_raw_lora": stats.get("worker_has_raw_lora"),
        "gpu_visible_plaintext_fields": stats.get("gpu_visible_plaintext_fields"),
        "leaked_secret_fields": stats.get("leaked_secret_fields"),
        "gpu_calls": stats.get("gpu_calls"),
        "dry_run": is_dry,
        "paper_ready": (not is_dry),
    }
    report.update(sfields)
    # per-token decode profile -> target metrics + honest bottleneck localisation
    report.update(decode_metrics)
    report["bottleneck_stage"] = decode_metrics.get("bottleneck_stage")
    report["boundary_calls_reduced"] = bool(
        decode_metrics.get("boundary_calls_reduced"))
    report["schedule_used_for_metadata_only"] = bool(
        report.get("schedule_used_for_metadata_only", enabled))
    report["online_remask_still_performed"] = True
    report["decode_trace_jsonl"] = (args.trace_output_jsonl
                                    if args.trace_decode_steps else None)
    if args.report_schedule_stats and last_schedule is not None:
        report["schedule_stats"] = last_schedule.stats()

    rp = Path(args.output_response_jsonl)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as fh:
        for r in responses:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    jp = Path(args.output_report_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== ifeval generation (%s / %s) ===" % (args.backend, nb))
    print("num_examples=%d dry_run=%s paper_ready=%s audit_passed=%s"
          % (report["num_examples"], report["dry_run"], report["paper_ready"],
             report["audit_passed"]))
    print("precompute_obfuscation_schedule=%s slots_precomputed=%s "
          "slots_consumed=%s precompute_latency_s=%s online_latency_s=%s"
          % (report["precompute_obfuscation_schedule"],
             report["schedule_slots_precomputed"],
             report["schedule_slots_consumed"],
             report["schedule_precompute_latency_s"],
             report["online_generation_latency_s"]))
    print("schedule_used_for_metadata_only=%s online_remask_still_performed=%s "
          "schedule_secret_leaked_to_gpu=%s"
          % (report["schedule_used_for_metadata_only"],
             report["online_remask_still_performed"],
             report["schedule_secret_leaked_to_gpu"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
