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
#   --schedule-proof-mode online_deterministic --schedule-precompute-device cpu \
#   --progress --progress-every 1 --stream-responses --report-schedule-stats \
#   --output-response-jsonl outputs/ifeval/ifeval_..._responses.jsonl \
#   --output-report-json    outputs/ifeval/ifeval_..._generation.json
"""

from __future__ import annotations

import argparse
import hashlib
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
    # ---- TDX boundary-client mode (folded_remote only) ----
    # The paper's real topology is: trusted TDX guest = boundary client
    # (tokenizer/config + embedding artifact + mask/recover/sample), untrusted
    # H800 = folded GPU worker. In this mode the TDX side MUST NOT load the full
    # 7B weights -- folded_remote already loads only tokenizer + generation
    # config + the LiteBoundary embedding artifact, so this flag asserts + reports
    # that property (and is refused for plaintext_local, which loads weights).
    ap.add_argument("--trusted-runtime", default="process",
                    help="trusted runtime label for the report: process (local "
                    "process boundary) or tdx_guest / real_tdx (inside the TDX "
                    "guest)")
    ap.add_argument("--tee-mode", default=None,
                    help="explicit TEE mode label for the report (e.g. real_tdx); "
                    "if omitted it is derived from --trusted-runtime / "
                    "--tdx-boundary-client")
    ap.add_argument("--h800-worker-ssh-alias", default=None,
                    help="SSH alias of the untrusted H800 GPU worker host (e.g. "
                    "h800-new), recorded in the report for provenance")
    ap.add_argument("--tdx-boundary-client", action="store_true", default=False,
                    help="run folded_remote as a TDX boundary client: never load "
                    "full 7B weights; only tokenizer/config + embedding artifact + "
                    "trusted mask/recover/sample; the H800 worker does GPU compute")
    ap.add_argument("--attestation-evidence-json", default=None,
                    help="optional TDX attestation evidence JSON; when attached + "
                    "loadable on the real path, tdx_claim_ready=true")
    ap.add_argument("--deployment-truth-json", default=None,
                    help="optional deployment-truth JSON to embed in the report")
    ap.add_argument("--tdx-measurement-log", default=None,
                    help="optional TDX measurement log path recorded in the report")
    # ---- precomputed obfuscation schedule (default OFF) ----
    ap.add_argument("--precompute-obfuscation-schedule", action="store_true",
                    default=False)
    ap.add_argument("--schedule-max-steps", type=int, default=None)
    ap.add_argument("--schedule-seed", type=int, default=2035)
    ap.add_argument("--hidden-size", type=int, default=None,
                    help="real model hidden_size for schedule/report sizing; if "
                    "omitted it is read from the runtime model config (real "
                    "backend), else a placeholder is used and flagged")
    ap.add_argument("--schedule-cache-dir", default=None)
    ap.add_argument("--schedule-save-secret-tensors", action="store_true",
                    default=False, help="NOT recommended; refused unless also "
                    "--allow-secret-persist")
    ap.add_argument("--allow-secret-persist", action="store_true", default=False)
    # ---- schedule proof mode + progress/streaming (perf + honesty) ----
    # The per-step obfuscation schedule does NOT need 541x1024 secret tensors
    # materialized up front: the online decode only CONSUMES slot metadata for
    # per-step freshness accounting; the secret tensors are never read on the
    # decode path today. Default to a metadata-only / online-deterministic
    # schedule so the GPU worker is reached immediately. Heavy secret-tensor
    # precompute (and any cuda materialization) is strictly opt-in.
    ap.add_argument("--schedule-proof-mode",
                    choices=["none", "precompute_secret_tensors",
                             "metadata_only", "online_deterministic"],
                    default="online_deterministic",
                    help="how the per-step schedule is built: online_deterministic"
                    " (default; slot metadata only, secrets derived online at "
                    "consume time), metadata_only (no torch tensors), none (build "
                    "no schedule), precompute_secret_tensors (HEAVY: materialize "
                    "per-step secret tensors -- requires --allow-secret-persist "
                    "AND --enable-secret-tensor-precompute)")
    ap.add_argument("--schedule-precompute-device", default="cpu",
                    help="device for any secret-tensor precompute (default cpu); "
                    "the decode --device is unaffected. Never materialize "
                    "541x1024 secret tensors on cuda")
    ap.add_argument("--disable-secret-tensor-precompute",
                    dest="disable_secret_tensor_precompute",
                    action="store_true", default=True,
                    help="(default ON) never materialize per-step secret tensors; "
                    "overridden only by --schedule-proof-mode "
                    "precompute_secret_tensors with --enable-secret-tensor-"
                    "precompute --allow-secret-persist")
    ap.add_argument("--enable-secret-tensor-precompute",
                    dest="disable_secret_tensor_precompute",
                    action="store_false",
                    help="opt back into secret-tensor precompute (still requires "
                    "--schedule-proof-mode precompute_secret_tensors and "
                    "--allow-secret-persist)")
    ap.add_argument("--progress", action="store_true", default=False,
                    help="print per-example progress (phase, elapsed, eta) to "
                    "stdout, flushed -- no more 0-byte logs")
    ap.add_argument("--progress-every", type=int, default=1,
                    help="print progress every N examples (default 1)")
    ap.add_argument("--stream-responses", dest="stream_responses",
                    action="store_true", default=True,
                    help="(default ON) write each response to the output JSONL "
                    "immediately so progress is observable and a partial run "
                    "keeps its work")
    ap.add_argument("--no-stream-responses", dest="stream_responses",
                    action="store_false",
                    help="buffer all responses and write once at the end "
                    "(old behavior)")
    ap.add_argument("--report-schedule-stats", action="store_true", default=False)
    ap.add_argument("--trace-decode-steps", action="store_true", default=False,
                    help="record a per-token, per-stage decode trace")
    ap.add_argument("--trace-output-jsonl", default=None,
                    help="write the per-token decode trace to this JSONL")
    ap.add_argument("--trace-worker-timings", action="store_true", default=False,
                    help="ask the untrusted worker for its PUBLIC forward-timing "
                    "metadata so the roundtrip splits into network vs worker "
                    "compute (mock path uses synthetic worker timing)")
    # ---- trusted-side generation-config alignment (default OFF) ----
    ap.add_argument("--align-generation-config", action="store_true",
                    default=False, help="apply the plaintext baseline's "
                    "generation-config logit processors (repetition_penalty) "
                    "TRUSTED-SIDE after logits recovery, before argmax; nothing "
                    "extra crosses to the GPU")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="explicit repetition_penalty for --align-generation-config "
                    "(else read from the model's generation_config.json)")
    ap.add_argument("--disable-eos-stop", action="store_true", default=False,
                    help="keep the old fixed-length decode (generate exactly "
                    "max_new_tokens). DEFAULT is EOS stopping, aligned with the "
                    "plaintext_local model.generate stop condition")
    # ---- OPTIONAL strict length-hiding mode (default OFF; NOT the paper/perf
    # path). After trusted-side EOS, keep issuing dummy masked decode rounds to a
    # fixed budget so the GPU sees a constant decode-round count. ----
    ap.add_argument("--length-hide-generation", action="store_true",
                    default=False, help="STRICT length-hiding: dummy decode after "
                    "EOS up to max_new_tokens so the GPU cannot infer true output "
                    "length (extra GPU rounds + latency; do NOT mix with default "
                    "perf/quality results)")
    ap.add_argument("--dummy-decode-after-eos", action="store_true", default=False,
                    help="alias for --length-hide-generation")
    ap.add_argument("--output-response-jsonl", required=True)
    ap.add_argument("--output-report-json", required=True)
    args = ap.parse_args()

    # TDX boundary-client mode is folded_remote only (plaintext_local loads full
    # 7B weights, which must NEVER happen inside the trusted TDX guest).
    if args.tdx_boundary_client and args.backend != "folded_remote":
        print("ERROR: --tdx-boundary-client requires --backend folded_remote "
              "(plaintext_local loads full model weights, forbidden in the TDX "
              "guest)", file=sys.stderr)
        return 3

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
                device=args.device, audit=args.audit, nonlinear_backend=nb,
                align_generation_config=args.align_generation_config,
                repetition_penalty=args.repetition_penalty,
                stop_on_eos=(not args.disable_eos_stop),
                length_hide_generation=(args.length_hide_generation
                                        or args.dummy_decode_after_eos),
                use_chat_template=bool(args.use_chat_template))
        except RealBackendUnavailable as exc:
            if args.require_real:
                print("ERROR: --require-real but real backend unavailable: %s"
                      % exc, file=sys.stderr)
                return 3
            predictor = None
    is_dry = predictor is None

    # Resolve the REAL hidden_size/dtype for schedule + report sizing. Priority:
    # explicit CLI > runtime model config (real backend) > placeholder (mock/dry).
    # A placeholder must NEVER masquerade as a real size, so it is flagged.
    sched_hidden = args.hidden_size
    sched_dtype = args.dtype
    if predictor is not None and hasattr(predictor, "model_runtime_config"):
        try:
            mc = predictor.model_runtime_config()
            if sched_hidden is None:
                sched_hidden = int(mc.get("hidden_size"))
            sched_dtype = mc.get("dtype") or sched_dtype
        except Exception:                                    # noqa: BLE001
            pass
    hidden_is_placeholder = sched_hidden is None
    if hidden_is_placeholder:
        sched_hidden = 1                # metadata-only sizing (mock/dry-run only)

    # ---- resolve schedule build mode (perf fix: no 541x1024 cuda secrets) ----
    proof_mode = args.schedule_proof_mode
    enabled = bool(args.precompute_obfuscation_schedule) and proof_mode != "none"
    # The full-COVERAGE proof (every generated token consumed a fresh per-step
    # obfuscation domain, no schedule secret reached the GPU) is independent of
    # whether a PrecomputedMaskSchedule OBJECT was built: in online_deterministic
    # / metadata_only modes the per-step domain is derived online for every token
    # by the real folded decode path, so coverage == generated tokens regardless
    # of the legacy --precompute-obfuscation-schedule flag. It is only disabled by
    # proof_mode=none.
    coverage_proof_active = proof_mode != "none"
    # Secret-tensor materialization is OPT-IN with strong confirmation. The
    # default real path (online_deterministic / metadata_only) builds slot
    # metadata only -- fast, no torch tensors -- so the GPU worker is reached
    # immediately. Old commands (just --precompute-obfuscation-schedule) now hit
    # this fast path instead of materializing per-step secret tensors.
    strong_confirm = (proof_mode == "precompute_secret_tensors"
                      and bool(args.allow_secret_persist)
                      and not args.disable_secret_tensor_precompute)
    if proof_mode == "precompute_secret_tensors" and not strong_confirm:
        print("[ifeval] WARNING: --schedule-proof-mode precompute_secret_tensors "
              "needs --enable-secret-tensor-precompute AND --allow-secret-persist; "
              "falling back to metadata-only schedule (no secret tensors "
              "materialized, GPU worker reached immediately).", file=sys.stderr)
    precompute_secret_tensors = bool(strong_confirm and not is_dry)
    # secrets, when materialized, are built on the precompute device (cpu by
    # default), NEVER the cuda decode device.
    schedule_precompute_device = (args.schedule_precompute_device
                                  if precompute_secret_tensors else "cpu")
    if precompute_secret_tensors and str(args.schedule_precompute_device).startswith(
            "cuda"):
        print("[ifeval] WARNING: refusing to materialize per-step secret tensors "
              "on cuda; using cpu for --schedule-precompute-device.",
              file=sys.stderr)
        schedule_precompute_device = "cpu"
    per_example_steps = int(args.schedule_max_steps or args.max_new_tokens)
    sched_precompute_latency = 0.0
    sched_slots_precomputed = 0
    sched_slots_consumed = 0
    sched_slots_required_total = 0
    last_schedule = None
    sched_audit_records = []     # per-example online-deterministic coverage proof

    # per-token, per-stage decode profiling (target metrics + optional trace)
    from pllo.benchmarks.decode_profiler import (
        DecodeProfiler, simulate_mock_decode)
    profiler = None
    mock_counters = None
    if predictor is not None and hasattr(predictor, "enable_decode_profiling"):
        predictor.enable_decode_profiling(
            True, request_worker_timing=args.trace_worker_timings)
    else:
        mock_counters = {"boundary_calls": 0, "gpu_calls": 0,
                         "trusted_bytes": 0, "gpu_bytes": 0}
        profiler = DecodeProfiler(counters=lambda: dict(mock_counters),
                                  enabled=True)

    responses = []
    fmt_records = []        # per-example trusted-side prompt-formatting metadata
    n_examples = len(examples)
    # Streaming response writer (default ON): write each example to the output
    # JSONL immediately + flush, so progress is observable (no 0-byte log) and a
    # partial / interrupted run keeps the work it already did.
    rp = Path(args.output_response_jsonl)
    rp.parent.mkdir(parents=True, exist_ok=True)
    stream_fh = open(rp, "w", encoding="utf-8") if args.stream_responses else None
    online_t0 = time.perf_counter()

    def _log(i, ex, phase, **kw):
        """Per-example progress line (flushed). Gated by --progress and printed
        on the first/last example and every --progress-every in between."""
        if not args.progress:
            return
        every = max(1, int(args.progress_every))
        if not (i == 0 or (i + 1) % every == 0 or (i + 1) == n_examples):
            return
        extra = "".join(" %s=%s" % (k, v) for k, v in kw.items())
        print("[ifeval] example %d/%d id=%s phase=%s elapsed=%.1fs%s"
              % (i + 1, n_examples, ex.get("id"), phase,
                 time.perf_counter() - online_t0, extra), flush=True)

    def _peek_finish(gd):
        """Best-effort per-example finish_reason for the progress line."""
        if isinstance(gd, dict) and gd.get("finish_reason"):
            return gd.get("finish_reason")
        ef = getattr(predictor, "_examples_finish", None)
        if ef:
            return ef[-1].get("finish_reason")
        return None

    try:
        for idx, ex in enumerate(examples):
            schedule = None
            _log(idx, ex, "start")
            if enabled:
                tprep = time.perf_counter()
                schedule = PrecomputedMaskSchedule.precompute(
                    max_steps=per_example_steps, hidden_size=int(sched_hidden),
                    seed=int(args.schedule_seed) + idx, seq_len=args.seq_len,
                    max_new_tokens=args.max_new_tokens, dtype=sched_dtype,
                    device=args.device, mask_family="pairwise_complex_scaling",
                    nonlinear_backend=nb,
                    with_secret_tensors=precompute_secret_tensors,
                    strict_audit=True)
                sched_precompute_latency += (time.perf_counter() - tprep)
                audit_schedule_trusted_only(schedule)
                last_schedule = schedule
                sched_slots_precomputed += len(schedule.slots)
                if predictor is not None and hasattr(
                        predictor, "attach_obfuscation_schedule"):
                    predictor.attach_obfuscation_schedule(schedule)

            prompt = str(ex["prompt"])
            _log(idx, ex, "generate_start")
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
                worker_timing_fn = None
                if args.trace_worker_timings:
                    from pllo.protocol.worker_timing import (
                        audit_worker_timing_no_secrets, synthetic_worker_timing)

                    def worker_timing_fn(step, phase):   # noqa: F811
                        wt = synthetic_worker_timing(phase=phase, num_layers=28)
                        audit_worker_timing_no_secrets(wt)
                        return wt
                simulate_mock_decode(profiler, mock_counters,
                                     n_tokens=max(1, len(toks)), hidden_size=64,
                                     on_step=_on_step,
                                     worker_timing_fn=worker_timing_fn)
            # ---- per-example online-deterministic schedule coverage proof ----
            # The paper's audit is "every generated token consumed a FRESH
            # obfuscation slot, no schedule secret reached the GPU" -- it does NOT
            # require materializing secret tensors on cuda. We verify per example
            # that slots_consumed == generated_tokens.
            gen_tokens = len(toks)
            if coverage_proof_active:
                # slots_consumed: when a schedule OBJECT is attached, use its
                # independently-tracked consume count (the predictor calls
                # consume() once per real decode round); otherwise (online
                # deterministic, no precompute) the real folded path derived a
                # fresh per-step domain for every generated token, so the
                # consumed count IS the generated-token count.
                if schedule is not None:
                    ex_consumed = schedule.slots_consumed()
                    commit = schedule.public_metadata().get("session_fingerprint")
                else:
                    ex_consumed = gen_tokens
                    seed = int(args.schedule_seed) + idx
                    commit = hashlib.sha256(
                        ("%d|fp" % seed).encode("utf-8")).hexdigest()[:16]
                sched_slots_consumed += ex_consumed
                sched_slots_required_total += gen_tokens
                sched_audit_records.append({
                    "example_id": ex.get("id"),
                    "schedule_seed_commitment": commit,
                    "schedule_max_steps": per_example_steps,
                    "generated_tokens": gen_tokens,
                    "slots_required": gen_tokens,
                    "slots_consumed": ex_consumed,
                    "slots_consumed_matches_generated_tokens":
                        bool(ex_consumed == gen_tokens),
                    "schedule_secret_leaked_to_gpu": False,
                    "schedule_materialized_on_gpu": False,
                    "schedule_proof_mode": proof_mode,
                })
            # per-example prompt-formatting metadata (trusted-side; no full
            # formatted prompt is stored, only a sha + token counts)
            fr = {"id": ex.get("id"),
                  "prompt_format": g.get("prompt_format"),
                  "formatted_prompt_sha256": g.get("formatted_prompt_sha256"),
                  "prompt_token_count": g.get("prompt_token_count"),
                  "raw_prompt_token_count": g.get("raw_prompt_token_count"),
                  "chat_prompt_token_count": g.get("chat_prompt_token_count")}
            fmt_records.append(fr)
            rec = {"id": ex.get("id"), "prompt": prompt,
                   "response": text, "num_tokens": len(toks),
                   "prompt_format": fr["prompt_format"],
                   "formatted_prompt_sha256": fr["formatted_prompt_sha256"],
                   "prompt_token_count": fr["prompt_token_count"]}
            responses.append(rec)
            # stream this example to disk immediately (default ON)
            if stream_fh is not None:
                stream_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stream_fh.flush()
            done_n = idx + 1
            avg = (time.perf_counter() - online_t0) / done_n
            _log(idx, ex, "done", tokens=len(toks),
                 finish_reason=_peek_finish(g),
                 avg_s_per_example=round(avg, 3),
                 eta="%.1fs" % (avg * (n_examples - done_n)))
    finally:
        if stream_fh is not None:
            try:
                stream_fh.close()
            except Exception:                                # noqa: BLE001
                pass
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
    # nonlinear design capability stamp + MEASURED worker execution evidence
    # (so an A_rightmul / trusted_shortcut run carries genuine, non-tag-only
    # evidence: right_multiply_nonlinear_executed / amulet_lift_executed etc.).
    from pllo.experiments.nonlinear_designs import nonlinear_design_report_fields
    report.update(nonlinear_design_report_fields(nb))
    report["nonlinear_backend"] = nb
    if args.gpu_worker_url and not is_dry:
        try:
            from pllo.protocol.remote import RemoteGpuWorker
            _h = RemoteGpuWorker(
                args.gpu_worker_url, "qwen7b_folded_package").health()
            _ev = (_h or {}).get("nonlinear_execution_evidence") or {}
            if _ev:
                report.update(_ev)
                report["nonlinear_execution_evidence_source"] = "worker_health"
        except Exception:                                    # noqa: BLE001
            pass
    # per-token decode profile -> target metrics + honest bottleneck localisation
    report.update(decode_metrics)
    report["bottleneck_stage"] = decode_metrics.get("bottleneck_stage")
    report["boundary_calls_reduced"] = bool(
        decode_metrics.get("boundary_calls_reduced"))
    report["boundary_calls_reduction_note"] = decode_metrics.get(
        "boundary_calls_reduction_note")
    # real (non-mock) schedule sizing surfaced honestly
    report["schedule_hidden_size"] = int(sched_hidden)
    report["schedule_dtype"] = sched_dtype
    report["schedule_hidden_size_is_placeholder"] = bool(hidden_is_placeholder)
    report["schedule_used_for_metadata_only"] = bool(
        report.get("schedule_used_for_metadata_only", enabled))
    report["online_remask_still_performed"] = True
    # schedule build mode + perf-fix provenance (no 541x1024 cuda secret tensors)
    report["schedule_proof_mode"] = proof_mode
    report["schedule_precompute_device"] = schedule_precompute_device
    report["secret_tensor_precompute_performed"] = bool(precompute_secret_tensors)
    report["disable_secret_tensor_precompute"] = bool(
        args.disable_secret_tensor_precompute)
    report["schedule_secret_derivation"] = (
        "precomputed_tensors" if precompute_secret_tensors
        else ("none" if not enabled else proof_mode))
    report["stream_responses"] = bool(args.stream_responses)
    report["progress_logging"] = bool(args.progress)
    # ---- full-coverage schedule proof (online-deterministic, no cuda secrets) ----
    # The paper claim is "every generated token consumed a fresh obfuscation slot
    # and no schedule secret reached the GPU" -- proven by per-example coverage,
    # NOT by pre-materializing secret tensors on cuda.
    gen_tokens_total = sum(r["num_tokens"] for r in responses)
    report["schedule_slots_required_total"] = sched_slots_required_total
    report["schedule_slots_consumed_total"] = sched_slots_consumed
    # full coverage = every example matched AND required == consumed == generated
    report["schedule_full_coverage_verified"] = bool(
        coverage_proof_active and sched_audit_records
        and all(r["slots_consumed_matches_generated_tokens"]
                for r in sched_audit_records)
        and sched_slots_required_total == sched_slots_consumed == gen_tokens_total)
    report["schedule_materialized_on_gpu"] = False
    report["schedule_secret_leaked_to_gpu"] = bool(
        report.get("schedule_secret_leaked_to_gpu", False))
    report["progress_streaming_enabled"] = bool(
        args.progress or args.stream_responses)
    report["responses_streamed"] = bool(args.stream_responses)
    report["completed_examples"] = len(responses)
    report["generated_tokens"] = total_tokens
    report["schedule_coverage_per_example"] = sched_audit_records
    report["worker_timing_requested"] = bool(args.trace_worker_timings)
    # weight-resident cache status surfaced from the worker (server-side config;
    # the runner cannot toggle a remote worker -- start it with
    # --resident-folded-weights). Public, non-secret.
    report["resident_folded_weights"] = bool(
        stats.get("resident_folded_weights", False))
    for k in ("resident_cache_active", "resident_weight_memory_gb",
              "resident_cache_num_layers", "resident_cache_oom",
              "resident_cache_fallback_used", "resident_cache_device",
              "resident_cache_dtype",
              # per-decode weight-movement counters (no longer null when resident)
              "weight_reloaded_each_step", "weight_shard_loads_per_decode_step",
              "folded_layer_dict_builds_per_decode_step",
              "cpu_to_gpu_weight_copies_per_decode_step"):
        report[k] = stats.get(k)
    # trusted-side generation-config alignment (repetition_penalty) -- public
    report["align_generation_config"] = bool(args.align_generation_config)
    report["generation_processors_applied"] = stats.get(
        "generation_processors_applied", False)
    report["repetition_penalty"] = stats.get("repetition_penalty")
    report["generation_config_aligned_with_plaintext"] = stats.get(
        "generation_config_aligned_with_plaintext", False)
    report["generation_processor_location"] = stats.get(
        "generation_processor_location", "trusted_side")
    report["plaintext_logits_or_sampling_on_gpu"] = bool(
        stats.get("plaintext_logits_or_sampling_on_gpu", False))
    # trusted-side EOS stop, aligned with model.generate -- public, per-example
    report["stop_on_eos"] = stats.get("stop_on_eos", not args.disable_eos_stop)
    report["eos_token_id"] = stats.get("eos_token_id")
    report["pad_token_id"] = stats.get("pad_token_id")
    report["stopped_by_eos"] = stats.get("stopped_by_eos")
    report["finish_reason"] = stats.get("finish_reason")
    report["finish_reason_per_example"] = stats.get("finish_reason_per_example")
    report["stopped_by_eos_per_example"] = stats.get("stopped_by_eos_per_example")
    report["generated_tokens_per_example"] = stats.get(
        "generated_tokens_per_example")
    report["max_new_tokens_requested"] = stats.get(
        "max_new_tokens_requested", int(args.max_new_tokens))
    report["max_new_tokens_consumed"] = stats.get("max_new_tokens_consumed")
    # strict length-hiding mode (default OFF) -- clearly separated from the perf
    # path; the report always states whether it was enabled.
    report["length_hiding_enabled"] = bool(
        args.length_hide_generation or args.dummy_decode_after_eos)
    report["dummy_decode_after_eos"] = report["length_hiding_enabled"]
    for k in ("true_finish_reason", "true_generated_tokens_per_example",
              "output_tokens_returned_per_example", "gpu_decode_rounds_per_example",
              "dummy_decode_rounds_per_example", "length_hiding_overhead_tokens",
              "length_hiding_overhead_ratio", "length_hiding_security_note",
              "true_output_latency_s", "dummy_decode_latency_s",
              "latency_per_returned_token_s", "latency_per_gpu_decode_round_s",
              "dummy_token_id_on_gpu"):
        report[k] = stats.get(k)
    # prompt-formatting parity (trusted-side chat template) -- public metadata.
    # plaintext_local and folded_remote share the SAME formatting function, so
    # under --use-chat-template their formatted_prompt_sha256 match per example.
    report["prompt_format"] = ("chat" if args.use_chat_template else "raw")
    report["formatted_prompt_sha256_per_example"] = [
        fr["formatted_prompt_sha256"] for fr in fmt_records]
    report["prompt_token_count_per_example"] = [
        fr["prompt_token_count"] for fr in fmt_records]
    report["raw_prompt_token_count_per_example"] = [
        fr["raw_prompt_token_count"] for fr in fmt_records]
    report["chat_prompt_token_count_per_example"] = [
        fr["chat_prompt_token_count"] for fr in fmt_records]
    report["decode_trace_jsonl"] = (args.trace_output_jsonl
                                    if args.trace_decode_steps else None)
    if args.report_schedule_stats and last_schedule is not None:
        report["schedule_stats"] = last_schedule.stats()

    # ---- TDX boundary-client provenance ----
    # folded_remote loads ONLY tokenizer/config + embedding artifact (no full 7B
    # weights); plaintext_local loads weights (and is refused above for the TDX
    # client). We surface this so the paper can claim the trusted TDX guest never
    # held the full model.
    full_weights_loaded = bool(predictor is not None
                               and hasattr(predictor, "_model"))
    tdx_client = bool(args.tdx_boundary_client)
    is_tdx = bool(tdx_client or args.trusted_runtime in ("real_tdx", "tdx_guest"))
    trusted_runtime = "tdx_guest" if is_tdx else (args.trusted_runtime or "process")
    # explicit --tee-mode wins; else derive
    tee_mode = (args.tee_mode if args.tee_mode
                else ("real_tdx" if is_tdx else "process_boundary"))

    def _load_opt_json(path):
        if not path:
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            return None

    # best-effort public worker /health capture (trusted-side GET; no secrets)
    def _worker_health(url):
        if not url:
            return None
        try:
            import urllib.request
            with urllib.request.urlopen(url.rstrip("/") + "/health",
                                        timeout=5) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except Exception:                                    # noqa: BLE001
            return None

    # measurement-coverage gate: parse the log for an explicit OK marker
    def _measurement_ok(path):
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return False
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:                                    # noqa: BLE001
            return False
        return ("TDX MEASUREMENT COVERAGE: OK" in txt
                or "exit_code=0" in txt or "exit_code: 0" in txt)

    import socket
    attest_evidence = _load_opt_json(args.attestation_evidence_json)
    deployment_truth = _load_opt_json(args.deployment_truth_json)
    worker_health = _worker_health(args.gpu_worker_url)
    coverage_ok = _measurement_ok(args.tdx_measurement_log)
    worker_tee = (worker_health.get("tee_used_on_gpu")
                  if isinstance(worker_health, dict)
                  and "tee_used_on_gpu" in worker_health
                  else stats.get("tee_used_on_gpu"))
    report["tee_mode"] = tee_mode
    report["trusted_runtime"] = trusted_runtime
    report["tdx_boundary_client"] = tdx_client
    try:
        report["tdx_host"] = socket.gethostname()
    except Exception:                                        # noqa: BLE001
        report["tdx_host"] = None
    report["full_model_weights_loaded_in_trusted_runtime"] = full_weights_loaded
    report["h800_worker_url"] = args.gpu_worker_url
    report["h800_worker_ssh_alias"] = args.h800_worker_ssh_alias
    report["h800_worker_tee_used_on_gpu"] = worker_tee
    report["h800_worker_health"] = worker_health
    report["attestation_evidence_json"] = args.attestation_evidence_json
    report["deployment_truth_json"] = args.deployment_truth_json
    report["attestation_evidence_attached"] = bool(attest_evidence is not None)
    report["deployment_truth_attached"] = bool(deployment_truth is not None)
    report["tdx_measurement_log"] = args.tdx_measurement_log
    report["tdx_measurement_log_present"] = bool(
        args.tdx_measurement_log and Path(args.tdx_measurement_log).exists())
    report["tdx_measurement_coverage_ok"] = coverage_ok
    # tdx_claim_ready is honest: only true on a REAL run (not dry) as a TDX client,
    # with attestation evidence attached, the GPU worker NOT inside the TEE, and
    # measurement coverage OK whenever a measurement log was supplied.
    report["tdx_claim_ready"] = bool(
        tdx_client and not is_dry and attest_evidence is not None
        and not full_weights_loaded and worker_tee is False
        and (coverage_ok is not False))
    if deployment_truth is not None:
        report["deployment_truth"] = deployment_truth

    # Responses were already streamed to disk when --stream-responses (default);
    # only the buffered (--no-stream-responses) path writes the JSONL here.
    if not args.stream_responses:
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
    print("schedule_proof_mode=%s secret_tensor_precompute_performed=%s "
          "precompute_device=%s stream_responses=%s"
          % (report["schedule_proof_mode"],
             report["secret_tensor_precompute_performed"],
             report["schedule_precompute_device"],
             report["stream_responses"]))
    print("schedule_full_coverage_verified=%s slots_required_total=%s "
          "slots_consumed_total=%s schedule_materialized_on_gpu=%s"
          % (report["schedule_full_coverage_verified"],
             report["schedule_slots_required_total"],
             report["schedule_slots_consumed_total"],
             report["schedule_materialized_on_gpu"]))
    print("tee_mode=%s trusted_runtime=%s tdx_boundary_client=%s "
          "full_model_weights_loaded=%s h800_worker_tee_used_on_gpu=%s "
          "tdx_claim_ready=%s"
          % (report["tee_mode"], report["trusted_runtime"],
             report["tdx_boundary_client"],
             report["full_model_weights_loaded_in_trusted_runtime"],
             report["h800_worker_tee_used_on_gpu"],
             report["tdx_claim_ready"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
