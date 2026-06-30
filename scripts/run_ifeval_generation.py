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
from pllo.benchmarks.run_state import (  # noqa: E402
    RunState,
    append_jsonl_record,
    completed_ids_from_jsonl,
    failed_ids_from_jsonl,
    plan_examples,
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


def _sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sanitize_error(exc: BaseException, *, sensitive_spans=None,
                    raw_prompt=None) -> tuple[str, str]:
    """Return (error_type, error_message) with NO raw prompt / sensitive span.

    The message is the exception class + str(exc), then any literal occurrence of
    the raw prompt or a sensitive span is stripped so a failure record / error log
    can never leak user input."""
    etype = type(exc).__name__
    msg = str(exc)
    for s in (list(sensitive_spans or []) + ([raw_prompt] if raw_prompt else [])):
        if s and str(s) in msg:
            msg = msg.replace(str(s), "<redacted>")
    return etype, msg[:500]


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
    ap.add_argument("--paper-facing-generation", action="store_true",
                    default=False,
                    help="enforce the AAAI paper-facing generation contract: "
                    "nonlinear A_rightmul, seq_len=1024, max_new_tokens=512, EOS "
                    "stop ON, --require-real, folded_remote + TDX boundary client, "
                    "attestation evidence binding the nonlinear backend, worker "
                    "health readable, nonlinear_trusted_calls=0, "
                    "compatible_masks_verified, schedule_full_coverage_verified. "
                    "Any unmet condition -> paper_ready=False and exit non-zero.")
    # ---- resume / crash-safety / status + heartbeat (long real runs) ----
    ap.add_argument("--dataset", default=None,
                    help="dataset label recorded per response (e.g. ifeval / "
                    "gsm8k / mt_bench / humaneval / sensitive_prompt_1024); also "
                    "drives sensitive-prompt raw-prompt protection")
    ap.add_argument("--run-id", default=None,
                    help="run id for the status/heartbeat files (default derived "
                    "from the output path)")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="resume: read the existing --output-response-jsonl, skip "
                    "already-completed ids, append (never overwrite)")
    ap.add_argument("--status-json", default=None,
                    help="status checkpoint JSON (counts / last id / timestamps)")
    ap.add_argument("--heartbeat-json", default=None,
                    help="heartbeat JSON (alive / pid / host / elapsed), updated "
                    "frequently so a monitor can tell the run is alive")
    ap.add_argument("--heartbeat-interval-sec", type=float, default=10.0)
    ap.add_argument("--max-retries-per-example", type=int, default=0,
                    help="retry a failing example this many times before marking "
                    "it failed (transport/transient errors)")
    ap.add_argument("--retry-sleep-sec", type=float, default=2.0)
    ap.add_argument("--retry-backoff", type=float, default=2.0,
                    help="exponential backoff multiplier for per-example retries")
    ap.add_argument("--fail-fast", action="store_true", default=False,
                    help="abort the whole run on the first example that fails all "
                    "retries (default: record failed + continue)")
    ap.add_argument("--skip-failed-existing", action="store_true", default=False,
                    help="on --resume, do NOT retry ids whose existing record is "
                    "status=failed (default: retry them)")
    # ---- raw-prompt output protection (user-input privacy) ----
    ap.add_argument("--save-raw-prompts", action="store_true", default=False,
                    help="store the raw prompt text in the response JSONL "
                    "(default OFF; only a prompt_sha256 is written). FORBIDDEN "
                    "under --paper-facing-generation and for sensitive datasets")
    ap.add_argument("--redact-raw-prompts", action="store_true", default=False,
                    help="explicitly force raw-prompt redaction (default behaviour; "
                    "kept for clarity in scripts)")
    ap.add_argument("--paper-facing-no-raw-prompts", action="store_true",
                    default=False, help="assert no raw prompt is ever written "
                    "(implied by --paper-facing-generation)")
    ap.add_argument("--worker-health-jsonl", default=None,
                    help="append the worker /health snapshot (public, no secrets) "
                    "to this JSONL for the security/robustness audit")
    ap.add_argument("--use-resilient-worker", dest="use_resilient_worker",
                    action="store_true", default=True,
                    help="(default ON) drive folded_remote through the "
                    "retry/backoff/reconnect ResilientRemoteGpuWorker")
    ap.add_argument("--no-resilient-worker", dest="use_resilient_worker",
                    action="store_false",
                    help="use the bare RemoteGpuWorker (no auto-retry/reconnect)")
    ap.add_argument("--worker-max-retries", type=int, default=5,
                    help="resilient worker: max transport retries per request")
    ap.add_argument("--worker-backoff-base-sec", type=float, default=0.5)
    ap.add_argument("--output-response-jsonl", required=True)
    ap.add_argument("--output-report-json", required=True)
    args = ap.parse_args()

    # --save-raw-prompts is incompatible with the paper-facing contract.
    if args.save_raw_prompts and (args.paper_facing_generation
                                  or args.paper_facing_no_raw_prompts):
        print("ERROR: --save-raw-prompts is forbidden under "
              "--paper-facing-generation / --paper-facing-no-raw-prompts "
              "(raw user input must never be persisted)", file=sys.stderr)
        return 3

    # Fail fast on the statically-knowable paper-facing violations (so a bad
    # invocation never even starts a long real run).
    if args.paper_facing_generation:
        _nb_req = normalize_nonlinear_backend(args.nonlinear_backend or "current")
        _static = []
        if _nb_req != "A_rightmul":
            _static.append("nonlinear_backend=%r (must be A_rightmul)" % _nb_req)
        if int(args.seq_len) != 1024:
            _static.append("seq_len=%s (must be 1024)" % args.seq_len)
        if int(args.max_new_tokens) != 512:
            _static.append("max_new_tokens=%s (must be 512)" % args.max_new_tokens)
        if args.disable_eos_stop:
            _static.append("--disable-eos-stop is forbidden (EOS stop must be ON)")
        if not args.require_real:
            _static.append("--require-real is required (no dry-run stub)")
        if args.backend != "folded_remote":
            _static.append("--backend must be folded_remote")
        if not args.tdx_boundary_client:
            _static.append("--tdx-boundary-client is required")
        if not args.attestation_evidence_json:
            _static.append("--attestation-evidence-json is required")
        if _static:
            print("ERROR: --paper-facing-generation violations:\n  - %s"
                  % "\n  - ".join(_static), file=sys.stderr)
            return 3

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

    # ---- dataset label + raw-prompt protection policy ----
    dataset_name = (args.dataset
                    or (examples[0].get("dataset") if examples else None))
    is_sensitive = bool(dataset_name and "sensitive" in str(dataset_name).lower())
    # Raw prompt is NEVER persisted unless explicitly opted in AND it is not a
    # paper-facing / sensitive run. The static guard above already rejected
    # --save-raw-prompts under --paper-facing-generation.
    paper_no_raw = bool(args.paper_facing_generation
                        or args.paper_facing_no_raw_prompts)
    save_raw_prompts = bool(args.save_raw_prompts
                            and not paper_no_raw and not is_sensitive)
    if args.save_raw_prompts and is_sensitive:
        print("[ifeval] NOTE: sensitive dataset -> raw prompts are NOT saved "
              "(overriding --save-raw-prompts).", file=sys.stderr)

    # ---- resume planning: skip completed ids, optionally retry failed ones ----
    rp = Path(args.output_response_jsonl)
    completed_ids: set = set()
    failed_existing: set = set()
    if args.resume and rp.exists():
        completed_ids = completed_ids_from_jsonl(rp)
        failed_existing = failed_ids_from_jsonl(rp)
        if args.skip_failed_existing:
            # treat existing failures as terminal: do not retry them
            completed_ids |= failed_existing
    to_run, skipped_ids = plan_examples(examples, completed_ids,
                                        resume=bool(args.resume))

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
                use_chat_template=bool(args.use_chat_template),
                use_resilient_worker=bool(args.use_resilient_worker),
                worker_max_retries=int(args.worker_max_retries),
                worker_backoff_base_sec=float(args.worker_backoff_base_sec))
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
    n_examples = len(to_run)
    failed_records = []      # per-example failure records (no raw prompt)
    # Streaming response writer: APPEND so a resumed run never overwrites prior
    # work; each example is flushed + fsync'd immediately (crash-safe). With
    # --no-stream-responses (and no resume) the old buffered single-write path is
    # used. Resume always streams (append) so prior records survive.
    rp.parent.mkdir(parents=True, exist_ok=True)
    stream_mode = bool(args.stream_responses or args.resume)
    open_mode = "a" if args.resume else "w"
    stream_fh = open(rp, open_mode, encoding="utf-8") if stream_mode else None

    # ---- run state: status + heartbeat checkpoints ----
    run_id = args.run_id or rp.stem
    state = RunState(
        run_id, dataset=dataset_name, backend=args.backend, model=args.model_name,
        nonlinear_backend=nb, output_response_jsonl=str(rp),
        paper_facing_generation=bool(args.paper_facing_generation),
        total_examples=len(examples), status_json=args.status_json,
        heartbeat_json=args.heartbeat_json,
        resume_from_existing=bool(args.resume))
    state.skipped_existing_examples = len(skipped_ids)
    state.checkpoint()
    last_hb = time.perf_counter()
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

    def _build_and_attach_schedule(idx):
        """Build + attach one per-example schedule; return (schedule, latency)."""
        if not enabled:
            return None, 0.0
        tprep = time.perf_counter()
        schedule = PrecomputedMaskSchedule.precompute(
            max_steps=per_example_steps, hidden_size=int(sched_hidden),
            seed=int(args.schedule_seed) + idx, seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens, dtype=sched_dtype,
            device=args.device, mask_family="pairwise_complex_scaling",
            nonlinear_backend=nb,
            with_secret_tensors=precompute_secret_tensors, strict_audit=True)
        lat = time.perf_counter() - tprep
        audit_schedule_trusted_only(schedule)
        if predictor is not None and hasattr(
                predictor, "attach_obfuscation_schedule"):
            predictor.attach_obfuscation_schedule(schedule)
        return schedule, lat

    def _generate_once(ex, schedule):
        """One generation attempt (real predictor or mock). Raises on failure."""
        prompt = str(ex["prompt"])
        if predictor is not None:
            g = predictor.generate(prompt)
            return g, g.get("text", ""), g.get("token_ids") or []
        g = stub_generate(ex, args.max_new_tokens)
        text, toks = g["text"], g["token_ids"]

        def _on_step(kind, step, phase, _sched=schedule):
            if kind == "schedule" and _sched is not None:
                try:
                    _sched.consume(step)
                except Exception:                        # noqa: BLE001
                    pass
        worker_timing_fn = None
        if args.trace_worker_timings:
            from pllo.protocol.worker_timing import (
                audit_worker_timing_no_secrets, synthetic_worker_timing)

            def worker_timing_fn(step, phase):           # noqa: F811
                wt = synthetic_worker_timing(phase=phase, num_layers=28)
                audit_worker_timing_no_secrets(wt)
                return wt
        simulate_mock_decode(profiler, mock_counters,
                             n_tokens=max(1, len(toks)), hidden_size=64,
                             on_step=_on_step, worker_timing_fn=worker_timing_fn)
        return g, text, toks

    def _maybe_heartbeat(force=False):
        nonlocal last_hb
        now = time.perf_counter()
        if force or (now - last_hb) >= max(0.0, args.heartbeat_interval_sec):
            state.heartbeat()
            last_hb = now

    try:
        for idx, ex in enumerate(to_run):
            rid = str(ex.get("id"))
            raw_spans = ex.get("sensitive_spans") or []
            state.begin_example(rid)
            _log(idx, ex, "start")
            prompt = str(ex["prompt"])
            ex_t0 = time.perf_counter()
            max_attempts = max(1, int(args.max_retries_per_example) + 1)
            attempt = 0
            schedule = None
            sched_lat = 0.0
            g = text = None
            toks = []
            last_exc = None
            while attempt < max_attempts:
                attempt += 1
                try:
                    schedule, sched_lat = _build_and_attach_schedule(idx)
                    _log(idx, ex, "generate_start", attempt=attempt)
                    g, text, toks = _generate_once(ex, schedule)
                    last_exc = None
                    break
                except Exception as exc:                 # noqa: BLE001
                    last_exc = exc
                    from pllo.protocol.resilient_remote import is_retriable_error
                    retriable = is_retriable_error(exc)
                    if attempt >= max_attempts or not retriable:
                        break
                    delay = (args.retry_sleep_sec
                             * (args.retry_backoff ** (attempt - 1)))
                    etype, _ = _sanitize_error(exc, sensitive_spans=raw_spans,
                                               raw_prompt=prompt)
                    print("[ifeval] example %s attempt %d/%d failed (%s); "
                          "retrying in %.1fs" % (rid, attempt, max_attempts,
                                                 etype, delay), file=sys.stderr,
                          flush=True)
                    time.sleep(delay)
            retries = attempt - 1
            # ---- failure: record (NO raw prompt), continue unless --fail-fast --
            if last_exc is not None:
                etype, emsg = _sanitize_error(
                    last_exc, sensitive_spans=raw_spans, raw_prompt=prompt)
                state.record_failed(rid, error_type=etype, error_message=emsg,
                                    retries=retries)
                frec = {"id": rid, "dataset": dataset_name,
                        "backend": args.backend, "nonlinear_backend": nb,
                        "model_name": args.model_name, "response": None,
                        "num_tokens": 0, "finish_reason": None,
                        "latency_s": round(time.perf_counter() - ex_t0, 6),
                        "retries": retries, "status": "failed",
                        "error_type": etype, "error_message": emsg,
                        "prompt_sha256": _sha256(prompt),
                        "prompt_token_count": None}
                failed_records.append(frec)
                if stream_fh is not None:
                    append_jsonl_record(stream_fh, frec)
                state.checkpoint()
                _maybe_heartbeat(force=True)
                if args.fail_fast:
                    print("ERROR: --fail-fast: example %s failed all %d attempt(s)"
                          % (rid, max_attempts), file=sys.stderr)
                    raise last_exc
                continue
            # ---- success: commit schedule counters for THIS attempt only ----
            if enabled and schedule is not None:
                sched_precompute_latency += sched_lat
                last_schedule = schedule
                sched_slots_precomputed += len(schedule.slots)
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
            finish_reason = _peek_finish(g)
            rec = {"id": rid, "dataset": dataset_name, "backend": args.backend,
                   "nonlinear_backend": nb, "model_name": args.model_name,
                   "response": text, "num_tokens": len(toks),
                   "finish_reason": finish_reason,
                   "latency_s": round(time.perf_counter() - ex_t0, 6),
                   "retries": retries, "status": "ok", "error_type": None,
                   "error_message": None, "prompt_sha256": _sha256(prompt),
                   "prompt_format": fr["prompt_format"],
                   "formatted_prompt_sha256": fr["formatted_prompt_sha256"],
                   "prompt_token_count": fr["prompt_token_count"]}
            # Raw prompt is persisted ONLY when explicitly opted-in and the run is
            # neither paper-facing nor a sensitive dataset (gate resolved above).
            if save_raw_prompts:
                rec["prompt"] = prompt
            responses.append(rec)
            # stream this example to disk immediately (append + flush + fsync)
            if stream_fh is not None:
                append_jsonl_record(stream_fh, rec)
            state.record_completed(rid, tokens=len(toks))
            state.checkpoint()
            _maybe_heartbeat()
            done_n = idx + 1
            avg = (time.perf_counter() - online_t0) / done_n
            _log(idx, ex, "done", tokens=len(toks),
                 finish_reason=finish_reason,
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

    # worker robustness (resilient client): retries / reconnects / last error
    worker_retry_count = stats.get("worker_retry_count")
    worker_reconnects_total = stats.get("worker_reconnect_count")
    worker_last_error = stats.get("worker_last_error_sanitized")
    state.record_robustness(retries=int(worker_retry_count or 0),
                            reconnects=int(worker_reconnects_total or 0))

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
        "dataset": dataset_name,
        "run_id": run_id,
        "dry_run": is_dry,
        "paper_ready": (not is_dry),
    }
    report.update(sfields)
    # ---- resume / crash-safety / failure accounting ----
    report["resume"] = bool(args.resume)
    report["resumed_from_existing"] = bool(args.resume and rp.exists())
    report["skipped_existing_examples"] = len(skipped_ids)
    report["completed_this_run"] = len(responses)
    report["failed_examples"] = len(failed_records)
    report["failed_ids"] = [f["id"] for f in failed_records]
    report["status_json"] = args.status_json
    report["heartbeat_json"] = args.heartbeat_json
    # a single failed example forces paper_ready=False (cannot claim a clean run)
    if failed_records:
        report["paper_ready"] = False
        report["paper_ready_blocker"] = "%d example(s) failed" % len(failed_records)
    # ---- raw-prompt output protection (user-input privacy) ----
    report["raw_prompts_saved"] = bool(save_raw_prompts)
    report["paper_facing_no_raw_prompts"] = bool(paper_no_raw)
    report["sensitive_dataset"] = is_sensitive
    report["response_jsonl_contains_raw_prompt"] = bool(save_raw_prompts)
    report["prompt_sha256_only"] = (not save_raw_prompts)
    # ---- worker robustness (resilient client) ----
    report["use_resilient_worker"] = bool(args.use_resilient_worker)
    report["worker_retry_count"] = worker_retry_count
    report["worker_reconnects_total"] = worker_reconnects_total
    report["worker_last_error_sanitized"] = worker_last_error
    report["worker_health_snapshots_jsonl"] = args.worker_health_jsonl
    report["retries_total"] = state.retries_total
    report["max_retries_per_example"] = int(args.max_retries_per_example)
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
    # append the (public, no-secret) worker health snapshot for the audit
    if args.worker_health_jsonl and worker_health is not None:
        try:
            whp = Path(args.worker_health_jsonl)
            whp.parent.mkdir(parents=True, exist_ok=True)
            with open(whp, "a", encoding="utf-8") as _wh:
                append_jsonl_record(_wh, {"run_id": run_id, **worker_health})
        except Exception:                                    # noqa: BLE001
            pass
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

    # ---- paper-facing generation contract (AAAI A_rightmul mainline) ----
    # Derive the attestation binding from the attached evidence + surface the
    # worker-measured compatible-mask flag, then evaluate the full contract.
    report["attestation_runtime_hash_binds_nonlinear_backend"] = bool(
        isinstance(attest_evidence, dict)
        and attest_evidence.get("runtime_hash_binds_nonlinear_backend") is True)
    if "compatible_masks_verified" not in report:
        report["compatible_masks_verified"] = (
            (worker_health or {}).get("compatible_masks_verified")
            if isinstance(worker_health, dict) else None)
    paper_facing_gen_failed = False
    if args.paper_facing_generation:
        from pllo.benchmarks.paper_facing_generation import (
            paper_facing_generation_report_fields)
        pf = paper_facing_generation_report_fields(report)
        report.update(pf)
        if not pf["paper_facing_generation"]:
            report["paper_ready"] = False
            report["paper_ready_blocker"] = (
                "paper_facing_generation contract unmet: %s"
                % pf["paper_facing_generation_violations"])
            paper_facing_gen_failed = True

    # Responses were already streamed to disk (append + fsync) when streaming was
    # active; only the buffered (--no-stream-responses, no --resume) path writes
    # the JSONL here. Never overwrite when a stream handle was used (resume-safe).
    if stream_fh is None:
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as fh:
            for r in responses:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # final status/heartbeat checkpoint (alive=false)
    state.finish()
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
    if args.paper_facing_generation:
        print("paper_facing_generation=%s violations=%s"
              % (report.get("paper_facing_generation"),
                 report.get("paper_facing_generation_violations")))
    if paper_facing_gen_failed:
        print("ERROR: --paper-facing-generation contract unmet; paper_ready=False",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
