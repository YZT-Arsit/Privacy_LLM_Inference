"""AAAI generation benchmark runner (crash-safe, resumable, paper-facing).

Drives IFEval / GSM8K / MT-Bench over plaintext-GPU or ours (A_rightmul, folded
remote + TDX boundary client) with the robustness a memory-constrained H800 needs:

* **resume** -- re-running the same command skips already-completed ids;
* per-example **retry with exponential backoff** + worker reconnect (a single bad
  example is marked ``failed`` and the run continues, unless ``--fail-fast``);
* each response is flushed + fsync'd immediately;
* a **status** file + **heartbeat** file are checkpointed every ``--progress-every``
  examples so ``scripts/monitor_aaai_run.py`` can watch progress live;
* MT-Bench runs **two turns** (turn 2 conditioned on turn 1) per question;
* ``--paper-facing-aaai`` enforces the full AAAI contract (A_rightmul, seq_len
  1024, max_new_tokens 512, EOS, real run, TDX boundary client + valid bound
  attestation evidence, zero trusted nonlinear, compatible masks, pad coverage).

No downloads; all paths are CLI args. dry_run / mock_runtime are forbidden under
``--paper-facing-aaai``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_datasets import (  # noqa: E402
    extract_gsm8k_answer, gsm8k_exact_match, load_dataset)
from pllo.benchmarks.real_predictors import (  # noqa: E402
    RealBackendUnavailable, build_predictor)
from pllo.benchmarks.run_state import (  # noqa: E402
    RunState, append_jsonl_record, completed_ids_from_jsonl, plan_examples)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    normalize_nonlinear_backend)
from pllo.protocol.resilient_remote import (  # noqa: E402
    ResilientRemoteGpuWorker, is_retriable_error)

_BACKENDS = ("plaintext_local", "folded_remote")
_DATASETS = ("ifeval", "gsm8k", "mt_bench", "humaneval", "mbpp",
             "sensitive_prompt_1024", "longbench_1024_lite")


def _load_json(path):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None


def _worker_health(url, *, timeout=10.0):
    if not url:
        return None
    try:
        client = ResilientRemoteGpuWorker(url, per_request_timeout=timeout,
                                          max_retries=2, backoff_base_sec=0.3)
        h = client.health()
        client.close()
        return h
    except Exception:                                            # noqa: BLE001
        return None


def _build_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(_DATASETS))
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--backend", required=True, choices=list(_BACKENDS))
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--nonlinear-backend", default="A_rightmul")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-chat-template", action="store_true", default=False)
    ap.add_argument("--audit", action="store_true", default=False)
    ap.add_argument("--require-real", action="store_true", default=False)
    ap.add_argument("--mock-runtime", action="store_true", default=False)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--disable-eos-stop", action="store_true", default=False)
    ap.add_argument("--paper-facing-aaai", action="store_true", default=False)
    ap.add_argument("--tdx-boundary-client", action="store_true", default=False)
    ap.add_argument("--trusted-runtime", default="process")
    ap.add_argument("--attestation-evidence-json", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    # resume / status / heartbeat
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--output-response-jsonl", required=True)
    ap.add_argument("--output-report-json", required=True)
    ap.add_argument("--status-json", default=None)
    ap.add_argument("--heartbeat-json", default=None)
    # resilience
    ap.add_argument("--max-retries-per-example", type=int, default=3)
    ap.add_argument("--retry-sleep-sec", type=float, default=2.0)
    ap.add_argument("--heartbeat-interval-sec", type=float, default=10.0)
    ap.add_argument("--fail-fast", action="store_true", default=False)
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--trace-worker-timings", action="store_true", default=False)
    ap.add_argument("--trace-output-jsonl", default=None)
    # GPU-staged (non-secret) obfuscation schedule
    ap.add_argument("--gpu-staged-schedule-dir", default=None)
    ap.add_argument("--use-gpu-staged-schedule", action="store_true",
                    default=False)
    ap.add_argument("--require-staged-schedule", action="store_true",
                    default=False)
    ap.add_argument("--staged-schedule-audit-json", default=None)
    return ap.parse_args()


def _static_paper_facing_checks(args, nb) -> list[str]:
    v = []
    if args.dataset not in _DATASETS:
        v.append("dataset must be one of %s" % list(_DATASETS))
    if int(args.seq_len) != 1024:
        v.append("seq_len must be 1024")
    if int(args.max_new_tokens) != 512:
        v.append("max_new_tokens must be 512")
    if args.disable_eos_stop:
        v.append("--disable-eos-stop is forbidden")
    if args.mock_runtime:
        v.append("--mock-runtime is forbidden")
    if args.backend == "folded_remote":
        if nb != "A_rightmul":
            v.append("folded_remote requires --nonlinear-backend A_rightmul")
        if not args.require_real:
            v.append("--require-real is required for folded_remote")
        if not args.tdx_boundary_client:
            v.append("--tdx-boundary-client is required for folded_remote")
        if not args.attestation_evidence_json:
            v.append("--attestation-evidence-json is required for folded_remote")
    elif args.backend == "plaintext_local":
        if args.tdx_boundary_client:
            v.append("plaintext_local must NOT use --tdx-boundary-client")
    return v


def _generate_with_retry(predictor, prompt, *, max_retries, sleep_sec):
    """Generate one prompt, retrying transient transport failures with backoff.
    Returns (gen_dict, retries, error). On success error is None."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return predictor.generate(prompt), attempt, None
        except Exception as exc:                                # noqa: BLE001
            last_err = exc
            if attempt >= max_retries or not is_retriable_error(exc):
                return None, attempt, exc
            time.sleep(min(60.0, sleep_sec * (2 ** attempt)))
    return None, max_retries, last_err


def main() -> int:
    args = _build_args()
    nb = normalize_nonlinear_backend(args.nonlinear_backend or "A_rightmul")
    run_id = args.run_id or ("%s_%s_%s" % (args.dataset, args.backend, nb))

    # paper-facing static fail-fast (never start a long bad run)
    if args.paper_facing_aaai:
        if args.mock_runtime:
            print("ERROR: --paper-facing-aaai forbids --mock-runtime",
                  file=sys.stderr)
            return 3
        static = _static_paper_facing_checks(args, nb)
        if static:
            print("ERROR: --paper-facing-aaai violations:\n  - %s"
                  % "\n  - ".join(static), file=sys.stderr)
            return 3

    examples = load_dataset(args.dataset, args.dataset_jsonl,
                            max_examples=args.max_examples)
    if not examples:
        print("ERROR: no examples in %s" % args.dataset_jsonl, file=sys.stderr)
        return 3

    # ---- GPU-staged (non-secret) obfuscation schedule ----
    staged_fields: dict = {}
    if args.use_gpu_staged_schedule or args.require_staged_schedule:
        from pllo.runtime.gpu_staged_schedule import (
            StagedScheduleSecretLeak, audit_gpu_staged_schedule_no_secrets,
            load_gpu_staged_schedule, staged_schedule_report_fields)
        sdir = args.gpu_staged_schedule_dir
        if not sdir:
            print("ERROR: --use/--require-staged-schedule needs "
                  "--gpu-staged-schedule-dir", file=sys.stderr)
            return 3
        try:
            manifest = load_gpu_staged_schedule(sdir)
            audit = audit_gpu_staged_schedule_no_secrets(manifest)
        except (StagedScheduleSecretLeak, FileNotFoundError, OSError) as exc:
            print("ERROR: staged schedule audit/load failed (no unsafe fallback): "
                  "%s" % exc, file=sys.stderr)
            if args.require_staged_schedule or args.paper_facing_aaai:
                return 3
            audit, manifest = None, None
        if manifest is not None:
            staged_fields = staged_schedule_report_fields(
                manifest, audit, slots_consumed=manifest.get("num_slots"))
            staged_fields["require_staged_schedule"] = bool(
                args.require_staged_schedule)
            if args.staged_schedule_audit_json:
                Path(args.staged_schedule_audit_json).parent.mkdir(
                    parents=True, exist_ok=True)
                Path(args.staged_schedule_audit_json).write_text(
                    json.dumps({**audit, **staged_fields}, indent=2),
                    encoding="utf-8")

    resp_path = Path(args.output_response_jsonl)
    resp_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_ids_from_jsonl(resp_path) if args.resume else set()
    to_run, skipped = plan_examples(examples, completed, resume=args.resume)

    # predictor (or stub) -------------------------------------------------
    if args.backend == "plaintext_local":
        have_real = bool(args.model_path) and not args.mock_runtime
    else:
        have_real = bool(args.model_path and args.gpu_worker_url
                         and args.embedding_path) and not args.mock_runtime
    attest_evidence = _load_json(args.attestation_evidence_json)
    predictor = None
    if have_real or args.require_real:
        try:
            predictor = build_predictor(
                args.backend, model_path=args.model_path,
                model_name=args.model_name, gpu_worker_url=args.gpu_worker_url,
                embedding_path=args.embedding_path,
                attestation_evidence=attest_evidence,
                expected_mr_td=args.expected_mr_td, seq_len=args.seq_len,
                max_new_tokens=args.max_new_tokens, dtype=args.dtype,
                device=args.device, audit=args.audit, nonlinear_backend=nb,
                stop_on_eos=(not args.disable_eos_stop),
                use_chat_template=bool(args.use_chat_template))
        except RealBackendUnavailable as exc:
            if args.require_real:
                print("ERROR: --require-real but backend unavailable: %s" % exc,
                      file=sys.stderr)
                return 3
            predictor = None
    is_dry = predictor is None
    if args.paper_facing_aaai and is_dry:
        print("ERROR: --paper-facing-aaai requires a real run (no dry-run/stub)",
              file=sys.stderr)
        return 3

    if predictor is not None and hasattr(predictor, "enable_decode_profiling"):
        try:
            predictor.enable_decode_profiling(
                True, request_worker_timing=args.trace_worker_timings)
        except Exception:                                        # noqa: BLE001
            pass

    state = RunState(
        run_id, dataset=args.dataset, backend=args.backend, model=args.model_name,
        total_examples=len(examples), status_json=args.status_json,
        heartbeat_json=args.heartbeat_json,
        resume_from_existing=bool(args.resume and completed))
    state.skipped_existing_examples = len(skipped)
    state.checkpoint()

    fh = open(resp_path, "a", encoding="utf-8")
    last_hb = time.perf_counter()
    n = len(to_run)
    failed_records = []
    try:
        for idx, ex in enumerate(to_run):
            rid = str(ex.get("id"))
            state.begin_example(rid)
            turns = ex.get("turns") or [ex.get("prompt", "")]
            two_turn = (args.dataset == "mt_bench" and len(turns) > 1)
            try:
                recs = _run_example(
                    predictor, ex, turns, two_turn, args, nb, is_dry)
            except _ExampleFailure as ef:
                failed = {"id": rid, "dataset": args.dataset, "status": "failed",
                          "error_type": ef.error_type,
                          "error_message": ef.message, "retries": ef.retries}
                append_jsonl_record(fh, failed)
                failed_records.append(failed)
                state.record_failed(rid, error_type=ef.error_type,
                                    error_message=ef.message, retries=ef.retries)
                if args.fail_fast:
                    print("ERROR: --fail-fast: example %s failed: %s"
                          % (rid, ef.message), file=sys.stderr)
                    state.checkpoint()
                    return 1
                continue
            tokens = 0
            for rec in recs:
                append_jsonl_record(fh, rec)
                tokens += int(rec.get("num_tokens") or 0)
            state.record_completed(rid, tokens=tokens)
            now = time.perf_counter()
            if ((idx + 1) % max(1, args.progress_every) == 0
                    or (now - last_hb) >= args.heartbeat_interval_sec
                    or (idx + 1) == n):
                state.checkpoint()
                last_hb = now
                print("[aaai] %s %d/%d id=%s completed=%d failed=%d skipped=%d"
                      % (args.dataset, idx + 1, n, rid, state.completed_examples,
                         state.failed_examples, state.skipped_existing_examples),
                      flush=True)
    finally:
        fh.close()
        if predictor is not None and hasattr(predictor, "close"):
            try:
                predictor.close()
            except Exception:                                    # noqa: BLE001
                pass

    state.finish()

    # ---- report ----
    stats = {}
    if predictor is not None and hasattr(predictor, "stats"):
        try:
            stats = predictor.stats() or {}
        except Exception:                                        # noqa: BLE001
            stats = {}
    worker_health = _worker_health(args.gpu_worker_url) if args.gpu_worker_url \
        else None
    report = _build_report(args, nb, state, stats, worker_health,
                           attest_evidence, is_dry, failed_records, staged_fields)
    Path(args.output_report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report_json).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== aaai generation (%s / %s / %s) ===" % (
        args.dataset, args.backend, nb))
    print("total=%d completed=%d failed=%d skipped=%d paper_ready=%s"
          % (report["total_examples"], report["completed_examples"],
             report["failed_examples"], report["skipped_existing_examples"],
             report["paper_ready"]))
    if args.paper_facing_aaai:
        print("paper_facing_aaai=%s violations=%s"
              % (report.get("paper_facing_aaai"),
                 report.get("paper_facing_aaai_violations")))
        if not report.get("paper_facing_aaai"):
            return 1
    if report["failed_examples"] and args.fail_fast:
        return 1
    return 0


class _ExampleFailure(Exception):
    def __init__(self, error_type, message, retries):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retries = retries


def _gen(predictor, prompt, args, is_dry):
    if is_dry:
        from pllo.benchmarks.generation_metrics import stub_generate
        g = stub_generate({"id": "x", "prompt": prompt}, args.max_new_tokens)
        return {"text": g["text"], "token_ids": g["token_ids"],
                "finish_reason": "length"}, 0, None
    return _generate_with_retry(
        predictor, prompt, max_retries=args.max_retries_per_example,
        sleep_sec=args.retry_sleep_sec)


def _run_example(predictor, ex, turns, two_turn, args, nb, is_dry):
    """Generate 1 (single-turn) or 2 (mt_bench) response records for an example.
    Raises _ExampleFailure on unrecoverable error."""
    rid = str(ex.get("id"))
    recs = []
    history = ""
    n_turns = len(turns) if two_turn else 1
    for ti in range(n_turns):
        if two_turn:
            prompt = (turns[ti] if ti == 0
                      else "%s\n\n%s\n\n%s" % (turns[0], history, turns[ti]))
        else:
            prompt = ex.get("prompt", "")
        t0 = time.perf_counter()
        g, retries, err = _gen(predictor, prompt, args, is_dry)
        if g is None:
            raise _ExampleFailure(type(err).__name__ if err else "GenerationError",
                                  str(err), retries)
        text = g.get("text", "")
        toks = g.get("token_ids") or []
        rec = {"id": rid, "dataset": args.dataset, "status": "ok",
               "turn_index": ti, "prompt_hash": _hash(prompt),
               "response": text, "token_ids": toks if toks else None,
               "num_tokens": len(toks),
               "finish_reason": g.get("finish_reason"),
               "latency_sec": round(time.perf_counter() - t0, 6),
               "category": ex.get("category")}
        if args.dataset == "gsm8k":
            gold = ex.get("final_answer") or ex.get("reference")
            rec["predicted_answer"] = extract_gsm8k_answer(text)
            rec["gold_answer"] = gold
            rec["exact_match"] = gsm8k_exact_match(text, gold)
        recs.append(rec)
        history = text
    return recs


def _hash(s):
    import hashlib
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def _build_report(args, nb, state, stats, worker_health, attest_evidence, is_dry,
                  failed_records, staged_fields=None):
    nonlinear_ev = {}
    if isinstance(worker_health, dict):
        nonlinear_ev = worker_health.get("nonlinear_execution_evidence") or {}
    report = {
        "stage": "aaai_generation", "run_id": state.run_id,
        "dataset": args.dataset, "backend": args.backend, "nonlinear_backend": nb,
        "model_name": args.model_name, "decoding": "greedy",
        "use_chat_template": bool(args.use_chat_template),
        "seq_len": int(args.seq_len), "max_new_tokens": int(args.max_new_tokens),
        "stop_on_eos": (not args.disable_eos_stop),
        "dry_run": is_dry, "mock_runtime": bool(args.mock_runtime),
        "require_real": bool(args.require_real),
        # run-state summary
        "total_examples": state.total_examples,
        "completed_examples": state.completed_examples,
        "failed_examples": state.failed_examples,
        "skipped_existing_examples": state.skipped_existing_examples,
        "resume_from_existing": state.resume_from_existing,
        "last_completed_id": state.last_completed_id,
        "generated_tokens_total": state.generated_tokens_total,
        "start_time": state.start_time, "update_time": state.update_time,
        "end_time": state.end_time, "failed_records": failed_records,
        # tdx / worker
        "tdx_boundary_client": bool(args.tdx_boundary_client),
        "trusted_runtime": ("tdx_guest" if args.tdx_boundary_client
                            else args.trusted_runtime),
        "attestation_evidence_json": args.attestation_evidence_json,
        "attestation_evidence_attached": bool(attest_evidence is not None),
        "h800_worker_url": args.gpu_worker_url,
        "h800_worker_health": worker_health,
        "h800_worker_tee_used_on_gpu": (
            worker_health.get("tee_used_on_gpu")
            if isinstance(worker_health, dict) else stats.get("tee_used_on_gpu")),
        "full_model_weights_loaded_in_trusted_runtime": bool(
            args.backend == "plaintext_local" and not is_dry),
        "paper_ready": (not is_dry),
    }
    # merge measured nonlinear evidence + pad/compatible flags from worker
    for k in ("nonlinear_trusted_calls", "trusted_nonlinear_ops_count",
              "nonlinear_single_tee_entry_exit", "compatible_masks_verified",
              "nonlinear_op_backend", "nonlinear_execution_status"):
        if k in nonlinear_ev:
            report[k] = nonlinear_ev[k]
        elif isinstance(worker_health, dict) and k in worker_health:
            report[k] = worker_health[k]
    if isinstance(worker_health, dict):
        report["base_linear_pad_all_modules_covered"] = worker_health.get(
            "base_linear_pad_all_modules_covered",
            _all_pad_covered(worker_health.get("linear_pad_coverage")))
        report["compatible_masks_verified"] = worker_health.get(
            "compatible_masks_verified", report.get("compatible_masks_verified"))
    # gsm8k accuracy summary
    if args.dataset == "gsm8k":
        report.update(_gsm8k_summary(args.output_response_jsonl))
    # GPU-staged (non-secret) schedule fields
    if staged_fields:
        report.update(staged_fields)
    # paper-facing-aaai verdict
    if args.paper_facing_aaai:
        from pllo.benchmarks.aaai_paper_facing import (
            aaai_paper_facing_report_fields)
        pf = aaai_paper_facing_report_fields(
            report, evidence=attest_evidence,
            expected_mr_td=args.expected_mr_td)
        report.update(pf)
        if not pf["paper_facing_aaai"]:
            report["paper_ready"] = False
            report["paper_ready_blocker"] = pf["paper_facing_aaai_violations"]
    return report


def _all_pad_covered(cov):
    if not isinstance(cov, dict):
        return None
    return all(cov.get(m) for m in ("q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj",
                                    "lm_head"))


def _gsm8k_summary(resp_path):
    n = correct = scored = 0
    try:
        with open(resp_path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                if r.get("status") == "failed":
                    continue
                n += 1
                if r.get("gold_answer") is not None:
                    scored += 1
                    correct += int(bool(r.get("exact_match")))
    except Exception:                                            # noqa: BLE001
        pass
    return {"gsm8k_num_scored": scored, "gsm8k_num_correct": correct,
            "gsm8k_exact_match_accuracy": (correct / scored) if scored else None}


if __name__ == "__main__":
    raise SystemExit(main())
