"""Local mock probe for the precomputed per-step obfuscation schedule.

Builds a trusted-side schedule, then in ``--mock-runtime`` simulates an
autoregressive decode that CONSUMES one fresh slot per generated token and sends
synthetic *masked* GPU requests (public metadata only). It audits, every step,
that NO schedule secret reaches the GPU payload, and that the schedule's
serializable surface holds no secrets.

Mock only -- NO model, NO GPU, NO server paths, NO Qwen weights. Synthetic
tensors. Honest reporting: the mock does not move the real online remask off the
critical path, so it reports ``schedule_used_for_metadata_only=true`` and
``online_remask_still_performed=true``.

Example::

    python scripts/run_precomputed_schedule_probe.py \\
        --hidden-size 128 --seq-len 128 --max-new-tokens 8 --dtype float32 \\
        --device cpu --precompute-obfuscation-schedule --schedule-max-steps 128 \\
        --schedule-seed 2035 --mock-runtime --report-schedule-stats \\
        --output-json outputs/precompute/precomputed_schedule_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.runtime.obfuscation_schedule import (  # noqa: E402
    PrecomputedMaskSchedule,
    ScheduleSecretLeak,
    audit_gpu_payload_no_schedule_secrets,
    audit_schedule_trusted_only,
    default_schedule_report_fields,
    schedule_report_fields,
)
from pllo.benchmarks.decode_profiler import (  # noqa: E402
    DecodeProfiler,
    simulate_mock_decode,
)


def _mock_gpu_request(slot, step, hidden_size):
    """A synthetic MASKED GPU request for one decode step: public metadata + a
    masked (already-obfuscated) embedding only. Carries the slot's PUBLIC ids so
    the worker can route, but NEVER any schedule secret tensor / pad / inverse."""
    return {
        "session_id": "mock",
        "step": int(step),
        "position": int(step),
        # masked/obfuscated payload (synthetic placeholder, not a real tensor)
        "masked_embedding": [0.0] * min(8, hidden_size),
        # PUBLIC schedule routing metadata only
        "obfuscation_mask_id": slot.mask_id,
        "obfuscation_domain_id": slot.domain_id,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mask-family", default="pairwise_complex_scaling")
    ap.add_argument("--nonlinear-backend", default="current")
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
    ap.add_argument("--mock-runtime", action="store_true", default=False)
    ap.add_argument("--trace-decode-steps", action="store_true", default=False)
    ap.add_argument("--trace-output-jsonl", default=None)
    ap.add_argument("--trace-worker-timings", action="store_true", default=False,
                    help="attach SYNTHETIC worker-side forward timing per step "
                    "(mock only; splits the roundtrip into network vs worker)")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    if not args.mock_runtime:
        print("ERROR: this probe only supports --mock-runtime locally "
              "(no model/GPU/server).", file=sys.stderr)
        return 3

    n_new = int(args.max_new_tokens)
    enabled = bool(args.precompute_obfuscation_schedule)
    max_steps = int(args.schedule_max_steps or max(n_new, args.seq_len))

    schedule = None
    if enabled:
        schedule = PrecomputedMaskSchedule.precompute(
            max_steps=max_steps, hidden_size=args.hidden_size,
            seed=args.schedule_seed, seq_len=args.seq_len,
            max_new_tokens=n_new, dtype=args.dtype, device=args.device,
            mask_family=args.mask_family,
            nonlinear_backend=args.nonlinear_backend,
            with_secret_tensors=True, strict_audit=True)
        audit_schedule_trusted_only(schedule)
        if args.schedule_cache_dir:
            schedule.to_disk(
                Path(args.schedule_cache_dir) / "obfuscation_schedule.json",
                save_secret_tensors=args.schedule_save_secret_tensors,
                allow_secret_persist=args.allow_secret_persist)

    # ---- mock profiled decode: one fresh slot + 9-stage timing per token ----
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    prof = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    leak_detected = False

    def _slot(step):
        if schedule is not None:
            return schedule.slot(step)
        return type("S", (), {"mask_id": "n/a", "domain_id": "n/a"})()

    def _on_step(kind, step, phase):
        nonlocal leak_detected
        if kind == "schedule" and schedule is not None:
            schedule.consume(step)
        elif kind == "serialize":
            try:
                audit_gpu_payload_no_schedule_secrets(
                    _mock_gpu_request(_slot(step), step, args.hidden_size))
            except ScheduleSecretLeak:
                leak_detected = True

    # optional SYNTHETIC worker timing (mock only): split the roundtrip into
    # network vs worker compute so the client-side merge/aggregate is exercised.
    worker_timing_fn = None
    if args.trace_worker_timings:
        from pllo.protocol.worker_timing import (
            audit_worker_timing_no_secrets, synthetic_worker_timing)

        def worker_timing_fn(step, phase):       # noqa: F811
            wt = synthetic_worker_timing(phase=phase, num_layers=28)
            audit_worker_timing_no_secrets(wt)   # mock metadata carries no secret
            return wt

    simulate_mock_decode(prof, counters, n_tokens=n_new,
                         hidden_size=args.hidden_size, on_step=_on_step,
                         worker_timing_fn=worker_timing_fn)

    agg = prof.aggregate(
        generated_tokens=n_new,
        schedule_precompute_latency_s=(schedule.precompute_latency_s
                                       if schedule else None))
    online_latency = agg["online_decode_latency_s"]

    fields = schedule_report_fields(
        schedule, enabled=enabled, online_generation_latency_s=online_latency,
        boundary_calls=agg["total_boundary_calls"], generated_tokens=n_new,
        trusted_bytes=None, gpu_bytes=None,
        # the mock does NOT move real remask offline -> honest flags
        schedule_used_for_metadata_only=True,
        online_remask_still_performed=True,
        schedule_secret_leaked_to_gpu=leak_detected,
        gpu_request_contains_schedule_secret=leak_detected)
    if not enabled:
        fields = default_schedule_report_fields()
        fields["online_generation_latency_s"] = round(online_latency, 6)

    if args.trace_decode_steps and args.trace_output_jsonl:
        prof.write_trace(args.trace_output_jsonl)

    report = {
        "stage": "precomputed_schedule_probe",
        "mock_runtime": True,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "max_new_tokens": n_new,
        "dtype": args.dtype,
        "device": args.device,
        "mask_family": args.mask_family,
        "nonlinear_backend": args.nonlinear_backend,
        "gpu_calls": agg["total_gpu_calls"],
        "audit_passed": (not leak_detected),
        "schedule_secret_leaked_to_gpu": leak_detected,
        "gpu_request_contains_schedule_secret": leak_detected,
        # honest bottleneck localisation
        "bottleneck_stage": agg["bottleneck_stage"],
        "boundary_calls_reduced": agg["boundary_calls_reduced"],
        "boundary_calls_reduction_note": agg.get("boundary_calls_reduction_note"),
        "online_remask_still_performed": True,
        "schedule_used_for_metadata_only": True,
        "worker_timing_requested": bool(args.trace_worker_timings),
        "trace_output_jsonl": (args.trace_output_jsonl
                               if args.trace_decode_steps else None),
    }
    report.update(fields)
    report.update(agg)           # all target per-token metrics
    if args.report_schedule_stats and schedule is not None:
        report["schedule_stats"] = schedule.stats()

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== precomputed obfuscation schedule probe (mock) ===")
    print("precompute=%s slots_precomputed=%s slots_consumed=%s "
          "precompute_latency_s=%s"
          % (report["precompute_obfuscation_schedule"],
             report["schedule_slots_precomputed"],
             report["schedule_slots_consumed"],
             report["schedule_precompute_latency_s"]))
    print("boundary_calls=%s gpu_calls=%s audit_passed=%s leak=%s"
          % (report["boundary_calls"], report["gpu_calls"],
             report["audit_passed"], report["schedule_secret_leaked_to_gpu"]))
    print("schedule_used_for_metadata_only=%s online_remask_still_performed=%s"
          % (report["schedule_used_for_metadata_only"],
             report["online_remask_still_performed"]))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
