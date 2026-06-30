"""Build (and optionally execute) the two-design experiment matrix (Task 3).

Plans the FULL experiment suite under BOTH nonlinear designs (``current`` and
``trusted_shortcut``), namespaced per design so they never collide on disk and
TDX evidence is regenerated per design. The planner is pure (no torch); this CLI
wraps it and can optionally execute / resume / verify the plan.

Examples::

    python scripts/run_dual_nonlinear_experiment_matrix.py \\
        --nonlinear-backends current,trusted_shortcut \\
        --model-path <P> --model-name Qwen2.5-7B-Instruct \\
        --base-output-root <R> --outputs-dir outputs/dual_nonlinear \\
        --seq-len 128 --max-new-tokens-list 1,4,8,16 --run-mode plan \\
        --include-build true --include-local-probes true \\
        --include-remote-decode true --include-tdx-lite true \\
        --include-tdx-attested true --include-lora true \\
        --include-public-benchmarks true --include-latency true \\
        --include-security true \\
        --output-json outputs/dual_nonlinear/dual_matrix_plan.json \\
        --output-md outputs/dual_nonlinear/dual_matrix_plan.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.dual_nonlinear_matrix import (  # noqa: E402
    build_matrix_plan,
    iter_commands,
    render_md,
)


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _all_outputs_exist(step) -> bool:
    files = step.get("expected_output_files") or []
    return bool(files) and all(Path(f).exists() for f in files)


def _missing_outputs(step):
    return [f for f in (step.get("expected_output_files") or [])
            if not Path(f).exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nonlinear-backends", default="A_rightmul,amulet_secure_R")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--base-output-root", required=True)
    ap.add_argument("--outputs-dir", default="outputs/dual_nonlinear")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-new-tokens-list", default="1,4,8,16")
    ap.add_argument("--run-mode", default="plan",
                    choices=["plan", "execute", "resume", "verify-only"])
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--attestation-evidence", default=None)
    ap.add_argument("--include-build", default="false")
    ap.add_argument("--include-local-probes", default="false")
    ap.add_argument("--include-remote-decode", default="false")
    ap.add_argument("--include-tdx-lite", default="false")
    ap.add_argument("--include-tdx-attested", default="false")
    ap.add_argument("--include-lora", default="false")
    ap.add_argument("--include-public-benchmarks", default="false")
    ap.add_argument("--include-latency", default="false")
    ap.add_argument("--include-security", default="false")
    ap.add_argument("--keep-going", action="store_true",
                    help="in execute/resume, do not stop on first failure")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    mnt = [int(t) for t in str(args.max_new_tokens_list).split(",")
           if str(t).strip()]
    include = {
        "build": _bool(args.include_build),
        "local_probes": _bool(args.include_local_probes),
        "remote_decode": _bool(args.include_remote_decode),
        "tdx_lite": _bool(args.include_tdx_lite),
        "tdx_attested": _bool(args.include_tdx_attested),
        "lora": _bool(args.include_lora),
        "public_benchmarks": _bool(args.include_public_benchmarks),
        "latency": _bool(args.include_latency),
        "security": _bool(args.include_security),
    }

    plan = build_matrix_plan(
        nonlinear_backends=args.nonlinear_backends,
        model_path=args.model_path or "<MODEL_PATH>",
        model_name=args.model_name,
        base_output_root=args.base_output_root,
        outputs_dir=args.outputs_dir, seq_len=args.seq_len,
        max_new_tokens_list=mnt, run_mode=args.run_mode, include=include,
        gpu_worker_url=args.gpu_worker_url,
        expected_mr_td=args.expected_mr_td,
        attestation_evidence=args.attestation_evidence)

    # Always write the plan JSON (and MD if requested).
    if args.output_json:
        _write_json(args.output_json, plan)
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_md(plan), encoding="utf-8")

    if args.run_mode == "plan":
        print("=== dual-nonlinear experiment matrix (plan) ===")
        print("backends=%s total_steps=%d"
              % (",".join(plan["nonlinear_backends"]),
                 plan["total_step_count"]))
        for backend in plan["nonlinear_backends"]:
            print("  %s: %d steps -> %s"
                  % (backend, len(plan["per_backend"][backend]["steps"]),
                     plan["per_backend"][backend]["namespaced_paths"][
                         "outputs_dir"]))
        if args.output_json:
            print("plan_json=%s" % args.output_json)
        if args.output_md:
            print("plan_md=%s" % args.output_md)
        return 0

    if args.run_mode == "verify-only":
        results = []
        any_missing = False
        for backend, step in iter_commands(plan):
            missing = _missing_outputs(step)
            if missing:
                any_missing = True
            results.append({"backend": backend, "id": step["id"],
                            "ok": not missing, "missing_files": missing})
        report = {"stage": "dual_nonlinear_experiment_matrix_verify",
                  "run_mode": "verify-only", "results": results,
                  "any_missing": any_missing}
        if args.output_json:
            _write_json(args.output_json + ".verify.json", report)
        print("=== verify-only ===")
        for r in results:
            print("  [%s] %s: %s%s"
                  % (r["backend"], r["id"], "OK" if r["ok"] else "MISSING",
                     "" if r["ok"] else " " + ",".join(r["missing_files"])))
        # verify-only never executes; it reports and returns 0.
        return 0

    # execute / resume
    exec_results = []
    failed = False
    for backend, step in iter_commands(plan):
        if args.run_mode == "resume" and _all_outputs_exist(step):
            exec_results.append({"backend": backend, "id": step["id"],
                                 "status": "skipped", "returncode": 0})
            print("SKIP [%s] %s (outputs exist)" % (backend, step["id"]))
            continue
        print("RUN  [%s] %s :: %s" % (backend, step["id"], step["command"]))
        proc = subprocess.run(step["command"], shell=True)
        rc = proc.returncode
        exec_results.append({"backend": backend, "id": step["id"],
                             "status": "ok" if rc == 0 else "failed",
                             "returncode": rc, "command": step["command"]})
        if rc != 0:
            failed = True
            print("FAIL [%s] %s rc=%d" % (backend, step["id"], rc))
            if not args.keep_going:
                break

    report = {"stage": "dual_nonlinear_experiment_matrix_execution",
              "run_mode": args.run_mode, "keep_going": bool(args.keep_going),
              "results": exec_results,
              "failed": failed}
    if args.output_json:
        _write_json(args.output_json + ".execution.json", report)
    print("=== %s done: %d steps, failed=%s ==="
          % (args.run_mode, len(exec_results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
