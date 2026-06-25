"""One-command E6 private-LoRA real-run pipeline for the H800 GPU host.

Chains the validated E6 steps so a real run needs minimal manual intervention:

1. build a synthetic or HF folded-LoRA package (trusted setup);
2. verify the folded-LoRA package (no raw A/B, no optimizer/training/mask secrets,
   target coverage, base-manifest compatibility);
3. H800 local folded-LoRA correctness probe (base+raw-LoRA reference vs folded);
4. start-or-check the remote folded worker (base package + folded-LoRA package);
5. H800 remote folded-LoRA decode probe (TDX-lite boundary client) vs the local
   reference token ids;
6. optionally emit TDX-lite replay inputs (input_ids + expected tokens + command);
7. write ONE consolidated JSON + Markdown report.

Nothing here weakens the threat model: the worker only ever loads the public base
folded package + the folded-LoRA operators; raw A/B, optimizer state, training
data, mask secrets, and input ids stay trusted-side.

``--plan-only`` prints the exact command plan (every sub-command, fully resolved)
without executing -- use it to review a real run before paying for it.
``--dry-run`` runs the whole chain tiny on CPU (starts a tiny CPU worker), never a
paper result.

Required real-run example::

    python scripts/run_e6_lora_real_h800_pipeline.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --model-name Qwen2.5-7B-Instruct \\
        --base-folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --embedding-artifact-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_cuda \\
        --lora-mode synthetic --lora-rank 4 --lora-alpha 8 \\
        --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\
        --output-lora-package /root/autodl-tmp/privacy_llm_packages/qwen7b_lora_folded_synth_r4 \\
        --gpu-worker-url http://127.0.0.1:18083 --listen-port 18083 \\
        --seq-len 128 --max-new-tokens 4 --dtype bfloat16 --device cuda \\
        --audit true \\
        --output-json outputs/e6_lora_real_h800_pipeline.json \\
        --output-md outputs/e6_lora_real_h800_pipeline.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_designs import (  # noqa: E402
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Plan construction (pure: no side effects, so --plan-only + tests can use it)
# ---------------------------------------------------------------------------


def build_plan(args) -> list:
    work = Path(args.work_dir)
    out_dir = Path(args.output_json).parent if args.output_json else Path("outputs")
    dry = bool(args.dry_run)
    lora_hf = (args.lora_mode == "hf")
    tm = args.target_modules
    nb = args.nonlinear_backend

    paths = {
        "build": (work / "e6_build.json").as_posix(),
        "verify": (work / "e6_verify.json").as_posix(),
        "local": (work / "e6_local_probe.json").as_posix(),
        "remote": (work / "e6_remote_decode.json").as_posix(),
        "validate": (work / "e6_validate_effect.json").as_posix(),
    }

    def _model_args():
        a = []
        if args.model_path and not dry:
            a += ["--model-path", args.model_path]
        a += ["--model-name", args.model_name]
        return a

    # 1. build folded-LoRA package
    build_argv = ["build_qwen7b_lora_folded_package.py"] + _model_args() + [
        "--base-folded-package-path", args.base_folded_package_path,
        "--target-modules", tm, "--rank", str(args.lora_rank),
        "--alpha", str(args.lora_alpha), "--dtype", args.dtype,
        "--device", args.device, "--output-dir", args.output_lora_package,
        "--nonlinear-backend", nb,
        "--output-json", paths["build"]]
    if lora_hf:
        build_argv += ["--raw-lora-adapter-path", args.raw_lora_adapter_path,
                       "--adapter-format", args.adapter_format]
    if dry:
        build_argv += ["--dry-run"]

    # 2. verify folded-LoRA package
    verify_argv = ["verify_qwen7b_lora_folded_package.py",
                   "--lora-folded-package-path", args.output_lora_package,
                   "--base-folded-package-path", args.base_folded_package_path,
                   "--expected-nonlinear-backend", nb,
                   "--output-json", paths["verify"]]

    # 3. H800 local folded-LoRA correctness probe
    local_argv = ["run_qwen7b_lora_folded_local_probe.py"] + _model_args() + [
        "--base-folded-package-path", args.base_folded_package_path,
        "--target-modules", tm, "--rank", str(args.lora_rank),
        "--alpha", str(args.lora_alpha), "--max-new-tokens",
        str(args.max_new_tokens), "--seq-len", str(args.seq_len),
        "--dtype", args.dtype, "--device", args.device,
        "--nonlinear-backend", nb,
        "--output-json", paths["local"]]
    if lora_hf:
        local_argv += ["--adapter-path", args.raw_lora_adapter_path]
    if dry:
        local_argv += ["--dry-run"]

    # 5. H800 remote folded-LoRA decode probe (TDX-lite boundary client)
    remote_argv = ["run_qwen7b_lora_folded_remote_decode_probe.py",
                   "--gpu-worker-url", args.gpu_worker_url,
                   "--model-name", args.model_name,
                   "--embedding-path", args.embedding_artifact_path,
                   "--input-ids-file", paths["local"],
                   "--expected-token-ids-file", paths["local"],
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--seq-len", str(args.seq_len), "--dtype", args.dtype,
                   "--device", args.boundary_device, "--audit",
                   str(args.audit), "--nonlinear-backend", nb,
                   "--output-json", paths["remote"]]
    if dry:
        remote_argv += ["--dry-run"]

    plan = [
        {"name": "build_lora_package", "kind": "script", "argv": build_argv,
         "output_json": paths["build"], "optional": False,
         "description": "fold the %s LoRA adapter into the base masked basis"
                        % args.lora_mode},
        {"name": "verify_lora_package", "kind": "script", "argv": verify_argv,
         "output_json": paths["verify"], "optional": False,
         "description": "verify no raw A/B, no optimizer/training/mask secrets, "
                        "target coverage, base-manifest compatibility"},
        {"name": "local_lora_probe", "kind": "script", "argv": local_argv,
         "output_json": paths["local"], "optional": False,
         "description": "H800 local correctness: base+raw-LoRA vs base-folded+"
                        "folded-LoRA (allclose / top1 / tokens_exact_match)"},
        {"name": "worker_check_or_start", "kind": "worker", "argv": None,
         "output_json": None, "optional": False,
         "description": "start-or-check the remote worker with base package + "
                        "folded-LoRA package at %s" % args.gpu_worker_url},
        {"name": "remote_lora_decode", "kind": "script", "argv": remote_argv,
         "output_json": paths["remote"], "optional": False,
         "description": "H800 remote folded-LoRA decode (TDX-lite boundary) vs "
                        "the local reference token ids"},
    ]

    if args.no_lora_decode_json:
        plan.append({
            "name": "validate_lora_effect", "kind": "script",
            "argv": ["validate_lora_effect.py", "--no-lora-json",
                     args.no_lora_decode_json, "--lora-json", paths["remote"],
                     "--output-json", paths["validate"]],
            "output_json": paths["validate"], "optional": True,
            "description": "compare LoRA vs no-LoRA tokens (effect observed?)"})

    if _bool(args.emit_tdx_inputs):
        plan.append({
            "name": "prepare_tdx_lite_inputs", "kind": "script",
            "argv": ["prepare_tdx_lora_lite_inputs.py", "--reference-json",
                     paths["remote"], "--embedding-path",
                     args.embedding_artifact_path,
                     "--lora-folded-package-path", args.output_lora_package,
                     "--gpu-worker-url", args.gpu_worker_url, "--max-new-tokens",
                     str(args.max_new_tokens), "--dtype", args.dtype,
                     "--output-dir", out_dir.as_posix()],
            "output_json": (out_dir / "tdx_lora_replay.json").as_posix(),
            "optional": True,
            "description": "emit TDX-lite replay input_ids + expected tokens + "
                           "run_tdx_lora_lite_decode.sh"})
    return plan


def _resolved_cmd(step):
    if step["kind"] != "script":
        return None
    return [sys.executable, str(SCRIPTS / step["argv"][0])] + step["argv"][1:]


# ---------------------------------------------------------------------------
# Worker start-or-check
# ---------------------------------------------------------------------------


def _worker_healthy(url, timeout=3.0):
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health",
                                    timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def _start_worker(args):
    """Spawn the demo gpu_worker_server (base + folded LoRA) as a child process."""
    port = args.listen_port
    argv = [sys.executable, str(SCRIPTS / "run_tee_gpu_protocol_demo.py"),
            "--mode", "gpu_worker_server", "--gpu-backend",
            "qwen7b_folded_package", "--folded-package-path",
            args.base_folded_package_path, "--folded-lora-package-path",
            args.output_lora_package, "--listen-host", "127.0.0.1",
            "--listen-port", str(port), "--device", args.device,
            "--dtype", args.dtype, "--audit", str(args.audit),
            "--nonlinear-backend", args.nonlinear_backend]
    if args.model_path and not args.dry_run:
        argv += ["--model-path", args.model_path]
    proc = subprocess.Popen(argv)
    return proc


def _ensure_worker(args, log):
    health = _worker_healthy(args.gpu_worker_url)
    if health is not None:
        log("worker already healthy: %s" % json.dumps(health))
        return None, True
    if not _bool(args.start_worker):
        log("worker NOT reachable at %s and --start-worker is false"
            % args.gpu_worker_url)
        return None, False
    log("starting worker (base + folded LoRA) on port %d ..." % args.listen_port)
    proc = _start_worker(args)
    deadline = time.time() + float(args.worker_start_timeout)
    while time.time() < deadline:
        if proc.poll() is not None:
            log("worker process exited early (rc=%s)" % proc.returncode)
            return proc, False
        health = _worker_healthy(args.gpu_worker_url)
        if health is not None:
            log("worker healthy: %s" % json.dumps(health))
            return proc, True
        time.sleep(1.0)
    log("worker did not become healthy within %ss" % args.worker_start_timeout)
    return proc, False


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def consolidate(args, results: dict, step_status: dict) -> dict:
    build = results.get("build_lora_package")
    verify = results.get("verify_lora_package")
    local = results.get("local_lora_probe")
    remote = results.get("remote_lora_decode")
    validate = results.get("validate_lora_effect")

    lora_tokens = (_g(remote, "package_token_ids")
                   or _g(local, "package_token_ids"))
    no_lora_tokens = (_g(validate, "no_lora_token_ids"))
    lora_differs = _g(validate, "tokens_differ")

    rep = {
        "stage": "e6_lora_real_h800_pipeline",
        "dry_run": bool(args.dry_run),
        "model_name": args.model_name,
        "lora_mode": args.lora_mode,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "output_lora_package": args.output_lora_package,
        "gpu_worker_url": args.gpu_worker_url,
        "step_status": step_status,
        # ---- package build/verify ----
        "lora_package_built": bool(_g(build, "lora_package_built", default=False)),
        "lora_package_valid": bool(_g(verify, "lora_package_valid",
                                      default=False)),
        "contains_raw_lora": bool(_g(verify, "contains_raw_lora",
                                     default=_g(build, "contains_raw_lora",
                                                default=True))),
        "contains_optimizer_state": bool(_g(
            verify, "contains_optimizer_state",
            default=_g(build, "contains_optimizer_state", default=True))),
        "contains_training_data": bool(_g(
            verify, "contains_training_data",
            default=_g(build, "contains_training_data", default=True))),
        "contains_mask_secrets": bool(_g(
            verify, "contains_mask_secrets",
            default=_g(build, "contains_mask_secrets", default=True))),
        # ---- worker / remote decode ----
        "lora_enabled": _g(remote, "lora_enabled"),
        "folded_lora_loaded": _g(remote, "folded_lora_loaded"),
        "folded_lora_valid": _g(remote, "folded_lora_valid"),
        "local_lora_probe_passed": step_status.get("local_lora_probe") == "ok",
        "remote_lora_decode_passed": step_status.get("remote_lora_decode")
        == "ok",
        "tokens_exact_match": _g(remote, "tokens_exact_match"),
        "token_match_rate": _g(remote, "token_match_rate"),
        "lora_tokens": lora_tokens,
        "no_lora_tokens": no_lora_tokens,
        "lora_differs_from_no_lora": lora_differs,
        # ---- security ----
        "worker_has_raw_lora": bool(_g(remote, "worker_has_raw_lora",
                                       default=False)),
        "worker_has_mask_secrets": bool(_g(remote, "worker_has_mask_secrets",
                                           default=False)),
        "tee_used_on_gpu": bool(_g(remote, "tee_used_on_gpu", default=False)),
        "gpu_visible_plaintext_fields": _g(remote, "gpu_visible_plaintext_fields",
                                           default=[]),
        "leaked_secret_fields": _g(remote, "leaked_secret_fields", default=[]),
        "audit_passed": _g(remote, "audit_passed"),
        # ---- cost ----
        "latency_s": _g(remote, "latency_s"),
        "trusted_bytes": _g(remote, "trusted_bytes"),
        "gpu_bytes": _g(remote, "gpu_bytes"),
        "boundary_calls": _g(remote, "boundary_calls"),
        "peak_gpu_memory_mb": _g(remote, "peak_gpu_memory_mb"),
        # local probe correctness echo
        "local_allclose": _g(local, "allclose"),
        "local_max_abs_error": _g(local, "max_abs_error"),
        "local_tokens_exact_match": _g(local, "tokens_exact_match"),
    }
    rep.update(nonlinear_design_report_fields(args.nonlinear_backend))

    security_ok = (rep["worker_has_raw_lora"] is False
                   and rep["worker_has_mask_secrets"] is False
                   and rep["tee_used_on_gpu"] is False
                   and not rep["gpu_visible_plaintext_fields"]
                   and not rep["leaked_secret_fields"]
                   and not rep["contains_raw_lora"]
                   and not rep["contains_optimizer_state"]
                   and not rep["contains_training_data"]
                   and not rep["contains_mask_secrets"])
    correctness_ok = (rep["lora_package_built"] and rep["lora_package_valid"]
                      and rep["local_lora_probe_passed"]
                      and rep["remote_lora_decode_passed"]
                      and rep["tokens_exact_match"] is not False)
    rep["security_ok"] = bool(security_ok)
    rep["pipeline_passed"] = bool(security_ok and correctness_ok)
    return rep


def _write_md(path, rep, plan):
    L = ["# E6 private-LoRA real-run pipeline (%s)"
         % ("dry-run" if rep["dry_run"] else "H800"), "",
         "- model_name=`%s`  lora_mode=%s  rank=%s  alpha=%s"
         % (rep["model_name"], rep["lora_mode"], rep["lora_rank"],
            rep["lora_alpha"]),
         "- target_modules=`%s`" % rep["target_modules"],
         "- output_lora_package=`%s`" % rep["output_lora_package"], "",
         "## Step status", ""]
    for name, st in rep["step_status"].items():
        L.append("- %s: **%s**" % (name, st))
    L += ["", "## Result", "",
          "- lora_package_built=%s  lora_package_valid=%s"
          % (rep["lora_package_built"], rep["lora_package_valid"]),
          "- lora_enabled=%s folded_lora_loaded=%s folded_lora_valid=%s"
          % (rep["lora_enabled"], rep["folded_lora_loaded"],
             rep["folded_lora_valid"]),
          "- local_lora_probe_passed=%s remote_lora_decode_passed=%s"
          % (rep["local_lora_probe_passed"], rep["remote_lora_decode_passed"]),
          "- **tokens_exact_match=%s** token_match_rate=%s"
          % (rep["tokens_exact_match"], rep["token_match_rate"]),
          "- lora_tokens=%s" % rep["lora_tokens"],
          "- no_lora_tokens=%s lora_differs_from_no_lora=%s"
          % (rep["no_lora_tokens"], rep["lora_differs_from_no_lora"]),
          "- worker_has_raw_lora=%s worker_has_mask_secrets=%s tee_used_on_gpu=%s"
          % (rep["worker_has_raw_lora"], rep["worker_has_mask_secrets"],
             rep["tee_used_on_gpu"]),
          "- contains_raw_lora=%s contains_optimizer_state=%s "
          "contains_training_data=%s contains_mask_secrets=%s"
          % (rep["contains_raw_lora"], rep["contains_optimizer_state"],
             rep["contains_training_data"], rep["contains_mask_secrets"]),
          "- gpu_visible_plaintext_fields=%s leaked_secret_fields=%s"
          % (rep["gpu_visible_plaintext_fields"] or "[]",
             rep["leaked_secret_fields"] or "[]"),
          "- audit_passed=%s  latency_s=%s  peak_gpu_memory_mb=%s"
          % (rep["audit_passed"], rep["latency_s"], rep["peak_gpu_memory_mb"]),
          "- trusted_bytes=%s gpu_bytes=%s boundary_calls=%s"
          % (rep["trusted_bytes"], rep["gpu_bytes"], rep["boundary_calls"]),
          "", "## Verdict", "",
          "- security_ok=%s" % rep["security_ok"],
          "- **pipeline_passed=%s**" % rep["pipeline_passed"], ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")


def _write_plan(path, plan):
    out = {"stage": "e6_lora_real_h800_pipeline_plan",
           "steps": [{"name": s["name"], "kind": s["kind"],
                      "optional": s["optional"],
                      "description": s["description"],
                      "command": (" ".join([sys.executable, "scripts/"
                                            + s["argv"][0]] + s["argv"][1:])
                                  if s["kind"] == "script" else None),
                      "output_json": s["output_json"]} for s in plan]}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--base-folded-package-path", required=True)
    ap.add_argument("--embedding-artifact-path", required=True)
    ap.add_argument("--lora-mode", default="synthetic",
                    choices=["synthetic", "hf"])
    ap.add_argument("--raw-lora-adapter-path", default=None,
                    help="required for --lora-mode hf")
    ap.add_argument("--adapter-format", default="hf_peft",
                    choices=["hf_peft", "auto"])
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--target-modules",
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,"
                            "down_proj")
    ap.add_argument("--output-lora-package", required=True)
    ap.add_argument("--gpu-worker-url", default="http://127.0.0.1:18083")
    ap.add_argument("--listen-port", type=int, default=18083)
    ap.add_argument("--start-worker", default="false",
                    help="spawn the worker if --gpu-worker-url is unreachable")
    ap.add_argument("--worker-start-timeout", type=float, default=180.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda", help="worker / local-probe device")
    ap.add_argument("--boundary-device", default="cpu",
                    help="TDX-lite boundary device for the remote probe")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--no-lora-decode-json", default=None,
                    help="prior no-LoRA decode JSON -> run validate_lora_effect")
    ap.add_argument("--emit-tdx-inputs", default="true")
    ap.add_argument("--work-dir", default=None,
                    help="intermediate step JSONs (default: <outputs>/e6_pipeline)")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json", default="outputs/e6_lora_real_h800_pipeline.json")
    ap.add_argument("--output-md", default="outputs/e6_lora_real_h800_pipeline.md")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    if args.lora_mode == "hf" and not args.raw_lora_adapter_path:
        ap.error("--lora-mode hf requires --raw-lora-adapter-path")
    if args.work_dir is None:
        base = Path(args.output_json).parent if args.output_json else Path("outputs")
        args.work_dir = str(base / "e6_pipeline")

    plan = build_plan(args)

    if args.plan_only:
        out = _write_plan(args.output_json or "outputs/e6_lora_plan.json", plan)
        print("=== E6 LoRA real-run pipeline PLAN (%d steps) ===" % len(plan))
        for s in out["steps"]:
            print("\n[%s]%s %s" % (s["name"], " (optional)" if s["optional"]
                                   else "", s["description"]))
            if s["command"]:
                print("  $ " + s["command"])
        return 0

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    def log(msg):
        print("[pipeline] " + msg, flush=True)

    results: dict = {}
    step_status: dict = {}
    worker_proc = None
    failed = False
    try:
        for step in plan:
            name = step["name"]
            if failed and not args.continue_on_error and not step["optional"]:
                step_status[name] = "skipped"
                continue
            if step["kind"] == "worker":
                worker_proc, ok = _ensure_worker(args, log)
                step_status[name] = "ok" if ok else "fail"
                if not ok and not args.continue_on_error:
                    failed = True
                continue
            cmd = _resolved_cmd(step)
            log("running %s: %s" % (name, " ".join(cmd)))
            t0 = time.time()
            rc = subprocess.call(cmd)
            dt = round(time.time() - t0, 2)
            if step["output_json"]:
                results[name] = _load_json(step["output_json"])
            if rc == 0:
                step_status[name] = "ok"
                log("%s OK (%.1fs)" % (name, dt))
            else:
                step_status[name] = "fail"
                log("%s FAILED rc=%d (%.1fs)" % (name, rc, dt))
                if not step["optional"]:
                    failed = True
    finally:
        if worker_proc is not None:
            log("terminating worker started by the pipeline")
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=15)
            except Exception:                               # noqa: BLE001
                worker_proc.kill()

    rep = consolidate(args, results, step_status)
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(args.output_md, rep, plan)

    print("\n=== E6 LoRA pipeline consolidated ===")
    for name, st in step_status.items():
        print("  %-22s %s" % (name, st))
    print("lora_package_built=%s lora_package_valid=%s lora_enabled=%s"
          % (rep["lora_package_built"], rep["lora_package_valid"],
             rep["lora_enabled"]))
    print("tokens_exact_match=%s token_match_rate=%s"
          % (rep["tokens_exact_match"], rep["token_match_rate"]))
    print("security_ok=%s pipeline_passed=%s"
          % (rep["security_ok"], rep["pipeline_passed"]))
    print("\nE6 LoRA PIPELINE %s"
          % ("PASSED" if rep["pipeline_passed"] else "FAILED"))
    return 0 if rep["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
