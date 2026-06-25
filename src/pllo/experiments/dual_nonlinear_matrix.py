"""Two-design experiment matrix planner (Task 3).

The advisor requires the *full* experiment suite to run under BOTH nonlinear
designs (``current`` and ``trusted_shortcut``) so either can be chosen later
with complete results. This module is a pure, stdlib-only *planner*: given a set
of nonlinear backends and a set of include flags, it builds an ordered,
namespaced plan of shell commands that map onto the real repo scripts, with the
``--nonlinear-backend {backend}`` flag wired in where applicable.

Crucially, every artifact path is namespaced by the nonlinear design so the two
designs never collide on disk, and every TDX-attested step is flagged so its
runtime hash + quote is regenerated per design (the design metadata hash binds
into the attestation runtime hash -- design A evidence cannot be reused for
design B).

No torch / model / GPU / network imports here. The module only *builds* the
plan; the runner script may optionally execute it via subprocess.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from pllo.experiments.nonlinear_designs import (
    nonlinear_design_report_fields,
    parse_nonlinear_backends,
)

__all__ = [
    "MATRIX_STAGE",
    "INCLUDE_FLAGS",
    "default_include",
    "namespaced_paths",
    "build_backend_steps",
    "build_matrix_plan",
    "render_md",
    "iter_commands",
]

MATRIX_STAGE = "dual_nonlinear_experiment_matrix"

# All include-flag keys (every one defaults False; the runner/caller opts in).
INCLUDE_FLAGS = (
    "build",
    "local_probes",
    "remote_decode",
    "tdx_lite",
    "tdx_attested",
    "lora",
    "public_benchmarks",
    "latency",
    "security",
)

# Public benchmark dataset placeholders for E9 task-utility runs.
_E9_DATASETS = ("mmlu", "gsm8k", "boolq", "sst2")


def default_include() -> Dict[str, bool]:
    """A fresh include-dict with every flag False."""
    return {k: False for k in INCLUDE_FLAGS}


def _norm_include(include: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    out = default_include()
    if include:
        for k, v in include.items():
            if k in out:
                out[k] = bool(v)
    return out


def namespaced_paths(backend: str, *, base_output_root: str,
                     outputs_dir: str) -> Dict[str, str]:
    """The per-backend namespaced artifact paths (EXACT name templates)."""
    return {
        "base_folded_package_dir":
            "%s/qwen7b_folded_full_%s" % (base_output_root, backend),
        "boundary_artifact":
            "%s/qwen7b_boundary_artifact_%s" % (base_output_root, backend),
        "lora_folded_package_dir":
            "%s/qwen7b_lora_folded_%s" % (base_output_root, backend),
        "outputs_dir": "%s/%s" % (outputs_dir, backend),
    }


def _step(step_id: str, title: str, backend: str, command: str, *,
          expected_output_files: Optional[List[str]] = None,
          required_input_files: Optional[List[str]] = None,
          required_server_state: str = "none",
          side: str = "local",
          tdx_evidence_must_be_regenerated: bool = False) -> Dict[str, Any]:
    return {
        "id": step_id,
        "title": title,
        "backend": backend,
        "command": command,
        "expected_output_files": list(expected_output_files or []),
        "required_input_files": list(required_input_files or []),
        "required_server_state": required_server_state,
        "side": side,
        "tdx_evidence_must_be_regenerated": bool(
            tdx_evidence_must_be_regenerated),
    }


def build_backend_steps(backend: str, *, model_path: str, model_name: str,
                        base_output_root: str, outputs_dir: str,
                        seq_len: int, max_new_tokens_list: List[int],
                        include: Dict[str, bool],
                        gpu_worker_url: Optional[str] = None,
                        expected_mr_td: Optional[str] = None,
                        attestation_evidence: Optional[str] = None,
                        ) -> List[Dict[str, Any]]:
    """Ordered list of steps (only those whose include flag is on) for ONE
    nonlinear backend, with paths namespaced by ``backend``."""
    inc = _norm_include(include)
    paths = namespaced_paths(backend, base_output_root=base_output_root,
                             outputs_dir=outputs_dir)
    base_pkg = paths["base_folded_package_dir"]
    boundary = paths["boundary_artifact"]
    lora_pkg = paths["lora_folded_package_dir"]
    out = paths["outputs_dir"]
    nb = "--nonlinear-backend %s" % backend
    mnt = ",".join(str(t) for t in max_new_tokens_list)
    worker = gpu_worker_url or "http://127.0.0.1:8800"

    steps: List[Dict[str, Any]] = []

    # 1. build folded package
    if inc["build"]:
        manifest = "%s/manifest.json" % base_pkg
        steps.append(_step(
            "build_folded_package", "Build folded base package", backend,
            "python scripts/build_qwen7b_folded_package.py %s "
            "--output-dir %s --model-path %s --model-name %s "
            "--seq-len %d --output-json %s/build_folded_package.json"
            % (nb, base_pkg, model_path, model_name, seq_len, out),
            expected_output_files=[manifest,
                                   "%s/build_folded_package.json" % out],
            required_input_files=[model_path],
            side="trusted_host"))
        # 2. verify folded package
        steps.append(_step(
            "verify_folded_package", "Verify folded base package", backend,
            "python scripts/verify_folded_package.py "
            "--package-path %s --expected-nonlinear-backend %s "
            "--output-json %s/verify_folded_package.json"
            % (base_pkg, backend, out),
            expected_output_files=["%s/verify_folded_package.json" % out],
            required_input_files=[manifest],
            side="trusted_host"))

    # 3. local prefill/logits/decode probes
    if inc["local_probes"]:
        for sid, script, title in (
            ("local_prefill_probe",
             "run_qwen7b_folded_package_prefill_probe.py",
             "Local prefill probe"),
            ("local_logits_probe",
             "run_qwen7b_folded_package_onestep_logits_probe.py",
             "Local one-step logits probe"),
            ("local_decode_probe",
             "run_qwen7b_folded_package_decode_probe.py",
             "Local decode probe"),
        ):
            steps.append(_step(
                sid, title, backend,
                "python scripts/%s %s --package-path %s --model-path %s "
                "--seq-len %d --output-json %s/%s.json"
                % (script, nb, base_pkg, model_path, seq_len, out, sid),
                expected_output_files=["%s/%s.json" % (out, sid)],
                required_input_files=["%s/manifest.json" % base_pkg],
                side="local"))

    # 4. remote H800 package-backed decode
    if inc["remote_decode"]:
        steps.append(_step(
            "remote_decode", "Remote H800 package-backed decode", backend,
            "python scripts/run_tee_gpu_protocol_demo.py "
            "--mode boundary_client --gpu-backend qwen7b_folded_package %s "
            "--package-path %s --gpu-worker-url %s --seq-len %d "
            "--max-new-tokens %d --output-json %s/remote_decode.json"
            % (nb, base_pkg, worker, seq_len,
               max(max_new_tokens_list), out),
            expected_output_files=["%s/remote_decode.json" % out],
            required_input_files=["%s/manifest.json" % base_pkg],
            required_server_state="h800_worker_running", side="h800"))

    # 5. tdx-lite decode (boundary backend, no evidence)
    if inc["tdx_lite"]:
        steps.append(_step(
            "tdx_lite_decode", "TDX-lite decode (no evidence)", backend,
            "python scripts/run_tee_gpu_protocol_demo.py "
            "--mode boundary_client --gpu-backend boundary %s "
            "--boundary-artifact %s --seq-len %d --max-new-tokens %d "
            "--output-json %s/tdx_lite_decode.json"
            % (nb, boundary, seq_len, max(max_new_tokens_list), out),
            expected_output_files=["%s/tdx_lite_decode.json" % out],
            required_input_files=[boundary],
            required_server_state="tdx_guest", side="tdx"))

    # 6. tdx-attested decode (preceded by runtime-hash write)
    if inc["tdx_attested"]:
        steps.append(_step(
            "runtime_hash", "Write TEE boundary runtime hash", backend,
            "python scripts/write_tee_boundary_runtime_hash.py %s "
            "--output-json %s/runtime_hash.json" % (nb, out),
            expected_output_files=["%s/runtime_hash.json" % out],
            required_input_files=[],
            required_server_state="tdx_guest", side="tdx",
            tdx_evidence_must_be_regenerated=True))
        ev = attestation_evidence or "%s/attestation_evidence.json" % out
        mrtd = expected_mr_td or "<EXPECTED_MR_TD>"
        steps.append(_step(
            "tdx_attested_decode", "TDX-attested decode", backend,
            "python scripts/run_tee_gpu_protocol_demo.py "
            "--mode boundary_client --gpu-backend boundary %s "
            "--boundary-artifact %s --attestation-evidence %s "
            "--expected-mr-td %s --seq-len %d --max-new-tokens %d "
            "--output-json %s/tdx_attested_decode.json"
            % (nb, boundary, ev, mrtd, seq_len,
               max(max_new_tokens_list), out),
            expected_output_files=["%s/tdx_attested_decode.json" % out],
            required_input_files=[boundary, "%s/runtime_hash.json" % out],
            required_server_state="tdx_quote_bound", side="tdx",
            tdx_evidence_must_be_regenerated=True))

    # 7. E3 scaling
    if inc["remote_decode"]:
        steps.append(_step(
            "e3_scaling", "E3 remote-decode scaling", backend,
            "python scripts/run_e3_remote_decode_scaling.py %s "
            "--package-path %s --max-new-tokens-list %s "
            "--output-json %s/e3_scaling.json"
            % (nb, base_pkg, mnt, out),
            expected_output_files=["%s/e3_scaling.json" % out],
            required_input_files=["%s/manifest.json" % base_pkg],
            required_server_state="h800_worker_running", side="h800"))

    # 8. E4 setup cost
    if inc["build"]:
        steps.append(_step(
            "e4_setup_cost", "E4 setup-cost report", backend,
            "python scripts/run_e4_setup_cost_report.py %s "
            "--package-path %s --output-json %s/e4_setup_cost.json"
            % (nb, base_pkg, out),
            expected_output_files=["%s/e4_setup_cost.json" % out],
            required_input_files=["%s/manifest.json" % base_pkg],
            side="trusted_host"))

    # 9. E5 comparison
    if inc["remote_decode"]:
        steps.append(_step(
            "e5_comparison", "E5 final-comparison report", backend,
            "python scripts/run_e5_final_comparison_report.py %s "
            "--remote-scaling-json %s/e3_scaling.json "
            "--output-json %s/e5_comparison.json "
            "--output-md %s/e5_comparison.md"
            % (nb, out, out, out),
            expected_output_files=["%s/e5_comparison.json" % out],
            required_input_files=["%s/e3_scaling.json" % out],
            side="local"))

    # 10. E6 LoRA package/probe/remote
    if inc["lora"]:
        lora_manifest = "%s/manifest.json" % lora_pkg
        steps.append(_step(
            "lora_build", "Build LoRA folded package", backend,
            "python scripts/build_qwen7b_lora_folded_package.py %s "
            "--output-dir %s --model-path %s --model-name %s "
            "--output-json %s/lora_build.json"
            % (nb, lora_pkg, model_path, model_name, out),
            expected_output_files=[lora_manifest,
                                   "%s/lora_build.json" % out],
            required_input_files=[model_path], side="trusted_host"))
        steps.append(_step(
            "lora_verify", "Verify LoRA folded package", backend,
            "python scripts/verify_qwen7b_lora_folded_package.py "
            "--package-path %s --expected-nonlinear-backend %s "
            "--output-json %s/lora_verify.json"
            % (lora_pkg, backend, out),
            expected_output_files=["%s/lora_verify.json" % out],
            required_input_files=[lora_manifest], side="trusted_host"))
        steps.append(_step(
            "lora_local_probe", "LoRA folded local probe", backend,
            "python scripts/run_qwen7b_lora_folded_local_probe.py %s "
            "--package-path %s --model-path %s "
            "--output-json %s/lora_local_probe.json"
            % (nb, lora_pkg, model_path, out),
            expected_output_files=["%s/lora_local_probe.json" % out],
            required_input_files=[lora_manifest], side="local"))
        steps.append(_step(
            "lora_remote_probe", "LoRA folded remote decode probe", backend,
            "python scripts/run_qwen7b_lora_folded_remote_decode_probe.py %s "
            "--package-path %s --gpu-worker-url %s "
            "--output-json %s/lora_remote_probe.json"
            % (nb, lora_pkg, worker, out),
            expected_output_files=["%s/lora_remote_probe.json" % out],
            required_input_files=[lora_manifest],
            required_server_state="h800_worker_running", side="h800"))

    # 11. E9 public benchmark runs (one per dataset placeholder)
    if inc["public_benchmarks"]:
        for ds in _E9_DATASETS:
            steps.append(_step(
                "e9_benchmark_%s" % ds, "E9 public benchmark: %s" % ds,
                backend,
                "python scripts/run_e9_task_utility_benchmark.py "
                "--require-real %s --dataset %s "
                "--output-json %s/e9_%s.json" % (nb, ds, out, ds),
                expected_output_files=["%s/e9_%s.json" % (out, ds)],
                required_input_files=["%s/manifest.json" % base_pkg],
                required_server_state="h800_worker_running", side="h800"))
        # 12. E9 pairwise utility preservation
        steps.append(_step(
            "e9_pairwise", "E9 pairwise utility preservation", backend,
            "python scripts/run_e9_pairwise_utility_preservation.py "
            "--output-json %s/e9_pairwise.json" % out,
            expected_output_files=["%s/e9_pairwise.json" % out],
            required_input_files=["%s/e9_%s.json" % (out, ds)
                                  for ds in _E9_DATASETS],
            side="local"))

    # 13. E10 LoRA utility
    if inc["lora"]:
        steps.append(_step(
            "e10_lora_utility", "E10 LoRA utility benchmark", backend,
            "python scripts/run_e10_lora_utility_benchmark.py %s "
            "--package-path %s --output-json %s/e10_lora_utility.json"
            % (nb, lora_pkg, out),
            expected_output_files=["%s/e10_lora_utility.json" % out],
            required_input_files=["%s/manifest.json" % lora_pkg],
            side="local"))

    # 14. E12 latency baselines
    if inc["latency"]:
        steps.append(_step(
            "e12_latency", "E12 latency baselines", backend,
            "python scripts/run_e12_latency_baselines.py %s "
            "--package-path %s --output-json %s/e12_latency.json"
            % (nb, base_pkg, out),
            expected_output_files=["%s/e12_latency.json" % out],
            required_input_files=["%s/manifest.json" % base_pkg],
            side="local"))

    # 15. transcript scan
    if inc["security"]:
        steps.append(_step(
            "transcript_scan", "Security transcript scan", backend,
            "python scripts/scan_security_transcript.py "
            "--output-json %s/transcript_scan.json" % out,
            expected_output_files=["%s/transcript_scan.json" % out],
            side="local"))
        # 16. security negative tests
        steps.append(_step(
            "security_negative_tests", "Security negative tests", backend,
            "python scripts/run_security_negative_tests.py "
            "--output-json %s/security_negative_tests.json" % out,
            expected_output_files=["%s/security_negative_tests.json" % out],
            side="local"))

    # 17. deployment truth (always)
    steps.append(_step(
        "deployment_truth", "Deployment-truth check", backend,
        "python scripts/check_deployment_truth.py "
        "--output-json %s/deployment_truth.json" % out,
        expected_output_files=["%s/deployment_truth.json" % out],
        side="local"))

    # 18. claim validator (always; backend-tagged required claims)
    required_claims = "public_benchmark_utility_preserved[%s]" % backend
    steps.append(_step(
        "claim_validator", "Paper claim validator", backend,
        "python scripts/validate_paper_claims.py --inputs-dir %s "
        "--required-claims %s --output-json %s/claim_validator.json"
        % (out, required_claims, out),
        expected_output_files=["%s/claim_validator.json" % out],
        side="local"))

    # 19. final artifact packaging (always)
    steps.append(_step(
        "package_final_artifacts", "Package final artifacts", backend,
        "python scripts/package_final_artifacts.py --inputs-dir %s "
        "--output-json %s/package_final_artifacts.json" % (out, out),
        expected_output_files=["%s/package_final_artifacts.json" % out],
        side="local"))

    return steps


def build_matrix_plan(*, nonlinear_backends, model_path: str,
                      model_name: str, base_output_root: str,
                      outputs_dir: str, seq_len: int,
                      max_new_tokens_list: List[int], run_mode: str,
                      include: Dict[str, bool],
                      gpu_worker_url: Optional[str] = None,
                      expected_mr_td: Optional[str] = None,
                      attestation_evidence: Optional[str] = None,
                      ) -> Dict[str, Any]:
    """Build the full two-design matrix plan (pure; nothing executed)."""
    if isinstance(nonlinear_backends, str):
        backends = parse_nonlinear_backends(nonlinear_backends)
    else:
        backends = parse_nonlinear_backends(
            ",".join(str(b) for b in nonlinear_backends))

    if run_mode not in {"plan", "execute", "resume", "verify-only"}:
        raise ValueError("unknown run_mode %r" % run_mode)

    inc = _norm_include(include)

    per_backend: Dict[str, Any] = {}
    total = 0
    for backend in backends:
        steps = build_backend_steps(
            backend, model_path=model_path, model_name=model_name,
            base_output_root=base_output_root, outputs_dir=outputs_dir,
            seq_len=seq_len, max_new_tokens_list=max_new_tokens_list,
            include=inc, gpu_worker_url=gpu_worker_url,
            expected_mr_td=expected_mr_td,
            attestation_evidence=attestation_evidence)
        total += len(steps)
        per_backend[backend] = {
            "steps": steps,
            "namespaced_paths": namespaced_paths(
                backend, base_output_root=base_output_root,
                outputs_dir=outputs_dir),
            "nonlinear_design": nonlinear_design_report_fields(backend),
        }

    notes = [
        "Full experiment suite runs under EACH nonlinear design so either "
        "can be chosen later with complete results.",
        "All artifact paths are namespaced by nonlinear design "
        "(qwen7b_folded_full_<backend>, outputs/<backend>/...) so the two "
        "designs never collide on disk.",
        "TDX runtime hash + quote must be regenerated separately per "
        "nonlinear design: the design metadata hash binds into the "
        "attestation runtime hash, so design A evidence cannot be reused "
        "for design B.",
    ]
    limitations = [
        "trusted_shortcut (design B) security is not formally claimed "
        "(selector-leakage caveat); only correctness is exact.",
        "Steps reference real H800/TDX/CUDA scripts and require a running "
        "worker / TDX guest / bound quote where indicated; the planner does "
        "not execute them.",
        "E9 public-benchmark datasets are placeholders (mmlu/gsm8k/boolq/"
        "sst2); real --require-real runs need actual datasets + models.",
    ]

    return {
        "stage": MATRIX_STAGE,
        "run_mode": run_mode,
        "nonlinear_backends": backends,
        "model_path": model_path,
        "model_name": model_name,
        "base_output_root": base_output_root,
        "outputs_dir": outputs_dir,
        "seq_len": seq_len,
        "max_new_tokens_list": list(max_new_tokens_list),
        "include": inc,
        "gpu_worker_url": gpu_worker_url,
        "expected_mr_td": expected_mr_td,
        "attestation_evidence": attestation_evidence,
        "per_backend": per_backend,
        "total_step_count": total,
        "notes": notes,
        "limitations": limitations,
    }


def iter_commands(plan: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(backend, step)`` in plan order."""
    for backend in plan.get("nonlinear_backends", []):
        for step in plan["per_backend"][backend]["steps"]:
            yield backend, step


def render_md(plan: Dict[str, Any]) -> str:
    """A per-backend numbered command list (Markdown)."""
    lines: List[str] = []
    lines.append("# Dual-nonlinear experiment matrix")
    lines.append("")
    lines.append("- stage: `%s`" % plan["stage"])
    lines.append("- run_mode: `%s`" % plan["run_mode"])
    lines.append("- nonlinear_backends: %s"
                 % ", ".join("`%s`" % b for b in plan["nonlinear_backends"]))
    lines.append("- total_step_count: %d" % plan["total_step_count"])
    lines.append("")
    for backend in plan["nonlinear_backends"]:
        pb = plan["per_backend"][backend]
        design = pb["nonlinear_design"]
        lines.append("## Design `%s` (%s)"
                     % (backend, design.get("nonlinear_design_label", "")))
        lines.append("")
        lines.append("- security_status: `%s`"
                     % design["nonlinear_design_metadata_summary"][
                         "security_status"])
        lines.append("- outputs dir: `%s`"
                     % pb["namespaced_paths"]["outputs_dir"])
        lines.append("- folded package: `%s`"
                     % pb["namespaced_paths"]["base_folded_package_dir"])
        lines.append("")
        for i, step in enumerate(pb["steps"], 1):
            tag = ""
            if step["tdx_evidence_must_be_regenerated"]:
                tag = "  _(regenerate TDX evidence per design)_"
            lines.append("%d. **%s** [`%s`, side=%s, server=%s]%s"
                         % (i, step["title"], step["id"], step["side"],
                            step["required_server_state"], tag))
            lines.append("   ```")
            lines.append("   %s" % step["command"])
            lines.append("   ```")
        lines.append("")
    if plan.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for n in plan["notes"]:
            lines.append("- %s" % n)
        lines.append("")
    if plan.get("limitations"):
        lines.append("## Limitations")
        lines.append("")
        for n in plan["limitations"]:
            lines.append("- %s" % n)
        lines.append("")
    return "\n".join(lines)
