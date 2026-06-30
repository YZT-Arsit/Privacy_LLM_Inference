"""Strict AAAI paper-facing gate for the generation benchmark.

Extends :mod:`pllo.benchmarks.paper_facing_generation` with the FULL AAAI contract
the runner and validator enforce: dataset whitelist, greedy + EOS + length config,
real (no dry-run / mock), backend-specific rules, and DEEP validation of the
attached TDX attestation evidence (tee/debug/sim/binding/report_data) plus the
worker-side A_rightmul evidence (zero trusted nonlinear, compatible masks, pad
coverage).

A run is paper-facing only when :func:`aaai_generation_violations` returns ``[]``.
Both backends are allowed (``plaintext_local`` is the GPU baseline; ``folded_remote``
is ours_A_rightmul_TDX_H800); the folded-remote branch carries the heavy security
contract. stdlib only, pure, unit-testable.
"""

from __future__ import annotations

from typing import Any

from pllo.benchmarks.paper_facing_generation import (
    PAPER_FACING_GENERATION_DESIGN, PAPER_FACING_MAX_NEW_TOKENS,
    PAPER_FACING_SEQ_LEN)

__all__ = [
    "AAAI_DATASETS",
    "AAAI_BACKENDS",
    "aaai_attestation_evidence_violations",
    "aaai_generation_violations",
    "is_aaai_paper_facing",
    "aaai_paper_facing_report_fields",
]

AAAI_DATASETS = ("ifeval", "gsm8k", "mt_bench", "humaneval", "mbpp",
                 "sensitive_prompt_1024", "longbench_1024_lite")
AAAI_BACKENDS = ("plaintext_local", "folded_remote")


def _t(v) -> bool:
    return v is True


def aaai_attestation_evidence_violations(
        evidence: Any, *, expected_mr_td: str | None = None) -> list[str]:
    """Validate a TDX attestation evidence dict for an A_rightmul paper-facing run."""
    v: list[str] = []
    if not isinstance(evidence, dict) or not evidence:
        return ["attestation evidence is missing / not a dict"]
    if str(evidence.get("tee") or "").lower() != "tdx":
        v.append("attestation tee=%r (must be tdx)" % evidence.get("tee"))
    if _t(evidence.get("simulated_unsigned")):
        v.append("attestation simulated_unsigned=True (off-TDX / simulated quote "
                 "is not paper-facing)")
    if evidence.get("paper_facing") is False:
        v.append("attestation paper_facing=False")
    if not _t(evidence.get("runtime_hash_binds_nonlinear_backend")):
        v.append("attestation runtime_hash_binds_nonlinear_backend != True")
    if evidence.get("nonlinear_backend") != PAPER_FACING_GENERATION_DESIGN:
        v.append("attestation nonlinear_backend=%r (must be %r)"
                 % (evidence.get("nonlinear_backend"),
                    PAPER_FACING_GENERATION_DESIGN))
    rd, rh = evidence.get("report_data"), evidence.get("runtime_hash")
    if not (rd and rh and str(rd).lower() == str(rh).lower()):
        v.append("attestation report_data != runtime_hash")
    debug = ((evidence.get("tdx") or {}).get("td_attributes") or {}).get("debug")
    if debug is not False:
        v.append("attestation td_attributes.debug=%r (must be false)" % debug)
    if expected_mr_td and str(evidence.get("mr_td") or "").lower() \
            != expected_mr_td.lower():
        v.append("attestation mr_td != expected_mr_td")
    return v


def aaai_generation_violations(report: dict[str, Any], *,
                               evidence: Any = None,
                               expected_mr_td: str | None = None) -> list[str]:
    """Return unmet AAAI paper-facing conditions for a generation ``report``
    (empty == paper-facing). ``evidence`` is the loaded attestation evidence dict
    (folded_remote only)."""
    v: list[str] = []

    ds = report.get("dataset") or report.get("current_dataset")
    if ds not in AAAI_DATASETS:
        v.append("dataset=%r (must be one of %s)" % (ds, list(AAAI_DATASETS)))

    if int(report.get("seq_len") or 0) != PAPER_FACING_SEQ_LEN:
        v.append("seq_len=%s (must be %d)" % (report.get("seq_len"),
                                              PAPER_FACING_SEQ_LEN))
    if int(report.get("max_new_tokens") or 0) != PAPER_FACING_MAX_NEW_TOKENS:
        v.append("max_new_tokens=%s (must be %d)"
                 % (report.get("max_new_tokens"), PAPER_FACING_MAX_NEW_TOKENS))
    if not _t(report.get("stop_on_eos")):
        v.append("stop_on_eos != True (EOS stopping must be ON)")
    if (report.get("decoding") or "greedy") != "greedy":
        v.append("decoding=%r (must be greedy)" % report.get("decoding"))
    if _t(report.get("dry_run")):
        v.append("dry_run=True (must be a real run)")
    if _t(report.get("mock_runtime")):
        v.append("mock_runtime=True (forbidden in paper-facing AAAI)")

    backend = report.get("backend")
    if backend not in AAAI_BACKENDS:
        v.append("backend=%r (must be one of %s)" % (backend, list(AAAI_BACKENDS)))

    if backend == "folded_remote":
        if report.get("nonlinear_backend") != PAPER_FACING_GENERATION_DESIGN:
            v.append("nonlinear_backend=%r (folded_remote must be %r)"
                     % (report.get("nonlinear_backend"),
                        PAPER_FACING_GENERATION_DESIGN))
        if not _t(report.get("tdx_boundary_client")):
            v.append("tdx_boundary_client != True")
        if _t(report.get("full_model_weights_loaded_in_trusted_runtime")):
            v.append("full_model_weights_loaded_in_trusted_runtime=True (the TDX "
                     "guest must not load the full model)")
        # attestation evidence (deep)
        if not _t(report.get("attestation_evidence_attached")):
            v.append("attestation_evidence_attached != True")
        ev_viol = aaai_attestation_evidence_violations(
            evidence, expected_mr_td=expected_mr_td)
        v.extend("attestation: " + x for x in ev_viol)
        # worker health
        wh = report.get("h800_worker_health")
        if not (isinstance(wh, dict) and wh):
            v.append("h800_worker_health unreadable")
        elif wh.get("nonlinear_backend") not in (None,
                                                 PAPER_FACING_GENERATION_DESIGN):
            v.append("worker health nonlinear_backend=%r (must be %r)"
                     % (wh.get("nonlinear_backend"),
                        PAPER_FACING_GENERATION_DESIGN))
        if report.get("h800_worker_tee_used_on_gpu") is not False:
            v.append("h800_worker_tee_used_on_gpu != False")
        # nonlinear accounting
        if (report.get("nonlinear_trusted_calls") or 0) != 0 \
                or report.get("nonlinear_trusted_calls") is None:
            v.append("nonlinear_trusted_calls=%s (must be 0)"
                     % report.get("nonlinear_trusted_calls"))
        if (report.get("trusted_nonlinear_ops_count") or 0) != 0 \
                or report.get("trusted_nonlinear_ops_count") is None:
            v.append("trusted_nonlinear_ops_count=%s (must be 0)"
                     % report.get("trusted_nonlinear_ops_count"))
        if report.get("nonlinear_single_tee_entry_exit") is not True:
            v.append("nonlinear_single_tee_entry_exit != True")
        if not _t(report.get("compatible_masks_verified")):
            v.append("compatible_masks_verified != True")
        if not _t(report.get("base_linear_pad_all_modules_covered")):
            v.append("base_linear_pad_all_modules_covered != True")
        # schedule full-coverage proof (only enforced when a schedule was enabled)
        if report.get("precompute_obfuscation_schedule") and \
                not _t(report.get("schedule_full_coverage_verified")):
            v.append("schedule_full_coverage_verified != True")
        # GPU-staged schedule (only enforced when the run REQUIRES it). The audit
        # must pass and no raw secret may be staged -- never fall back to unsafe.
        if _t(report.get("require_staged_schedule")):
            if not _t(report.get("staged_schedule_used")):
                v.append("require_staged_schedule but staged_schedule_used != True")
            if not _t(report.get("staged_schedule_no_secret_audit_passed")):
                v.append("staged_schedule_no_secret_audit_passed != True")
            for flag in ("gpu_staged_schedule_contains_raw_masks",
                         "gpu_staged_schedule_contains_raw_pad",
                         "gpu_staged_schedule_contains_plaintext_input",
                         "gpu_staged_schedule_contains_token_ids"):
                if report.get(flag) is True:
                    v.append("%s=True (staged schedule must carry no secrets)"
                             % flag)

    elif backend == "plaintext_local":
        if _t(report.get("tdx_boundary_client")):
            v.append("plaintext_local must NOT run as a TDX boundary client")
        if report.get("trusted_runtime") in ("tdx_guest", "real_tdx"):
            v.append("plaintext_local must run on the H800, not in the TDX guest")

    return v


def is_aaai_paper_facing(report: dict[str, Any], *, evidence: Any = None,
                         expected_mr_td: str | None = None) -> bool:
    return not aaai_generation_violations(
        report, evidence=evidence, expected_mr_td=expected_mr_td)


def aaai_paper_facing_report_fields(report: dict[str, Any], *, evidence: Any = None,
                                    expected_mr_td: str | None = None
                                    ) -> dict[str, Any]:
    viol = aaai_generation_violations(report, evidence=evidence,
                                      expected_mr_td=expected_mr_td)
    return {
        "paper_facing_aaai": len(viol) == 0,
        "paper_facing_aaai_requested": True,
        "paper_facing_aaai_violations": viol,
    }
