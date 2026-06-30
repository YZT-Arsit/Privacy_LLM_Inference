"""Paper-facing generation gate for the AAAI A_rightmul + Qwen2.5-7B mainline.

A generation run is *paper-facing* only when it meets the full AAAI contract:

* nonlinear design is **A_rightmul** (the AAAI mainline; ``current`` /
  ``trusted_shortcut`` / ``amulet_secure_R`` are NOT AAAI-default here);
* ``seq_len == 1024`` and ``max_new_tokens == 512``;
* trusted-side **EOS stopping is ON** (``--disable-eos-stop`` is forbidden);
* the run is **real** (``--require-real``; no dry-run stub);
* the backend is ``folded_remote`` in **TDX boundary-client** mode (the trusted
  TDX guest never loads the full 7B weights; the H800 worker does GPU compute);
* a TDX **attestation evidence** file is attached AND its runtime hash binds the
  nonlinear backend (``runtime_hash_binds_nonlinear_backend == True``);
* the H800 worker ``/health`` is readable;
* ``nonlinear_trusted_calls == 0`` (no nonlinear ever crosses the TEE);
* ``compatible_masks_verified == True`` (A_rightmul compatible-mask assumption);
* ``schedule_full_coverage_verified == True``.

This module is stdlib-only and pure so it is unit-testable and shared by every
generation runner. :func:`paper_facing_generation_violations` returns the list of
unmet conditions (empty == paper-facing); a runner sets ``paper_ready=False`` and
exits non-zero when it is non-empty under ``--paper-facing-generation``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PAPER_FACING_GENERATION_DESIGN",
    "PAPER_FACING_SEQ_LEN",
    "PAPER_FACING_MAX_NEW_TOKENS",
    "paper_facing_generation_violations",
    "is_paper_facing_generation",
    "paper_facing_generation_report_fields",
]

PAPER_FACING_GENERATION_DESIGN = "A_rightmul"
PAPER_FACING_SEQ_LEN = 1024
PAPER_FACING_MAX_NEW_TOKENS = 512


def _truthy(v: Any) -> bool:
    return v is True


def paper_facing_generation_violations(report: dict[str, Any]) -> list[str]:
    """Return the list of unmet paper-facing-generation conditions for ``report``
    (empty list == fully paper-facing). Reads only fields the generation runner
    already stamps into its report dict."""
    v: list[str] = []

    nb = report.get("nonlinear_backend")
    if nb != PAPER_FACING_GENERATION_DESIGN:
        v.append("nonlinear_backend=%r (paper-facing generation requires %r)"
                 % (nb, PAPER_FACING_GENERATION_DESIGN))

    if int(report.get("seq_len") or 0) != PAPER_FACING_SEQ_LEN:
        v.append("seq_len=%s (must be %d)"
                 % (report.get("seq_len"), PAPER_FACING_SEQ_LEN))

    if int(report.get("max_new_tokens") or 0) != PAPER_FACING_MAX_NEW_TOKENS:
        v.append("max_new_tokens=%s (must be %d)"
                 % (report.get("max_new_tokens"), PAPER_FACING_MAX_NEW_TOKENS))

    if not _truthy(report.get("stop_on_eos")):
        v.append("stop_on_eos=%s (EOS stopping must be ON; --disable-eos-stop is "
                 "forbidden)" % report.get("stop_on_eos"))

    if _truthy(report.get("dry_run")):
        v.append("dry_run=True (paper-facing generation must be a real run; pass "
                 "--require-real with a real model + worker)")

    if report.get("backend") != "folded_remote":
        v.append("backend=%r (must be folded_remote)" % report.get("backend"))

    if not _truthy(report.get("tdx_boundary_client")):
        v.append("tdx_boundary_client=%s (must run as a TDX boundary client)"
                 % report.get("tdx_boundary_client"))

    if _truthy(report.get("full_model_weights_loaded_in_trusted_runtime")):
        v.append("full_model_weights_loaded_in_trusted_runtime=True (the trusted "
                 "TDX guest must NOT load the full 7B weights)")

    if not _truthy(report.get("attestation_evidence_attached")):
        v.append("attestation_evidence_attached is not True (a real TDX "
                 "attestation evidence file must be attached)")
    if not _truthy(report.get("attestation_runtime_hash_binds_nonlinear_backend")):
        v.append("attestation_runtime_hash_binds_nonlinear_backend is not True "
                 "(the TDX runtime hash must bind the nonlinear backend)")

    wh = report.get("h800_worker_health")
    if not (isinstance(wh, dict) and wh):
        v.append("h800_worker_health is unreadable (the H800 worker /health must "
                 "be reachable)")
    if report.get("h800_worker_tee_used_on_gpu") is not False:
        v.append("h800_worker_tee_used_on_gpu=%s (the GPU worker must NOT be a TEE)"
                 % report.get("h800_worker_tee_used_on_gpu"))

    tc = report.get("nonlinear_trusted_calls")
    if tc is None or int(tc) != 0:
        v.append("nonlinear_trusted_calls=%s (must be 0; no nonlinear may cross "
                 "the TEE)" % tc)

    if not _truthy(report.get("compatible_masks_verified")):
        v.append("compatible_masks_verified=%s (A_rightmul compatible-mask "
                 "assumption must be verified)"
                 % report.get("compatible_masks_verified"))

    if not _truthy(report.get("schedule_full_coverage_verified")):
        v.append("schedule_full_coverage_verified=%s (every generated token must "
                 "consume a fresh obfuscation slot)"
                 % report.get("schedule_full_coverage_verified"))

    return v


def is_paper_facing_generation(report: dict[str, Any]) -> bool:
    return not paper_facing_generation_violations(report)


def paper_facing_generation_report_fields(report: dict[str, Any]
                                          ) -> dict[str, Any]:
    """The audit block a runner stamps when ``--paper-facing-generation`` was
    requested (records the verdict + every unmet condition)."""
    viol = paper_facing_generation_violations(report)
    return {
        "paper_facing_generation": len(viol) == 0,
        "paper_facing_generation_requested": True,
        "paper_facing_generation_violations": viol,
        "paper_facing_generation_contract": {
            "nonlinear_backend": PAPER_FACING_GENERATION_DESIGN,
            "seq_len": PAPER_FACING_SEQ_LEN,
            "max_new_tokens": PAPER_FACING_MAX_NEW_TOKENS,
            "eos_stop": True, "require_real": True,
            "backend": "folded_remote", "tdx_boundary_client": True,
            "attestation_binds_nonlinear_backend": True,
            "worker_health_readable": True, "nonlinear_trusted_calls": 0,
            "compatible_masks_verified": True,
            "schedule_full_coverage_verified": True,
        },
    }
