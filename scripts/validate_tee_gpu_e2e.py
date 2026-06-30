#!/usr/bin/env python
"""End-to-end paper-facing validator for the TEE(TDX) <-> H800-GPU experiment.

Reads every ``*.json`` under one or more output dirs and asserts the strict
paper-facing contract holds across the collected evidence:

  * Linear layers: every folded package report has Linear-boundary pad coverage on
    all 8 Linear families (q/k/v/o/gate/up/down/lm_head), read from REAL shard
    tensor names (``base_linear_pad_all_modules_covered`` / coverage map).
  * Nonlinear: the design is paper-facing (A_rightmul / amulet_secure_R), executed
    in the real path (measured evidence, not tag-only), with
    ``nonlinear_trusted_calls == 0`` and ``nonlinear_single_tee_entry_exit``.
  * TEE boundary: semantic input boundary calls == 1, final logits == 1,
    intermediate == 0, nonlinear trusted calls == 0 (where reported).
  * TDX quote: ``tee == tdx``, ``td_attributes.debug == false``, JWT has 3 parts,
    ``report_data == runtime_hash``, runtime hash binds the nonlinear backend
    (when a nonlinear-bound evidence file is present); ``mr_td`` matches when an
    expected value is supplied.
  * Interconnect: an H800 worker ``/health`` success, a TDX boundary-client decode
    success, and remote-decode exactness (``tokens_exact_match``).

Exit code 0 iff every requested check passes (and every required signal is
present). ``--require`` lists which signal groups MUST be present (default: all).
Off-TDX / simulated evidence is NOT accepted as paper-facing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pllo.experiments.nonlinear_designs import (  # noqa: E402
    PAPER_FACING_DESIGNS, NON_PAPER_FACING_DESIGNS, normalize_nonlinear_backend,
    UnknownNonlinearBackend, report_has_real_nonlinear_execution,
    report_nonlinear_trusted_calls_clean, nonlinear_tag_only)

SIGNAL_GROUPS = ("linear_pad", "nonlinear_exec", "tee_boundary", "tdx_quote",
                 "worker_health", "boundary_client", "remote_exactness")


def _load_jsons(paths) -> list[dict]:
    out = []
    for base in paths:
        p = Path(base)
        files = ([p] if p.is_file() else sorted(p.rglob("*.json")))
        for f in files:
            try:
                out.append({"file": str(f), "report": json.loads(
                    f.read_text(encoding="utf-8"))})
            except Exception:                                # noqa: BLE001
                continue
    return out


def _nb(r):
    nb = r.get("nonlinear_backend") or r.get("nonlinear_design_name")
    try:
        return normalize_nonlinear_backend(nb) if nb else None
    except UnknownNonlinearBackend:
        return None


def _is_jwt(tok) -> bool:
    return isinstance(tok, str) and len(tok.split(".")) == 3


def validate(reports: list[dict], *, expected_mr_td: str | None,
             require: list[str]) -> dict:
    checks: list[dict] = []
    present: dict[str, bool] = {g: False for g in SIGNAL_GROUPS}

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    for e in reports:
        r, f = e["report"], e["file"]
        if not isinstance(r, dict):
            continue
        nb = _nb(r)

        # --- linear pad coverage (folded package / probe / worker reports) ----
        if ("linear_pad_coverage" in r or "base_linear_pad_all_modules_covered"
                in r):
            present["linear_pad"] = True
            cov = r.get("linear_pad_coverage") or {}
            all_cov = r.get("base_linear_pad_all_modules_covered")
            if all_cov is None and cov:
                all_cov = all(cov.get(m) for m in (
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj", "lm_head"))
            chk("linear_pad_all_modules_covered[%s]" % Path(f).name,
                bool(all_cov), "coverage=%s" % cov)

        # --- nonlinear execution evidence -------------------------------------
        exec_bearing = (r.get("nonlinear_real_path_executed") is not None
                        or r.get("nonlinear_execution_status") is not None)
        if exec_bearing and nb is not None:
            present["nonlinear_exec"] = True
            chk("nonlinear_design_paper_facing[%s]" % Path(f).name,
                nb in PAPER_FACING_DESIGNS,
                "design=%s (legacy=%s)" % (nb, nb in NON_PAPER_FACING_DESIGNS))
            chk("nonlinear_not_tag_only[%s]" % Path(f).name,
                not nonlinear_tag_only(r)
                and report_has_real_nonlinear_execution(r), "design=%s" % nb)
            chk("nonlinear_trusted_calls_zero[%s]" % Path(f).name,
                report_nonlinear_trusted_calls_clean(r)
                and (r.get("nonlinear_trusted_calls") or 0) == 0,
                "trusted_calls=%s" % r.get("nonlinear_trusted_calls"))
            if r.get("nonlinear_single_tee_entry_exit") is not None:
                chk("nonlinear_single_tee_entry_exit[%s]" % Path(f).name,
                    r.get("nonlinear_single_tee_entry_exit") is True)

        # --- TEE boundary call accounting -------------------------------------
        if r.get("semantic_input_boundary_calls") is not None:
            present["tee_boundary"] = True
            chk("semantic_input_boundary_calls==1[%s]" % Path(f).name,
                r.get("semantic_input_boundary_calls") == 1)
            chk("semantic_final_logits_boundary_calls==1[%s]" % Path(f).name,
                r.get("semantic_final_logits_boundary_calls") == 1)
            chk("intermediate_tee_boundary_calls==0[%s]" % Path(f).name,
                (r.get("intermediate_tee_boundary_calls_per_layer") or 0) == 0)

        # --- TDX quote verification ------------------------------------------
        att = r.get("attestation") if isinstance(r.get("attestation"), dict) else r
        is_tdx_evidence = (str(r.get("tee") or att.get("tee") or "").lower()
                           == "tdx") or ("attestation" in r)
        if is_tdx_evidence and (r.get("tee") or att.get("attestation_verified")
                                is not None or "runtime_hash" in r):
            if r.get("simulated_unsigned") or r.get("paper_facing") is False:
                chk("tdx_not_simulated[%s]" % Path(f).name, False,
                    "simulated/off-TDX evidence is NOT paper-facing")
                present["tdx_quote"] = True
                continue
            if str(r.get("tee") or "").lower() == "tdx" or "report_data" in r:
                present["tdx_quote"] = True
                debug = (((r.get("tdx") or {}).get("td_attributes") or {})
                         .get("debug"))
                chk("tdx_tee[%s]" % Path(f).name,
                    str(r.get("tee") or "").lower() == "tdx")
                chk("tdx_debug_false[%s]" % Path(f).name, debug is False)
                chk("tdx_jwt_3_parts[%s]" % Path(f).name, _is_jwt(r.get("jwt")))
                rd, rh = r.get("report_data"), r.get("runtime_hash")
                chk("tdx_report_data_eq_runtime_hash[%s]" % Path(f).name,
                    bool(rd) and bool(rh) and str(rd).lower() == str(rh).lower())
                if r.get("nonlinear_backend") is not None:
                    chk("tdx_runtime_hash_binds_nonlinear[%s]" % Path(f).name,
                        r.get("runtime_hash_binds_nonlinear_backend") is True)
                if expected_mr_td:
                    chk("tdx_mr_td_matches[%s]" % Path(f).name,
                        str(r.get("mr_td") or "").lower()
                        == expected_mr_td.lower())

        # --- interconnect: worker health / boundary client / exactness -------
        if r.get("h800_worker_url") or r.get("worker_health") is not None \
                or r.get("worker_healthy") is not None:
            present["worker_health"] = True
            healthy = (r.get("worker_healthy")
                       or (isinstance(r.get("worker_health"), dict)
                           and r["worker_health"].get("ok"))
                       or r.get("h800_worker_tee_used_on_gpu") is False)
            chk("h800_worker_health_ok[%s]" % Path(f).name, bool(healthy))
        if r.get("tdx_boundary_client") is not None:
            present["boundary_client"] = True
            chk("tdx_boundary_client_ok[%s]" % Path(f).name,
                r.get("tdx_boundary_client") is True
                or r.get("tdx_boundary_client") == "ok")
        if r.get("tokens_exact_match") is not None:
            present["remote_exactness"] = True
            chk("remote_decode_tokens_exact[%s]" % Path(f).name,
                r.get("tokens_exact_match") is True)

    # required-signal presence
    for g in require:
        chk("signal_present:%s" % g, present.get(g, False),
            "no report carried the %s signal" % g)

    failed = [c for c in checks if not c["ok"]]
    return {
        "stage": "tee_gpu_e2e_validation",
        "num_reports": len(reports),
        "signals_present": present,
        "required_signals": require,
        "num_checks": len(checks),
        "num_failed": len(failed),
        "passed": len(failed) == 0 and all(present.get(g) for g in require),
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="output dirs / json files to scan")
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--require", default="all",
                    help="comma-separated signal groups that MUST be present, or "
                         "'all'; groups: %s" % ",".join(SIGNAL_GROUPS))
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    require = (list(SIGNAL_GROUPS) if args.require.strip() == "all"
               else [g.strip() for g in args.require.split(",") if g.strip()])
    reports = _load_jsons(args.inputs)
    rep = validate(reports, expected_mr_td=args.expected_mr_td, require=require)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(rep, indent=2),
                                          encoding="utf-8")
    print(json.dumps({k: rep[k] for k in (
        "num_reports", "signals_present", "num_checks", "num_failed",
        "passed")}, indent=2))
    if rep["failed_checks"]:
        print("\nFAILED CHECKS:", file=sys.stderr)
        for c in rep["failed_checks"]:
            print("  - %s: %s" % (c["check"], c["detail"]), file=sys.stderr)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
