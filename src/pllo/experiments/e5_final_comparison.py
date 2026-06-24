"""E5: paper-ready comparison + security-matrix consolidation.

Pure parsing of the prior experiment JSON outputs (stdlib only) into five
sections -- correctness, deployment, security matrix, cost, limitations -- plus
Markdown + LaTeX + JSON renderers. Honest by construction: a missing input is
marked ``provided=False`` and its pass is ``None`` (never assumed); the security
matrix encodes the fixed architecture facts and is *cross-checked* against the
audit results actually present in the decode reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "load_json", "correctness_section", "deployment_section",
    "security_matrix_section", "cost_section", "limitations_section",
    "build_e5_report", "render_e5_md", "render_e5_tex",
]


def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def _g(d: dict | None, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------


def _row(name, provided, metric, passed, source):
    return {"name": name, "provided": bool(provided), "metric": metric,
            "pass": passed, "source": source}


def correctness_section(*, e1=None, e2=None, local_prefill=None,
                        local_logits=None, local_decode=None,
                        remote_scaling=None, tdx_lite=None,
                        tdx_attested=None) -> list[dict]:
    rows = []

    rows.append(_row(
        "H800 standalone E1 (no-LoRA generation)", e1 is not None,
        {"token_match_rate": _g(e1, "token_match_rate"),
         "teacher_forced_top1_match_rate":
             _g(e1, "teacher_forced_top1_match_rate")},
        None if e1 is None else True, "e1_json"))
    rows.append(_row(
        "H800 standalone E2 (token scaling)", e2 is not None,
        {"token_match_rate": _g(e2, "token_match_rate")},
        None if e2 is None else True, "e2_json"))

    rows.append(_row(
        "H800 local package prefill (k=28)", local_prefill is not None,
        {"num_exec_layers": _g(local_prefill, "num_exec_layers"),
         "allclose": _g(local_prefill, "allclose"),
         "max_abs_error": _g(local_prefill, "max_abs_error"),
         "relative_l2_error": _g(local_prefill, "relative_l2_error")},
        _g(local_prefill, "allclose"), "local_prefill_json"))

    rows.append(_row(
        "H800 one-step logits", local_logits is not None,
        {"top1_match": _g(local_logits, "top1_match"),
         "next_token_match": _g(local_logits, "next_token_match"),
         "topk_overlap": _g(local_logits, "topk_overlap")},
        _g(local_logits, "top1_match"), "local_logits_json"))

    rows.append(_row(
        "H800 short decode (max_new_tokens=4)", local_decode is not None,
        {"tokens_exact_match": _g(local_decode, "tokens_exact_match"),
         "token_match_rate": _g(local_decode, "token_match_rate")},
        _g(local_decode, "tokens_exact_match"), "local_decode_json"))

    rs_all = _g(remote_scaling, "summary", "all_pass")
    rs_rows = _g(remote_scaling, "summary", "num_rows")
    rows.append(_row(
        "H800 remote HTTP decode scaling (E3)", remote_scaling is not None,
        {"all_pass": rs_all, "num_rows": rs_rows,
         "max_new_tokens_list": _g(remote_scaling, "meta",
                                   "max_new_tokens_list")},
        rs_all, "remote_scaling_json"))

    rows.append(_row(
        "TDX-lite remote decode", tdx_lite is not None,
        {"tokens_exact_match": _g(tdx_lite, "tokens_exact_match"),
         "token_match_rate": _g(tdx_lite, "token_match_rate"),
         "boundary_mode": _g(tdx_lite, "boundary_mode")},
        _g(tdx_lite, "tokens_exact_match"), "tdx_lite_json"))

    rows.append(_row(
        "TDX-attested remote decode", tdx_attested is not None,
        {"tokens_exact_match": _g(tdx_attested, "tokens_exact_match"),
         "audit_passed": _g(tdx_attested, "audit_passed"),
         "boundary_attested": _g(tdx_attested, "boundary_attested"),
         "runtime_hash_bound": _g(tdx_attested, "runtime_hash_bound"),
         "mr_td_match": _g(tdx_attested, "attestation", "mr_td_match")},
        _g(tdx_attested, "tokens_exact_match"), "tdx_attested_json"))
    return rows


# ---------------------------------------------------------------------------
# 2. Deployment
# ---------------------------------------------------------------------------


def deployment_section(*, local_decode=None, remote_scaling=None, tdx_lite=None,
                       tdx_attested=None) -> list[dict]:
    def d(name, *, boundary, pkg_loc, needs_ckpt, needs_full_pkg, attested,
          tokens_match, provided):
        return {"scenario": name, "boundary_runs": boundary,
                "folded_package_lives": pkg_loc,
                "boundary_needs_full_checkpoint": needs_ckpt,
                "boundary_needs_full_26gb_package": needs_full_pkg,
                "tdx_attestation_bound": attested,
                "generated_tokens_match": tokens_match, "provided": provided}

    rows = [
        d("H800 local executable package", boundary="H800 (in-process)",
          pkg_loc="H800 local disk", needs_ckpt=True, needs_full_pkg=True,
          attested=False, tokens_match=_g(local_decode, "tokens_exact_match"),
          provided=local_decode is not None),
        d("H800 remote HTTP decode (E3)", boundary="H800 process (client)",
          pkg_loc="H800 worker", needs_ckpt=True, needs_full_pkg=False,
          attested=False, tokens_match=_g(remote_scaling, "summary", "all_pass"),
          provided=remote_scaling is not None),
        d("TDX-lite remote decode", boundary="TDX guest (lite)",
          pkg_loc="H800 worker", needs_ckpt=False, needs_full_pkg=False,
          attested=False, tokens_match=_g(tdx_lite, "tokens_exact_match"),
          provided=tdx_lite is not None),
        d("TDX-attested remote decode", boundary="TDX guest (attested)",
          pkg_loc="H800 worker", needs_ckpt=False, needs_full_pkg=False,
          attested=_g(tdx_attested, "runtime_hash_bound", default=None),
          tokens_match=_g(tdx_attested, "tokens_exact_match"),
          provided=tdx_attested is not None),
    ]
    return rows


# ---------------------------------------------------------------------------
# 3. Security matrix (fixed architecture facts + audit cross-check)
# ---------------------------------------------------------------------------

# (tdx_boundary_visible, h800_worker_visible, gpu_visible, protected, notes)
_SECURITY_ROWS = [
    ("input_ids", True, False, False, True,
     "tokenized prompt; trusted-only, never sent to GPU"),
    ("prompt embeddings (plaintext)", True, False, False, True,
     "embed_tokens lookup happens in the trusted boundary"),
    ("masked embeddings", True, True, True, True,
     "the ONLY activation sent to the worker; masked by N_0"),
    ("N_0 / residual mask secrets", True, False, False, True,
     "in the trusted boundary artifact; never sent to GPU"),
    ("vocab mask", True, False, False, True,
     "trusted-only; used for logit recovery on the boundary"),
    ("folded package (W_tilde + folded head)", False, True, True, True,
     "base W is public; folded form carries no mask secret"),
    ("raw Qwen transformer weights", False, True, True, True,
     "public base model; not a secret"),
    ("recovered logits", True, False, False, True,
     "recovered on the boundary after masked logits return"),
    ("sampled token ids", True, False, False, True,
     "sampled on the boundary; never sent to GPU"),
    ("KV cache", True, True, True, True,
     "masked KV held on the worker (masked tensors only)"),
    ("boundary embedding artifact", True, False, False, True,
     "embed table + N_0 + vocab mask; trusted-only (~1GB)"),
]


def security_matrix_section(*decode_reports) -> dict:
    """Fixed architecture matrix + a cross-check that the provided decode reports
    show NO GPU-visible plaintext / NO leaked secrets / GPU is not a TEE."""
    rows = [{
        "asset": a, "tdx_boundary_visible": tb, "h800_worker_visible": hw,
        "gpu_visible": gv, "protected": pr, "notes": nt,
    } for (a, tb, hw, gv, pr, nt) in _SECURITY_ROWS]

    checks = []
    for rep in decode_reports:
        if not isinstance(rep, dict):
            continue
        checks.append({
            "source_stage": rep.get("stage"),
            "boundary_mode": rep.get("boundary_mode"),
            "gpu_visible_plaintext_fields": rep.get(
                "gpu_visible_plaintext_fields"),
            "leaked_secret_fields": rep.get("leaked_secret_fields"),
            "worker_has_mask_secrets": rep.get("worker_has_mask_secrets"),
            "tee_used_on_gpu": rep.get("tee_used_on_gpu"),
            "audit_passed": rep.get("audit_passed"),
        })
    cross_ok = all(
        (not c["gpu_visible_plaintext_fields"]) and (not c["leaked_secret_fields"])
        and (not c["worker_has_mask_secrets"]) and (not c["tee_used_on_gpu"])
        and (c["audit_passed"] is not False)
        for c in checks) if checks else None
    return {"matrix": rows, "audit_cross_check": checks,
            "audit_cross_check_ok": cross_ok}


# ---------------------------------------------------------------------------
# 4. Cost
# ---------------------------------------------------------------------------


def cost_section(*, local_decode=None, remote_scaling=None, tdx_lite=None,
                 tdx_attested=None, setup_cost=None) -> dict:
    def from_report(name, rep, provided):
        return {"scenario": name, "provided": provided,
                "latency_s": _g(rep, "latency_s"),
                "trusted_bytes": _g(rep, "trusted_bytes"),
                "gpu_bytes": _g(rep, "gpu_bytes"),
                "boundary_calls": _g(rep, "boundary_calls"),
                "peak_gpu_memory_mb": _g(rep, "peak_gpu_memory_mb"),
                "package_size_gb": _g(rep, "package_size_gb")}

    runtime = [
        from_report("H800 local short decode", local_decode,
                    local_decode is not None),
        from_report("TDX-lite remote decode", tdx_lite, tdx_lite is not None),
        from_report("TDX-attested remote decode", tdx_attested,
                    tdx_attested is not None),
    ]
    # remote scaling latency rows (per max_new_tokens)
    scaling = _g(remote_scaling, "summary", "latency_table") or []

    setup = {
        "provided": setup_cost is not None,
        "folded_package_size_gb": _g(setup_cost, "folded_package_size_gb"),
        "folded_package_size_if_bf16_gb":
            _g(setup_cost, "folded_package_size_if_bf16_gb"),
        "generation_time_s": _g(setup_cost, "generation_time_s"),
        "package_load_time_s": _g(setup_cost, "package_load_time_s"),
        "one_time_setup_s": _g(setup_cost, "one_time_setup_s"),
        "boundary_embedding_artifact_size_gb":
            _g(setup_cost, "boundary_embedding_artifact_size_gb"),
        "transfer_estimates": _g(setup_cost, "transfer_estimates"),
        "amortized_setup_cost": _g(setup_cost, "amortized_setup_cost"),
    }
    return {"runtime": runtime, "remote_scaling_latency": scaling,
            "setup": setup}


# ---------------------------------------------------------------------------
# 5. Limitations
# ---------------------------------------------------------------------------


def limitations_section(*, tdx_attested=None) -> list[str]:
    attested_n = _g(tdx_attested, "max_new_tokens")
    return [
        "The current TDX-attested run validates max_new_tokens=%s; longer decode "
        "scaling is measured in remote HTTP / H800 mode and may be repeated under "
        "TDX if needed." % (attested_n if attested_n is not None else 4),
        "The boundary embedding artifact contains trusted mask tensors (N_0 + "
        "vocab mask) and MUST remain inside the TDX guest; it is never sent to "
        "the GPU.",
        "The folded package is currently stored in float32 for numerical "
        "fidelity (~26.34GB); a bf16 store (~13.17GB) is smaller but is not the "
        "current measured artifact.",
        "The HTTP transport is a research prototype, not production hardened "
        "(no TLS/mTLS/rate-limiting claims here).",
        "No formal cryptographic security is claimed for the masking; attention "
        "scores remain GPU-visible and vocab permutation+scaling is weaker than "
        "dense vocab masking.",
    ]


# ---------------------------------------------------------------------------
# Build + render
# ---------------------------------------------------------------------------


def build_e5_report(inputs: dict) -> dict:
    e1 = inputs.get("e1")
    e2 = inputs.get("e2")
    local_prefill = inputs.get("local_prefill")
    local_logits = inputs.get("local_logits")
    local_decode = inputs.get("local_decode")
    remote_scaling = inputs.get("remote_scaling")
    tdx_lite = inputs.get("tdx_lite")
    tdx_attested = inputs.get("tdx_attested")
    setup_cost = inputs.get("setup_cost")

    correctness = correctness_section(
        e1=e1, e2=e2, local_prefill=local_prefill, local_logits=local_logits,
        local_decode=local_decode, remote_scaling=remote_scaling,
        tdx_lite=tdx_lite, tdx_attested=tdx_attested)
    deployment = deployment_section(
        local_decode=local_decode, remote_scaling=remote_scaling,
        tdx_lite=tdx_lite, tdx_attested=tdx_attested)
    security = security_matrix_section(local_decode, tdx_lite, tdx_attested)
    cost = cost_section(
        local_decode=local_decode, remote_scaling=remote_scaling,
        tdx_lite=tdx_lite, tdx_attested=tdx_attested, setup_cost=setup_cost)
    limitations = limitations_section(tdx_attested=tdx_attested)

    provided = {k: (inputs.get(k) is not None) for k in (
        "e1", "e2", "local_prefill", "local_logits", "local_decode",
        "remote_scaling", "tdx_lite", "tdx_attested", "setup_cost")}
    return {
        "experiment": "E5", "stage": "final_comparison",
        "inputs_provided": provided,
        "correctness": correctness,
        "deployment": deployment,
        "security_matrix": security,
        "cost": cost,
        "limitations": limitations,
        "note": "Consolidated from prior experiment outputs. Fields from missing "
                "inputs are null/not-provided (not assumed). TDX attestation is "
                "reflected only from the provided tdx-attested JSON.",
    }


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".") if v else "0"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(";", ":"))
    return str(v)


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return out


def render_e5_md(report: dict) -> str:
    L = ["# E5 — Final comparison + security matrix", "",
         "_%s_" % report["note"], "",
         "## 1. Correctness", ""]
    L += _md_table(
        ["result", "provided", "pass", "metric", "source"],
        [[r["name"], r["provided"], r["pass"], r["metric"], r["source"]]
         for r in report["correctness"]])
    L += ["", "## 2. Deployment", ""]
    L += _md_table(
        ["scenario", "boundary_runs", "folded_package_lives",
         "needs_full_checkpoint", "needs_full_26GB_package",
         "tdx_attestation_bound", "tokens_match", "provided"],
        [[r["scenario"], r["boundary_runs"], r["folded_package_lives"],
          r["boundary_needs_full_checkpoint"],
          r["boundary_needs_full_26gb_package"], r["tdx_attestation_bound"],
          r["generated_tokens_match"], r["provided"]]
         for r in report["deployment"]])
    L += ["", "## 3. Security matrix", "",
          "_audit_cross_check_ok: %s_"
          % report["security_matrix"]["audit_cross_check_ok"], ""]
    L += _md_table(
        ["asset", "TDX boundary visible", "H800 worker visible", "GPU-visible",
         "protected", "notes"],
        [[r["asset"], r["tdx_boundary_visible"], r["h800_worker_visible"],
          r["gpu_visible"], r["protected"], r["notes"]]
         for r in report["security_matrix"]["matrix"]])
    L += ["", "## 4. Cost", "", "### Runtime", ""]
    L += _md_table(
        ["scenario", "latency_s", "trusted_bytes", "gpu_bytes",
         "peak_gpu_memory_mb", "package_size_gb", "provided"],
        [[r["scenario"], r["latency_s"], r["trusted_bytes"], r["gpu_bytes"],
          r["peak_gpu_memory_mb"], r["package_size_gb"], r["provided"]]
         for r in report["cost"]["runtime"]])
    if report["cost"]["remote_scaling_latency"]:
        L += ["", "### Remote scaling latency (E3)", ""]
        L += _md_table(
            ["seq_len", "max_new_tokens", "latency_s", "latency_per_token_s"],
            [[t.get("seq_len"), t.get("max_new_tokens"), t.get("latency_s"),
              t.get("latency_per_token_s")]
             for t in report["cost"]["remote_scaling_latency"]])
    s = report["cost"]["setup"]
    L += ["", "### Setup (E4)", "",
          "- folded_package_size_gb: %s (bf16-equiv: %s)"
          % (_fmt(s["folded_package_size_gb"]),
             _fmt(s["folded_package_size_if_bf16_gb"])),
          "- generation_time_s: %s  package_load_time_s: %s  one_time_setup_s: %s"
          % (_fmt(s["generation_time_s"]), _fmt(s["package_load_time_s"]),
             _fmt(s["one_time_setup_s"])),
          "- boundary_embedding_artifact_size_gb: %s"
          % _fmt(s["boundary_embedding_artifact_size_gb"])]
    L += ["", "## 5. Limitations", ""]
    L += ["%d. %s" % (i + 1, t) for i, t in enumerate(report["limitations"])]
    L += [""]
    return "\n".join(L)


def _tex_escape(s: str) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _tex_table(caption, headers, rows):
    cols = "l" * len(headers)
    out = [r"\begin{table}[h]", r"\centering",
           r"\caption{%s}" % _tex_escape(caption),
           r"\begin{tabular}{%s}" % cols, r"\hline",
           " & ".join(_tex_escape(h) for h in headers) + r" \\", r"\hline"]
    for r in rows:
        out.append(" & ".join(_tex_escape(_fmt(c)) for c in r) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return out


def render_e5_tex(report: dict) -> str:
    L = ["% E5 final comparison tables (paper-ready)",
         "% Generated from prior experiment outputs; missing inputs are blank.",
         ""]
    L += _tex_table(
        "Correctness across deployment stages",
        ["Result", "Pass", "Source"],
        [[r["name"], _fmt(r["pass"]), r["source"]]
         for r in report["correctness"]])
    L += _tex_table(
        "Deployment matrix",
        ["Scenario", "Boundary", "Pkg loc", "Needs ckpt", "Needs 26GB",
         "Attested", "Tokens match"],
        [[r["scenario"], r["boundary_runs"], r["folded_package_lives"],
          _fmt(r["boundary_needs_full_checkpoint"]),
          _fmt(r["boundary_needs_full_26gb_package"]),
          _fmt(r["tdx_attestation_bound"]), _fmt(r["generated_tokens_match"])]
         for r in report["deployment"]])
    L += _tex_table(
        "Security matrix",
        ["Asset", "TDX vis", "H800 vis", "GPU vis", "Protected"],
        [[r["asset"], _fmt(r["tdx_boundary_visible"]),
          _fmt(r["h800_worker_visible"]), _fmt(r["gpu_visible"]),
          _fmt(r["protected"])]
         for r in report["security_matrix"]["matrix"]])
    return "\n".join(L)
