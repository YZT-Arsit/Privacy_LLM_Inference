"""E8: private-LoRA ablation / reporting consolidation.

Pure parsing of prior LoRA experiment JSON outputs (stdlib only) into four
tables -- inference correctness, security matrix, cost, training prototype --
plus Markdown + JSON renderers. Honest: a missing input is ``provided=False`` and
its pass is ``None``; the security matrix is cross-checked against the audit
results actually present in the provided probe outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "load_json", "lora_correctness_section", "lora_security_matrix_section",
    "lora_cost_section", "lora_training_section", "build_e8_report",
    "render_e8_md",
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


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# 1. Inference correctness
# ---------------------------------------------------------------------------


def lora_correctness_section(*, local=None, remote=None, attested=None):
    def row(name, rep, *, pass_key, extra):
        return {"name": name, "provided": rep is not None,
                "pass": _g(rep, pass_key), "metrics": {k: _g(rep, k)
                                                       for k in extra},
                "worker_has_raw_lora": _g(rep, "worker_has_raw_lora"),
                "worker_has_mask_secrets": _g(rep, "worker_has_mask_secrets"),
                "tee_used_on_gpu": _g(rep, "tee_used_on_gpu")}
    return [
        row("local folded LoRA", local, pass_key="tokens_exact_match",
            extra=("allclose", "max_abs_error", "relative_l2_error",
                   "top1_match", "tokens_exact_match", "token_match_rate")),
        row("remote folded LoRA", remote, pass_key="tokens_exact_match",
            extra=("lora_enabled", "folded_lora_loaded", "folded_lora_valid",
                   "tokens_exact_match", "token_match_rate")),
        row("TDX-attested folded LoRA", attested, pass_key="tokens_exact_match",
            extra=("tokens_exact_match", "audit_passed", "boundary_attested",
                   "runtime_hash_bound")),
    ]


# ---------------------------------------------------------------------------
# 2. Security matrix
# ---------------------------------------------------------------------------

# (asset, tdx_boundary_visible, gpu_visible, protected, notes)
_LORA_SECURITY_ROWS = [
    ("raw adapter A / B", True, False, True,
     "raw LoRA stays trusted; only folded a_tilde/b_tilde reach the GPU"),
    ("folded adapter (a_tilde/b_tilde)", True, True, True,
     "folded with N masks + rank mask; carries no raw A/B (base W is public)"),
    ("optimizer state", True, False, True,
     "Adam/SGD state stays trusted (training); never sent to GPU"),
    ("gradients dA / dB", True, False, True,
     "computed + clipped trusted-side; never sent to GPU"),
    ("training data (X)", True, False, True,
     "private inputs stay trusted; GPU sees only masked activations"),
    ("labels (Y)", True, False, True,
     "private labels stay trusted; loss computed trusted-side"),
]


def lora_security_matrix_section(*reports):
    rows = [{"asset": a, "tdx_boundary_visible": tb, "gpu_visible": gv,
             "protected": pr, "notes": nt}
            for (a, tb, gv, pr, nt) in _LORA_SECURITY_ROWS]
    checks = []
    for rep in reports:
        if not isinstance(rep, dict):
            continue
        checks.append({
            "source_stage": rep.get("stage"),
            "worker_has_raw_lora": rep.get("worker_has_raw_lora"),
            "worker_has_mask_secrets": rep.get("worker_has_mask_secrets"),
            "tee_used_on_gpu": rep.get("tee_used_on_gpu"),
            "gpu_visible_plaintext_fields": rep.get(
                "gpu_visible_plaintext_fields"),
            "leaked_secret_fields": rep.get("leaked_secret_fields"),
            # training-probe flags (present only for E7)
            "raw_lora_visible_to_gpu": rep.get("raw_lora_visible_to_gpu"),
            "optimizer_state_visible_to_gpu": rep.get(
                "optimizer_state_visible_to_gpu"),
            "training_data_visible_to_gpu": rep.get(
                "training_data_visible_to_gpu"),
            "labels_visible_to_gpu": rep.get("labels_visible_to_gpu"),
        })

    def _bad(c):
        return bool(c.get("worker_has_raw_lora") or c.get("worker_has_mask_secrets")
                    or c.get("tee_used_on_gpu")
                    or c.get("gpu_visible_plaintext_fields")
                    or c.get("leaked_secret_fields")
                    or c.get("raw_lora_visible_to_gpu")
                    or c.get("optimizer_state_visible_to_gpu")
                    or c.get("training_data_visible_to_gpu")
                    or c.get("labels_visible_to_gpu"))
    cross_ok = (None if not checks else not any(_bad(c) for c in checks))
    return {"matrix": rows, "audit_cross_check": checks,
            "audit_cross_check_ok": cross_ok}


# ---------------------------------------------------------------------------
# 3. Cost
# ---------------------------------------------------------------------------


def lora_cost_section(*, lora_build=None, lora_verify=None, remote=None,
                      base_decode=None, local=None):
    lora_size = _g(lora_build, "size_gb")
    if lora_size is None:
        lora_size = _g(lora_verify, "package_size_gb")
    lora_lat = _g(remote, "latency_s")
    if lora_lat is None:
        lora_lat = _g(local, "latency_s")
    base_lat = _g(base_decode, "latency_s")
    overhead = (None if (lora_lat is None or base_lat is None)
                else round(float(lora_lat) - float(base_lat), 6))
    overhead_pct = (None if (overhead is None or not base_lat)
                    else round(100.0 * overhead / float(base_lat), 3))
    lora_peak = _g(remote, "peak_gpu_memory_mb")
    if lora_peak is None:
        lora_peak = _g(local, "peak_gpu_memory_mb")
    base_peak = _g(base_decode, "peak_gpu_memory_mb")
    mem_overhead = (None if (lora_peak is None or base_peak is None)
                    else round(float(lora_peak) - float(base_peak), 3))
    return {
        "folded_lora_package_size_gb": lora_size,
        "folded_lora_setup_time_s": _g(lora_build, "build_time_s"),
        "lora_rank": _g(lora_build, "rank") or _g(lora_verify, "rank"),
        "lora_target_modules": (_g(lora_build, "target_modules")
                                or _g(lora_verify, "target_modules")),
        "decode_latency_s_lora": lora_lat,
        "decode_latency_s_base": base_lat,
        "decode_latency_overhead_s": overhead,
        "decode_latency_overhead_pct": overhead_pct,
        "peak_gpu_memory_mb_lora": lora_peak,
        "peak_gpu_memory_mb_base": base_peak,
        "memory_overhead_mb": mem_overhead,
    }


# ---------------------------------------------------------------------------
# 4. Training prototype
# ---------------------------------------------------------------------------


def lora_training_section(*, training=None):
    if training is None:
        return {"provided": False}
    return {
        "provided": True,
        "training_steps": training.get("training_steps"),
        "rank": training.get("rank"), "alpha": training.get("alpha"),
        "target_modules": training.get("target_modules"),
        "loss_before": training.get("loss_before"),
        "loss_after": training.get("loss_after"),
        "loss_decreased": training.get("loss_decreased"),
        "adapter_delta_norm": training.get("adapter_delta_norm"),
        "update_correct": training.get("loss_decreased"),
        "audit_passed": training.get("audit_passed"),
        "raw_lora_visible_to_gpu": training.get("raw_lora_visible_to_gpu"),
        "optimizer_state_visible_to_gpu": training.get(
            "optimizer_state_visible_to_gpu"),
        "training_data_visible_to_gpu": training.get(
            "training_data_visible_to_gpu"),
        "labels_visible_to_gpu": training.get("labels_visible_to_gpu"),
        "limitations": training.get("limitations", []),
    }


def build_e8_report(inputs: dict) -> dict:
    local = inputs.get("local")
    remote = inputs.get("remote")
    attested = inputs.get("attested")
    training = inputs.get("training")
    provided = {k: (inputs.get(k) is not None) for k in (
        "local", "remote", "attested", "lora_build", "lora_verify",
        "base_decode", "training")}
    # ---- Linear-boundary pad provenance (read from the LoRA build report) ----
    lora_build = inputs.get("lora_build")
    base_pad_enabled = bool(
        _g(lora_build, "base_linear_boundary_pad_enabled", default=False)
        or _g(remote, "base_linear_boundary_pad_enabled", default=False))
    lora_inherits = bool(
        _g(lora_build, "lora_inherits_linear_boundary_pad_from_base",
           default=False)
        or _g(remote, "lora_inherits_linear_boundary_pad_from_base",
              default=False))
    lora_recomputes = bool(
        _g(lora_build, "lora_merge_recomputes_cpad", default=True))
    pad_section = {
        "main_scheme": "linear_boundary_additive_pad",
        "lora_case_study_uses_pad_enabled_base": base_pad_enabled,
        "base_linear_boundary_pad_enabled": base_pad_enabled,
        "lora_inherits_linear_boundary_pad_from_base": lora_inherits,
        "lora_merge_recomputes_cpad": lora_recomputes,
        "lora_package_contains_pad": False,
        "pad_scope": "base_folded_linear_boundary",
    }
    report = {
        "experiment": "E8", "stage": "lora_final_report",
        "inputs_provided": provided,
        "correctness": lora_correctness_section(
            local=local, remote=remote, attested=attested),
        "security_matrix": lora_security_matrix_section(
            local, remote, attested, training),
        "cost": lora_cost_section(
            lora_build=lora_build,
            lora_verify=inputs.get("lora_verify"), remote=remote,
            base_decode=inputs.get("base_decode"), local=local),
        "training": lora_training_section(training=training),
        "linear_boundary_pad": pad_section,
        "note": "Consolidated from prior LoRA experiment outputs; missing inputs "
                "are not-provided (not assumed). TDX attestation is reflected only "
                "from the provided attested JSON.",
    }
    report.update(pad_section)
    if not (base_pad_enabled and lora_inherits):
        report["paper_ready"] = False
        report["paper_ready_blocker"] = (
            "E8 final report: LoRA case study is not over a pad-enabled base "
            "folded package (main_scheme=linear_boundary_additive_pad requires "
            "base_linear_boundary_pad_enabled and "
            "lora_inherits_linear_boundary_pad_from_base)")
    return report


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


def render_e8_md(report: dict) -> str:
    L = ["# E8 — Private LoRA final report", "", "_%s_" % report["note"], "",
         "## 1. LoRA inference correctness", ""]
    L += _md_table(
        ["result", "provided", "pass", "worker_has_raw_lora",
         "worker_has_mask_secrets", "tee_used_on_gpu", "metrics"],
        [[r["name"], r["provided"], r["pass"], r["worker_has_raw_lora"],
          r["worker_has_mask_secrets"], r["tee_used_on_gpu"], r["metrics"]]
         for r in report["correctness"]])
    L += ["", "## 2. LoRA security matrix", "",
          "_audit_cross_check_ok: %s_"
          % report["security_matrix"]["audit_cross_check_ok"], ""]
    L += _md_table(
        ["asset", "TDX boundary visible", "GPU-visible", "protected", "notes"],
        [[r["asset"], r["tdx_boundary_visible"], r["gpu_visible"],
          r["protected"], r["notes"]]
         for r in report["security_matrix"]["matrix"]])
    c = report["cost"]
    L += ["", "## 3. LoRA cost", "",
          "- folded_lora_package_size_gb: %s  setup_time_s: %s  rank: %s  "
          "target_modules: %s" % (_fmt(c["folded_lora_package_size_gb"]),
                                  _fmt(c["folded_lora_setup_time_s"]),
                                  _fmt(c["lora_rank"]),
                                  _fmt(c["lora_target_modules"])),
          "- decode_latency_s (lora=%s, base=%s, overhead=%s s / %s%%)"
          % (_fmt(c["decode_latency_s_lora"]), _fmt(c["decode_latency_s_base"]),
             _fmt(c["decode_latency_overhead_s"]),
             _fmt(c["decode_latency_overhead_pct"])),
          "- peak_gpu_memory_mb (lora=%s, base=%s, overhead=%s MB)"
          % (_fmt(c["peak_gpu_memory_mb_lora"]),
             _fmt(c["peak_gpu_memory_mb_base"]), _fmt(c["memory_overhead_mb"]))]
    t = report["training"]
    L += ["", "## 4. LoRA training prototype", ""]
    if not t.get("provided"):
        L += ["- not provided"]
    else:
        L += ["- steps=%s rank=%s target_modules=%s"
              % (_fmt(t["training_steps"]), _fmt(t["rank"]),
                 _fmt(t["target_modules"])),
              "- **loss_before=%s loss_after=%s loss_decreased=%s**"
              % (_fmt(t["loss_before"]), _fmt(t["loss_after"]),
                 _fmt(t["loss_decreased"])),
              "- adapter_delta_norm=%s  audit_passed=%s"
              % (_fmt(t["adapter_delta_norm"]), _fmt(t["audit_passed"])),
              "- raw_lora_visible_to_gpu=%s optimizer_state_visible_to_gpu=%s "
              "training_data_visible_to_gpu=%s labels_visible_to_gpu=%s"
              % (_fmt(t["raw_lora_visible_to_gpu"]),
                 _fmt(t["optimizer_state_visible_to_gpu"]),
                 _fmt(t["training_data_visible_to_gpu"]),
                 _fmt(t["labels_visible_to_gpu"])), "", "### Limitations", ""]
        L += ["%d. %s" % (i + 1, x) for i, x in enumerate(t.get("limitations",
                                                                []))]
    L += [""]
    return "\n".join(L)
