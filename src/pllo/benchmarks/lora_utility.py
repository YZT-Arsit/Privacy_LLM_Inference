"""E10: private folded-LoRA utility-preservation analysis (pure parsing).

Shows that the **private folded LoRA preserves the utility gain of plaintext
LoRA** on a real public task, while leaking nothing to the GPU. Consumes the E9
task-utility benchmark reports (which carry ``metric_value``) for:

* base (plaintext or base folded, no adapter),
* plaintext LoRA reference,
* folded LoRA remote,
* (optional) TDX-attested folded LoRA,

plus an optional ``validate_lora_effect`` comparison and the folded-LoRA package
verify report (for the ``contains_*`` no-secret guarantees). Honest labeling:
``dry_run`` / ``paper_ready`` are inherited from the folded-LoRA source report, so
a fixture-derived E10 report is never ``paper_ready``.

stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["metric_of", "build_lora_utility_report", "render_lora_utility_md"]


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def metric_of(report, override=None):
    """A scalar task metric from an E9 report (``metric_value``), or override."""
    if override is not None:
        return float(override)
    if isinstance(report, dict):
        for k in ("metric_value", "accuracy", "score"):
            v = report.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _infer(field, *reports, default=None):
    for r in reports:
        v = _g(r, field)
        if v is not None:
            return v
    return default


def build_lora_utility_report(inputs: dict) -> dict:
    """Consolidate base / plaintext-LoRA / folded-LoRA (/ attested) into one
    utility-preservation report. ``inputs`` keys: ``base, plaintext_lora,
    folded_lora, tdx_attested_folded_lora`` (E9 report dicts or None),
    ``effect`` (a validate_lora_effect dict or None), ``lora_verify`` (package
    verify report or None), ``dataset_name/task_type/metric_name`` (optional;
    inferred from folded_lora/base otherwise), ``preserve_threshold`` (default
    0.9), ``metric_overrides`` (optional dict of explicit floats)."""
    base = inputs.get("base")
    plain = inputs.get("plaintext_lora")
    folded = inputs.get("folded_lora")
    attested = inputs.get("tdx_attested_folded_lora")
    effect = inputs.get("effect")
    verify = inputs.get("lora_verify")
    ov = inputs.get("metric_overrides") or {}
    thr = float(inputs.get("preserve_threshold", 0.9))

    base_m = metric_of(base, ov.get("base"))
    plain_m = metric_of(plain, ov.get("plaintext_lora"))
    folded_m = metric_of(folded, ov.get("folded_lora"))
    attested_m = metric_of(attested, ov.get("tdx_attested_folded_lora"))

    gain_plain = (None if (base_m is None or plain_m is None)
                  else round(plain_m - base_m, 6))
    gain_folded = (None if (base_m is None or folded_m is None)
                   else round(folded_m - base_m, 6))
    if gain_plain is None or gain_folded is None or abs(gain_plain) < 1e-9:
        ratio = None
    else:
        ratio = round(gain_folded / gain_plain, 6)
    preserves = (None if ratio is None else bool(ratio >= thr))

    # effect (token-level / metric-level) -- accept a precomputed dict
    lora_differs = _g(effect, "tokens_differ")
    if lora_differs is None:
        lora_differs = _g(effect, "lora_has_effect")
    no_lora_tokens = _g(effect, "no_lora_token_ids")
    lora_tokens = _g(effect, "lora_token_ids")

    # security posture pulled from the actual folded/attested reports + verify
    sec_src = attested if attested is not None else folded
    worker_has_raw_lora = _infer("worker_has_raw_lora", sec_src, default=None)
    worker_has_mask_secrets = _infer("worker_has_mask_secrets", sec_src,
                                     default=None)
    gpu_plain = _infer("gpu_visible_plaintext_fields", sec_src, default=None)
    leaked = _infer("leaked_secret_fields", sec_src, default=None)
    audit_passed = _infer("audit_passed", attested, folded, default=None)
    tee_used_on_gpu = _infer("tee_used_on_gpu", sec_src, default=None)
    contains_raw = _g(verify, "contains_raw_lora")
    contains_opt = _g(verify, "contains_optimizer_state")
    contains_train = _g(verify, "contains_training_data")
    contains_mask = _g(verify, "contains_mask_secrets")

    security_ok = (worker_has_raw_lora is False
                   and worker_has_mask_secrets is False
                   and (not gpu_plain) and (not leaked)
                   and tee_used_on_gpu is False
                   and contains_raw is False and contains_opt is False
                   and contains_train is False and contains_mask is False)

    dataset_name = inputs.get("dataset_name") or _infer("dataset", folded, base,
                                                        plain)
    task_type = inputs.get("task_type") or _infer("task_type", folded, base,
                                                  plain)
    metric_name = inputs.get("metric_name") or _infer("metric_name", folded,
                                                      base, plain)

    dry_run = bool(_infer("dry_run", folded, attested, base, plain,
                          default=True))
    paper_ready = bool(_infer("paper_ready", attested, folded, default=False)
                       ) and not dry_run

    return {
        "stage": "e10_lora_utility_benchmark",
        "dataset_name": dataset_name, "task_type": task_type,
        "metric_name": metric_name,
        "base_metric": base_m, "plaintext_lora_metric": plain_m,
        "folded_lora_metric": folded_m,
        "tdx_attested_folded_lora_metric": attested_m,
        "lora_gain_plaintext": gain_plain, "lora_gain_folded": gain_folded,
        "lora_gain_preserved_ratio": ratio,
        "folded_lora_preserves_gain": preserves,
        "lora_differs_from_no_lora": lora_differs,
        "no_lora_token_ids": no_lora_tokens, "lora_token_ids": lora_tokens,
        "worker_has_raw_lora": worker_has_raw_lora,
        "worker_has_mask_secrets": worker_has_mask_secrets,
        "contains_raw_lora": contains_raw,
        "contains_optimizer_state": contains_opt,
        "contains_training_data": contains_train,
        "contains_mask_secrets": contains_mask,
        "gpu_visible_plaintext_fields": gpu_plain if gpu_plain is not None
        else [],
        "leaked_secret_fields": leaked if leaked is not None else [],
        "audit_passed": audit_passed, "tee_used_on_gpu": tee_used_on_gpu,
        "security_ok": security_ok,
        "utility_preserved": bool(preserves and security_ok),
        "preserve_threshold": thr,
        "inputs_provided": {
            "base": base is not None, "plaintext_lora": plain is not None,
            "folded_lora": folded is not None,
            "tdx_attested_folded_lora": attested is not None,
            "effect": effect is not None, "lora_verify": verify is not None,
        },
        "dry_run": dry_run, "paper_ready": paper_ready,
        "note": "folded_lora_preserves_gain compares the folded-LoRA metric gain "
                "to the plaintext-LoRA gain over the same base; security flags are "
                "read from the actual folded/attested reports + package verify "
                "(not assumed). Missing inputs are null, not assumed.",
    }


def render_lora_utility_md(r: dict) -> str:
    L = ["# E10 — private LoRA utility preservation", "",
         "_%s_" % r["note"], "",
         "- dataset=`%s`  task_type=%s  metric=%s  (dry_run=%s, paper_ready=%s)"
         % (r["dataset_name"], r["task_type"], r["metric_name"], r["dry_run"],
            r["paper_ready"]), "",
         "## Utility", "",
         "| variant | metric | gain vs base |",
         "| --- | --- | --- |",
         "| base | %s | — |" % r["base_metric"],
         "| plaintext LoRA | %s | %s |" % (r["plaintext_lora_metric"],
                                           r["lora_gain_plaintext"]),
         "| folded LoRA | %s | %s |" % (r["folded_lora_metric"],
                                        r["lora_gain_folded"]),
         "| TDX-attested folded LoRA | %s | — |"
         % r["tdx_attested_folded_lora_metric"],
         "",
         "- **lora_gain_preserved_ratio=%s** (threshold %s) -> "
         "folded_lora_preserves_gain=%s"
         % (r["lora_gain_preserved_ratio"], r["preserve_threshold"],
            r["folded_lora_preserves_gain"]),
         "- lora_differs_from_no_lora=%s" % r["lora_differs_from_no_lora"],
         "", "## Security", "",
         "- worker_has_raw_lora=%s worker_has_mask_secrets=%s tee_used_on_gpu=%s"
         % (r["worker_has_raw_lora"], r["worker_has_mask_secrets"],
            r["tee_used_on_gpu"]),
         "- contains_raw_lora=%s contains_optimizer_state=%s "
         "contains_training_data=%s contains_mask_secrets=%s"
         % (r["contains_raw_lora"], r["contains_optimizer_state"],
            r["contains_training_data"], r["contains_mask_secrets"]),
         "- gpu_visible_plaintext_fields=%s leaked_secret_fields=%s "
         "audit_passed=%s" % (r["gpu_visible_plaintext_fields"] or "[]",
                              r["leaked_secret_fields"] or "[]",
                              r["audit_passed"]),
         "- **security_ok=%s  utility_preserved=%s**"
         % (r["security_ok"], r["utility_preserved"]), ""]
    return "\n".join(L)
