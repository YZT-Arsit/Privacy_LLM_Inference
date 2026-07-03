"""Generation-task evaluation for the current vs trusted_shortcut nonlinear
backends (Amulet-style ``amulet_migrated`` migration).

This module is the *library* half of ``scripts/run_generation_backend_eval.py``.
Everything here is torch-free and importable in unit tests: backend resolution,
the no-silent-fallback verifier, the nonlinear-execution-evidence summariser, the
full dimension audit (model / attention / KV-cache / MLP / nonlinear / LM-head /
mask+pad), the seq-len compatibility gate, generation-quality metrics, the
current-vs-trusted_shortcut comparison, and the report writers.

Honesty conventions
-------------------
* Every dimension field is tagged with a ``source``:
  ``measured`` (an actual tensor shape captured client-side during generation),
  ``derived_from_config`` (deterministic from the public model config), or
  ``derived_from_exec_metadata`` (from the folded package's public exec metadata).
  We never label a derived shape as measured.
* If the worker does not return non-empty nonlinear execution evidence for a
  ``trusted_shortcut`` run, ``nonlinear_execution_evidence_missing`` is set True
  with a diagnostic list -- we never silently report success.
* ``trusted_shortcut`` that does not actually resolve to / execute
  ``amulet_migrated`` is a hard verification failure, not a warning.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pllo.experiments.nonlinear_designs import (
    normalize_nonlinear_backend,
    op_backend_for_design,
    report_has_amulet_execution,
    trusted_shortcut_tag_only,
)

# The two backends this evaluation stage compares. A_rightmul is intentionally
# excluded from this round.
SUPPORTED_BACKENDS = ("current", "trusted_shortcut")


# --------------------------------------------------------------------------- #
# Backend resolution + no-silent-fallback verification
# --------------------------------------------------------------------------- #
def resolve_backend(name: str) -> Dict[str, str]:
    """Map a user backend string to canonical design + op-backend.

    ``trusted_shortcut`` (and its alias ``amulet_migrated``) -> op_backend
    ``amulet_migrated``; ``current`` -> op_backend ``current``.
    """
    canonical = normalize_nonlinear_backend(name)
    op_backend = op_backend_for_design(canonical)
    return {"input": name, "nonlinear_backend": canonical,
            "op_backend": op_backend}


def verify_backend_execution(
    health: Dict[str, Any],
    expected_nonlinear_backend: str,
    expected_op_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare the *running* worker's advertised backend (from ``/health``)
    against what the caller asked for. Detects a silent fallback where a
    ``trusted_shortcut`` request is served by a ``current`` worker.
    """
    exp = resolve_backend(expected_nonlinear_backend)
    exp_nl = exp["nonlinear_backend"]
    exp_op = (op_backend_for_design(normalize_nonlinear_backend(
        expected_op_backend)) if expected_op_backend else exp["op_backend"])

    actual_nl_raw = health.get("nonlinear_backend")
    actual_nl = (normalize_nonlinear_backend(actual_nl_raw)
                 if actual_nl_raw else None)
    ev = health.get("nonlinear_execution_evidence") or {}
    actual_op = ev.get("nonlinear_op_backend")
    if actual_op is None and actual_nl is not None:
        # fall back to the design->op map when evidence has not populated yet
        actual_op = op_backend_for_design(actual_nl)

    reasons: List[str] = []
    if actual_nl is None:
        reasons.append("worker /health did not advertise a nonlinear_backend")
    elif actual_nl != exp_nl:
        reasons.append(
            f"nonlinear_backend mismatch: expected {exp_nl!r}, "
            f"worker reports {actual_nl!r}")
    if actual_op is not None and actual_op != exp_op:
        reasons.append(
            f"op_backend mismatch: expected {exp_op!r}, worker {actual_op!r}")

    silent_fallback = bool(
        exp_nl == "trusted_shortcut"
        and actual_nl == "current")

    return {
        "expected_nonlinear_backend": exp_nl,
        "expected_op_backend": exp_op,
        "actual_nonlinear_backend": actual_nl,
        "actual_op_backend": actual_op,
        "backend_verification_passed": len(reasons) == 0,
        "silent_fallback_detected": silent_fallback,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Nonlinear execution evidence
# --------------------------------------------------------------------------- #
# Counter fields the worker's execution evidence is expected to carry. Reported
# verbatim (with null if absent) so a reviewer sees exactly what was measured.
_NONLINEAR_EVIDENCE_FIELDS = (
    "nonlinear_backend", "nonlinear_op_backend", "nonlinear_execution_status",
    "amulet_lift_executed", "amulet_backend_used",
    "lifted_nonlinear_ops_count", "lift_k", "lifted_gpu_bytes",
    "trusted_nonlinear_ops_count", "trusted_calls", "trusted_bytes",
    "gpu_bytes", "migrated_ops_by_type", "unsupported_ops",
)


def summarize_nonlinear_evidence(
    evidence: Dict[str, Any],
    expected_nonlinear_backend: str,
    expected_op_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Build ``nonlinear_execution_evidence.json`` content.

    For ``trusted_shortcut`` we require genuine ``amulet_migrated`` lift
    evidence (``report_has_amulet_execution``). If it is absent we set
    ``nonlinear_execution_evidence_missing = True`` and enumerate candidate
    causes -- never a silent pass.
    """
    ev = dict(evidence or {})
    exp = resolve_backend(expected_nonlinear_backend)
    exp_nl = exp["nonlinear_backend"]
    exp_op = (op_backend_for_design(normalize_nonlinear_backend(
        expected_op_backend)) if expected_op_backend else exp["op_backend"])

    counters = {k: ev.get(k) for k in _NONLINEAR_EVIDENCE_FIELDS}
    by_type = ev.get("migrated_ops_by_type") or {}
    nonlinear_ops_count = {
        "rmsnorm": by_type.get("rmsnorm") or by_type.get("layernorm"),
        "softmax": by_type.get("softmax"),
        "silu": (ev.get("lifted_nonlinear_ops_count")
                 if by_type.get("silu") is None else by_type.get("silu")),
        "gelu": by_type.get("gelu"),
    }

    out: Dict[str, Any] = {
        "expected_nonlinear_backend": exp_nl,
        "expected_op_backend": exp_op,
        "nonlinear_backend": ev.get("nonlinear_backend"),
        "nonlinear_op_backend": ev.get("nonlinear_op_backend"),
        "counters": counters,
        "nonlinear_ops_count": nonlinear_ops_count,
        "per_op_evidence_sample": ev.get("per_op_evidence")
        or ev.get("per_op_evidence_sample"),
    }

    empty = (not ev) or all(ev.get(k) in (None, 0, {}, [])
                            for k in ("lifted_nonlinear_ops_count",
                                      "trusted_nonlinear_ops_count",
                                      "migrated_ops_by_type"))
    diagnostics: List[str] = []
    if exp_nl == "trusted_shortcut":
        amulet_ok = report_has_amulet_execution(ev)
        tag_only = trusted_shortcut_tag_only(ev)
        out["amulet_real_path_executed"] = bool(amulet_ok)
        out["trusted_shortcut_tag_only"] = bool(tag_only)
        if not amulet_ok:
            if empty:
                diagnostics.append(
                    "worker returned no nonlinear execution counters -- did "
                    "generation actually run a decode (health polled before "
                    "any forward)?")
            if ev.get("nonlinear_op_backend") not in (None, "amulet_migrated"):
                diagnostics.append(
                    "op_backend is not amulet_migrated -- trusted_shortcut may "
                    "have fallen back to current")
            if not ev.get("amulet_lift_executed"):
                diagnostics.append(
                    "amulet_lift_executed is falsey -- the SwiGLU SiLU lift "
                    "(the counter that sets it) never fired; MLP path may be "
                    "bypassed or worker lazy-init not triggered")
            if not ev.get("lifted_gpu_bytes"):
                diagnostics.append(
                    "lifted_gpu_bytes == 0 -- no tensor was lifted onto the "
                    "accelerator")
            if (ev.get("lift_k") or 0) < 2:
                diagnostics.append(
                    "lift_k < 2 -- lift is degenerate (needs >=1 valid + >=1 "
                    "decoy column)")
        out["nonlinear_execution_evidence_missing"] = not amulet_ok
    else:
        # current baseline: trusted path; evidence may legitimately be sparse.
        out["nonlinear_execution_evidence_missing"] = bool(empty)
        if empty:
            diagnostics.append(
                "current backend returned no counters (trusted path may not "
                "increment migration counters -- informational, not a failure)")

    out["diagnostics"] = diagnostics
    return out


# --------------------------------------------------------------------------- #
# Dimension audit
# --------------------------------------------------------------------------- #
# Required top-level sections; test_dimension_audit_schema asserts these exist.
REQUIRED_AUDIT_SECTIONS = (
    "model", "inputs", "attention", "kv_cache", "mlp", "nonlinear",
    "lm_head", "masks_pad_package",
)


def _shape(dims: Sequence[Optional[int]]) -> List[Optional[int]]:
    return [None if d is None else int(d) for d in dims]


def build_dimension_audit(
    config: Dict[str, Any],
    *,
    exec_meta: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    prompt_len: int,
    max_new_tokens: int,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    device: str = "cuda",
    nonlinear_backend: str = "current",
    lift_k: Optional[int] = None,
    measured: Optional[Dict[str, Any]] = None,
    representative_decode_steps: Sequence[int] = (1,),
) -> Dict[str, Any]:
    """Full per-dimension audit. Weight/activation/KV/MLP/nonlinear/LM-head
    shapes are *derived* from the public config (deterministic across all
    layers); a handful of client-side captured shapes are *measured*. Every
    field carries a ``source`` tag.
    """
    exec_meta = exec_meta or {}
    manifest = manifest or {}
    measured = measured or {}

    H = int(config["hidden_size"])
    I = int(config["intermediate_size"])
    n_layers = int(config["num_hidden_layers"])
    n_heads = int(config["num_attention_heads"])
    n_kv = int(config.get("num_key_value_heads", n_heads))
    head_dim = int(config.get("head_dim", H // n_heads))
    vocab = int(config["vocab_size"])
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    B = int(batch_size)
    T = int(prompt_len)
    op_backend = op_backend_for_design(
        normalize_nonlinear_backend(nonlinear_backend))

    def d(shape, src="derived_from_config", note=None):
        e = {"shape": _shape(shape), "source": src}
        if note:
            e["note"] = note
        return e

    def m(key, fallback_shape=None, note=None):
        """measured if captured client-side, else derived fallback."""
        if key in measured and measured[key] is not None:
            e = {"shape": _shape(measured[key]), "source": "measured"}
        else:
            e = {"shape": _shape(fallback_shape) if fallback_shape else None,
                 "source": "derived_from_config"}
        if note:
            e["note"] = note
        return e

    model = {
        "model_name": exec_meta.get("model_name") or manifest.get("model_name")
        or config.get("_name_or_path"),
        "num_layers": n_layers,
        "hidden_size": H,
        "intermediate_size": I,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "max_position_embeddings": int(
            config.get("max_position_embeddings", 0)) or None,
        "dtype": dtype,
        "device": device,
        "package_seq_len": exec_meta.get("seq_len") or manifest.get("seq_len"),
        "generation_max_new_tokens": int(max_new_tokens),
        "batch_size": B,
        "nonlinear_backend": normalize_nonlinear_backend(nonlinear_backend),
        "nonlinear_op_backend": op_backend,
    }

    inputs = {
        "input_ids": m("input_ids", [B, T]),
        "attention_mask": m("attention_mask", [B, T]),
        "embedding_output": m("embedding_output", [B, T, H],
                              note="masked embedding crossing to GPU"),
        "prefill_hidden": d([B, T, H]),
        "decode_hidden_per_step": d([B, 1, H]),
    }

    # per-layer attention (identical shapes across layers -> record template +
    # explicit per-layer list for the CSV)
    attn_template = {
        "hidden_input": d([B, T, H]),
        "q_proj_weight": d([q_dim, H]),
        "k_proj_weight": d([kv_dim, H]),
        "v_proj_weight": d([kv_dim, H]),
        "o_proj_weight": d([H, q_dim]),
        "q_activation": d([B, n_heads, T, head_dim]),
        "k_activation": d([B, n_kv, T, head_dim]),
        "v_activation": d([B, n_kv, T, head_dim]),
        "attention_score": d([B, n_heads, T, T]),
        "attention_probability": d([B, n_heads, T, T]),
        "attention_output_before_o_proj": d([B, T, q_dim]),
        "attention_output_after_o_proj": d([B, T, H]),
        "residual_output": d([B, T, H]),
    }
    attention = {
        "note": "shapes identical across all %d layers (prefill, B=%d, T=%d)"
        % (n_layers, B, T),
        "per_layer_template": attn_template,
        "num_layers": n_layers,
    }

    # KV cache
    kv_after_prefill = [B, n_kv, T, head_dim]
    kv_steps = {}
    for s in representative_decode_steps:
        kv_steps[f"after_decode_step_{s}"] = d([B, n_kv, T + int(s), head_dim])
    kv_cache = {
        "key_cache_after_prefill": d(kv_after_prefill),
        "value_cache_after_prefill": d(kv_after_prefill),
        "key_cache_representative_decode": kv_steps,
        "value_cache_representative_decode": kv_steps,
        "cache_dtype": exec_meta.get("fold_dtype") or dtype,
        "cache_device": device,
        "cache_num_layers": n_layers,
        "cache_sequence_length_after_prefill": T,
        "cache_head_dim": head_dim,
        "cache_num_key_value_heads": n_kv,
    }

    mlp_template = {
        "mlp_input": d([B, T, H]),
        "gate_proj_weight": d([I, H]),
        "up_proj_weight": d([I, H]),
        "down_proj_weight": d([H, I]),
        "gate_activation": d([B, T, I]),
        "up_activation": d([B, T, I]),
        "nonlinear_input": d([B, T, I]),
        "nonlinear_output": d([B, T, I]),
        "down_input": d([B, T, I]),
        "down_output": d([B, T, H]),
        "mlp_residual_output": d([B, T, H]),
    }
    mlp = {"per_layer_template": mlp_template, "num_layers": n_layers}

    # nonlinear ops
    lifted = op_backend == "amulet_migrated"
    lk = int(lift_k) if lift_k else (2 if lifted else None)
    silu_entry = {
        "op_type": "silu",
        "op_input_shape": _shape([B, T, I]),
        "op_output_shape": _shape([B, T, I]),
        "backend_used": op_backend,
        "lifted_representation_used": lifted,
        "lift_k": lk,
        "lifted_tensor_shape": _shape([B, T, I, lk]) if lifted else None,
        "auxiliary_matrix_shape": _shape([I, lk]) if lifted else None,
        "source": "derived_from_config",
    }
    softmax_entry = {
        "op_type": "softmax",
        "op_input_shape": _shape([B, n_heads, T, T]),
        "op_output_shape": _shape([B, n_heads, T, T]),
        "backend_used": op_backend,
        "lifted_representation_used": False,
        "note": ("amulet_migrated migrates the elementwise exp to the "
                 "accelerator, keeps a trusted row-max reduction"
                 if lifted else "trusted softmax"),
        "source": "derived_from_config",
    }
    rmsnorm_entry = {
        "op_type": "rmsnorm",
        "op_input_shape": _shape([B, T, H]),
        "op_output_shape": _shape([B, T, H]),
        "backend_used": op_backend,
        "lifted_representation_used": False,
        "note": ("input_layernorm + post_attention_layernorm per layer "
                 "(2 per layer) + final norm"),
        "source": "derived_from_config",
    }
    nonlinear = {
        "per_layer_ops": [rmsnorm_entry, softmax_entry, silu_entry],
        "ops_per_layer": {"rmsnorm": 2, "softmax": 1, "silu": 1},
        "num_layers": n_layers,
        "final_rmsnorm": rmsnorm_entry,
    }

    lm_head = {
        "final_hidden": m("final_hidden", [B, 1, H],
                          note="last-position hidden used for logits"),
        "lm_head_weight": d([vocab, H]),
        "logits": m("logits", [B, 1, vocab],
                    note="masked last-position logits over the wire"),
        "vocab_mask": (d([vocab], note="present iff package carries a "
                         "vocab/output mask")
                       if manifest.get("has_vocab_mask") else
                       {"shape": None, "source": "derived_from_config",
                        "note": "no vocab mask advertised"}),
        "recovered_logits": m("recovered_logits", [B, vocab],
                              note="trusted-side recovered logits"),
        "sampled_token": m("sampled_token", [B, 1]),
    }

    masks_pad = {
        "residual_mask_shape": _shape([H, H]) if exec_meta.get("mask_family")
        else None,
        "mask_family": exec_meta.get("mask_family"),
        "q_proj_local_mask": _shape([q_dim, q_dim]),
        "k_proj_local_mask": _shape([kv_dim, kv_dim]),
        "v_proj_local_mask": _shape([kv_dim, kv_dim]),
        "o_proj_local_mask": _shape([H, H]),
        "gate_proj_local_mask": _shape([I, I]),
        "up_proj_local_mask": _shape([I, I]),
        "down_proj_local_mask": _shape([H, H]),
        "lm_head_output_mask": _shape([vocab, vocab]),
        "pad_tensor_shape_per_module": {
            "q_proj": _shape([H]), "k_proj": _shape([H]),
            "v_proj": _shape([H]), "o_proj": _shape([q_dim]),
            "gate_proj": _shape([H]), "up_proj": _shape([H]),
            "down_proj": _shape([I]), "lm_head": _shape([H])},
        "pad_compensation_tensor_shape_per_module": {
            "q_proj": _shape([q_dim]), "k_proj": _shape([kv_dim]),
            "v_proj": _shape([kv_dim]), "o_proj": _shape([H]),
            "gate_proj": _shape([I]), "up_proj": _shape([I]),
            "down_proj": _shape([H]), "lm_head": _shape([vocab])},
        "linear_boundary_pad": manifest.get("linear_boundary_pad"),
        "pad_enabled": manifest.get("linear_boundary_pad"),
        "nonlinear_backend": manifest.get("nonlinear_backend")
        or normalize_nonlinear_backend(nonlinear_backend),
        "nonlinear_op_backend": op_backend,
        "package_seq_len": exec_meta.get("seq_len") or manifest.get("seq_len"),
        "package_dtype": manifest.get("dtype") or dtype,
        "num_shards": manifest.get("num_shards"),
        "package_size_gb": manifest.get("package_size_gb"),
        "source": ("derived_from_config+manifest; local/pad mask shapes are the "
                   "structural sizes -- raw mask tensors are NOT in the package "
                   "(contains_mask_secrets=false)"),
    }

    return {
        "model": model,
        "inputs": inputs,
        "attention": attention,
        "kv_cache": kv_cache,
        "mlp": mlp,
        "nonlinear": nonlinear,
        "lm_head": lm_head,
        "masks_pad_package": masks_pad,
    }


def dimension_audit_to_rows(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the per-layer templates into CSV rows (one per layer/op)."""
    rows: List[Dict[str, Any]] = []
    n_layers = int(audit["attention"].get("num_layers", 0))
    for li in range(n_layers):
        for group, tmpl in (("attention",
                             audit["attention"]["per_layer_template"]),
                            ("mlp", audit["mlp"]["per_layer_template"])):
            for name, entry in tmpl.items():
                rows.append({
                    "layer": li, "group": group, "tensor": name,
                    "shape": "x".join(
                        str(s) for s in (entry.get("shape") or [])),
                    "source": entry.get("source"),
                })
        for op in audit["nonlinear"]["per_layer_ops"]:
            rows.append({
                "layer": li, "group": "nonlinear", "tensor": op["op_type"],
                "shape": "x".join(str(s) for s in (op["op_input_shape"] or [])),
                "source": op.get("source"),
                "backend_used": op.get("backend_used"),
            })
    return rows


# --------------------------------------------------------------------------- #
# seq-len compatibility
# --------------------------------------------------------------------------- #
def seq_len_compatible(
    package_seq_len: Optional[int],
    prompt_len: int,
    max_new_tokens: int,
) -> Tuple[bool, str]:
    """The folded package fixes a rope/kv horizon = seq_len (+ max_new_tokens).
    A prompt longer than the package seq_len cannot be served.
    """
    if not package_seq_len:
        return True, "package seq_len unknown; skipping compatibility gate"
    need = int(prompt_len)
    if need > int(package_seq_len):
        return False, (
            f"prompt length {need} exceeds package seq_len "
            f"{package_seq_len}; rebuild the package with a larger --seq-len")
    horizon = int(prompt_len) + int(max_new_tokens)
    if horizon > int(package_seq_len):
        return True, (
            f"prompt+max_new_tokens ({horizon}) exceeds package seq_len "
            f"{package_seq_len}; generation may truncate at the horizon")
    return True, "ok"


# --------------------------------------------------------------------------- #
# generation-quality metrics
# --------------------------------------------------------------------------- #
def _looks_repeated(token_ids: Sequence[int], window: int = 8,
                    min_repeats: int = 6) -> bool:
    """Detect degenerate same-token / short-cycle repetition at the tail."""
    ids = list(token_ids)
    if len(ids) < window * 2:
        return False
    tail = ids[-window * min_repeats:] if len(ids) >= window * min_repeats \
        else ids
    # all-same-token collapse
    if len(set(ids[-min_repeats:])) == 1 and len(ids) >= min_repeats:
        return True
    # short-cycle repeat
    for p in range(1, window + 1):
        seg = tail[-p:]
        reps = 0
        i = len(tail) - p
        while i - p >= 0 and tail[i - p:i] == seg:
            reps += 1
            i -= p
        if reps >= min_repeats:
            return True
    return False


def generation_quality(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate malformed / empty / repeated / early-EOS / truncation counts."""
    n = len(records)
    empty = repeated = early_eos = truncated = malformed = 0
    for r in records:
        text = (r.get("generated_text") or "").strip()
        ids = r.get("generated_token_ids") or []
        fin = r.get("finish_reason")
        if not text or not ids:
            empty += 1
        if _looks_repeated(ids):
            repeated += 1
        if fin == "eos" and len(ids) <= 1:
            early_eos += 1
        if fin == "length":
            truncated += 1
        # malformed: non-empty but no printable content
        if text and not re.search(r"[A-Za-z0-9一-鿿]", text):
            malformed += 1
    return {
        "num_prompts": n,
        "empty_output_count": empty,
        "repeated_output_count": repeated,
        "early_eos_count": early_eos,
        "max_length_truncation_count": truncated,
        "malformed_output_count": malformed,
        "clean_output_count": n - empty - repeated - malformed,
    }


# --------------------------------------------------------------------------- #
# current vs trusted_shortcut comparison
# --------------------------------------------------------------------------- #
def _first_mismatch(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def compare_backends(
    records_a: Sequence[Dict[str, Any]],
    records_b: Sequence[Dict[str, Any]],
    *,
    label_a: str = "current",
    label_b: str = "trusted_shortcut",
) -> Dict[str, Any]:
    """Token/text-level comparison of two runs, matched by ``prompt_id``."""
    by_id_a = {r["prompt_id"]: r for r in records_a}
    by_id_b = {r["prompt_id"]: r for r in records_b}
    common = [pid for pid in by_id_a if pid in by_id_b]

    per_prompt = []
    exact_text = exact_tokens = 0
    tok_match_num = tok_match_den = 0
    for pid in common:
        ra, rb = by_id_a[pid], by_id_b[pid]
        ta = ra.get("generated_token_ids") or []
        tb = rb.get("generated_token_ids") or []
        mm = _first_mismatch(ta, tb)
        matched = sum(1 for x, y in zip(ta, tb) if x == y)
        denom = max(len(ta), len(tb))
        tok_match_num += matched
        tok_match_den += denom
        text_eq = (ra.get("generated_text") == rb.get("generated_text"))
        tok_eq = (ta == tb)
        exact_text += int(text_eq)
        exact_tokens += int(tok_eq)
        per_prompt.append({
            "prompt_id": pid,
            "first_mismatch_position": mm,
            "token_match": (matched / denom) if denom else 1.0,
            "exact_text_match": text_eq,
            "exact_token_match": tok_eq,
            "len_a": len(ta), "len_b": len(tb),
            "output_length_difference": len(ta) - len(tb),
        })
    n = len(common)
    return {
        "label_a": label_a, "label_b": label_b,
        "num_compared": n,
        "token_match_rate": (tok_match_num / tok_match_den)
        if tok_match_den else None,
        "exact_text_match_rate": (exact_text / n) if n else None,
        "exact_token_match_rate": (exact_tokens / n) if n else None,
        "per_prompt": per_prompt,
    }


# --------------------------------------------------------------------------- #
# output writers
# --------------------------------------------------------------------------- #
def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def write_generations_jsonl(path: str | Path,
                            records: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_metrics_csv(path: str | Path,
                      records: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["prompt_id", "backend", "output_length", "finish_reason",
            "latency_sec", "tokens_per_sec", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def write_per_layer_csv(path: str | Path,
                        rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["layer", "group", "tensor", "shape", "source", "backend_used"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _yn(v: Any) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "n/a" if v is None else str(v)


def write_report_md(
    path: str | Path,
    *,
    config: Dict[str, Any],
    backend: str,
    backend_verify: Dict[str, Any],
    nonlinear_summary: Dict[str, Any],
    quality: Dict[str, Any],
    perf: Dict[str, Any],
    dimension_audit: Dict[str, Any],
    generations: Sequence[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]] = None,
    ifeval: Optional[Dict[str, Any]] = None,
    seq_len_note: str = "",
    errors: Optional[Sequence[str]] = None,
) -> None:
    """Human-readable report.md for one backend run."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    ap = L.append
    ap(f"# Generation-task evaluation — `{backend}` backend\n")

    ap("## 1. Backend verification (no silent fallback)\n")
    bv = backend_verify
    ap(f"- expected nonlinear_backend: `{bv.get('expected_nonlinear_backend')}` "
       f"→ op_backend `{bv.get('expected_op_backend')}`")
    ap(f"- worker actual: nonlinear_backend `{bv.get('actual_nonlinear_backend')}`, "
       f"op_backend `{bv.get('actual_op_backend')}`")
    ap(f"- **verification passed: {_yn(bv.get('backend_verification_passed'))}** "
       f"| silent fallback detected: {_yn(bv.get('silent_fallback_detected'))}")
    for r in bv.get("reasons") or []:
        ap(f"  - ⚠️ {r}")
    ap("")

    ap("## 2. Nonlinear execution evidence\n")
    ns = nonlinear_summary
    ap(f"- nonlinear_backend / op_backend (measured): "
       f"`{ns.get('nonlinear_backend')}` / `{ns.get('nonlinear_op_backend')}`")
    if "amulet_real_path_executed" in ns:
        ap(f"- **amulet_migrated real path executed: "
           f"{_yn(ns.get('amulet_real_path_executed'))}** "
           f"| tag-only: {_yn(ns.get('trusted_shortcut_tag_only'))}")
    ap(f"- nonlinear_execution_evidence_missing: "
       f"{_yn(ns.get('nonlinear_execution_evidence_missing'))}")
    c = ns.get("counters") or {}
    ap(f"- lifted_nonlinear_ops_count: {c.get('lifted_nonlinear_ops_count')} "
       f"| lift_k: {c.get('lift_k')} | lifted_gpu_bytes: "
       f"{c.get('lifted_gpu_bytes')} | trusted_nonlinear_ops_count: "
       f"{c.get('trusted_nonlinear_ops_count')}")
    noc = ns.get("nonlinear_ops_count") or {}
    ap(f"- nonlinear ops by type: rmsnorm={noc.get('rmsnorm')}, "
       f"softmax={noc.get('softmax')}, silu={noc.get('silu')}, "
       f"gelu={noc.get('gelu')}")
    for dgn in ns.get("diagnostics") or []:
        ap(f"  - {dgn}")
    ap("")

    ap("## 3. Dimension audit completeness\n")
    present = [s for s in REQUIRED_AUDIT_SECTIONS if s in dimension_audit]
    ap(f"- sections present: {len(present)}/{len(REQUIRED_AUDIT_SECTIONS)} "
       f"({', '.join(present)})")
    md = dimension_audit.get("model", {})
    ap(f"- model: {md.get('num_layers')} layers, hidden={md.get('hidden_size')}, "
       f"inter={md.get('intermediate_size')}, heads={md.get('num_attention_heads')}, "
       f"kv_heads={md.get('num_key_value_heads')}, head_dim={md.get('head_dim')}, "
       f"vocab={md.get('vocab_size')}, dtype={md.get('dtype')}, "
       f"seq_len={md.get('package_seq_len')}")
    ap(f"- full per-layer/KV/MLP/nonlinear/LM-head/mask shapes in "
       f"`dimension_audit.json` + `per_layer_dimensions.csv`.")
    if seq_len_note:
        ap(f"- seq_len compatibility: {seq_len_note}")
    ap("")

    ap("## 4. Generation quality\n")
    q = quality
    ap(f"- prompts: {q.get('num_prompts')} | clean: {q.get('clean_output_count')} "
       f"| empty: {q.get('empty_output_count')} | repeated: "
       f"{q.get('repeated_output_count')} | early-EOS: {q.get('early_eos_count')} "
       f"| length-truncated: {q.get('max_length_truncation_count')} | malformed: "
       f"{q.get('malformed_output_count')}")
    if ifeval:
        ap(f"- IFEval (approx, coverage {ifeval.get('coverage_pct')}%): "
           f"strict_prompt={ifeval.get('strict_prompt_acc')} "
           f"loose_prompt={ifeval.get('loose_prompt_acc')} "
           f"strict_inst={ifeval.get('strict_inst_acc')} "
           f"loose_inst={ifeval.get('loose_inst_acc')}")
    ap("")

    ap("## 5. Performance\n")
    p = perf
    ap(f"- total prompts: {p.get('num_prompts')} | total tokens: "
       f"{p.get('total_tokens')} | total latency: {p.get('total_latency_sec')}s")
    ap(f"- tokens/sec: {p.get('tokens_per_sec')} | mean per-prompt latency: "
       f"{p.get('mean_latency_sec')}s | mean first-token latency: "
       f"{p.get('mean_first_token_latency_sec')}s")
    ap(f"- peak GPU memory: {p.get('peak_gpu_memory_mb')} MB | resident cache "
       f"active: {_yn(p.get('resident_cache_active'))}")
    ap(f"- boundary calls: {p.get('boundary_calls')} | gpu calls: "
       f"{p.get('gpu_calls')} | trusted bytes: {p.get('trusted_bytes')} | "
       f"gpu bytes: {p.get('gpu_bytes')}")
    if p.get("decode_bottleneck_stage"):
        ap(f"- decode bottleneck stage: {p.get('decode_bottleneck_stage')}")
    ap("")

    if comparison:
        ap("## 6. current vs trusted_shortcut comparison\n")
        cm = comparison
        ap(f"- compared {cm.get('num_compared')} prompts "
           f"({cm.get('label_a')} vs {cm.get('label_b')})")
        ap(f"- token match rate: {cm.get('token_match_rate')} | exact text "
           f"match: {cm.get('exact_text_match_rate')} | exact token match: "
           f"{cm.get('exact_token_match_rate')}")
        firsts = [pp for pp in cm.get("per_prompt", [])
                  if pp.get("first_mismatch_position") is not None]
        if firsts:
            ap(f"- {len(firsts)} prompts diverge; earliest divergences:")
            for pp in sorted(firsts,
                             key=lambda x: x["first_mismatch_position"])[:5]:
                ap(f"  - {pp['prompt_id']}: first mismatch @ token "
                   f"{pp['first_mismatch_position']} "
                   f"(len {pp['len_a']} vs {pp['len_b']})")
        ap("")

    ap("## 7. Generation examples\n")
    for r in list(generations)[:3]:
        ap(f"**{r.get('prompt_id')}** ({r.get('backend')}, "
           f"finish={r.get('finish_reason')}, {r.get('output_length')} tokens)")
        prm = (r.get("prompt") or "")[:200]
        gen = (r.get("generated_text") or "")[:400]
        ap(f"> prompt: {prm}")
        ap(f"> output: {gen}\n")

    ap("## 8. Errors / failures\n")
    errs = list(errors or [])
    err_recs = [r for r in generations if r.get("error")]
    if not errs and not err_recs:
        ap("- none")
    for e in errs:
        ap(f"- {e}")
    for r in err_recs:
        ap(f"- {r.get('prompt_id')}: {r.get('error')}")
    ap("")

    ap("## 9. Next steps\n")
    ap("- scale IFEval subset (20 → 100 → 541) once smoke passes")
    ap("- run the paired backend (current ↔ trusted_shortcut) for the "
       "comparison file if not already present")
    ap("")

    with open(path, "w") as f:
        f.write("\n".join(L))
