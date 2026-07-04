"""Metrics + unified comparison schema for the ObfuscaTune baseline.

Three metric families are provided:

  * correctness  -- error of an obfuscated run against its plaintext reference
    (max/mean abs error, relative L2, argmax/token match).
  * performance proxy -- wall time plus *simulated* TEE-boundary accounting
    (transfer bytes, boundary calls, TEE vs untrusted FLOP estimates). These
    are simulator counters, not measurements of a real TEE.
  * numerical stability -- condition-number stats, inverse method, NaN/Inf count.

The :func:`comparison_row` builder emits the unified schema from the task spec
so an ObfuscaTune row can sit in the same table as our amulet-style /
trusted-shortcut rows. ``security_proxy`` is a *structural declaration* with an
explicit ``notes`` field -- ObfuscaTune's threat model (protect model + data
under secret weights) is NOT our threat model (protect user input / LoRA / KV
cache with public base weights), and the two must never be conflated.
"""

from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def correctness_metrics(
    approx: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    a = approx.detach().to(torch.float64)
    r = reference.detach().to(torch.float64)
    diff = a - r
    denom = r.norm().item()
    rel_l2 = (diff.norm().item() / denom) if denom > 0 else 0.0
    return {
        "max_abs_error": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "relative_l2_error": float(rel_l2),
    }


def argmax_match_rate(
    approx_logits: torch.Tensor, reference_logits: torch.Tensor
) -> float:
    """Fraction of positions whose argmax over the last dim agrees."""
    a = approx_logits.reshape(-1, approx_logits.shape[-1]).argmax(-1)
    r = reference_logits.reshape(-1, reference_logits.shape[-1]).argmax(-1)
    if a.numel() == 0:
        return 1.0
    return float((a == r).float().mean().item())


def nan_or_inf_count(t: torch.Tensor) -> int:
    return int((~torch.isfinite(t)).sum().item())


# ---------------------------------------------------------------------------
# FLOP proxies (2*m*n*k for an (m,k)@(k,n) matmul)
# ---------------------------------------------------------------------------
def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * int(m) * int(k) * int(n)


# ---------------------------------------------------------------------------
# Unified comparison schema (task spec, section 7)
# ---------------------------------------------------------------------------
def security_proxy_obfuscatune(cond_note: str = "") -> dict[str, Any]:
    """Structural security declaration for ObfuscaTune (paper threat model).

    Under ObfuscaTune the protection target is the *proprietary model weights*
    and the *private data*, assuming an authenticated TEE. Q/K/V and the MLP
    intermediate are recovered in plaintext outside the TEE (the mask cancels),
    so it is NOT an open-weight defense.
    """
    notes = (
        "ObfuscaTune paper threat model: protect proprietary model weights + "
        "private data, secret weights + authenticated TEE assumed. Q/K/V and "
        "the MLP intermediate are exposed in plaintext outside the TEE (the "
        "mask cancels). NOT comparable to our user-asset threat model (public "
        "base weights; protect user input / LoRA / KV cache / logits)."
    )
    if cond_note:
        notes = f"{notes} {cond_note}"
    return {
        "protected_input": True,          # input embedding obfuscated leaving TEE
        "protected_model_weights": True,  # weights obfuscated (secret-weight assumption)
        "protected_lora": True,           # LoRA params obfuscated with the layer
        "protected_kv_cache": False,      # KV derived from plaintext Q/K/V outside TEE
        "public_qkv": True,               # TRUE Q/K/V exposed to untrusted side
        "public_attention_scores": True,  # softmax runs in TEE, but Q/K/V are public
        "tee_auth_assumed": True,
        "notes": notes,
    }


def comparison_row(
    *,
    method: str,
    correctness: dict[str, float],
    cost: dict[str, Any],
    numerical: dict[str, Any],
    token_match_rate: float | None = None,
    exact_token_match: bool | None = None,
    security_proxy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one row of the unified method-comparison schema."""
    corr = {
        "max_abs_error": correctness.get("max_abs_error"),
        "mean_abs_error": correctness.get("mean_abs_error"),
        "relative_l2_error": correctness.get("relative_l2_error"),
        "token_match_rate": token_match_rate,
        "exact_token_match": exact_token_match,
    }
    return {
        "method": method,
        "correctness": corr,
        "security_proxy": security_proxy or security_proxy_obfuscatune(),
        "cost": cost,
        "numerical": numerical,
    }


def cost_proxy(
    *,
    wall_time_ms: float,
    slowdown_vs_unprotected: float | None,
    boundary_calls_per_forward: int,
    trusted_transfer_bytes: int,
    trusted_compute_flops: int,
    untrusted_compute_flops: int,
    tee_param_ratio: float,
    gpu_param_ratio: float,
) -> dict[str, Any]:
    return {
        "wall_time_ms": wall_time_ms,
        "slowdown_vs_unprotected": slowdown_vs_unprotected,
        "boundary_calls_per_forward": boundary_calls_per_forward,
        "trusted_transfer_bytes": trusted_transfer_bytes,
        "trusted_compute_flops": trusted_compute_flops,
        "untrusted_compute_flops": untrusted_compute_flops,
        "tee_param_ratio": tee_param_ratio,
        "gpu_param_ratio": gpu_param_ratio,
    }


def numerical_proxy(
    *,
    random_matrix_type: str,
    condition_number_mean: float,
    condition_number_max: float,
    inverse_method: str,
    dtype: str,
    nan_or_inf_count: int,
) -> dict[str, Any]:
    return {
        "random_matrix_type": random_matrix_type,
        "condition_number_mean": condition_number_mean,
        "condition_number_max": condition_number_max,
        "inverse_method": inverse_method,
        "dtype": dtype,
        "nan_or_inf_count": nan_or_inf_count,
    }


__all__ = [
    "correctness_metrics",
    "argmax_match_rate",
    "nan_or_inf_count",
    "matmul_flops",
    "security_proxy_obfuscatune",
    "comparison_row",
    "cost_proxy",
    "numerical_proxy",
]
