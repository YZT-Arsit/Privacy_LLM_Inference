"""Unified ObfuscaTune-Qwen result schema (task spec, section 10).

Builds the single JSON record shape shared with our amulet-style /
trusted-shortcut outputs so rows are directly mergeable. The ``security_proxy``
block is deliberately conservative and HONEST for ObfuscaTune's threat model:
Q/K/V, attention scores and the KV cache are plaintext intermediates exposed
outside the simulated TEE, so ``public_qkv=true``, ``public_attention_scores=true``
and ``protected_kv_cache`` is at most ``"partial"`` -- never ``"true"`` unless a
real cache-obfuscation primitive is implemented and tested.
"""

from __future__ import annotations

from typing import Any

from .config import method_cond

_RANDOM_MATRIX_TYPE = {
    "orthogonal": "orthogonal",
    "random": "gaussian",
    "cond": "conditioned",
}
_INVERSE_METHOD = {
    "orthogonal": "transpose",
    "random": "inverse",
    "cond": "svd_constructed",
}


def method_id(random_matrix_type: str, condition_number: float = 1.0) -> str:
    if random_matrix_type == "random":
        return "obfuscatune_qwen_random"
    if random_matrix_type == "cond" and condition_number > 1.0:
        c = int(condition_number) if float(condition_number).is_integer() else condition_number
        return f"obfuscatune_qwen_cond_{c}"
    return "obfuscatune_qwen_orthogonal"


def qwen_security_proxy(*, embedding_obfuscated: bool = True) -> dict[str, Any]:
    """Conservative, honest security proxy for ObfuscaTune-Qwen."""
    return {
        "protected_input": "partial" if embedding_obfuscated else "false",
        "protected_model_weights": "true",
        "protected_lora": "not_implemented",
        "protected_kv_cache": "false",       # plaintext K/V exposed outside TEE
        "protected_logits": "true",          # LM head runs in the simulated TEE
        "public_qkv": True,
        "public_attention_scores": True,
        "tee_auth_assumed": True,
        "notes": (
            "ObfuscaTune's security model differs from amulet-style: it protects "
            "proprietary (secret) model weights + private data under an "
            "authenticated TEE. Q/K/V, attention scores and KV cache are "
            "plaintext intermediates exposed OUTSIDE the simulated TEE. Not "
            "equivalent to our scheme (public base weights; protect user input / "
            "LoRA / KV cache / logits)."
        ),
    }


def qwen_result_row(
    *,
    method: str,
    model_name_or_path: str,
    mode: str,                       # "prefill" | "decode" | "generation"
    dtype: str,
    batch_size: int,
    seq_len: int,
    arch,                            # QwenArch
    correctness: dict[str, Any],
    cache: dict[str, Any] | None,
    cost: dict[str, Any],
    numerical: dict[str, Any],
    security_proxy: dict[str, Any] | None = None,
    token_match_rate: float | None = None,
    exact_token_match: bool | None = None,
) -> dict[str, Any]:
    corr = {
        "max_abs_error": correctness.get("max_abs_error"),
        "mean_abs_error": correctness.get("mean_abs_error"),
        "relative_l2_error": correctness.get("relative_l2_error"),
        "logits_argmax_match_rate": correctness.get("logits_argmax_match_rate"),
        "token_match_rate": token_match_rate,
        "exact_token_match": exact_token_match,
    }
    return {
        "method": method,
        "model_family": "qwen",
        "model_name_or_path": model_name_or_path,
        "mode": mode,
        "dtype": dtype,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_layers": arch.num_layers,
        "hidden_size": arch.hidden_size,
        "num_attention_heads": arch.num_attention_heads,
        "num_key_value_heads": arch.num_key_value_heads,
        "intermediate_size": arch.intermediate_size,
        "correctness": corr,
        "cache": cache,
        "security_proxy": security_proxy or qwen_security_proxy(),
        "cost": cost,
        "numerical": numerical,
    }


def numerical_block(
    *,
    random_matrix_type: str,
    condition_number_target: float | None,
    condition_number_mean: float,
    condition_number_max: float,
    nan_or_inf_count: int,
) -> dict[str, Any]:
    return {
        "random_matrix_type": _RANDOM_MATRIX_TYPE.get(random_matrix_type, random_matrix_type),
        "condition_number_target": condition_number_target,
        "condition_number_mean": condition_number_mean,
        "condition_number_max": condition_number_max,
        "inverse_method": _INVERSE_METHOD.get(random_matrix_type, "inverse"),
        "nan_or_inf_count": nan_or_inf_count,
    }


def cost_block(
    *,
    wall_time_ms: float,
    slowdown_vs_unprotected: float | None,
    phase_metrics: dict[str, Any],
    tee_param_ratio: float,
    gpu_param_ratio: float,
) -> dict[str, Any]:
    return {
        "wall_time_ms": wall_time_ms,
        "slowdown_vs_unprotected": slowdown_vs_unprotected,
        "boundary_calls_per_forward": phase_metrics.get("boundary_calls"),
        "trusted_transfer_bytes": phase_metrics.get("tee_transfer_bytes"),
        "trusted_compute_flops": None,   # not separately estimated in the simulator
        "untrusted_compute_flops": None,
        "tee_param_ratio": tee_param_ratio,
        "gpu_param_ratio": gpu_param_ratio,
    }


def estimate_param_ratios(model) -> tuple[float, float]:
    """TEE vs GPU parameter split: obfuscated big linears -> GPU; norms/embed/head -> TEE."""
    tee, gpu = 0, 0
    big = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for name, p in model.named_parameters():
        if any(b in name for b in big) and "weight" in name:
            gpu += p.numel()
        else:
            tee += p.numel()
    total = tee + gpu
    if total == 0:
        return 0.0, 0.0
    return tee / total, gpu / total


__all__ = [
    "method_id",
    "qwen_security_proxy",
    "qwen_result_row",
    "numerical_block",
    "cost_block",
    "estimate_param_ratios",
]
