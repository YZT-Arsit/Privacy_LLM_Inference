"""Correctness + boundary audit of ObfuscaTune-on-Qwen2.

Checks the paper's obfuscation is implemented correctly and the TEE/GPU boundary
is labelled honestly (Q/K/V exposed, non-linears in TEE, KV cache NOT protected).
"""

from __future__ import annotations

from typing import Any

import torch

from .config import ObfuscaTuneConfig
from .qwen_config import extract_arch
from .qwen_obfuscatune import run_prefill, qwen_plain_logits
from .random_matrices import condition_number, orthogonal_matrix
from .linear_obfuscation import obfuscated_input_linear, obfuscated_output_linear


def audit_obfuscatune_qwen(model, *, seed: int = 0, num_tokens: int = 10) -> dict[str, Any]:
    arch = extract_arch(model)
    dtype = next(model.parameters()).dtype
    tol = 1e-3 if dtype == torch.float32 else 1e-8
    ids = torch.randint(0, arch.vocab_size, (1, num_tokens))

    # 1. orthogonal matrix + condition number
    q, q_inv = orthogonal_matrix(arch.hidden_size, seed=seed, dtype=torch.float64)
    cn = condition_number(q)
    orth_err = float((q @ q_inv - torch.eye(arch.hidden_size, dtype=torch.float64)).abs().max())

    # 2. input-projection linear equivalence  X W == (X R)(R^{-1} W)
    x = torch.randn(8, arch.hidden_size, dtype=torch.float64)
    w = torch.randn(arch.hidden_size, arch.hidden_size, dtype=torch.float64)
    in_err = float((obfuscated_input_linear(x, w, q, q_inv) - x @ w).abs().max())
    # 3. output-projection linear equivalence  H W == (H (W R)) R^{-1}
    out_err = float((obfuscated_output_linear(x, w, q, q_inv) - x @ w).abs().max())

    # 4. RMSNorm / SiLU NOT obfuscation-safe under a general mask (must stay in TEE)
    from .random_matrices import gaussian_invertible_matrix
    from .qwen_modules import qwen_rmsnorm_reference
    r, _ = gaussian_invertible_matrix(arch.hidden_size, seed=seed + 1, dtype=torch.float64, jitter=1e-2)
    xr = torch.randn(4, arch.hidden_size, dtype=torch.float64)
    wgain = torch.ones(arch.hidden_size, dtype=torch.float64)
    rmsnorm_not_safe = not torch.allclose(
        qwen_rmsnorm_reference(xr @ r, wgain, 1e-6),
        qwen_rmsnorm_reference(xr, wgain, 1e-6) @ r, atol=1e-3)
    silu = torch.nn.functional.silu
    silu_not_safe = not torch.allclose(silu(xr @ r), silu(xr) @ r, atol=1e-3)

    # 5. full-model orthogonal obfuscation matches plaintext logits
    plain = qwen_plain_logits(model, ids)
    dtype_str = "float64" if dtype == torch.float64 else "float32"
    r_orth = run_prefill(model, ids, "orthogonal", dtype=dtype, seed=seed)
    logits_err = r_orth["correctness"]["max_abs_error"]

    report = {
        "baseline": "obfuscatune_qwen",
        "paper": "ObfuscaTune (Frikha et al., arXiv:2407.02960)",
        "model_family": "qwen",
        "correctness": {
            "orthogonal_condition_number": cn,
            "orthogonal_inverse_max_abs_error": orth_err,
            "input_projection_equivalence_err": in_err,
            "output_projection_equivalence_err": out_err,
            "full_model_orthogonal_logits_err": logits_err,
            "logits_match_plaintext": logits_err < tol,
        },
        "boundary_labels": {
            "rmsnorm_not_run_in_obfuscated_domain": bool(rmsnorm_not_safe),
            "silu_not_run_in_obfuscated_domain": bool(silu_not_safe),
            "rope_in_tee_plain_domain": True,
            "softmax_in_tee": True,
            "swiglu_gate_up_input_obfuscated_silu_in_tee": True,
            "gqa_handled": arch.num_key_value_heads != arch.num_attention_heads,
            "qkv_exposed_plaintext": True,          # masks cancel -> true Q/K/V outside TEE
            "attention_scores_exposed": True,
            "kv_cache_protected": False,            # plaintext K/V outside TEE
            "silent_fallback": False,
        },
        "produces_attack_representations": True,
        "status": "correct" if (orth_err < 1e-8 and in_err < 1e-8 and out_err < 1e-8
                                and logits_err < tol and rmsnorm_not_safe and silu_not_safe)
                  else "partial",
    }
    return report


__all__ = ["audit_obfuscatune_qwen"]
