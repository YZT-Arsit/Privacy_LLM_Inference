"""Tests for the ObfuscaTune baseline (arXiv:2407.02960).

Verifies (a) the obfuscated linear / block is utility-preserving with orthogonal
matrices (kappa=1), (b) the condition-number ablation (their Table 2 trend:
high kappa -> large numerical error), and (c) the structural facts that make it a
contrast to our A_rightmul scheme: TRUE Q/K/V exposed to the untrusted side and
every non-linearity inside the TEE (per-block boundary crossings).
"""

from __future__ import annotations

import torch

from pllo.baselines.obfuscatune import (
    ObfuscaTune,
    ObfuscaTuneConfig,
    matrix_with_condition_number,
)


def _weights(d, di, g, dtype):
    def r(a, b):
        return torch.randn(a, b, generator=g, dtype=dtype) * 0.05
    return {"Wq": r(d, d), "Wk": r(d, d), "Wv": r(d, d), "Wo": r(d, d),
            "W1": r(d, di), "W2": r(di, d)}


def test_condition_number_matrix():
    g = torch.Generator().manual_seed(0)
    for kappa in (1.0, 8.0, 128.0):
        R, Rinv = matrix_with_condition_number(16, kappa, torch.float64, g)
        assert torch.allclose(R @ Rinv, torch.eye(16, dtype=torch.float64), atol=1e-6)
        cond = torch.linalg.cond(R).item()
        assert abs(cond - kappa) / kappa < 0.1 or kappa == 1.0


def test_obfuscated_linear_exact_orthogonal():
    b = ObfuscaTune(ObfuscaTuneConfig(dtype="float64", condition_number=1.0))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 32, generator=g, dtype=torch.float64)
    w = torch.randn(32, 32, generator=g, dtype=torch.float64)
    out = b.obfuscated_linear(x, w)
    assert out["max_abs_error"] < 1e-9
    assert out["gpu_output_is_plaintext_true_value"] is True


def test_block_utility_preserving_orthogonal():
    b = ObfuscaTune(ObfuscaTuneConfig(dtype="float64", condition_number=1.0))
    g = torch.Generator().manual_seed(2)
    x = torch.randn(6, 32, generator=g, dtype=torch.float64)
    w = _weights(32, 64, g, torch.float64)
    out = b.attention_mlp_block(x, w, n_heads=4)
    assert out["max_abs_error"] < 1e-8


def test_condition_number_error_trend():
    # reproduce the paper's Table 2 trend: higher kappa -> larger error (float32)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(6, 32, generator=g, dtype=torch.float32)
    w = _weights(32, 64, g, torch.float32)
    errs = {}
    for kappa in (1.0, 32.0, 160.0):
        b = ObfuscaTune(ObfuscaTuneConfig(dtype="float32", condition_number=kappa))
        errs[kappa] = b.attention_mlp_block(x, w, n_heads=4)["max_abs_error"]
    assert errs[1.0] <= errs[32.0] <= errs[160.0]
    assert errs[160.0] > errs[1.0]


def test_structural_true_qkv_exposed_and_tee_crossings():
    b = ObfuscaTune(ObfuscaTuneConfig(dtype="float64", condition_number=1.0))
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 32, generator=g, dtype=torch.float64)
    w = _weights(32, 64, g, torch.float64)
    out = b.attention_mlp_block(x, w, n_heads=4)
    assert "Q_plaintext" in out["exposed_plaintext_tensors"]
    assert "K_plaintext" in out["exposed_plaintext_tensors"]
    assert "V_plaintext" in out["exposed_plaintext_tensors"]
    assert out["boundary_crossings"] > 0            # nonlinears in TEE
    assert set(out["tee_nonlinear_ops"]) == {"layernorm", "softmax", "gelu"}


def test_structural_audit_contrast():
    b = ObfuscaTune()
    a = b.structural_audit(n_layers=28, total_params=100, tee_params=5)
    assert a["true_qkv_exposed_to_untrusted"] is True
    assert a["open_weight_secure"] is False
    assert a["nonlinears_in_tee"] is True
    assert a["tee_boundary_crossings_total"] > a["contrast_ours_A_rightmul"]["tee_boundary_crossings_total"]
    assert a["contrast_ours_A_rightmul"]["nonlinears_in_tee"] is False


def test_self_declaration():
    b = ObfuscaTune()
    d = b.declare.to_dict()
    assert d["name"] == "obfuscatune"
    assert d["exact_primitive_implemented"] is True
    assert d["full_system_reproduced"] is False
