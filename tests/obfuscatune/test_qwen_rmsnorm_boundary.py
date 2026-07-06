"""RMSNorm stays in the simulated TEE / plaintext domain (never obfuscated)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.qwen_config import (  # noqa: E402
    load_qwen, make_tiny_qwen_config, discover_layer_norms, is_rmsnorm)
from pllo.baselines.obfuscatune.qwen_modules import qwen_rmsnorm_reference  # noqa: E402
from pllo.baselines.obfuscatune.random_matrices import (  # noqa: E402
    gaussian_invertible_matrix, orthogonal_matrix)


def _model():
    cfg = make_tiny_qwen_config()
    return load_qwen(tiny_random_qwen=cfg, seed=0, dtype=torch.float64)


def test_layer_norms_are_rmsnorm():
    model = _model()
    norms = discover_layer_norms(model.model.layers[0])
    assert is_rmsnorm(norms["input_layernorm"])
    assert is_rmsnorm(norms["post_attention_layernorm"])


def test_reference_rmsnorm_matches_module():
    model = _model()
    norm = model.model.layers[0].input_layernorm
    x = torch.randn(1, 6, model.config.hidden_size, dtype=torch.float64)
    ref = qwen_rmsnorm_reference(x, norm.weight, model.config.rms_norm_eps)
    assert torch.allclose(ref, norm(x), atol=1e-9)


def test_rmsnorm_not_safe_under_general_mask():
    # For a GENERAL (non-orthogonal) obfuscation matrix, RMSNorm(X R) != RMSNorm(X) R,
    # so RMSNorm cannot run in the obfuscated domain -> it stays in the TEE.
    # (Note: an orthogonal R happens to be norm-preserving, so RMSNorm is
    # equivariant under it; ObfuscaTune keeps RMSNorm in the TEE regardless,
    # which is required once a non-orthogonal / conditioned mask is used.)
    torch.manual_seed(0)
    d = 16
    x = torch.randn(4, d, dtype=torch.float64)
    r, _ = gaussian_invertible_matrix(d, seed=1, dtype=torch.float64, jitter=1e-2)
    w = torch.ones(d, dtype=torch.float64)
    eps = 1e-6
    lhs = qwen_rmsnorm_reference(x @ r, w, eps)
    rhs = qwen_rmsnorm_reference(x, w, eps) @ r
    assert not torch.allclose(lhs, rhs, atol=1e-3)


def test_rmsnorm_is_equivariant_under_orthogonal():
    # Mathematical fact: RMSNorm depends only on the L2 norm, which an orthogonal
    # matrix preserves -> RMSNorm(X R) == RMSNorm(X) R for orthogonal R.
    torch.manual_seed(0)
    d = 16
    x = torch.randn(4, d, dtype=torch.float64)
    r, _ = orthogonal_matrix(d, seed=1, dtype=torch.float64)
    w = torch.ones(d, dtype=torch.float64)
    eps = 1e-12
    lhs = qwen_rmsnorm_reference(x @ r, w, eps)
    rhs = qwen_rmsnorm_reference(x, w, eps) @ r
    assert torch.allclose(lhs, rhs, atol=1e-8)
