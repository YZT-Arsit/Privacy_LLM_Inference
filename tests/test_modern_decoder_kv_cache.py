"""Stage 6.4c — KV cache tests."""

from __future__ import annotations

import json
import math
import re

import pytest
import torch
import torch.nn.functional as F

from pllo.cache.modern_decoder_kv_cache import (
    ModernDecoderKVCache,
    ModernDecoderLayerKVCache,
    init_empty_modern_decoder_kv_cache,
)
from pllo.experiments.gqa_probe import repeat_kv
from pllo.masks.mask_generator import generate_invertible_matrix


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_empty_cache_shape() -> None:
    cache = init_empty_modern_decoder_kv_cache(
        num_layers=3, batch_size=2,
        num_kv_heads=4, head_dim=8, attention_variant="mha",
        dtype=torch.float32, device=torch.device("cpu"),
    )
    assert len(cache.layers) == 3
    assert cache.total_seq_len == 0
    for layer in cache.layers:
        assert layer.seq_len == 0
        assert layer.num_kv_heads == 4
        assert layer.head_dim == 8
        assert layer.key_tilde.shape == (2, 4, 0, 8)
        assert layer.value_tilde.shape == (2, 4, 0, 8)


# ---------------------------------------------------------------------------
# append seq_len
# ---------------------------------------------------------------------------


def test_append_increases_seq_len() -> None:
    cache = init_empty_modern_decoder_kv_cache(
        num_layers=1, batch_size=1,
        num_kv_heads=2, head_dim=8, attention_variant="gqa",
        dtype=torch.float32, device=torch.device("cpu"),
    )
    layer = cache.layers[0]
    layer.key_tilde = torch.randn(1, 2, 4, 8)
    layer.value_tilde = torch.randn(1, 2, 4, 8)
    layer.seq_len = 4
    layer.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
    assert layer.seq_len == 5
    assert layer.key_tilde.shape == (1, 2, 5, 8)
    assert layer.value_tilde.shape == (1, 2, 5, 8)


# ---------------------------------------------------------------------------
# K_tilde / V_tilde append invariant
# ---------------------------------------------------------------------------


def test_k_tilde_append_invariant() -> None:
    """concat(K_tilde, k_new @ N_K) must equal (concat(K, k_new)) @ N_K."""
    H = 4
    D = 8
    n_k, _ = generate_invertible_matrix(D, torch.float32, torch.device("cpu"))
    n_k_stack = n_k.unsqueeze(0).expand(2, D, D)
    K_history = torch.randn(1, 2, 4, D)
    k_new = torch.randn(1, 2, 1, D)
    K_tilde_history = K_history @ n_k_stack.unsqueeze(0)
    k_tilde_new = k_new @ n_k_stack.unsqueeze(0)
    K_tilde_full = torch.cat([K_tilde_history, k_tilde_new], dim=-2)
    K_full = torch.cat([K_history, k_new], dim=-2)
    K_full_tilde_direct = K_full @ n_k_stack.unsqueeze(0)
    assert torch.allclose(K_tilde_full, K_full_tilde_direct, atol=1e-5)


def test_v_tilde_append_invariant() -> None:
    D = 8
    n_v, _ = generate_invertible_matrix(D, torch.float32, torch.device("cpu"))
    n_v_stack = n_v.unsqueeze(0).expand(2, D, D)
    V_history = torch.randn(1, 2, 4, D)
    v_new = torch.randn(1, 2, 1, D)
    V_tilde_history = V_history @ n_v_stack.unsqueeze(0)
    v_tilde_new = v_new @ n_v_stack.unsqueeze(0)
    V_tilde_full = torch.cat([V_tilde_history, v_tilde_new], dim=-2)
    V_full_tilde_direct = torch.cat([V_history, v_new], dim=-2) @ n_v_stack.unsqueeze(0)
    assert torch.allclose(V_tilde_full, V_full_tilde_direct, atol=1e-5)


# ---------------------------------------------------------------------------
# GQA repeat_kv on masked cache
# ---------------------------------------------------------------------------


def test_gqa_repeat_kv_masked_cache_score_invariant() -> None:
    """Score `q_tilde @ k_tilde_rep.T = q_rope @ k_rep.T` under GQA + masked cache."""
    B, Hq, Hk, S, D = 1, 4, 2, 6, 8
    group = Hq // Hk
    # Per-kv-head N_K with N_Q = N_K^{-T}.
    N_K_list, N_K_inv_list = [], []
    for _ in range(Hk):
        nk, nki = generate_invertible_matrix(D, torch.float32, torch.device("cpu"))
        N_K_list.append(nk)
        N_K_inv_list.append(nki)
    N_K_stack = torch.stack(N_K_list, dim=0)
    N_Q_per_q = torch.stack(
        [N_K_inv_list[i // group].transpose(-2, -1) for i in range(Hq)], dim=0
    )

    Q_rope = torch.randn(B, Hq, 1, D)
    K_rope = torch.randn(B, Hk, S, D)

    Q_tilde = Q_rope @ N_Q_per_q.unsqueeze(0)
    K_tilde = K_rope @ N_K_stack.unsqueeze(0)
    K_tilde_rep = repeat_kv(K_tilde, group)
    K_rep = repeat_kv(K_rope, group)

    scores_plain = Q_rope @ K_rep.transpose(-2, -1)
    scores_tilde = Q_tilde @ K_tilde_rep.transpose(-2, -1)
    assert torch.allclose(scores_plain, scores_tilde, atol=1e-4)


# ---------------------------------------------------------------------------
# JSON summary excludes raw tensors
# ---------------------------------------------------------------------------


def test_summary_dict_is_json_safe() -> None:
    cache = init_empty_modern_decoder_kv_cache(
        num_layers=2, batch_size=1,
        num_kv_heads=2, head_dim=8, attention_variant="gqa",
        dtype=torch.float32, device=torch.device("cpu"),
    )
    for layer in cache.layers:
        layer.key_tilde = torch.randn(1, 2, 4, 8)
        layer.value_tilde = torch.randn(1, 2, 4, 8)
        layer.n_k_stack = torch.randn(2, 8, 8)
        layer.n_v_stack = torch.randn(2, 8, 8)
        layer.n_v_inv_stack = torch.randn(2, 8, 8)
        layer.seq_len = 4
    summary = cache.summary_dict()
    blob = json.dumps(summary)
    assert "tensor(" not in blob
    assert "torch.Tensor" not in blob
    assert _LONG_NUMBER_ARRAY.search(blob) is None
    assert summary["num_layers"] == 2
    for layer_sum in summary["layers"]:
        assert "key_tilde_shape" in layer_sum
        assert "n_k_stack_fingerprint" in layer_sum
        assert "n_v_stack_fingerprint" in layer_sum
