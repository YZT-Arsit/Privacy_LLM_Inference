"""ObfuscaTune-Qwen attention correctness (q/k/v/o, RoPE, GQA)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.config import ObfuscaTuneConfig  # noqa: E402
from pllo.baselines.obfuscatune.qwen_config import (  # noqa: E402
    extract_arch, load_qwen, make_tiny_qwen_config)
from pllo.baselines.obfuscatune.qwen_modules import qwen_attention_obfuscated  # noqa: E402


def _model(n_heads=4, n_kv=2, dtype=torch.float64):
    cfg = make_tiny_qwen_config(num_attention_heads=n_heads, num_key_value_heads=n_kv,
                                hidden_size=32, head_dim=8)
    return load_qwen(tiny_random_qwen=cfg, seed=0, dtype=dtype)


def _cos_sin(model, hidden, S):
    pos = torch.arange(S).unsqueeze(0)
    return model.model.rotary_emb(hidden, pos)


def _reference_attn(model, hidden, S):
    """Reference: HF attention path is exercised via full block reconstruction.

    We reconstruct plaintext q/k/v/o with the same RoPE + GQA and compare the
    obfuscated attention to it.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    arch = extract_arch(model)
    attn = model.model.layers[0].self_attn
    cos, sin = _cos_sin(model, hidden, S)
    B = hidden.shape[0]
    q = attn.q_proj(hidden).view(B, S, arch.num_attention_heads, arch.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(B, S, arch.num_key_value_heads, arch.head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(B, S, arch.num_key_value_heads, arch.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    n_rep = arch.num_attention_heads // arch.num_key_value_heads
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    scale = arch.head_dim ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    att = torch.softmax(scores, dim=-1)
    ctx = (att @ v).transpose(1, 2).reshape(B, S, arch.num_attention_heads * arch.head_dim)
    return attn.o_proj(ctx)


def test_attention_orthogonal_matches_reference():
    model = _model()
    arch = extract_arch(model)
    S = 8
    hidden = torch.randn(1, S, arch.hidden_size, dtype=torch.float64)
    cos, sin = _cos_sin(model, hidden, S)
    cfg = ObfuscaTuneConfig(dtype="float64", random_matrix_type="orthogonal")
    out, audit, _m, _kv = qwen_attention_obfuscated(
        hidden, model.model.layers[0].self_attn, cos, sin, cfg,
        n_heads=arch.num_attention_heads, n_kv_heads=arch.num_key_value_heads,
        head_dim=arch.head_dim)
    ref = _reference_attn(model, hidden, S)
    assert torch.allclose(out, ref, atol=1e-9)
    assert audit["public_qkv"] is True
    assert audit["is_gqa"] is True


def test_attention_gqa_shapes_and_kv():
    model = _model(n_heads=4, n_kv=2)
    arch = extract_arch(model)
    S = 6
    hidden = torch.randn(1, S, arch.hidden_size, dtype=torch.float64)
    cos, sin = _cos_sin(model, hidden, S)
    cfg = ObfuscaTuneConfig(dtype="float64", random_matrix_type="orthogonal")
    _out, _audit, _m, (nk, nv) = qwen_attention_obfuscated(
        hidden, model.model.layers[0].self_attn, cos, sin, cfg,
        n_heads=arch.num_attention_heads, n_kv_heads=arch.num_key_value_heads,
        head_dim=arch.head_dim)
    # new k/v are pre-GQA-repeat: (B, n_kv, S, head_dim)
    assert nk.shape == (1, arch.num_key_value_heads, S, arch.head_dim)
    assert nv.shape == (1, arch.num_key_value_heads, S, arch.head_dim)


def test_attention_condition_number_error_trend():
    model = _model(dtype=torch.float32)
    arch = extract_arch(model)
    S = 8
    hidden = torch.randn(1, S, arch.hidden_size, dtype=torch.float32)
    cos, sin = _cos_sin(model, hidden, S)
    ref = _reference_attn(model, hidden, S)
    errs = {}
    for kind, kappa in [("orthogonal", 1.0), ("cond", 128.0), ("random", None)]:
        cfg = ObfuscaTuneConfig(dtype="float32", random_matrix_type=kind,
                                condition_number=kappa or 1.0)
        out, _a, _m, _kv = qwen_attention_obfuscated(
            hidden, model.model.layers[0].self_attn, cos, sin, cfg,
            n_heads=arch.num_attention_heads, n_kv_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim)
        errs[kind] = float((out - ref).abs().max().item())
    assert errs["orthogonal"] <= errs["cond"]
    assert torch.isfinite(torch.tensor(errs["random"]))
