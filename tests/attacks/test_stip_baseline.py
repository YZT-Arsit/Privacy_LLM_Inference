"""STIP-on-Qwen numerical correctness + permutation attack + autoregressive decode."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.qwen_config import extract_arch, load_qwen, make_tiny_qwen_config
from pllo.baselines.stip.permutation import make_stip_permutations, whole_head_permutation
from pllo.baselines.stip.qwen_stip import (build_transformed_qwen, stip_protect_qwen,
                                           stip_recover_logits, verify_linear_equivalence)
from pllo.attacks.permutation_multiset_attack import run_autoregressive_decode


def _model(dtype=torch.float64):
    cfg = make_tiny_qwen_config(num_attention_heads=4, num_key_value_heads=2,
                                hidden_size=32, head_dim=8, num_hidden_layers=3)
    return load_qwen(tiny_random_qwen=cfg, seed=0, dtype=dtype)


def test_theorem1_transformed_forward_recovers_plaintext():
    model = _model()
    arch = extract_arch(model)
    perms = make_stip_permutations(arch, seed=0)
    ids = torch.randint(0, arch.vocab_size, (1, 10))
    with torch.no_grad():
        plain = model(ids).logits
        cloud = build_transformed_qwen(model, perms)
        rec = stip_recover_logits(cloud(ids).logits, perms.pi_c)
    assert float((rec - plain).abs().max()) < 1e-5


def test_whole_head_permutation_is_gqa_grouped():
    p_q, p_kv = whole_head_permutation(4, 2, 8, seed=0)
    assert p_q.numel() == 32 and p_kv.numel() == 16
    # within-head order preserved (RoPE-safe): each head block is contiguous identity
    for h in range(2):
        block = p_kv[h * 8:(h + 1) * 8]
        assert (block - block[0] == torch.arange(8)).all()


def test_per_linear_equivalence():
    model = _model()
    arch = extract_arch(model)
    perms = make_stip_permutations(arch, seed=0)
    layer = model.model.layers[0]
    h = torch.randn(1, 6, arch.hidden_size, dtype=torch.float64)
    err = verify_linear_equivalence(layer.self_attn.q_proj, perms.pi, perms.pi_attn_q[0], h)
    assert err < 1e-8


def test_stip_protect_produces_leaky_reps():
    model = _model()
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    out = stip_protect_qwen(model, ids)
    assert out["logits_recovery_max_abs_error"] < 1e-6
    assert out["protected_embedding"].shape == out["plaintext_embedding"].shape
    assert "layer0_hidden_perm" in out


def test_autoregressive_decode_recovers_tokens():
    # capture AFTER one attention layer (layer1 input) so the positional anchor
    # from causal attention exists (paper requires >=1 attention op). Alg.1
    # positional match (unpermuted).
    model = _model(dtype=torch.float32)
    from pllo.attacks.qwen_hooks import QwenActivationCapture
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    cap = QwenActivationCapture(model, layers=[1]); obs = cap.run(ids)["layer1_input"][0]; cap.remove()
    r = run_autoregressive_decode(model, obs, ids[0], target_method="stip_qwen",
                                  layer=1, epsilon=1e-3, permuted=False)
    assert r.status == "measured"
    assert r.metrics["mean_token_accuracy"] >= 0.5    # anchor recovers the sequence
