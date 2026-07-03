"""CONJFORMER reproduction correctness: the conjugated (server) model run on
rotated embeddings must be functionally identical to the retrofit (scalar-RMSNorm)
model run on plain embeddings, to floating point. This is the paper's Lemma 2.1
(block equivalence) end-to-end, including RoPE + GQA + SwiGLU.

Uses a tiny random Qwen2 config so it runs on CPU with no model download.
"""
from __future__ import annotations

import torch
import pytest

from pllo.baselines.conjformer import (
    ScalarRMSNorm,
    build_conjformer_server,
    retrofit_scalar_rmsnorm,
    sample_secrets,
    verify_equivariance,
    rotate_in,
    rotate_out,
    structural_audit,
)


def _tiny_qwen2(dtype=torch.float64):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,     # head_dim = 8
        num_key_value_heads=2,     # GQA: n_rep = 4
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(dtype).eval()
    # de-trivialise the q/k/v biases so bias conjugation is actually exercised
    with torch.no_grad():
        for layer in model.model.layers:
            for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                if p.bias is not None:
                    p.bias.normal_(0, 0.5)
    return model


def test_scalar_rmsnorm_is_equivariant():
    torch.manual_seed(1)
    d = 32
    norm = ScalarRMSNorm(d, gamma=1.3).double()
    x = torch.randn(4, d, dtype=torch.float64)
    a = torch.randn(d, d, dtype=torch.float64)
    U, _ = torch.linalg.qr(a)                      # orthogonal
    lhs = norm(x @ U.T)
    rhs = norm(x) @ U.T
    # HF RMSNorm computes the variance internally in float32 (we mirror it
    # faithfully), so equivariance holds to float32 precision, not float64.
    assert torch.allclose(lhs, rhs, atol=1e-5), (lhs - rhs).abs().max().item()


def test_rope_compatible_O_preserves_attention_logits():
    """O_b built on the (i, i+h2) planes must commute with HF rotate_half RoPE."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, rotate_half  # noqa: F401
    from pllo.baselines.conjformer import _rope_compatible_head, _gen

    hd, S = 16, 5
    g = _gen(0)
    O = _rope_compatible_head(hd, g)               # [hd, hd]
    assert torch.allclose(O @ O.T, torch.eye(hd, dtype=torch.float64), atol=1e-10)

    q = torch.randn(1, 2, S, hd, dtype=torch.float64)   # [b, heads, S, hd]
    k = torch.randn(1, 2, S, hd, dtype=torch.float64)
    pos = torch.arange(S).unsqueeze(0)
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    freqs = pos.unsqueeze(-1).double() * inv                     # [1, S, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    def rope(x):
        return apply_rotary_pos_emb(x, x, cos, sin)[0]

    # rope(q O^T) O ?= rope(q)   (i.e. logits with O cancel post-RoPE)
    qO = q @ O.T
    kO = k @ O.T
    logits_plain = rope(q) @ rope(k).transpose(-1, -2)
    logits_obf = rope(qO) @ rope(kO).transpose(-1, -2)
    err = (logits_plain - logits_obf).abs().max().item()
    assert err < 1e-9, f"RoPE-obf attention logits diverge: {err}"


def test_full_equivariance_end_to_end():
    model = _tiny_qwen2(torch.float64)
    retro, server, secrets, audit = build_conjformer_server(model, seed=3)
    assert audit["norm_layers_retrofitted"] == 3 * 2 + 1  # 2 norms/layer + final

    ids = torch.randint(0, 256, (2, 12))
    rep = verify_equivariance(retro, server, secrets, ids)
    # exact up to the norm's float32 internal compute (HF-faithful) + orthogonal
    # conjugation round-off; the decisive signal is 100% top-1 logit agreement.
    assert rep["hidden_rel_err"] < 1e-5, rep
    assert rep["logit_top1_agreement"] == 1.0, rep
    assert rep["equivariant"] is True, rep


def test_rotate_in_out_roundtrip():
    secrets = _dummy_secrets(48)
    x = torch.randn(3, 48, dtype=torch.float64)
    back = rotate_out(rotate_in(x, secrets), secrets)
    assert torch.allclose(x, back, atol=1e-10)


def test_retrofit_scalar_init_matches_mean():
    model = _tiny_qwen2(torch.float64)
    # record a per-channel mean before retrofit
    first = model.model.layers[0].input_layernorm.weight.data.mean().item()
    n = retrofit_scalar_rmsnorm(model, init="mean")
    assert n == 3 * 2 + 1
    new = model.model.layers[0].input_layernorm
    assert isinstance(new, ScalarRMSNorm)
    assert abs(float(new.weight.item()) - first) < 1e-9


def test_structural_audit_is_honest():
    a = structural_audit()
    assert a["obfuscation_is_exact"] is True
    assert a["needs_finetune_to_recover_on_pretrained_model"] is True
    assert a["paper_measured_task_accuracy"] is False
    assert a["extra_per_token_tee_crossings"] == 0
    assert "attention logits" in a["server_observable_invariants"]


def _dummy_secrets(hidden):
    from pllo.baselines.conjformer import _full_orthogonal, _gen, ConjFormerSecrets
    g = _gen(0)
    return ConjFormerSecrets(U=_full_orthogonal(hidden, g), O_kv=[], R_kv=[], P=[])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
