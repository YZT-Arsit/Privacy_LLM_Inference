"""Stage 6.5 tests -- LLaMA/Qwen-like synthetic decoder block (CPU, float64)."""

from __future__ import annotations

import pytest
import torch

from pllo.experiments.llama_synthetic_block_probe import (
    LlamaSyntheticBlockProbeConfig,
    run_llama_synthetic_block_probe,
)
from pllo.ops.llama_synthetic_block import (
    SyntheticLlamaBlockConfig,
    generate_block_masks,
    init_synthetic_llama_block_weights,
    llama_attention_plain_prefill,
    llama_block_masked_decode,
    llama_block_masked_prefill,
    llama_block_plain_prefill,
    rmsnorm_plain,
    swiglu_plain,
)
from pllo.ops.rope import build_rope_cache

ATOL = 1e-8
RTOL = 1e-8
DTYPE = torch.float64


def _cfg(num_kv: int = 2, seq_len: int = 8) -> SyntheticLlamaBlockConfig:
    return SyntheticLlamaBlockConfig(
        batch_size=2, seq_len=seq_len, decode_steps=3, hidden_size=32,
        intermediate_size=64, num_heads=4, num_key_value_heads=num_kv,
        dtype=DTYPE, device="cpu", seed=2028,
    )


def _g(seed: int = 2028) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _setup(num_kv: int = 2, seq_len: int = 8):
    cfg = _cfg(num_kv, seq_len)
    g = _g(cfg.seed)
    weights = init_synthetic_llama_block_weights(cfg, g)
    masks = generate_block_masks(cfg, g)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, generator=g,
                    dtype=DTYPE)
    return cfg, weights, masks, x, g


# 1.
def test_synthetic_config_validates_shapes() -> None:
    cfg = _cfg()
    cfg.validate()
    assert cfg.head_dim == 8
    assert cfg.head_dim % 2 == 0
    w = init_synthetic_llama_block_weights(cfg, _g())
    assert w.rms1_weight.shape == (32,)
    assert w.wq.shape == (32, 4 * 8)
    assert w.wk.shape == (32, 2 * 8)
    assert w.wv.shape == (32, 2 * 8)
    assert w.wo.shape == (4 * 8, 32)
    assert w.w_gate.shape == (32, 64)
    assert w.w_down.shape == (64, 32)
    with pytest.raises(ValueError):
        SyntheticLlamaBlockConfig(hidden_size=30, num_heads=4).validate()


# 2.
def test_rmsnorm_plain_matches_reference_formula() -> None:
    x = torch.randn(2, 5, 32, generator=_g(1), dtype=DTYPE)
    w = torch.randn(32, generator=_g(2), dtype=DTYPE)
    eps = 1e-5
    ref = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w
    torch.testing.assert_close(rmsnorm_plain(x, w, eps), ref, atol=ATOL,
                               rtol=RTOL)


# 3.
def test_swiglu_plain_shapes() -> None:
    x = torch.randn(2, 5, 32, generator=_g(3), dtype=DTYPE)
    wg = torch.randn(32, 64, generator=_g(4), dtype=DTYPE)
    wu = torch.randn(32, 64, generator=_g(5), dtype=DTYPE)
    wd = torch.randn(64, 32, generator=_g(6), dtype=DTYPE)
    out = swiglu_plain(x, wg, wu, wd)
    assert out["gate"].shape == (2, 5, 64)
    assert out["up"].shape == (2, 5, 64)
    assert out["hidden"].shape == (2, 5, 64)
    assert out["out"].shape == (2, 5, 32)


# 4.
def test_llama_attention_plain_prefill_shapes() -> None:
    cfg, weights, _, x, _ = _setup()
    cos, sin = build_rope_cache(cfg.seq_len + 4, cfg.head_dim, cfg.rope_base,
                                DTYPE, "cpu")
    r1 = rmsnorm_plain(x, weights.rms1_weight, cfg.rms_norm_eps)
    attn = llama_attention_plain_prefill(r1, weights, cfg, cos, sin)
    assert attn["out"].shape == (2, 8, 32)
    assert attn["q_rope"].shape == (2, 4, 8, 8)
    assert attn["k_rope"].shape == (2, 2, 8, 8)
    assert attn["v"].shape == (2, 2, 8, 8)
    assert attn["scores"].shape == (2, 4, 8, 8)


# 5.
def test_llama_block_plain_prefill_shapes() -> None:
    cfg, weights, _, x, _ = _setup()
    cos, sin = build_rope_cache(cfg.seq_len + 4, cfg.head_dim, cfg.rope_base,
                                DTYPE, "cpu")
    out = llama_block_plain_prefill(x, weights, cfg, cos, sin)
    assert out["y"].shape == (2, 8, 32)
    assert out["x1"].shape == (2, 8, 32)
    assert out["mlp_out"].shape == (2, 8, 32)


def _prefill(num_kv: int = 2, seq_len: int = 8):
    cfg, weights, masks, x, _ = _setup(num_kv, seq_len)
    return llama_block_masked_prefill(x, weights, masks, cfg)


# 6.
def test_masked_prefill_rms1_core_invariant() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["tilde"]["r1_core"], res["expected"]["r1_core"],
        atol=ATOL, rtol=RTOL)


# 7.
def test_masked_prefill_attention_scores_match() -> None:
    res = _prefill()
    m = res["metrics"]
    assert m["attention_score_max_abs_error"] < 1e-8
    assert m["attention_prob_max_abs_error"] < 1e-8
    assert m["q_mask_max_abs_error"] < 1e-8
    assert m["k_mask_max_abs_error"] < 1e-8
    assert m["v_mask_max_abs_error"] < 1e-8


# 8.
def test_masked_prefill_attention_output_invariant() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["tilde"]["attn_out"], res["expected"]["attn_out"],
        atol=ATOL, rtol=RTOL)


# 9.
def test_masked_prefill_residual1_invariant() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["tilde"]["x1"], res["expected"]["x1"], atol=ATOL, rtol=RTOL)


# 10.
def test_masked_prefill_rms2_core_invariant() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["tilde"]["r2_core"], res["expected"]["r2_core"],
        atol=ATOL, rtol=RTOL)


# 11.
def test_masked_prefill_swiglu_gate_up_permutation() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["tilde"]["gate"], res["expected"]["gate"], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        res["tilde"]["up"], res["expected"]["up"], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        res["tilde"]["hidden"], res["expected"]["hidden"], atol=ATOL,
        rtol=RTOL)


# 12.
def test_masked_prefill_mlp_output_invariant() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["tilde"]["mlp_out"], res["expected"]["mlp_out"], atol=ATOL,
        rtol=RTOL)


# 13.
def test_masked_prefill_final_output_correctness() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL)
    assert res["metrics"]["allclose"] is True


# 14.
def test_masked_prefill_cache_invariant() -> None:
    res = _prefill()
    torch.testing.assert_close(
        res["cache_tilde"]["key_rope_tilde"], res["expected"]["cache_key"],
        atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        res["cache_tilde"]["value_tilde"], res["expected"]["cache_value"],
        atol=ATOL, rtol=RTOL)


# 15.
def test_masked_decode_one_step_output_correctness() -> None:
    cfg, weights, masks, x, g = _setup()
    pre = llama_block_masked_prefill(x, weights, masks, cfg)
    x_new = torch.randn(cfg.batch_size, 1, cfg.hidden_size, generator=g,
                        dtype=DTYPE)
    dec = llama_block_masked_decode(
        x_new, pre["cache_tilde"], pre["cache_plain"], weights, masks, cfg,
        position_id=cfg.seq_len)
    torch.testing.assert_close(
        dec["y_new_tilde"], dec["expected_y_new_tilde"], atol=ATOL, rtol=RTOL)


# 16.
def test_masked_decode_cache_append_invariant() -> None:
    cfg, weights, masks, x, g = _setup()
    pre = llama_block_masked_prefill(x, weights, masks, cfg)
    x_new = torch.randn(cfg.batch_size, 1, cfg.hidden_size, generator=g,
                        dtype=DTYPE)
    dec = llama_block_masked_decode(
        x_new, pre["cache_tilde"], pre["cache_plain"], weights, masks, cfg,
        position_id=cfg.seq_len)
    torch.testing.assert_close(
        dec["appended_key_tilde"], dec["expected_appended_key_tilde"],
        atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        dec["appended_value_tilde"], dec["expected_appended_value_tilde"],
        atol=ATOL, rtol=RTOL)


# 17.
def test_masked_decode_multi_step_correctness() -> None:
    cfg, weights, masks, x, g = _setup()
    pre = llama_block_masked_prefill(x, weights, masks, cfg)
    cache_tilde = pre["cache_tilde"]
    cache_plain = pre["cache_plain"]
    for step in range(cfg.decode_steps):
        x_new = torch.randn(cfg.batch_size, 1, cfg.hidden_size, generator=g,
                            dtype=DTYPE)
        dec = llama_block_masked_decode(
            x_new, cache_tilde, cache_plain, weights, masks, cfg,
            position_id=cfg.seq_len + step)
        torch.testing.assert_close(
            dec["y_new_tilde"], dec["expected_y_new_tilde"], atol=ATOL,
            rtol=RTOL)
        assert dec["metrics"]["allclose"] is True
        cache_tilde = dec["cache_tilde"]
        cache_plain = dec["cache_plain"]
    # Cache grew by decode_steps along the seq axis.
    assert cache_tilde["key_rope_tilde"].shape[2] == cfg.seq_len + cfg.decode_steps


# 18.
def test_probe_runs_and_reports_allclose() -> None:
    report = run_llama_synthetic_block_probe(LlamaSyntheticBlockProbeConfig())
    assert report["status"] == "ok"
    assert report["all_allclose"] is True
    assert report["gqa"]["allclose"] is True
    assert report["mha"]["allclose"] is True


# 19.
def test_probe_metadata_has_no_intermediate_tee_and_no_security_claim() -> None:
    report = run_llama_synthetic_block_probe(LlamaSyntheticBlockProbeConfig())
    md = report["metadata"]
    assert md["no_intermediate_tee"] is True
    assert md["no_hf_dependency"] is True
    assert md["selector_lifted_swiglu_default"] is False
    assert md["security_status"] == (
        "operator_compatible_leakage_reduction_not_semantic_security")
    assert "semantic security" in report["statement"].lower()


# 20.
def test_mha_variant_num_heads_equals_num_kv_heads() -> None:
    res = _prefill(num_kv=4)  # MHA: num_heads == num_key_value_heads
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL)
    assert res["metrics"]["allclose"] is True
