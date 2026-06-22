"""Stage 6.4 tests -- RoPE-compatible masked GQA/MHA attention (CPU, float64)."""

from __future__ import annotations

import math

import torch

from pllo.experiments.rope_gqa_probe import (
    RopeGQAProbeConfig,
    run_rope_gqa_probe,
    run_rope_leakage_proxy,
)
from pllo.ops.gqa_attention import (
    apply_head_masks,
    generate_gqa_rope_masks,
    masked_rope_gqa_attention_decode,
    masked_rope_gqa_attention_prefill,
    merge_heads,
    repeat_kv,
    split_heads,
)
from pllo.ops.rope import (
    apply_rope,
    build_rope_cache,
    is_pairwise_complex_scaling_mask,
    make_pairwise_complex_scaling_mask,
    make_pairwise_complex_scaling_masks,
    make_pairwise_rotation_mask,
    pairwise_complex_scaling_inverse,
    rope_commutation_error,
    rotate_half,
)

ATOL = 1e-8
RTOL = 1e-8
DTYPE = torch.float64


def _g(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _orthogonal(dim: int, g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=DTYPE))
    return q


# 1.
def test_rotate_half_pair_convention() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=DTYPE)
    # [x0,x1,x2,x3] -> [-x1, x0, -x3, x2]
    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]], dtype=DTYPE)
    torch.testing.assert_close(rotate_half(x), expected, atol=ATOL, rtol=RTOL)


# 2.
def test_rope_cache_shapes() -> None:
    cos, sin = build_rope_cache(8, 6, dtype=DTYPE)
    assert cos.shape == (8, 6)
    assert sin.shape == (8, 6)
    # Adjacent pairs share a frequency: cos[:,0]==cos[:,1], etc.
    torch.testing.assert_close(cos[:, 0], cos[:, 1], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(sin[:, 2], sin[:, 3], atol=ATOL, rtol=RTOL)


# 3.
def test_pairwise_rotation_mask_is_orthogonal() -> None:
    M = make_pairwise_rotation_mask(8, DTYPE, "cpu", _g(1))
    eye = torch.eye(8, dtype=DTYPE)
    torch.testing.assert_close(M @ M.T, eye, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(M.T @ M, eye, atol=ATOL, rtol=RTOL)


# 4.
def test_pairwise_rotation_mask_commutes_with_rope() -> None:
    g = _g(2)
    B, H, T, D = 2, 3, 8, 8
    x = torch.randn(B, H, T, D, generator=g, dtype=DTYPE)
    M = make_pairwise_rotation_mask(D, DTYPE, "cpu", g)
    cos, sin = build_rope_cache(T, D, dtype=DTYPE)
    err = rope_commutation_error(x, M, cos, sin)
    assert err < 1e-8, err
    # Also the inverse-transpose mask (used on Q) commutes.
    m_inv_t = torch.linalg.inv(M).T
    err2 = rope_commutation_error(x, m_inv_t, cos, sin)
    assert err2 < 1e-8, err2


# 5.
def test_repeat_kv_shape_and_values() -> None:
    g = _g(3)
    B, n_kv, T, D = 2, 2, 5, 4
    x = torch.randn(B, n_kv, T, D, generator=g, dtype=DTYPE)
    rep = repeat_kv(x, num_heads=4, num_key_value_heads=2)
    assert rep.shape == (B, 4, T, D)
    # group_size=2: heads 0,1 share kv 0; heads 2,3 share kv 1.
    torch.testing.assert_close(rep[:, 0], x[:, 0], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(rep[:, 1], x[:, 0], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(rep[:, 2], x[:, 1], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(rep[:, 3], x[:, 1], atol=ATOL, rtol=RTOL)


def _setup(num_heads: int, num_key_value_heads: int, seed: int = 2027):
    g = _g(seed)
    B, T, hidden = 2, 8, 32
    head_dim = hidden // num_heads
    kv_dim = num_key_value_heads * head_dim

    def rn(*s: int) -> torch.Tensor:
        return torch.randn(*s, generator=g, dtype=DTYPE)

    x = rn(B, T, hidden)
    w_q, b_q = rn(hidden, num_heads * head_dim), rn(num_heads * head_dim)
    w_k, b_k = rn(hidden, kv_dim), rn(kv_dim)
    w_v, b_v = rn(hidden, kv_dim), rn(kv_dim)
    w_o, b_o = rn(num_heads * head_dim, hidden), rn(hidden)
    n_out = _orthogonal(hidden, g)
    masks = generate_gqa_rope_masks(
        num_heads, num_key_value_heads, head_dim, DTYPE, "cpu", g,
    )
    cos, sin = build_rope_cache(T + 5, head_dim, dtype=DTYPE)
    return x, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, n_out, masks, cos, sin


# 6.
def test_mha_rope_masked_scores_match_plain() -> None:
    args = _setup(4, 4)
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    assert res["score_max_abs_error"] < 1e-8
    torch.testing.assert_close(
        res["probs_plain"], res["probs_tilde"], atol=ATOL, rtol=RTOL)


# 7.
def test_gqa_rope_masked_scores_match_plain() -> None:
    args = _setup(4, 2)
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    assert res["score_max_abs_error"] < 1e-8
    torch.testing.assert_close(
        res["probs_plain"], res["probs_tilde"], atol=ATOL, rtol=RTOL)


# 8.
def test_gqa_v_aggregation_mask_invariant() -> None:
    args = _setup(4, 2)
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    torch.testing.assert_close(
        res["av_tilde"], res["expected_av_tilde"], atol=ATOL, rtol=RTOL)


# 9.
def test_gqa_prefill_output_correctness() -> None:
    args = _setup(4, 2)
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    torch.testing.assert_close(
        res["out_tilde"], res["expected_out_tilde"], atol=ATOL, rtol=RTOL)


# 10.
def test_gqa_prefill_cache_invariant() -> None:
    args = _setup(4, 2)
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    torch.testing.assert_close(
        res["cache"]["key_rope_tilde"], res["expected_cache_key_tilde"],
        atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        res["cache"]["value_tilde"], res["expected_cache_value_tilde"],
        atol=ATOL, rtol=RTOL)


# 11.
def test_gqa_decode_one_step_correctness() -> None:
    args = _setup(4, 2)
    x = args[0]
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    hidden = x.shape[-1]
    x_new = torch.randn(x.shape[0], 1, hidden, generator=_g(99), dtype=DTYPE)
    dec = masked_rope_gqa_attention_decode(x_new, res["cache"], position=8)
    torch.testing.assert_close(
        dec["out_tilde"], dec["expected_out_tilde"], atol=ATOL, rtol=RTOL)


# 12.
def test_gqa_decode_cache_append_invariant() -> None:
    args = _setup(4, 2)
    x = args[0]
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    cache = res["cache"]
    hidden = x.shape[-1]
    gg = _g(123)
    for step in range(3):
        x_new = torch.randn(x.shape[0], 1, hidden, generator=gg, dtype=DTYPE)
        dec = masked_rope_gqa_attention_decode(x_new, cache, position=8 + step)
        torch.testing.assert_close(
            dec["appended_key_tilde"], dec["expected_appended_key_tilde"],
            atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            dec["appended_value_tilde"], dec["expected_appended_value_tilde"],
            atol=ATOL, rtol=RTOL)
        cache = dec["cache"]
    # Cache grew by 3 along the seq axis.
    assert cache["key_rope_tilde"].shape[2] == 8 + 3


# 13.
def test_probe_runs_and_reports_allclose() -> None:
    cfg = RopeGQAProbeConfig()
    report = run_rope_gqa_probe(cfg)
    assert report["status"] == "ok"
    assert report["all_allclose"] is True
    assert report["mha"]["allclose"] is True
    assert report["gqa"]["allclose"] is True
    # MHA degenerates to num_key_value_heads == num_heads.
    assert report["mha"]["num_key_value_heads"] == cfg.num_heads
    assert report["gqa"]["num_key_value_heads"] == cfg.num_key_value_heads
    assert report["mask_structure"]["gqa_supported"] is True


def test_split_merge_heads_roundtrip() -> None:
    g = _g(7)
    x = torch.randn(2, 6, 32, generator=g, dtype=DTYPE)
    back = merge_heads(split_heads(x, 4))
    torch.testing.assert_close(back, x, atol=ATOL, rtol=RTOL)


def test_score_invariant_manual_gqa() -> None:
    # Direct check that RoPE(Q M^{-T}) @ RoPE(K M)^T == RoPE(Q) @ RoPE(K)^T.
    g = _g(11)
    B, T, D = 2, 5, 8
    Q = torch.randn(B, 1, T, D, generator=g, dtype=DTYPE)
    K = torch.randn(B, 1, T, D, generator=g, dtype=DTYPE)
    M = make_pairwise_rotation_mask(D, DTYPE, "cpu", g).unsqueeze(0)
    cos, sin = build_rope_cache(T, D, dtype=DTYPE)
    m_inv_t = torch.linalg.inv(M).transpose(-2, -1)
    qr = apply_rope(Q, cos, sin)
    kr = apply_rope(K, cos, sin)
    qr_t = apply_rope(apply_head_masks(Q, m_inv_t), cos, sin)
    kr_t = apply_rope(apply_head_masks(K, M), cos, sin)
    s_plain = qr @ kr.transpose(-2, -1) / math.sqrt(D)
    s_masked = qr_t @ kr_t.transpose(-2, -1) / math.sqrt(D)
    torch.testing.assert_close(s_plain, s_masked, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Stage 6.4.1 -- pairwise complex-scaling masks + leakage proxy
# ===========================================================================


# 6.4.1 / 1.
def test_pairwise_complex_scaling_mask_structure() -> None:
    D = 8
    M = make_pairwise_complex_scaling_mask(D, DTYPE, "cpu", _g(21))
    assert M.shape == (D, D)
    assert is_pairwise_complex_scaling_mask(M)
    half = D // 2
    two_i = torch.arange(half) * 2
    a = M[two_i, two_i]
    b = M[two_i + 1, two_i]
    # block form [[a,-b],[b,a]]
    torch.testing.assert_close(M[two_i, two_i + 1], -b, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(M[two_i + 1, two_i + 1], a, atol=ATOL, rtol=RTOL)
    # off-block entries are zero (reconstruct and compare).
    ref = torch.zeros_like(M)
    ref[two_i, two_i] = a
    ref[two_i, two_i + 1] = -b
    ref[two_i + 1, two_i] = b
    ref[two_i + 1, two_i + 1] = a
    torch.testing.assert_close(M, ref, atol=ATOL, rtol=RTOL)
    # determinant per block = a^2 + b^2 > 0.
    assert torch.all(a * a + b * b > 0.0)


# 6.4.1 / 2.
def test_pairwise_complex_scaling_inverse_closed_form() -> None:
    D = 8
    M = make_pairwise_complex_scaling_mask(D, DTYPE, "cpu", _g(22))
    M_inv = pairwise_complex_scaling_inverse(M)
    eye = torch.eye(D, dtype=DTYPE)
    torch.testing.assert_close(M @ M_inv, eye, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(M_inv @ M, eye, atol=ATOL, rtol=RTOL)
    # Closed-form matches torch.linalg.inv.
    torch.testing.assert_close(
        M_inv, torch.linalg.inv(M), atol=ATOL, rtol=RTOL)
    # Batched form is supported too.
    stack = make_pairwise_complex_scaling_masks(3, D, DTYPE, "cpu", _g(23))
    inv_stack = pairwise_complex_scaling_inverse(stack)
    torch.testing.assert_close(
        inv_stack, torch.linalg.inv(stack), atol=ATOL, rtol=RTOL)


# 6.4.1 / 3.
def test_pairwise_complex_scaling_commutes_with_rope() -> None:
    g = _g(24)
    B, H, T, D = 2, 3, 8, 8
    x = torch.randn(B, H, T, D, generator=g, dtype=DTYPE)
    M = make_pairwise_complex_scaling_mask(D, DTYPE, "cpu", g)
    cos, sin = build_rope_cache(T, D, dtype=DTYPE)
    lhs = apply_rope(x @ M, cos, sin)
    rhs = apply_rope(x, cos, sin) @ M
    torch.testing.assert_close(lhs, rhs, atol=ATOL, rtol=RTOL)


def _setup_family(num_heads: int, num_kv: int, family: str, seed: int = 2027):
    g = _g(seed)
    B, T, hidden = 2, 8, 32
    head_dim = hidden // num_heads
    kv_dim = num_kv * head_dim

    def rn(*s: int) -> torch.Tensor:
        return torch.randn(*s, generator=g, dtype=DTYPE)

    x = rn(B, T, hidden)
    w_q, b_q = rn(hidden, num_heads * head_dim), rn(num_heads * head_dim)
    w_k, b_k = rn(hidden, kv_dim), rn(kv_dim)
    w_v, b_v = rn(hidden, kv_dim), rn(kv_dim)
    w_o, b_o = rn(num_heads * head_dim, hidden), rn(hidden)
    n_out = _orthogonal(hidden, g)
    masks = generate_gqa_rope_masks(
        num_heads, num_kv, head_dim, DTYPE, "cpu", g, mask_family=family,
    )
    cos, sin = build_rope_cache(T + 5, head_dim, dtype=DTYPE)
    return x, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, n_out, masks, cos, sin


# 6.4.1 / 4.
def test_qk_score_invariant_complex_scaling_mha() -> None:
    args = _setup_family(4, 4, "pairwise_complex_scaling")
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    assert res["score_max_abs_error"] < 1e-8
    torch.testing.assert_close(
        res["probs_plain"], res["probs_tilde"], atol=ATOL, rtol=RTOL)


# 6.4.1 / 5.
def test_qk_score_invariant_complex_scaling_gqa() -> None:
    args = _setup_family(4, 2, "pairwise_complex_scaling")
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    assert res["score_max_abs_error"] < 1e-8
    torch.testing.assert_close(
        res["probs_plain"], res["probs_tilde"], atol=ATOL, rtol=RTOL)


# 6.4.1 / 6.
def test_value_aggregation_and_output_projection_complex_scaling() -> None:
    args = _setup_family(4, 2, "pairwise_complex_scaling")
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    # AV_tilde == AV @ S (per-Q-head value mask).
    torch.testing.assert_close(
        res["av_tilde"], res["expected_av_tilde"], atol=ATOL, rtol=RTOL)
    # output_tilde == output_plain @ n_out (true inverse folded into W_o).
    torch.testing.assert_close(
        res["out_tilde"], res["expected_out_tilde"], atol=ATOL, rtol=RTOL)


# 6.4.1 / 7.
def test_prefill_cache_invariant_complex_scaling() -> None:
    args = _setup_family(4, 2, "pairwise_complex_scaling")
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    torch.testing.assert_close(
        res["cache"]["key_rope_tilde"], res["expected_cache_key_tilde"],
        atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        res["cache"]["value_tilde"], res["expected_cache_value_tilde"],
        atol=ATOL, rtol=RTOL)


# 6.4.1 / 8.
def test_decode_cache_append_invariant_complex_scaling() -> None:
    args = _setup_family(4, 2, "pairwise_complex_scaling")
    x = args[0]
    res = masked_rope_gqa_attention_prefill(*args[:10], args[10], args[11],
                                            args[12])
    cache = res["cache"]
    hidden = x.shape[-1]
    gg = _g(321)
    for step in range(3):
        x_new = torch.randn(x.shape[0], 1, hidden, generator=gg, dtype=DTYPE)
        dec = masked_rope_gqa_attention_decode(x_new, cache, position=8 + step)
        torch.testing.assert_close(
            dec["appended_key_tilde"], dec["expected_appended_key_tilde"],
            atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            dec["appended_value_tilde"], dec["expected_appended_value_tilde"],
            atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            dec["out_tilde"], dec["expected_out_tilde"], atol=ATOL, rtol=RTOL)
        cache = dec["cache"]
    assert cache["key_rope_tilde"].shape[2] == 8 + 3


# 6.4.1 / 9.
def test_rope_gqa_probe_reports_complex_scaling() -> None:
    report = run_rope_gqa_probe(RopeGQAProbeConfig())
    assert report["security_status"] == (
        "operator_compatible_leakage_reduction_not_semantic_security")
    assert report["no_intermediate_tee"] is True
    assert "pairwise_complex_scaling" in report["correctness"]
    assert "pairwise_rotation" in report["correctness"]
    for fam in report["correctness"].values():
        assert fam["mha"]["allclose"] is True
        assert fam["gqa"]["allclose"] is True
    assert report["leakage_proxy"] is not None
    assert report["all_allclose"] is True


# 6.4.1 / 10.
def test_leakage_proxy_rotation_preserves_pair_norm_more_than_complex_scaling(
) -> None:
    cfg = RopeGQAProbeConfig(leakage_num_samples=256, seed=4242)
    leak = run_rope_leakage_proxy(cfg)
    rot = leak["pairwise_rotation"]["cross_session_pair_norm_correlation"]
    cs = leak["pairwise_complex_scaling"][
        "cross_session_pair_norm_correlation"]
    # Rotation preserves per-pair norm across sessions; complex-scaling does
    # not (independent per-session scales). Relative comparison only.
    assert rot > cs
    # Rotation cross-session pair-norm correlation is ~1 (norm-preserving).
    assert rot > 0.999
    assert leak["leakage_proxy_is_not_security_proof"] is True
