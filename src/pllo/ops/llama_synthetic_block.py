"""Stage 6.5 -- LLaMA/Qwen-like synthetic decoder block (CPU, correctness).

Composes the existing building blocks into one synthetic decoder layer
(row-vector convention):

    r1  = RMSNorm(x)
    a   = RoPE-compatible GQA attention(r1)
    x1  = x + a
    r2  = RMSNorm(x1)
    m   = SwiGLU(r2)
    y   = x1 + m

A single orthogonal *residual mask* ``n_res`` is shared by the block input
and output. Because RMSNorm's core (the normalize step) preserves per-row
L2 norm, an orthogonal right-multiply commutes with it exactly:

    rmsnorm_core(x @ n_res) == rmsnorm_core(x) @ n_res.

The RMSNorm affine weight and the residual mask are folded into the
following linear projections, so the masked path operates entirely on the
masked residual stream ``x_tilde = x @ n_res`` and masked weights:

    Wq_tilde   = n_res^{-1} @ diag(rms1_w) @ Wq @ blockdiag(Mq_heads)
    Wk_tilde   = n_res^{-1} @ diag(rms1_w) @ Wk @ blockdiag(Mk_heads)
    Wv_tilde   = n_res^{-1} @ diag(rms1_w) @ Wv @ blockdiag(Mv_heads)
    Wo_tilde   = blockdiag(Vinv_per_qhead) @ Wo @ n_res
    Wgate_tilde= n_res^{-1} @ diag(rms2_w) @ Wgate[:, perm]
    Wup_tilde  = n_res^{-1} @ diag(rms2_w) @ Wup[:, perm]
    Wdown_tilde= Wdown[perm, :] @ n_res

The attention masks (``Mq/Mk/Mv``) are the Stage 6.4.1 RoPE-compatible
masks (default ``pairwise_complex_scaling``); the MLP uses the
paired-permutation SwiGLU compatibility path (selector-lifted SwiGLU is
NOT the default, since selector zero-row leakage remains a caveat).

This is a synthetic tensor-level probe: no HF/ModelScope models, no GPT-2
wrapper, no embeddings/LM-head/sampling, no NTK/YaRN RoPE scaling, CPU
only. No formal, cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from pllo.ops.gqa_attention import (
    block_diag_from_head_masks,
    generate_gqa_rope_masks,
    merge_heads,
    repeat_kv,
    split_heads,
)
from pllo.ops.nonlinear_islands import rmsnorm_core, silu_reference
from pllo.ops.rope import apply_rope, build_rope_cache

__all__ = [
    "SyntheticLlamaBlockConfig",
    "SyntheticLlamaBlockWeights",
    "fold_block_weights",
    "generate_block_masks",
    "init_synthetic_llama_block_weights",
    "llama_attention_plain_prefill",
    "llama_block_masked_decode",
    "llama_block_masked_prefill",
    "llama_block_plain_prefill",
    "rmsnorm_plain",
    "swiglu_plain",
]


# ---------------------------------------------------------------------------
# Config + weights
# ---------------------------------------------------------------------------


@dataclass
class SyntheticLlamaBlockConfig:
    batch_size: int = 2
    seq_len: int = 8
    decode_steps: int = 3
    hidden_size: int = 32
    intermediate_size: int = 64
    num_heads: int = 4
    num_key_value_heads: int = 2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    mask_family: str = "pairwise_complex_scaling"
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 2028

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (RoPE adjacent pairs)")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class SyntheticLlamaBlockWeights:
    rms1_weight: torch.Tensor
    rms2_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


def init_synthetic_llama_block_weights(
    config: SyntheticLlamaBlockConfig, generator: torch.Generator,
) -> SyntheticLlamaBlockWeights:
    """Deterministic, modestly-scaled (no-bias) synthetic block weights."""
    config.validate()
    dtype = config.dtype
    device = torch.device(config.device)
    h = config.hidden_size
    inter = config.intermediate_size
    hd = config.head_dim
    nh = config.num_heads
    nkv = config.num_key_value_heads
    scale = 1.0 / math.sqrt(h)

    def rn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype,
                           device=device) * scale

    # RMSNorm gains centred near 1 (positive, non-degenerate).
    rms1 = 1.0 + 0.1 * torch.randn(h, generator=generator, dtype=dtype,
                                   device=device)
    rms2 = 1.0 + 0.1 * torch.randn(h, generator=generator, dtype=dtype,
                                   device=device)
    return SyntheticLlamaBlockWeights(
        rms1_weight=rms1,
        rms2_weight=rms2,
        wq=rn(h, nh * hd),
        wk=rn(h, nkv * hd),
        wv=rn(h, nkv * hd),
        wo=rn(nh * hd, h),
        w_gate=rn(h, inter),
        w_up=rn(h, inter),
        w_down=rn(inter, h),
    )


# ---------------------------------------------------------------------------
# Plain primitives
# ---------------------------------------------------------------------------


def rmsnorm_plain(
    x: torch.Tensor, weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    """``rmsnorm_core(x) * weight``."""
    return rmsnorm_core(x, eps) * weight


def swiglu_plain(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """SwiGLU MLP; returns intermediates for invariants."""
    gate = x @ w_gate
    up = x @ w_up
    hidden = silu_reference(gate) * up
    out = hidden @ w_down
    return {"gate": gate, "up": up, "hidden": hidden, "out": out}


def _causal_bias(t_q: int, t_k: int, dtype: torch.dtype,
                 device: torch.device, offset: int) -> torch.Tensor:
    q_pos = torch.arange(offset, offset + t_q, device=device).unsqueeze(1)
    k_pos = torch.arange(t_k, device=device).unsqueeze(0)
    bias = torch.zeros(t_q, t_k, dtype=dtype, device=device)
    bias.masked_fill_(k_pos > q_pos, float("-inf"))
    return bias


def _sdpa(qr: torch.Tensor, kr_rep: torch.Tensor, v_rep: torch.Tensor,
          scale: float, causal_offset: int | None,
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled-dot-product attention core. ``causal_offset`` None == no mask."""
    scores = qr @ kr_rep.transpose(-2, -1) * scale
    if causal_offset is not None:
        t_q = qr.shape[-2]
        t_k = kr_rep.shape[-2]
        scores = scores + _causal_bias(
            t_q, t_k, scores.dtype, scores.device, causal_offset)
    probs = torch.softmax(scores, dim=-1)
    av = probs @ v_rep
    return scores, probs, av


def llama_attention_plain_prefill(
    r1: torch.Tensor, weights: SyntheticLlamaBlockWeights,
    config: SyntheticLlamaBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
) -> dict[str, Any]:
    """Plain RoPE-GQA attention over a normed input ``r1``."""
    nh = config.num_heads
    nkv = config.num_key_value_heads
    hd = config.head_dim
    scale = 1.0 / math.sqrt(hd)

    q = split_heads(r1 @ weights.wq, nh)
    k = split_heads(r1 @ weights.wk, nkv)
    v = split_heads(r1 @ weights.wv, nkv)
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    kr_rep = repeat_kv(kr, nh, nkv)
    v_rep = repeat_kv(v, nh, nkv)
    scores, probs, av = _sdpa(qr, kr_rep, v_rep, scale, causal_offset=0)
    out = merge_heads(av) @ weights.wo
    return {
        "out": out,
        "q": q, "k": k, "v": v,
        "q_rope": qr, "k_rope": kr,
        "scores": scores, "probs": probs, "av": av,
        "cache_plain": {"key_rope": kr, "value": v},
    }


def llama_block_plain_prefill(
    x: torch.Tensor, weights: SyntheticLlamaBlockWeights,
    config: SyntheticLlamaBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
) -> dict[str, Any]:
    """Full plain decoder block; returns all intermediates."""
    eps = config.rms_norm_eps
    r1_core = rmsnorm_core(x, eps)
    r1 = r1_core * weights.rms1_weight
    attn = llama_attention_plain_prefill(r1, weights, config, cos, sin)
    x1 = x + attn["out"]
    r2_core = rmsnorm_core(x1, eps)
    r2 = r2_core * weights.rms2_weight
    mlp = swiglu_plain(r2, weights.w_gate, weights.w_up, weights.w_down)
    y = x1 + mlp["out"]
    return {
        "r1_core": r1_core, "r1": r1,
        "attn": attn, "attn_out": attn["out"],
        "x1": x1,
        "r2_core": r2_core, "r2": r2,
        "mlp": mlp, "mlp_out": mlp["out"],
        "y": y,
        "cache_plain": attn["cache_plain"],
    }


# ---------------------------------------------------------------------------
# Masks + folded weights
# ---------------------------------------------------------------------------


def _orthogonal(dim: int, dtype: torch.dtype, device: torch.device,
                g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype,
                                       device=device))
    return q


def generate_block_masks(
    config: SyntheticLlamaBlockConfig, generator: torch.Generator,
) -> dict[str, Any]:
    """Weight-independent block masks: residual ``n_res``, GQA attention
    masks (Stage 6.4.1), and the shared SwiGLU permutation."""
    config.validate()
    dtype = config.dtype
    device = torch.device(config.device)
    n_res = _orthogonal(config.hidden_size, dtype, device, generator)
    n_res_inv = n_res.transpose(-2, -1).contiguous()  # orthogonal: inv == T
    attn = generate_gqa_rope_masks(
        config.num_heads, config.num_key_value_heads, config.head_dim,
        dtype, device, generator, mask_family=config.mask_family,
    )
    perm = torch.randperm(config.intermediate_size, generator=generator,
                          device=device)
    return {
        "n_res": n_res,
        "n_res_inv": n_res_inv,
        "attn": attn,
        "perm": perm,
        "mask_family": config.mask_family,
        "residual_mask_family": "orthogonal",
    }


def fold_block_weights(
    weights: SyntheticLlamaBlockWeights, masks: dict[str, Any],
    config: SyntheticLlamaBlockConfig,
) -> dict[str, torch.Tensor]:
    """Fold RMSNorm affine + residual mask + per-head masks into weights."""
    n_res = masks["n_res"]
    n_res_inv = masks["n_res_inv"]
    attn = masks["attn"]
    perm = masks["perm"]

    mq_block = block_diag_from_head_masks(attn["q_masks"])
    mk_block = block_diag_from_head_masks(attn["key_masks"])
    mv_block = block_diag_from_head_masks(attn["value_masks"])
    # Per-query-head value-mask inverses for the output projection.
    v_inv_qhead = attn["value_mask_inverses"].index_select(0, attn["kv_index"])
    sv_block_inv = block_diag_from_head_masks(v_inv_qhead)

    rms1 = weights.rms1_weight.unsqueeze(1)  # [H,1] == diag(rms1) @ .
    rms2 = weights.rms2_weight.unsqueeze(1)

    wq_tilde = n_res_inv @ (rms1 * weights.wq) @ mq_block
    wk_tilde = n_res_inv @ (rms1 * weights.wk) @ mk_block
    wv_tilde = n_res_inv @ (rms1 * weights.wv) @ mv_block
    wo_tilde = sv_block_inv @ weights.wo @ n_res

    wgate_tilde = n_res_inv @ (rms2 * weights.w_gate).index_select(1, perm)
    wup_tilde = n_res_inv @ (rms2 * weights.w_up).index_select(1, perm)
    wdown_tilde = weights.w_down.index_select(0, perm) @ n_res

    return {
        "wq_tilde": wq_tilde, "wk_tilde": wk_tilde, "wv_tilde": wv_tilde,
        "wo_tilde": wo_tilde,
        "wgate_tilde": wgate_tilde, "wup_tilde": wup_tilde,
        "wdown_tilde": wdown_tilde,
    }


# ---------------------------------------------------------------------------
# Masked attention (operates on the masked, normed residual stream)
# ---------------------------------------------------------------------------


def _masked_attention(
    r1_core_tilde: torch.Tensor, folded: dict[str, torch.Tensor],
    config: SyntheticLlamaBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
    *, causal_offset: int | None, position_ids: torch.Tensor | None = None,
    past_key_rope: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
) -> dict[str, Any]:
    nh = config.num_heads
    nkv = config.num_key_value_heads
    hd = config.head_dim
    scale = 1.0 / math.sqrt(hd)

    q_t = split_heads(r1_core_tilde @ folded["wq_tilde"], nh)
    k_t = split_heads(r1_core_tilde @ folded["wk_tilde"], nkv)
    v_t = split_heads(r1_core_tilde @ folded["wv_tilde"], nkv)
    qr_t = apply_rope(q_t, cos, sin, position_ids=position_ids)
    kr_t_new = apply_rope(k_t, cos, sin, position_ids=position_ids)

    if past_key_rope is not None:
        kr_t = torch.cat([past_key_rope, kr_t_new], dim=2)
        v_full = torch.cat([past_value, v_t], dim=2)
    else:
        kr_t = kr_t_new
        v_full = v_t

    kr_rep = repeat_kv(kr_t, nh, nkv)
    v_rep = repeat_kv(v_full, nh, nkv)
    scores, probs, av = _sdpa(qr_t, kr_rep, v_rep, scale, causal_offset)
    out = merge_heads(av) @ folded["wo_tilde"]
    return {
        "out": out,
        "q_pre_rope": q_t, "k_pre_rope": k_t, "v": v_t,
        "q_rope": qr_t, "k_rope_new": kr_t_new,
        "scores": scores, "probs": probs, "av": av,
        "key_rope_full": kr_t, "value_full": v_full,
    }


# ---------------------------------------------------------------------------
# Masked block prefill
# ---------------------------------------------------------------------------


def _mx(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _finite_score_err(scores_plain: torch.Tensor,
                      scores_tilde: torch.Tensor) -> float:
    finite = torch.isfinite(scores_plain)
    return float((scores_plain[finite] - scores_tilde[finite]).abs().max().item())


def llama_block_masked_prefill(
    x: torch.Tensor, weights: SyntheticLlamaBlockWeights,
    masks: dict[str, Any], config: SyntheticLlamaBlockConfig,
) -> dict[str, Any]:
    """Full masked decoder block prefill with plain reference + invariants."""
    eps = config.rms_norm_eps
    dtype = config.dtype
    device = torch.device(config.device)
    n_res = masks["n_res"]
    perm = masks["perm"]
    attn_masks = masks["attn"]
    folded = fold_block_weights(weights, masks, config)

    max_pos = config.seq_len + config.decode_steps + 1
    cos, sin = build_rope_cache(max_pos, config.head_dim, config.rope_base,
                                dtype, device)

    # ---- plain reference ----
    plain = llama_block_plain_prefill(x, weights, config, cos, sin)

    # ---- masked path ----
    x_tilde = x @ n_res
    r1_core_tilde = rmsnorm_core(x_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, config, cos, sin,
                          causal_offset=0)
    attn_out_tilde = a["out"]
    x1_tilde = x_tilde + attn_out_tilde
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    gate_tilde = r2_core_tilde @ folded["wgate_tilde"]
    up_tilde = r2_core_tilde @ folded["wup_tilde"]
    hidden_tilde = silu_reference(gate_tilde) * up_tilde
    mlp_out_tilde = hidden_tilde @ folded["wdown_tilde"]
    y_tilde = x1_tilde + mlp_out_tilde

    # ---- expected (plain @ mask / permuted) ----
    v_masks_qhead = attn_masks["value_masks"].index_select(
        0, attn_masks["kv_index"])
    expected = {
        "r1_core": plain["r1_core"] @ n_res,
        "q": _apply(plain["attn"]["q"], attn_masks["q_masks"]),
        "k": _apply(plain["attn"]["k"], attn_masks["key_masks"]),
        "v": _apply(plain["attn"]["v"], attn_masks["value_masks"]),
        "av": _apply(plain["attn"]["av"], v_masks_qhead),
        "attn_out": plain["attn_out"] @ n_res,
        "x1": plain["x1"] @ n_res,
        "r2_core": plain["r2_core"] @ n_res,
        "gate": plain["mlp"]["gate"].index_select(-1, perm),
        "up": plain["mlp"]["up"].index_select(-1, perm),
        "hidden": plain["mlp"]["hidden"].index_select(-1, perm),
        "mlp_out": plain["mlp_out"] @ n_res,
        "y": plain["y"] @ n_res,
        "cache_key": _apply(plain["cache_plain"]["key_rope"],
                            attn_masks["key_masks"]),
        "cache_value": _apply(plain["cache_plain"]["value"],
                              attn_masks["value_masks"]),
    }

    metrics = {
        "rms1_core_max_abs_error": _mx(r1_core_tilde, expected["r1_core"]),
        "q_mask_max_abs_error": _mx(a["q_pre_rope"], expected["q"]),
        "k_mask_max_abs_error": _mx(a["k_pre_rope"], expected["k"]),
        "v_mask_max_abs_error": _mx(a["v"], expected["v"]),
        "attention_score_max_abs_error": _finite_score_err(
            plain["attn"]["scores"], a["scores"]),
        "attention_prob_max_abs_error": _mx(plain["attn"]["probs"], a["probs"]),
        "attention_av_max_abs_error": _mx(a["av"], expected["av"]),
        "attention_output_max_abs_error": _mx(attn_out_tilde,
                                              expected["attn_out"]),
        "residual1_max_abs_error": _mx(x1_tilde, expected["x1"]),
        "rms2_core_max_abs_error": _mx(r2_core_tilde, expected["r2_core"]),
        "swiglu_gate_max_abs_error": _mx(gate_tilde, expected["gate"]),
        "swiglu_up_max_abs_error": _mx(up_tilde, expected["up"]),
        "swiglu_hidden_max_abs_error": _mx(hidden_tilde, expected["hidden"]),
        "mlp_output_max_abs_error": _mx(mlp_out_tilde, expected["mlp_out"]),
        "final_output_max_abs_error": _mx(y_tilde, expected["y"]),
        "prefill_cache_key_max_abs_error": _mx(a["key_rope_full"],
                                               expected["cache_key"]),
        "prefill_cache_value_max_abs_error": _mx(a["value_full"],
                                                 expected["cache_value"]),
    }
    metrics["allclose"] = all(v <= 1e-8 for v in metrics.values()
                              if isinstance(v, float))

    cache_tilde = {
        "key_rope_tilde": a["key_rope_full"],
        "value_tilde": a["value_full"],
        "folded": folded,
        "seq_len": config.seq_len,
    }
    cache_plain = {
        "key_rope": plain["cache_plain"]["key_rope"],
        "value": plain["cache_plain"]["value"],
        "seq_len": config.seq_len,
    }

    return {
        "y_tilde": y_tilde,
        "expected_y_tilde": expected["y"],
        "y_plain": plain["y"],
        "cache_tilde": cache_tilde,
        "cache_plain": cache_plain,
        "tilde": {
            "r1_core": r1_core_tilde, "attn_out": attn_out_tilde,
            "x1": x1_tilde, "r2_core": r2_core_tilde,
            "gate": gate_tilde, "up": up_tilde, "hidden": hidden_tilde,
            "mlp_out": mlp_out_tilde,
        },
        "expected": expected,
        "plain": plain,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Masked block decode (one step appending to the cache)
# ---------------------------------------------------------------------------


def llama_block_masked_decode(
    x_new: torch.Tensor, cache_tilde: dict[str, Any],
    cache_plain: dict[str, Any], weights: SyntheticLlamaBlockWeights,
    masks: dict[str, Any], config: SyntheticLlamaBlockConfig,
    position_id: int,
) -> dict[str, Any]:
    """One masked decode step at absolute ``position_id`` (== past length)."""
    eps = config.rms_norm_eps
    dtype = config.dtype
    device = torch.device(config.device)
    n_res = masks["n_res"]
    perm = masks["perm"]
    attn_masks = masks["attn"]
    folded = cache_tilde.get("folded") or fold_block_weights(
        weights, masks, config)
    nh, nkv = config.num_heads, config.num_key_value_heads
    hd = config.head_dim
    scale = 1.0 / math.sqrt(hd)

    max_pos = position_id + 2
    cos, sin = build_rope_cache(max_pos, hd, config.rope_base, dtype, device)
    pid = torch.tensor([position_id], device=device)

    # ---- plain reference (append to plain cache) ----
    r1_new = rmsnorm_plain(x_new, weights.rms1_weight, eps)
    q_new = split_heads(r1_new @ weights.wq, nh)
    k_new = split_heads(r1_new @ weights.wk, nkv)
    v_new = split_heads(r1_new @ weights.wv, nkv)
    qr_new = apply_rope(q_new, cos, sin, position_ids=pid)
    kr_new = apply_rope(k_new, cos, sin, position_ids=pid)
    kr_full = torch.cat([cache_plain["key_rope"], kr_new], dim=2)
    v_full = torch.cat([cache_plain["value"], v_new], dim=2)
    _, _, av_new = _sdpa(qr_new, repeat_kv(kr_full, nh, nkv),
                         repeat_kv(v_full, nh, nkv), scale, causal_offset=None)
    attn_out_new = merge_heads(av_new) @ weights.wo
    x1_new = x_new + attn_out_new
    r2_new = rmsnorm_plain(x1_new, weights.rms2_weight, eps)
    mlp_new = swiglu_plain(r2_new, weights.w_gate, weights.w_up,
                           weights.w_down)
    y_new = x1_new + mlp_new["out"]

    # ---- masked path (append to masked cache) ----
    x_new_tilde = x_new @ n_res
    r1_core_tilde = rmsnorm_core(x_new_tilde, eps)
    a = _masked_attention(
        r1_core_tilde, folded, config, cos, sin, causal_offset=None,
        position_ids=pid, past_key_rope=cache_tilde["key_rope_tilde"],
        past_value=cache_tilde["value_tilde"])
    x1_new_tilde = x_new_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_new_tilde, eps)
    gate_t = r2_core_tilde @ folded["wgate_tilde"]
    up_t = r2_core_tilde @ folded["wup_tilde"]
    hidden_t = silu_reference(gate_t) * up_t
    mlp_out_t = hidden_t @ folded["wdown_tilde"]
    y_new_tilde = x1_new_tilde + mlp_out_t

    expected_appended_key = _apply(kr_new, attn_masks["key_masks"])
    expected_appended_value = _apply(v_new, attn_masks["value_masks"])

    metrics = {
        "output_max_abs_error": _mx(y_new_tilde, y_new @ n_res),
        "cache_append_key_max_abs_error": _mx(a["k_rope_new"],
                                              expected_appended_key),
        "cache_append_value_max_abs_error": _mx(a["v"],
                                                expected_appended_value),
    }
    metrics["allclose"] = all(v <= 1e-8 for v in metrics.values()
                              if isinstance(v, float))

    new_cache_tilde = dict(cache_tilde)
    new_cache_tilde["key_rope_tilde"] = a["key_rope_full"]
    new_cache_tilde["value_tilde"] = a["value_full"]
    new_cache_tilde["seq_len"] = cache_tilde["seq_len"] + 1
    new_cache_plain = dict(cache_plain)
    new_cache_plain["key_rope"] = kr_full
    new_cache_plain["value"] = v_full
    new_cache_plain["seq_len"] = cache_plain["seq_len"] + 1

    return {
        "y_new_tilde": y_new_tilde,
        "expected_y_new_tilde": y_new @ n_res,
        "y_new_plain": y_new,
        "appended_key_tilde": a["k_rope_new"],
        "appended_value_tilde": a["v"],
        "expected_appended_key_tilde": expected_appended_key,
        "expected_appended_value_tilde": expected_appended_value,
        "cache_tilde": new_cache_tilde,
        "cache_plain": new_cache_plain,
        "metrics": metrics,
    }


def _apply(x_heads: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Right-multiply each head by its mask (``[B,H,T,D] x [H,D,D]``)."""
    return torch.einsum("bhtd,hde->bhte", x_heads, masks)
