"""RoPE-compatible masked GQA/MHA attention (CPU tensor-level probe).

Stage 6.4 verifies that right-multiply masking commutes with RoPE and is
compatible with grouped-query attention (GQA), MHA, and KV-cache
prefill/decode. Masks are pair-wise rotation masks (orthogonal, so
``M^{-1} = M^T`` and ``M^{-T} = M``) which commute with the adjacent-pair
RoPE in :mod:`pllo.ops.rope`.

Per Q head ``h`` mapped to KV head ``kv = h // group_size``:
    q_mask[h]    = key_masks[kv]^{-T}     (applied to Q)
    k_mask[kv]   = key_masks[kv]          (applied to K)
    v_mask[kv]   = value_masks[kv]        (applied to V)

Score invariant (per Q head):
    RoPE(Q M^{-T}) @ RoPE(K M)^T = RoPE(Q) @ RoPE(K)^T.

V-aggregation: ``AV_tilde[h] = AV_plain[h] @ value_masks[kv(h)]``; the
output projection folds the (block-diagonal) value mask inverse so the
recovered output equals ``out_plain @ n_out``.

This is a synthetic tensor-level probe. Q/K/V are computed plainly and
then masked at the tensor level; q_proj/k_proj/v_proj/o_proj weight
folding is deferred to a later stage. CPU-only, no transformers.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pllo.ops.rope import (
    apply_rope,
    make_pairwise_complex_scaling_masks,
    make_pairwise_rotation_masks,
    pairwise_complex_scaling_inverse,
)

__all__ = [
    "apply_head_mask_inverses_transpose",
    "apply_head_masks",
    "block_diag_from_head_masks",
    "generate_gqa_rope_masks",
    "masked_rope_gqa_attention_decode",
    "masked_rope_gqa_attention_prefill",
    "merge_heads",
    "repeat_kv",
    "split_heads",
]


# ---------------------------------------------------------------------------
# Head reshaping
# ---------------------------------------------------------------------------


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[B, T, num_heads*head_dim] -> [B, num_heads, T, head_dim]``."""
    b, t, hidden = x.shape
    if hidden % num_heads != 0:
        raise ValueError(
            f"hidden ({hidden}) not divisible by num_heads ({num_heads})"
        )
    head_dim = hidden // num_heads
    return x.reshape(b, t, num_heads, head_dim).transpose(1, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``[B, num_heads, T, head_dim] -> [B, T, num_heads*head_dim]``."""
    b, h, t, d = x.shape
    return x.transpose(1, 2).contiguous().reshape(b, t, h * d)


def repeat_kv(
    x: torch.Tensor, num_heads: int, num_key_value_heads: int,
) -> torch.Tensor:
    """Repeat each KV head ``group_size`` times: ``[B,n_kv,T,D]->[B,H,T,D]``."""
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) not divisible by num_key_value_heads "
            f"({num_key_value_heads})"
        )
    group_size = num_heads // num_key_value_heads
    if group_size == 1:
        return x
    return x.repeat_interleave(group_size, dim=1)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def generate_gqa_rope_masks(
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    mask_family: str = "pairwise_complex_scaling",
) -> dict[str, torch.Tensor]:
    """Per-KV-head RoPE-compatible masks for keys and values.

    ``mask_family`` selects the local mask family:

    * ``"pairwise_rotation"`` -- orthogonal pair-wise rotation blocks
      (correctness baseline; preserves per-pair norm exactly).
    * ``"pairwise_complex_scaling"`` -- ``s R(alpha)`` blocks, the preferred
      RoPE-compatible family (changes per-pair magnitude; not orthogonal,
      closed-form inverse used).

    Returns ``key_masks``, ``key_mask_inverses``, ``value_masks``,
    ``value_mask_inverses`` each ``[num_key_value_heads, D, D]``, plus the
    derived per-Q-head ``q_masks = key_masks[kv]^{-T}`` ``[num_heads, D, D]``
    and the ``kv_index`` map ``[num_heads]``. Both families satisfy the
    score invariant ``RoPE(Q M^{-T}) @ RoPE(K M)^T = RoPE(Q) @ RoPE(K)^T``;
    for complex-scaling the true (closed-form) inverse is used, not the
    transpose. RoPE-compatible masks are weaker than dense masks and carry
    no semantic-security guarantee.
    """
    if num_heads % num_key_value_heads != 0:
        raise ValueError("num_heads must be divisible by num_key_value_heads")
    device = torch.device(device)
    if mask_family == "pairwise_rotation":
        key_masks = make_pairwise_rotation_masks(
            num_key_value_heads, head_dim, dtype, device, generator,
        )
        value_masks = make_pairwise_rotation_masks(
            num_key_value_heads, head_dim, dtype, device, generator,
        )
        # Orthogonal: closed-form inverse is the transpose.
        key_mask_inverses = key_masks.transpose(-2, -1).contiguous()
        value_mask_inverses = value_masks.transpose(-2, -1).contiguous()
    elif mask_family == "pairwise_complex_scaling":
        key_masks = make_pairwise_complex_scaling_masks(
            num_key_value_heads, head_dim, dtype, device, generator,
        )
        value_masks = make_pairwise_complex_scaling_masks(
            num_key_value_heads, head_dim, dtype, device, generator,
        )
        # Not orthogonal: use the closed-form block inverse, not transpose.
        key_mask_inverses = pairwise_complex_scaling_inverse(key_masks)
        value_mask_inverses = pairwise_complex_scaling_inverse(value_masks)
    else:
        raise ValueError(
            f"unknown mask_family {mask_family!r}; expected "
            "'pairwise_rotation' or 'pairwise_complex_scaling'"
        )
    # q_mask[h] = key_masks[kv]^{-T} (inverse-then-transpose; for rotation
    # this equals key_masks[kv]; for complex-scaling it does not).
    key_mask_inv_T = key_mask_inverses.transpose(-2, -1)
    group_size = num_heads // num_key_value_heads
    kv_index = torch.arange(num_heads, device=device) // group_size
    q_masks = key_mask_inv_T.index_select(0, kv_index)
    return {
        "key_masks": key_masks,
        "key_mask_inverses": key_mask_inverses,
        "value_masks": value_masks,
        "value_mask_inverses": value_mask_inverses,
        "q_masks": q_masks,
        "kv_index": kv_index,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "group_size": group_size,
        "mask_family": mask_family,
    }


def apply_head_masks(
    x_heads: torch.Tensor, masks: torch.Tensor,
) -> torch.Tensor:
    """Right-multiply each head by its mask: ``[B,H,T,D] x [H,D,D] -> [B,H,T,D]``."""
    return torch.einsum("bhtd,hde->bhte", x_heads, masks)


def apply_head_mask_inverses_transpose(
    x_heads: torch.Tensor, masks: torch.Tensor,
) -> torch.Tensor:
    """Right-multiply each head by ``masks[h]^{-T}``."""
    inv_t = torch.linalg.inv(masks).transpose(-2, -1)
    return torch.einsum("bhtd,hde->bhte", x_heads, inv_t)


def block_diag_from_head_masks(masks_per_qhead: torch.Tensor) -> torch.Tensor:
    """Assemble ``[H, D, D]`` per-Q-head masks into ``[H*D, H*D]`` block-diag."""
    return torch.block_diag(*[masks_per_qhead[h] for h in
                              range(masks_per_qhead.shape[0])])


# ---------------------------------------------------------------------------
# Internal attention core
# ---------------------------------------------------------------------------


def _causal_bias(t_q: int, t_k: int, dtype: torch.dtype,
                 device: torch.device, offset: int) -> torch.Tensor:
    """Additive causal bias ``[t_q, t_k]``; query i (abs pos offset+i) may
    attend to key j when ``j <= offset + i``."""
    q_pos = torch.arange(offset, offset + t_q, device=device).unsqueeze(1)
    k_pos = torch.arange(t_k, device=device).unsqueeze(0)
    allowed = k_pos <= q_pos
    bias = torch.zeros(t_q, t_k, dtype=dtype, device=device)
    bias.masked_fill_(~allowed, float("-inf"))
    return bias


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def masked_rope_gqa_attention_prefill(
    x: torch.Tensor,
    w_q: torch.Tensor, b_q: torch.Tensor,
    w_k: torch.Tensor, b_k: torch.Tensor,
    w_v: torch.Tensor, b_v: torch.Tensor,
    w_o: torch.Tensor, b_o: torch.Tensor,
    n_out: torch.Tensor,
    masks: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    causal: bool = True,
) -> dict[str, Any]:
    """Plain + masked GQA/MHA prefill with a returnable KV cache."""
    num_heads = masks["num_heads"]
    n_kv = masks["num_key_value_heads"]
    head_dim = masks["head_dim"]
    device = x.device
    dtype = x.dtype
    t = x.shape[1]
    scale = 1.0 / math.sqrt(head_dim)

    # ---- plain ----
    q = split_heads(x @ w_q + b_q, num_heads)
    k = split_heads(x @ w_k + b_k, n_kv)
    v = split_heads(x @ w_v + b_v, n_kv)
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    kr_rep = repeat_kv(kr, num_heads, n_kv)
    v_rep = repeat_kv(v, num_heads, n_kv)
    scores = qr @ kr_rep.transpose(-2, -1) * scale
    if causal:
        scores = scores + _causal_bias(t, t, dtype, device, 0)
    probs = torch.softmax(scores, dim=-1)
    av = probs @ v_rep
    out_plain = merge_heads(av) @ w_o + b_o

    # ---- masked ----
    q_tilde = apply_head_masks(q, masks["q_masks"])
    k_tilde = apply_head_masks(k, masks["key_masks"])
    v_tilde = apply_head_masks(v, masks["value_masks"])
    qr_tilde = apply_rope(q_tilde, cos, sin)
    kr_tilde = apply_rope(k_tilde, cos, sin)
    kr_tilde_rep = repeat_kv(kr_tilde, num_heads, n_kv)
    v_tilde_rep = repeat_kv(v_tilde, num_heads, n_kv)
    scores_tilde = qr_tilde @ kr_tilde_rep.transpose(-2, -1) * scale
    if causal:
        scores_tilde = scores_tilde + _causal_bias(t, t, dtype, device, 0)
    probs_tilde = torch.softmax(scores_tilde, dim=-1)
    av_tilde = probs_tilde @ v_tilde_rep

    # output projection: fold block-diagonal value-mask inverse into W_o.
    v_masks_per_qhead = masks["value_masks"].index_select(0, masks["kv_index"])
    v_inv_per_qhead = masks["value_mask_inverses"].index_select(
        0, masks["kv_index"])
    s_block_inv = block_diag_from_head_masks(v_inv_per_qhead)
    w_o_tilde = s_block_inv @ w_o @ n_out
    b_o_tilde = b_o @ n_out
    out_tilde = merge_heads(av_tilde) @ w_o_tilde + b_o_tilde

    # expected v-aggregation per Q head: AV_plain[h] @ value_masks[kv(h)]
    expected_av_tilde = apply_head_masks(av, v_masks_per_qhead)

    # Causal-masked entries are -inf in both; compare only finite scores.
    finite = torch.isfinite(scores)
    score_max_abs_error = float(
        (scores[finite] - scores_tilde[finite]).abs().max().item()
    )

    cache = {
        "key_rope_tilde": kr_tilde,          # [B, n_kv, T, D]
        "value_tilde": v_tilde,              # [B, n_kv, T, D]
        "seq_len": t,
        "masks": masks,
        "cos": cos,
        "sin": sin,
        "weights": {
            "w_q": w_q, "b_q": b_q, "w_k": w_k, "b_k": b_k,
            "w_v": w_v, "b_v": b_v, "w_o": w_o, "b_o": b_o,
        },
        "n_out": n_out,
        "w_o_tilde": w_o_tilde,
        "b_o_tilde": b_o_tilde,
    }

    return {
        "out_plain": out_plain,
        "out_tilde": out_tilde,
        "expected_out_tilde": out_plain @ n_out,
        "scores_plain": scores,
        "scores_tilde": scores_tilde,
        "score_max_abs_error": score_max_abs_error,
        "probs_plain": probs,
        "probs_tilde": probs_tilde,
        "av_plain": av,
        "av_tilde": av_tilde,
        "expected_av_tilde": expected_av_tilde,
        "kr_plain": kr,
        "v_plain": v,
        "expected_cache_key_tilde": apply_head_masks(kr, masks["key_masks"]),
        "expected_cache_value_tilde": apply_head_masks(v, masks["value_masks"]),
        "cache": cache,
        # RoPE commutes with the per-head Q mask: RoPE(Q M^{-T}) == RoPE(Q) M^{-T}.
        "rope_commutation_q_error": float(
            (qr_tilde - apply_head_masks(qr, masks["q_masks"]))
            .abs().max().item()
        ),
    }


# ---------------------------------------------------------------------------
# Decode (one step)
# ---------------------------------------------------------------------------


def masked_rope_gqa_attention_decode(
    x_new: torch.Tensor,
    cache: dict[str, Any],
    position: int,
) -> dict[str, Any]:
    """One masked decode step appending to ``cache``; also a plain ref.

    ``x_new`` is ``[B, 1, hidden]`` at absolute ``position`` (== past_len).
    """
    masks = cache["masks"]
    num_heads = masks["num_heads"]
    n_kv = masks["num_key_value_heads"]
    head_dim = masks["head_dim"]
    cos, sin = cache["cos"], cache["sin"]
    w = cache["weights"]
    n_out = cache["n_out"]
    device = x_new.device
    dtype = x_new.dtype
    scale = 1.0 / math.sqrt(head_dim)
    pid = torch.tensor([position], device=device)

    # new Q/K/V (plain).
    q_new = split_heads(x_new @ w["w_q"] + w["b_q"], num_heads)
    k_new = split_heads(x_new @ w["w_k"] + w["b_k"], n_kv)
    v_new = split_heads(x_new @ w["w_v"] + w["b_v"], n_kv)
    qr_new = apply_rope(q_new, cos, sin, position_ids=pid)
    kr_new = apply_rope(k_new, cos, sin, position_ids=pid)

    # masked new Q/K/V.
    q_new_tilde = apply_head_masks(q_new, masks["q_masks"])
    k_new_tilde = apply_head_masks(k_new, masks["key_masks"])
    v_new_tilde = apply_head_masks(v_new, masks["value_masks"])
    qr_new_tilde = apply_rope(q_new_tilde, cos, sin, position_ids=pid)
    kr_new_tilde = apply_rope(k_new_tilde, cos, sin, position_ids=pid)

    # append to cache (masked, RoPE'd K and masked V).
    key_cache_tilde = torch.cat([cache["key_rope_tilde"], kr_new_tilde], dim=2)
    value_cache_tilde = torch.cat([cache["value_tilde"], v_new_tilde], dim=2)

    # masked attention over the full (appended) cache; all positions valid.
    kr_tilde_rep = repeat_kv(key_cache_tilde, num_heads, n_kv)
    v_tilde_rep = repeat_kv(value_cache_tilde, num_heads, n_kv)
    scores_tilde = qr_new_tilde @ kr_tilde_rep.transpose(-2, -1) * scale
    probs_tilde = torch.softmax(scores_tilde, dim=-1)
    av_tilde = probs_tilde @ v_tilde_rep
    out_tilde = merge_heads(av_tilde) @ cache["w_o_tilde"] + cache["b_o_tilde"]

    # plain reference over the same appended (plain) cache.
    # Reconstruct plain cache by un-masking the stored tensors.
    key_cache_plain = apply_head_masks(key_cache_tilde, masks["key_mask_inverses"])
    value_cache_plain = apply_head_masks(value_cache_tilde,
                                         masks["value_mask_inverses"])
    kr_rep = repeat_kv(key_cache_plain, num_heads, n_kv)
    v_rep = repeat_kv(value_cache_plain, num_heads, n_kv)
    scores = qr_new @ kr_rep.transpose(-2, -1) * scale
    probs = torch.softmax(scores, dim=-1)
    av = probs @ v_rep
    out_plain = merge_heads(av) @ w["w_o"] + w["b_o"]

    new_cache = dict(cache)
    new_cache["key_rope_tilde"] = key_cache_tilde
    new_cache["value_tilde"] = value_cache_tilde
    new_cache["seq_len"] = cache["seq_len"] + 1

    return {
        "out_plain": out_plain,
        "out_tilde": out_tilde,
        "expected_out_tilde": out_plain @ n_out,
        "appended_key_tilde": kr_new_tilde,
        "appended_value_tilde": v_new_tilde,
        "expected_appended_key_tilde": apply_head_masks(
            kr_new, masks["key_masks"]),
        "expected_appended_value_tilde": apply_head_masks(
            v_new, masks["value_masks"]),
        "cache": new_cache,
    }
