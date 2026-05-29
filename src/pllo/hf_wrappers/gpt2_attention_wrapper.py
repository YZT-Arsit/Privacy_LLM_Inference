"""Obfuscated GPT-2 single-block attention helper."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.masks.mask_state import MaskState
from pllo.model_zoo.gpt2_conv1d_adapter import extract_conv1d_as_linear
from pllo.ops.attention import generate_head_masks, head_masks_to_block_diag, merge_heads, split_heads


def _qkv_output_masks(
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create block-diagonal Q/K/V masks for fused GPT-2 c_attn."""
    key_masks, key_mask_inverses = generate_head_masks(num_heads, head_dim, dtype, device)
    value_masks, value_mask_inverses = generate_head_masks(num_heads, head_dim, dtype, device)
    q_mask = head_masks_to_block_diag(key_mask_inverses.transpose(-2, -1))
    q_mask_inv = head_masks_to_block_diag(key_masks.transpose(-2, -1))
    k_mask = head_masks_to_block_diag(key_masks)
    k_mask_inv = head_masks_to_block_diag(key_mask_inverses)
    v_mask = head_masks_to_block_diag(value_masks)
    v_mask_inv = head_masks_to_block_diag(value_mask_inverses)
    qkv_mask = torch.block_diag(q_mask, k_mask, v_mask)
    qkv_mask_inv = torch.block_diag(q_mask_inv, k_mask_inv, v_mask_inv)
    return qkv_mask, qkv_mask_inv, v_mask, v_mask_inv


def _causal_additive_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    additive = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    return additive.masked_fill(mask, torch.finfo(dtype).min).reshape(1, 1, seq_len, seq_len)


def _build_fused_qkv_masks(
    key_masks: torch.Tensor,
    key_mask_inverses: torch.Tensor,
    value_masks: torch.Tensor,
    value_mask_inverses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble block-diagonal fused QKV mask from per-head K/V masks."""
    q_mask = head_masks_to_block_diag(key_mask_inverses.transpose(-2, -1))
    q_mask_inv = head_masks_to_block_diag(key_masks.transpose(-2, -1))
    k_mask = head_masks_to_block_diag(key_masks)
    k_mask_inv = head_masks_to_block_diag(key_mask_inverses)
    v_mask = head_masks_to_block_diag(value_masks)
    v_mask_inv = head_masks_to_block_diag(value_mask_inverses)
    qkv_mask = torch.block_diag(q_mask, k_mask, v_mask)
    qkv_mask_inv = torch.block_diag(q_mask_inv, k_mask_inv, v_mask_inv)
    return qkv_mask, qkv_mask_inv, v_mask, v_mask_inv


def _project_fused_qkv_tilde(
    x_plain: torch.Tensor,
    attn_module,
    qkv_mask: torch.Tensor,
    qkv_mask_inv: torch.Tensor,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool,
    pad_scale: float,
    pad_audit: dict[str, object] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run c_attn projection with a caller-supplied fused QKV mask."""
    batch, seq_len, hidden_size = x_plain.shape
    w_qkv, b_qkv = extract_conv1d_as_linear(attn_module.c_attn)
    w_qkv = w_qkv.to(dtype=x_plain.dtype, device=x_plain.device)
    b_qkv = None if b_qkv is None else b_qkv.to(dtype=x_plain.dtype, device=x_plain.device)

    flat = x_plain.reshape(-1, hidden_size)
    qkv_state = tee.create_linear_mask_state(
        flat, 3 * hidden_size, use_pad=use_pad, pad_scale=pad_scale
    )
    qkv_state.n_out = qkv_mask
    qkv_state.n_out_inv = qkv_mask_inv
    qkv_compensation = tee.make_linear_pad_compensation(w_qkv, qkv_state)
    if pad_audit is not None:
        pad_audit["attn_c_attn_pad"] = qkv_state.pad is not None
        if qkv_state.pad is not None:
            pad_audit.setdefault("pad_tensor_ids", []).append(id(qkv_state.pad))
    qkv_tilde = executor.linear_forward(
        tee.obfuscate_input(flat, qkv_state),
        *tee.transform_linear_weight(w_qkv, b_qkv, qkv_state),
        qkv_compensation,
    ).reshape(batch, seq_len, 3 * hidden_size)
    return torch.split(qkv_tilde, hidden_size, dim=-1)


def _project_attn_output_tilde(
    attn_output_v_tilde: torch.Tensor,
    attn_module,
    v_mask: torch.Tensor,
    v_mask_inv: torch.Tensor,
    residual_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool,
    pad_scale: float,
    pad_audit: dict[str, object] | None,
) -> torch.Tensor:
    """Apply c_proj to merged attention output, mapping V-mask space to residual mask."""
    w_o, b_o = extract_conv1d_as_linear(attn_module.c_proj)
    w_o = w_o.to(dtype=attn_output_v_tilde.dtype, device=attn_output_v_tilde.device)
    b_o = (
        None
        if b_o is None
        else b_o.to(dtype=attn_output_v_tilde.dtype, device=attn_output_v_tilde.device)
    )

    if use_pad:
        attn_output_plain = attn_output_v_tilde @ v_mask_inv
        o_state = tee.create_linear_mask_state(
            attn_output_plain,
            residual_state.n_out.shape[0],
            use_pad=True,
            pad_scale=pad_scale,
        )
        o_state.n_in = v_mask
        o_state.n_in_inv = v_mask_inv
        o_state.n_out = residual_state.n_out
        o_state.n_out_inv = residual_state.n_out_inv
        o_input_tilde = tee.obfuscate_input(attn_output_plain, o_state)
    else:
        o_state = MaskState(
            n_in=v_mask,
            n_in_inv=v_mask_inv,
            n_out=residual_state.n_out,
            n_out_inv=residual_state.n_out_inv,
        )
        o_input_tilde = attn_output_v_tilde
    o_compensation = tee.make_linear_pad_compensation(w_o, o_state)
    if pad_audit is not None:
        pad_audit["attn_c_proj_pad"] = o_state.pad is not None
        if o_state.pad is not None:
            pad_audit.setdefault("pad_tensor_ids", []).append(id(o_state.pad))
    return executor.linear_forward(
        o_input_tilde,
        *tee.transform_linear_weight(w_o, b_o, o_state),
        o_compensation,
    )


def obfuscated_gpt2_attention(
    x_plain: torch.Tensor,
    attn_module,
    residual_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    attention_mask: torch.Tensor | None = None,
    use_pad: bool = False,
    pad_scale: float = 1.0,
    pad_audit: dict[str, object] | None = None,
) -> torch.Tensor:
    """Run GPT-2 attention with fused c_attn and block-diagonal Q/K/V masks."""
    batch, seq_len, hidden_size = x_plain.shape
    num_heads = int(attn_module.num_heads)
    head_dim = int(attn_module.head_dim)
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            f"hidden_size must equal num_heads * head_dim, got {hidden_size}, "
            f"{num_heads}, {head_dim}"
        )

    w_qkv, b_qkv = extract_conv1d_as_linear(attn_module.c_attn)
    w_o, b_o = extract_conv1d_as_linear(attn_module.c_proj)
    w_qkv = w_qkv.to(dtype=x_plain.dtype, device=x_plain.device)
    b_qkv = None if b_qkv is None else b_qkv.to(dtype=x_plain.dtype, device=x_plain.device)
    w_o = w_o.to(dtype=x_plain.dtype, device=x_plain.device)
    b_o = None if b_o is None else b_o.to(dtype=x_plain.dtype, device=x_plain.device)

    flat = x_plain.reshape(-1, hidden_size)
    qkv_mask, qkv_mask_inv, v_mask, v_mask_inv = _qkv_output_masks(
        num_heads,
        head_dim,
        x_plain.dtype,
        x_plain.device,
    )
    qkv_state = tee.create_linear_mask_state(flat, 3 * hidden_size, use_pad=use_pad, pad_scale=pad_scale)
    qkv_state.n_out = qkv_mask
    qkv_state.n_out_inv = qkv_mask_inv
    qkv_compensation = tee.make_linear_pad_compensation(w_qkv, qkv_state)
    if pad_audit is not None:
        pad_audit["attn_c_attn_pad"] = qkv_state.pad is not None
        if qkv_state.pad is not None:
            pad_audit.setdefault("pad_tensor_ids", []).append(id(qkv_state.pad))
    qkv_tilde = executor.linear_forward(
        tee.obfuscate_input(flat, qkv_state),
        *tee.transform_linear_weight(w_qkv, b_qkv, qkv_state),
        qkv_compensation,
    ).reshape(batch, seq_len, 3 * hidden_size)

    q_tilde, k_tilde, v_tilde = torch.split(qkv_tilde, hidden_size, dim=-1)
    q_heads = split_heads(q_tilde, num_heads)
    k_heads = split_heads(k_tilde, num_heads)
    v_heads = split_heads(v_tilde, num_heads)

    scores = q_heads @ k_heads.transpose(-2, -1)
    if getattr(attn_module, "scale_attn_weights", True):
        scores = scores / math.sqrt(head_dim)
    scores = scores + _causal_additive_mask(seq_len, x_plain.dtype, x_plain.device)
    if attention_mask is not None:
        scores = scores + attention_mask.to(dtype=x_plain.dtype, device=x_plain.device)
    attn_weights = F.softmax(scores, dim=-1)
    attn_output_v_tilde = merge_heads(attn_weights @ v_heads).reshape(-1, hidden_size)

    if use_pad:
        attn_output_plain = attn_output_v_tilde @ v_mask_inv
        o_state = tee.create_linear_mask_state(
            attn_output_plain,
            residual_state.n_out.shape[0],
            use_pad=True,
            pad_scale=pad_scale,
        )
        o_state.n_in = v_mask
        o_state.n_in_inv = v_mask_inv
        o_state.n_out = residual_state.n_out
        o_state.n_out_inv = residual_state.n_out_inv
        o_input_tilde = tee.obfuscate_input(attn_output_plain, o_state)
    else:
        o_state = MaskState(
            n_in=v_mask,
            n_in_inv=v_mask_inv,
            n_out=residual_state.n_out,
            n_out_inv=residual_state.n_out_inv,
        )
        o_input_tilde = attn_output_v_tilde
    o_compensation = tee.make_linear_pad_compensation(w_o, o_state)
    if pad_audit is not None:
        pad_audit["attn_c_proj_pad"] = o_state.pad is not None
        if o_state.pad is not None:
            pad_audit.setdefault("pad_tensor_ids", []).append(id(o_state.pad))
    out_tilde = executor.linear_forward(
        o_input_tilde,
        *tee.transform_linear_weight(w_o, b_o, o_state),
        o_compensation,
    )
    return out_tilde.reshape(batch, seq_len, hidden_size)


def obfuscated_gpt2_attention_prefill(
    x_plain: torch.Tensor,
    attn_module,
    residual_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    key_masks: torch.Tensor,
    key_mask_inverses: torch.Tensor,
    value_masks: torch.Tensor,
    value_mask_inverses: torch.Tensor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
    pad_audit: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run causal attention prefill and return obfuscated K/V cache tensors.

    Returns ``(out_tilde, k_heads_tilde, v_heads_tilde)`` where the cache
    tensors have shape ``[batch, num_heads, seq_len, head_dim]``.
    """
    batch, seq_len, hidden_size = x_plain.shape
    num_heads = int(attn_module.num_heads)
    head_dim = int(attn_module.head_dim)
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            f"hidden_size must equal num_heads * head_dim, got {hidden_size}, "
            f"{num_heads}, {head_dim}"
        )

    qkv_mask, qkv_mask_inv, v_mask, v_mask_inv = _build_fused_qkv_masks(
        key_masks, key_mask_inverses, value_masks, value_mask_inverses
    )
    q_tilde, k_tilde, v_tilde = _project_fused_qkv_tilde(
        x_plain,
        attn_module,
        qkv_mask,
        qkv_mask_inv,
        tee,
        executor,
        use_pad,
        pad_scale,
        pad_audit,
    )
    q_heads = split_heads(q_tilde, num_heads)
    k_heads = split_heads(k_tilde, num_heads)
    v_heads = split_heads(v_tilde, num_heads)

    scores = q_heads @ k_heads.transpose(-2, -1)
    if getattr(attn_module, "scale_attn_weights", True):
        scores = scores / math.sqrt(head_dim)
    scores = scores + _causal_additive_mask(seq_len, x_plain.dtype, x_plain.device)
    attn_weights = F.softmax(scores, dim=-1)
    attn_output_v_tilde = merge_heads(attn_weights @ v_heads).reshape(-1, hidden_size)

    out_tilde_flat = _project_attn_output_tilde(
        attn_output_v_tilde,
        attn_module,
        v_mask,
        v_mask_inv,
        residual_state,
        tee,
        executor,
        use_pad,
        pad_scale,
        pad_audit,
    )
    return out_tilde_flat.reshape(batch, seq_len, hidden_size), k_heads, v_heads


def obfuscated_gpt2_attention_decode(
    x_plain: torch.Tensor,
    attn_module,
    residual_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    past_key_tilde: torch.Tensor,
    past_value_tilde: torch.Tensor,
    key_masks: torch.Tensor,
    key_mask_inverses: torch.Tensor,
    value_masks: torch.Tensor,
    value_mask_inverses: torch.Tensor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
    pad_audit: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one decode step using cached K/V.

    Returns ``(out_tilde, k_new_heads_tilde, v_new_heads_tilde)`` for the new
    token only. The caller is responsible for appending the new entries to the
    cache.
    """
    batch, query_len, hidden_size = x_plain.shape
    if query_len != 1:
        raise ValueError(f"decode_step expects query_len=1, got {query_len}")
    num_heads = int(attn_module.num_heads)
    head_dim = int(attn_module.head_dim)
    if hidden_size != num_heads * head_dim:
        raise ValueError(
            f"hidden_size must equal num_heads * head_dim, got {hidden_size}, "
            f"{num_heads}, {head_dim}"
        )

    qkv_mask, qkv_mask_inv, v_mask, v_mask_inv = _build_fused_qkv_masks(
        key_masks, key_mask_inverses, value_masks, value_mask_inverses
    )
    q_tilde, k_new_tilde, v_new_tilde = _project_fused_qkv_tilde(
        x_plain,
        attn_module,
        qkv_mask,
        qkv_mask_inv,
        tee,
        executor,
        use_pad,
        pad_scale,
        pad_audit,
    )
    q_heads = split_heads(q_tilde, num_heads)
    k_new_heads = split_heads(k_new_tilde, num_heads)
    v_new_heads = split_heads(v_new_tilde, num_heads)

    k_all = torch.cat([past_key_tilde, k_new_heads], dim=2)
    v_all = torch.cat([past_value_tilde, v_new_heads], dim=2)

    scores = q_heads @ k_all.transpose(-2, -1)
    if getattr(attn_module, "scale_attn_weights", True):
        scores = scores / math.sqrt(head_dim)
    # Decode: query is the latest token; all keys (past + current) are valid
    # attention targets, so no additive causal mask is required.
    attn_weights = F.softmax(scores, dim=-1)
    attn_output_v_tilde = merge_heads(attn_weights @ v_all).reshape(-1, hidden_size)

    out_tilde_flat = _project_attn_output_tilde(
        attn_output_v_tilde,
        attn_module,
        v_mask,
        v_mask_inv,
        residual_state,
        tee,
        executor,
        use_pad,
        pad_scale,
        pad_audit,
    )
    return out_tilde_flat.reshape(batch, query_len, hidden_size), k_new_heads, v_new_heads
