"""Causal self-attention helpers for the tiny Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.cache.cache_state import apply_head_masks
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.mask_state import MaskState
from pllo.ops.linear import linear_plain


def make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Create an additive causal mask for attention scores."""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=torch.device(device)), diagonal=1)
    additive = torch.zeros(seq_len, seq_len, dtype=dtype, device=torch.device(device))
    return additive.masked_fill(mask, torch.finfo(dtype).min)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert [batch, seq, hidden] to [batch, heads, seq, d_head]."""
    batch, seq_len, hidden = x.shape
    if hidden % num_heads != 0:
        raise ValueError(f"hidden size {hidden} must be divisible by num_heads {num_heads}")
    d_head = hidden // num_heads
    return x.reshape(batch, seq_len, num_heads, d_head).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert [batch, heads, seq, d_head] to [batch, seq, hidden]."""
    batch, num_heads, seq_len, d_head = x.shape
    return x.transpose(1, 2).reshape(batch, seq_len, num_heads * d_head)


def causal_self_attention_plain(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Compute standard full-sequence causal self-attention."""
    batch, seq_len, hidden = x.shape
    flat = x.reshape(-1, hidden)
    q = split_heads(linear_plain(flat, w_q).reshape(batch, seq_len, hidden), num_heads)
    k = split_heads(linear_plain(flat, w_k).reshape(batch, seq_len, hidden), num_heads)
    v = split_heads(linear_plain(flat, w_v).reshape(batch, seq_len, hidden), num_heads)
    d_head = hidden // num_heads
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    scores = scores + make_causal_mask(seq_len, x.dtype, x.device).reshape(1, 1, seq_len, seq_len)
    attn = F.softmax(scores, dim=-1)
    out = merge_heads(attn @ v)
    return linear_plain(out.reshape(-1, hidden), w_o).reshape(batch, seq_len, hidden)


def generate_qk_mask_pair(
    num_heads: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate block-diagonal Q/K masks satisfying N_Q N_K^T = I per head."""
    q_blocks = []
    q_inv_blocks = []
    k_blocks = []
    k_inv_blocks = []
    for _ in range(num_heads):
        n_q, n_q_inv = generate_invertible_matrix(d_head, dtype=dtype, device=device)
        n_k = n_q_inv.transpose(-2, -1)
        n_k_inv = n_q.transpose(-2, -1)
        q_blocks.append(n_q)
        q_inv_blocks.append(n_q_inv)
        k_blocks.append(n_k)
        k_inv_blocks.append(n_k_inv)
    return (
        torch.block_diag(*q_blocks),
        torch.block_diag(*q_inv_blocks),
        torch.block_diag(*k_blocks),
        torch.block_diag(*k_inv_blocks),
    )


def generate_head_block_mask(
    num_heads: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a block-diagonal mask that does not mix attention heads."""
    blocks = []
    inv_blocks = []
    for _ in range(num_heads):
        mask, mask_inv = generate_invertible_matrix(d_head, dtype=dtype, device=device)
        blocks.append(mask)
        inv_blocks.append(mask_inv)
    return torch.block_diag(*blocks), torch.block_diag(*inv_blocks)


def generate_head_masks(
    num_heads: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate per-head masks shaped [heads, d_head, d_head]."""
    masks = []
    inverses = []
    for _ in range(num_heads):
        mask, mask_inv = generate_invertible_matrix(d_head, dtype=dtype, device=device)
        masks.append(mask)
        inverses.append(mask_inv)
    return torch.stack(masks), torch.stack(inverses)


def head_masks_to_block_diag(masks: torch.Tensor) -> torch.Tensor:
    """Convert per-head masks to one hidden-dimension block-diagonal matrix."""
    return torch.block_diag(*[masks[idx] for idx in range(masks.shape[0])])


def qk_mask_constraint_error(n_q: torch.Tensor, n_k: torch.Tensor, num_heads: int) -> float:
    """Return max error for the per-head constraint N_Q N_K^T = I."""
    hidden = n_q.shape[0]
    d_head = hidden // num_heads
    max_error = 0.0
    for head in range(num_heads):
        start = head * d_head
        end = start + d_head
        block_q = n_q[start:end, start:end]
        block_k = n_k[start:end, start:end]
        eye = torch.eye(d_head, dtype=n_q.dtype, device=n_q.device)
        max_error = max(max_error, float((block_q @ block_k.transpose(-2, -1) - eye).abs().max().item()))
    return max_error


def qk_head_mask_constraint_error(key_masks: torch.Tensor, key_mask_inverses: torch.Tensor) -> float:
    """Return max error for N_Q N_K^T = I using N_Q = N_K^{-T} per head."""
    n_q = key_mask_inverses.transpose(-2, -1)
    products = n_q @ key_masks.transpose(-2, -1)
    eye = torch.eye(key_masks.shape[-1], dtype=key_masks.dtype, device=key_masks.device).reshape(
        1,
        key_masks.shape[-1],
        key_masks.shape[-1],
    )
    return float((products - eye).abs().max().item())


def project_qkv_plain(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project hidden states to Q/K/V head tensors."""
    batch, seq_len, hidden = x.shape
    flat = x.reshape(-1, hidden)
    q = split_heads(linear_plain(flat, w_q).reshape(batch, seq_len, hidden), num_heads)
    k = split_heads(linear_plain(flat, w_k).reshape(batch, seq_len, hidden), num_heads)
    v = split_heads(linear_plain(flat, w_v).reshape(batch, seq_len, hidden), num_heads)
    return q, k, v


def attention_from_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_o: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Compute attention from projected Q/K/V and apply output projection."""
    _, _, query_len, d_head = q.shape
    key_len = k.shape[2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    if causal:
        scores = scores + make_causal_mask(key_len, q.dtype, q.device)[-query_len:, :].reshape(
            1,
            1,
            query_len,
            key_len,
        )
    attn = F.softmax(scores, dim=-1)
    out = merge_heads(attn @ v)
    return linear_plain(out.reshape(-1, out.shape[-1]), w_o).reshape(*out.shape)


def causal_self_attention_prefill_plain(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run full-sequence attention and return output plus plain K/V cache tensors."""
    q, k, v = project_qkv_plain(x, w_q, w_k, w_v, num_heads)
    return attention_from_qkv(q, k, v, w_o, causal=True), k, v


def causal_self_attention_decode_plain(
    x: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one-token decode attention with plain cached K/V."""
    q, k_new, v_new = project_qkv_plain(x, w_q, w_k, w_v, num_heads)
    k_all = torch.cat([past_k, k_new], dim=2) if past_k is not None else k_new
    v_all = torch.cat([past_v, v_new], dim=2) if past_v is not None else v_new
    return attention_from_qkv(q, k_all, v_all, w_o, causal=False), k_new, v_new


def project_qkv_obfuscated_for_cache(
    x_plain: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    num_heads: int,
    key_masks: torch.Tensor,
    key_mask_inverses: torch.Tensor,
    value_masks: torch.Tensor,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool,
    pad_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q/K/V where Q is paired to K cache masks and K/V are cached obfuscated."""
    batch, seq_len, hidden = x_plain.shape
    flat = x_plain.reshape(-1, hidden)
    q_mask = head_masks_to_block_diag(key_mask_inverses.transpose(-2, -1))
    q_mask_inv = head_masks_to_block_diag(key_masks.transpose(-2, -1))
    k_mask = head_masks_to_block_diag(key_masks)
    k_mask_inv = head_masks_to_block_diag(key_mask_inverses)
    v_mask = head_masks_to_block_diag(value_masks)
    v_mask_inv = torch.linalg.inv(v_mask)

    def run_projection(weight: torch.Tensor, n_out: torch.Tensor, n_out_inv: torch.Tensor) -> torch.Tensor:
        state = tee.create_linear_mask_state(flat, hidden, use_pad=use_pad, pad_scale=pad_scale)
        state.n_out = n_out
        state.n_out_inv = n_out_inv
        return executor.linear_forward(
            tee.obfuscate_input(flat, state),
            *tee.transform_linear_weight(weight, None, state),
            tee.make_linear_pad_compensation(weight, state),
        )

    q_tilde = run_projection(w_q, q_mask, q_mask_inv)
    k_tilde = run_projection(w_k, k_mask, k_mask_inv)
    v_tilde = run_projection(w_v, v_mask, v_mask_inv)
    return (
        split_heads(q_tilde.reshape(batch, seq_len, hidden), num_heads),
        split_heads(k_tilde.reshape(batch, seq_len, hidden), num_heads),
        split_heads(v_tilde.reshape(batch, seq_len, hidden), num_heads),
    )


def attention_from_obfuscated_qkv(
    q_tilde: torch.Tensor,
    k_tilde: torch.Tensor,
    v_tilde: torch.Tensor,
    value_masks: torch.Tensor,
    w_o: torch.Tensor,
    output_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    causal: bool,
) -> torch.Tensor:
    """Compute attention from obfuscated Q/K/V and map output to residual mask."""
    _, _, query_len, d_head = q_tilde.shape
    key_len = k_tilde.shape[2]
    scores = q_tilde @ k_tilde.transpose(-2, -1) / math.sqrt(d_head)
    if causal:
        scores = scores + make_causal_mask(key_len, q_tilde.dtype, q_tilde.device)[-query_len:, :].reshape(
            1,
            1,
            query_len,
            key_len,
        )
    attn = F.softmax(scores, dim=-1)
    out_tilde_v = merge_heads(attn @ v_tilde).reshape(-1, value_masks.shape[0] * value_masks.shape[-1])
    n_v = head_masks_to_block_diag(value_masks)
    n_v_inv = torch.linalg.inv(n_v)
    o_state = MaskState(
        n_in=n_v,
        n_in_inv=n_v_inv,
        n_out=output_state.n_out,
        n_out_inv=output_state.n_out_inv,
    )
    w_o_tilde, b_o_tilde = tee.transform_linear_weight(w_o, None, o_state)
    out_tilde = executor.linear_forward(out_tilde_v, w_o_tilde, b_o_tilde)
    hidden = value_masks.shape[0] * value_masks.shape[-1]
    return out_tilde.reshape(q_tilde.shape[0], query_len, hidden)


def causal_self_attention_prefill_obfuscated(
    x_plain: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    key_masks: torch.Tensor,
    key_mask_inverses: torch.Tensor,
    value_masks: torch.Tensor,
    output_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run obfuscated full-sequence attention and return output plus obfuscated K/V."""
    q_tilde, k_tilde, v_tilde = project_qkv_obfuscated_for_cache(
        x_plain,
        w_q,
        w_k,
        w_v,
        num_heads,
        key_masks,
        key_mask_inverses,
        value_masks,
        tee,
        executor,
        use_pad,
        pad_scale,
    )
    out_tilde = attention_from_obfuscated_qkv(
        q_tilde,
        k_tilde,
        v_tilde,
        value_masks,
        w_o,
        output_state,
        tee,
        executor,
        causal=True,
    )
    return out_tilde, k_tilde, v_tilde


def causal_self_attention_decode_obfuscated(
    x_plain: torch.Tensor,
    past_k_tilde: torch.Tensor,
    past_v_tilde: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    key_masks: torch.Tensor,
    key_mask_inverses: torch.Tensor,
    value_masks: torch.Tensor,
    output_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one-token obfuscated decode attention with cached obfuscated K/V."""
    q_tilde, k_new_tilde, v_new_tilde = project_qkv_obfuscated_for_cache(
        x_plain,
        w_q,
        w_k,
        w_v,
        num_heads,
        key_masks,
        key_mask_inverses,
        value_masks,
        tee,
        executor,
        use_pad,
        pad_scale,
    )
    k_all = torch.cat([past_k_tilde, k_new_tilde], dim=2) if past_k_tilde is not None else k_new_tilde
    v_all = torch.cat([past_v_tilde, v_new_tilde], dim=2) if past_v_tilde is not None else v_new_tilde
    out_tilde = attention_from_obfuscated_qkv(
        q_tilde,
        k_all,
        v_all,
        value_masks,
        w_o,
        output_state,
        tee,
        executor,
        causal=False,
    )
    return out_tilde, k_new_tilde, v_new_tilde


def causal_self_attention_obfuscated(
    x_plain: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    output_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
) -> torch.Tensor:
    """Compute causal self-attention with obfuscated projections.

    Q and K use per-head masks constrained by N_Q N_K^T = I, so attention
    scores match plaintext scores. V remains masked through the softmax
    multiplication, and the output projection maps back to the residual hidden
    mask space.
    """
    batch, seq_len, hidden = x_plain.shape
    if hidden % num_heads != 0:
        raise ValueError(f"hidden size {hidden} must be divisible by num_heads {num_heads}")
    d_head = hidden // num_heads
    flat = x_plain.reshape(-1, hidden)

    n_q, n_q_inv, n_k, n_k_inv = generate_qk_mask_pair(num_heads, d_head, x_plain.dtype, x_plain.device)
    n_v, n_v_inv = generate_head_block_mask(num_heads, d_head, x_plain.dtype, x_plain.device)

    q_state = tee.create_linear_mask_state(flat, hidden, use_pad=use_pad, pad_scale=pad_scale)
    q_state.n_out = n_q
    q_state.n_out_inv = n_q_inv
    k_state = tee.create_linear_mask_state(flat, hidden, use_pad=use_pad, pad_scale=pad_scale)
    k_state.n_out = n_k
    k_state.n_out_inv = n_k_inv
    v_state = tee.create_linear_mask_state(flat, hidden, use_pad=use_pad, pad_scale=pad_scale)
    v_state.n_out = n_v
    v_state.n_out_inv = n_v_inv

    q_tilde = executor.linear_forward(
        tee.obfuscate_input(flat, q_state),
        *tee.transform_linear_weight(w_q, None, q_state),
        tee.make_linear_pad_compensation(w_q, q_state),
    )
    k_tilde = executor.linear_forward(
        tee.obfuscate_input(flat, k_state),
        *tee.transform_linear_weight(w_k, None, k_state),
        tee.make_linear_pad_compensation(w_k, k_state),
    )
    v_tilde = executor.linear_forward(
        tee.obfuscate_input(flat, v_state),
        *tee.transform_linear_weight(w_v, None, v_state),
        tee.make_linear_pad_compensation(w_v, v_state),
    )

    q_heads = split_heads(q_tilde.reshape(batch, seq_len, hidden), num_heads)
    k_heads = split_heads(k_tilde.reshape(batch, seq_len, hidden), num_heads)
    v_heads = split_heads(v_tilde.reshape(batch, seq_len, hidden), num_heads)

    scores = q_heads @ k_heads.transpose(-2, -1) / math.sqrt(d_head)
    scores = scores + make_causal_mask(seq_len, x_plain.dtype, x_plain.device).reshape(1, 1, seq_len, seq_len)
    attn = F.softmax(scores, dim=-1)
    out_tilde_v = merge_heads(attn @ v_heads).reshape(-1, hidden)

    o_state = MaskState(
        n_in=n_v,
        n_in_inv=n_v_inv,
        n_out=output_state.n_out,
        n_out_inv=output_state.n_out_inv,
    )
    w_o_tilde, b_o_tilde = tee.transform_linear_weight(w_o, None, o_state)
    out_tilde = executor.linear_forward(out_tilde_v, w_o_tilde, b_o_tilde)
    return out_tilde.reshape(batch, seq_len, hidden)
