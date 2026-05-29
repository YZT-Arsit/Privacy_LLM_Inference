"""Stage 5.0 Attention probe — verifies 6 attention invariants on GPT-2.

The probe deliberately reuses the existing GPT-2 attention wrapper helpers
rather than reimplementing attention. It captures intermediate tensors via the
same building blocks that `ObfuscatedGPT2BlockWrapper` uses, so any numerical
drift the probe reports also affects the production wrapper path.

Invariants validated (per the paper experiment plan):

1. Q_tilde K_tilde^T ≈ Q K^T                (per-head N_Q N_K^T = I)
2. softmax(Q_tilde K_tilde^T / sqrt(d)) ≈ softmax(Q K^T / sqrt(d))
3. A V_tilde ≈ (A V) N_V                    (right-multiply on each head)
4. AttnOut_tilde ≈ AttnOut N_res             (c_proj into residual space)
5. K_cache_tilde_new ≈ K_cache_new N_K       (cache append, prefill + decode)
6. V_cache_tilde_new ≈ V_cache_new N_V
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.cache.cache_state import apply_head_masks
from pllo.experiments.report_utils import compare
from pllo.hf_wrappers.gpt2_attention_wrapper import (
    _build_fused_qkv_masks,
    _causal_additive_mask,
    _project_attn_output_tilde,
    _project_fused_qkv_tilde,
    obfuscated_gpt2_attention,
    obfuscated_gpt2_attention_decode,
    obfuscated_gpt2_attention_prefill,
)
from pllo.masks.mask_state import MaskState
from pllo.model_zoo import ExternalModelConfig, get_model_loader, torch_dtype_from_string
from pllo.model_zoo.gpt2_conv1d_adapter import extract_conv1d_as_linear
from pllo.ops.attention import (
    generate_head_masks,
    merge_heads,
    qk_head_mask_constraint_error,
    split_heads,
)


@dataclass
class AttentionProbeConfig:
    model_id: str = "sshleifer/tiny-gpt2"
    batch_size: int = 2
    seq_len: int = 8
    decode_steps: int = 2
    use_pad: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_pad_audit() -> dict[str, Any]:
    return {
        "use_pad": False,
        "attn_c_attn_pad": False,
        "attn_c_proj_pad": False,
        "mlp_c_fc_pad": False,
        "mlp_c_proj_pad": False,
        "pad_tensor_ids": [],
    }


def _initial_residual_state(
    hidden_states: torch.Tensor, tee: SimulatedTEE
) -> tuple[MaskState, torch.Tensor]:
    """Replicate `ObfuscatedGPT2BlockWrapper._initial_hidden_state`."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    state = tee.create_linear_mask_state(flat, hidden_states.shape[-1], use_pad=False)
    state.n_out = state.n_in
    state.n_out_inv = state.n_in_inv
    return state, (flat @ state.n_out).reshape_as(hidden_states)


def _plain_qkv(
    ln_plain: torch.Tensor, attn_module, hidden_size: int, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (Q, K, V) per-head via plain c_attn."""
    w_qkv, b_qkv = extract_conv1d_as_linear(attn_module.c_attn)
    w_qkv = w_qkv.to(dtype=ln_plain.dtype, device=ln_plain.device)
    b_qkv = None if b_qkv is None else b_qkv.to(dtype=ln_plain.dtype, device=ln_plain.device)
    qkv = ln_plain @ w_qkv
    if b_qkv is not None:
        qkv = qkv + b_qkv
    q, k, v = torch.split(qkv, hidden_size, dim=-1)
    return split_heads(q, num_heads), split_heads(k, num_heads), split_heads(v, num_heads)


def _plain_attention_output(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    v_heads: torch.Tensor,
    attn_module,
    head_dim: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run plain causal (or non-causal) attention and the c_proj output.

    Returns ``(probs, av, attn_out)`` where ``probs`` is the softmax output,
    ``av = probs @ V``, and ``attn_out`` is the post-c_proj plaintext.
    """
    _, _, query_len, _ = q_heads.shape
    key_len = k_heads.shape[2]
    scores = q_heads @ k_heads.transpose(-2, -1)
    if getattr(attn_module, "scale_attn_weights", True):
        scores = scores / math.sqrt(head_dim)
    if causal:
        scores = scores + _causal_additive_mask(key_len, q_heads.dtype, q_heads.device)[
            ..., -query_len:, :
        ]
    probs = F.softmax(scores, dim=-1)
    av = probs @ v_heads
    merged = merge_heads(av)
    w_o, b_o = extract_conv1d_as_linear(attn_module.c_proj)
    w_o = w_o.to(dtype=q_heads.dtype, device=q_heads.device)
    b_o = None if b_o is None else b_o.to(dtype=q_heads.dtype, device=q_heads.device)
    attn_out = merged @ w_o
    if b_o is not None:
        attn_out = attn_out + b_o
    return probs, av, attn_out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_attention_probe(config: AttentionProbeConfig) -> dict[str, Any]:
    """Run the attention probe and return the structured report dict."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)

    loader_cfg = ExternalModelConfig(
        source="huggingface",
        model_id=config.model_id,
        device=config.device,
        dtype=config.dtype,
    )
    _, model = get_model_loader("hf").load(loader_cfg)
    model.eval()
    block = model.transformer.h[0]
    hf_cfg = model.config
    hidden_size = hf_cfg.n_embd
    num_heads = hf_cfg.n_head
    head_dim = hidden_size // num_heads

    hidden_states = torch.randn(
        config.batch_size,
        config.seq_len,
        hidden_size,
        dtype=dtype,
        device=device,
    )

    # Plain LayerNorm shortcut.
    with torch.no_grad():
        ln1_plain = block.ln_1(hidden_states)

    q_heads_plain, k_heads_plain, v_heads_plain = _plain_qkv(
        ln1_plain, block.attn, hidden_size, num_heads
    )

    # --- Obfuscated setup (same building blocks the wrapper uses) ---
    tee = SimulatedTEE(dtype=dtype, device=device)
    executor = UntrustedGPUExecutor()
    key_masks, key_mask_inverses = generate_head_masks(num_heads, head_dim, dtype, device)
    value_masks, value_mask_inverses = generate_head_masks(num_heads, head_dim, dtype, device)
    qkv_mask, qkv_mask_inv, v_mask, v_mask_inv = _build_fused_qkv_masks(
        key_masks, key_mask_inverses, value_masks, value_mask_inverses
    )

    residual_state, _ = _initial_residual_state(hidden_states, tee)

    # ---- Full attention (Stage 4.6 path) ----
    full_pad_audit = _new_pad_audit()
    full_out_tilde = obfuscated_gpt2_attention(
        ln1_plain,
        block.attn,
        residual_state,
        tee,
        executor,
        attention_mask=None,
        use_pad=config.use_pad,
        pad_scale=1.0,
        pad_audit=full_pad_audit,
    )
    full_out_recovered = full_out_tilde @ residual_state.n_out_inv

    # Plain reference output for full-sequence causal attention.
    probs_plain, av_plain, attn_out_plain = _plain_attention_output(
        q_heads_plain, k_heads_plain, v_heads_plain, block.attn, head_dim, causal=True
    )

    # ---- Score / probability / V-aggregation invariants (probe-only) ----
    probe_pad_audit = _new_pad_audit()
    q_tilde, k_tilde, v_tilde = _project_fused_qkv_tilde(
        ln1_plain,
        block.attn,
        qkv_mask,
        qkv_mask_inv,
        tee,
        executor,
        config.use_pad,
        1.0,
        probe_pad_audit,
    )
    q_heads_tilde = split_heads(q_tilde, num_heads)
    k_heads_tilde = split_heads(k_tilde, num_heads)
    v_heads_tilde = split_heads(v_tilde, num_heads)

    scores_plain = q_heads_plain @ k_heads_plain.transpose(-2, -1)
    scores_tilde = q_heads_tilde @ k_heads_tilde.transpose(-2, -1)
    if getattr(block.attn, "scale_attn_weights", True):
        scores_plain = scores_plain / math.sqrt(head_dim)
        scores_tilde = scores_tilde / math.sqrt(head_dim)
    causal_mask = _causal_additive_mask(config.seq_len, dtype, device)
    probs_tilde = F.softmax(scores_tilde + causal_mask, dim=-1)
    av_tilde = probs_tilde @ v_heads_tilde

    score_metrics = compare(scores_plain, scores_tilde, atol=atol, rtol=rtol)
    prob_metrics = compare(probs_plain, probs_tilde, atol=atol, rtol=rtol)
    expected_av_tilde = apply_head_masks(av_plain, value_masks)
    v_aggr_metrics = compare(expected_av_tilde, av_tilde, atol=atol, rtol=rtol)
    qk_err = qk_head_mask_constraint_error(key_masks, key_mask_inverses)

    full_output_metrics = compare(attn_out_plain, full_out_recovered, atol=atol, rtol=rtol)

    # ---- Prefill path with cache output ----
    prefill_pad_audit = _new_pad_audit()
    prefill_out_tilde, prefill_k_tilde, prefill_v_tilde = obfuscated_gpt2_attention_prefill(
        ln1_plain,
        block.attn,
        residual_state,
        tee,
        executor,
        key_masks=key_masks,
        key_mask_inverses=key_mask_inverses,
        value_masks=value_masks,
        value_mask_inverses=value_mask_inverses,
        use_pad=config.use_pad,
        pad_scale=1.0,
        pad_audit=prefill_pad_audit,
    )
    prefill_out_recovered = prefill_out_tilde @ residual_state.n_out_inv
    prefill_output_metrics = compare(attn_out_plain, prefill_out_recovered, atol=atol, rtol=rtol)
    expected_prefill_k_tilde = apply_head_masks(k_heads_plain, key_masks)
    expected_prefill_v_tilde = apply_head_masks(v_heads_plain, value_masks)
    prefill_cache_key_metrics = compare(expected_prefill_k_tilde, prefill_k_tilde, atol=atol, rtol=rtol)
    prefill_cache_value_metrics = compare(expected_prefill_v_tilde, prefill_v_tilde, atol=atol, rtol=rtol)

    # ---- Decode steps ----
    decode_per_step: list[dict[str, Any]] = []
    decode_pad_flags = {"attn_c_attn_pad": False, "attn_c_proj_pad": False}
    cum_k_plain = k_heads_plain.clone()
    cum_v_plain = v_heads_plain.clone()
    cum_k_tilde = prefill_k_tilde.clone()
    cum_v_tilde = prefill_v_tilde.clone()

    for step in range(config.decode_steps):
        new_hidden = torch.randn(config.batch_size, 1, hidden_size, dtype=dtype, device=device)
        with torch.no_grad():
            ln_new_plain = block.ln_1(new_hidden)
        q_new_plain, k_new_plain, v_new_plain = _plain_qkv(
            ln_new_plain, block.attn, hidden_size, num_heads
        )
        full_k_plain = torch.cat([cum_k_plain, k_new_plain], dim=2)
        full_v_plain = torch.cat([cum_v_plain, v_new_plain], dim=2)

        _, _, attn_out_step_plain = _plain_attention_output(
            q_new_plain, full_k_plain, full_v_plain, block.attn, head_dim, causal=False
        )

        decode_residual_state, _ = _initial_residual_state(new_hidden, tee)
        decode_pad_audit = _new_pad_audit()
        decode_out_tilde, k_new_tilde, v_new_tilde = obfuscated_gpt2_attention_decode(
            ln_new_plain,
            block.attn,
            decode_residual_state,
            tee,
            executor,
            past_key_tilde=cum_k_tilde,
            past_value_tilde=cum_v_tilde,
            key_masks=key_masks,
            key_mask_inverses=key_mask_inverses,
            value_masks=value_masks,
            value_mask_inverses=value_mask_inverses,
            use_pad=config.use_pad,
            pad_scale=1.0,
            pad_audit=decode_pad_audit,
        )
        decode_pad_flags["attn_c_attn_pad"] = decode_pad_audit["attn_c_attn_pad"]
        decode_pad_flags["attn_c_proj_pad"] = decode_pad_audit["attn_c_proj_pad"]

        decode_out_recovered = decode_out_tilde @ decode_residual_state.n_out_inv
        step_output_metrics = compare(attn_out_step_plain, decode_out_recovered, atol=atol, rtol=rtol)

        expected_k_new_tilde = apply_head_masks(k_new_plain, key_masks)
        expected_v_new_tilde = apply_head_masks(v_new_plain, value_masks)
        step_new_key_metrics = compare(expected_k_new_tilde, k_new_tilde, atol=atol, rtol=rtol)
        step_new_value_metrics = compare(expected_v_new_tilde, v_new_tilde, atol=atol, rtol=rtol)

        decode_per_step.append(
            {
                "step": step,
                "output_metrics": step_output_metrics,
                "new_key_metrics": step_new_key_metrics,
                "new_value_metrics": step_new_value_metrics,
            }
        )

        cum_k_plain = full_k_plain
        cum_v_plain = full_v_plain
        cum_k_tilde = torch.cat([cum_k_tilde, k_new_tilde], dim=2)
        cum_v_tilde = torch.cat([cum_v_tilde, v_new_tilde], dim=2)

    # Cumulative cache invariant after all decode steps.
    expected_cum_k_tilde = apply_head_masks(cum_k_plain, key_masks)
    expected_cum_v_tilde = apply_head_masks(cum_v_plain, value_masks)
    cache_append_key_metrics = compare(expected_cum_k_tilde, cum_k_tilde, atol=atol, rtol=rtol)
    cache_append_value_metrics = compare(expected_cum_v_tilde, cum_v_tilde, atol=atol, rtol=rtol)
    cache_append_allclose = bool(
        cache_append_key_metrics["allclose"] and cache_append_value_metrics["allclose"]
    )

    decode_output_max = (
        max(s["output_metrics"]["max_abs_error"] for s in decode_per_step)
        if decode_per_step
        else None
    )

    full_allclose = bool(
        full_output_metrics.get("allclose", False)
        and score_metrics.get("allclose", False)
        and prob_metrics.get("allclose", False)
        and v_aggr_metrics.get("allclose", False)
    )

    return {
        "config": asdict(config),
        "full_attention": {
            "output_metrics": full_output_metrics,
            "score_metrics": score_metrics,
            "prob_metrics": prob_metrics,
            "v_aggr_metrics": v_aggr_metrics,
            "qk_constraint_error": qk_err,
            "allclose": full_allclose,
        },
        "prefill_attention": {
            "output_metrics": prefill_output_metrics,
            "cache_key_metrics": prefill_cache_key_metrics,
            "cache_value_metrics": prefill_cache_value_metrics,
            "cache_invariant_allclose": bool(
                prefill_cache_key_metrics["allclose"]
                and prefill_cache_value_metrics["allclose"]
            ),
        },
        "decode_attention": {
            "per_step": decode_per_step,
            "decode_output_max_abs_error_max": decode_output_max,
            "cache_append_key_metrics": cache_append_key_metrics,
            "cache_append_value_metrics": cache_append_value_metrics,
            "cache_append_invariant_allclose": cache_append_allclose,
        },
        "mask_structure": {
            "right_multiply_mask": True,
            "fused_c_attn_block_diagonal": True,
            "qk_constraint": "N_Q N_K^T = I",
            "value_mask": "per-head block-diagonal",
            "same_cache_mask_within_session": True,
        },
        "pad_report": {
            "use_pad": config.use_pad,
            "attn_c_attn_pad": (
                prefill_pad_audit["attn_c_attn_pad"]
                or full_pad_audit["attn_c_attn_pad"]
                or decode_pad_flags["attn_c_attn_pad"]
            ),
            "attn_c_proj_pad": (
                prefill_pad_audit["attn_c_proj_pad"]
                or full_pad_audit["attn_c_proj_pad"]
                or decode_pad_flags["attn_c_proj_pad"]
            ),
            "compensation_formula": "C_T = T W N_out",
        },
    }
