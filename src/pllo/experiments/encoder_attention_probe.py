"""Stage 6.1 — Encoder-only bidirectional self-attention probe.

Validates the same mask + pad invariants the Stage 5.0 GPT-2 probe enforces,
but on BERT-style separate Q / K / V linear projections under a
padding-bidirectional attention mask (no causal mask, no KV cache).

Invariants checked, per ``(batch_size, seq_len, use_pad, attention_mask_kind)``:

* ``Q_tilde = Q N_Q``               (per-head N_Q = N_K^{-T})
* ``K_tilde = K N_K``
* ``V_tilde = V N_V``
* ``N_Q N_K^T = I``                  (per head)
* ``Q_tilde K_tilde^T = Q K^T``
* ``softmax(Q_tilde K_tilde^T / sqrt(d) + M) = softmax(Q K^T / sqrt(d) + M)``
  for both all-ones and padding masks
* ``AttnProb V_tilde = (AttnProb V) N_V``  (per head)
* ``W_O`` projects from V-mask space to encoder residual mask space, with
  ``Y_tilde = Y N_out`` and use-pad compensation when enabled.

The probe deliberately does not touch LayerNorm, GELU, the MLM head, or the
pooler — those are out of scope for Stage 6.1 and listed under Limitations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pllo.architectures import (
    DEFAULT_ARCHITECTURE_MODELS,
    load_for_architecture,
)
from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.cache.cache_state import apply_head_masks
from pllo.experiments.report_utils import compare
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.mask_state import MaskState
from pllo.model_zoo.base import torch_dtype_from_string
from pllo.ops.attention import (
    generate_head_masks,
    head_masks_to_block_diag,
    merge_heads,
    qk_head_mask_constraint_error,
    split_heads,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EncoderAttentionProbeConfig:
    model_id: str | None = None  # ``None`` ⇒ try registry candidates in order
    batch_size: int = 2
    seq_len: int = 8
    use_pad: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# BERT layer discovery + linear extraction
# ---------------------------------------------------------------------------


def _first_encoder_layer(model):
    base = getattr(model, "bert", None)
    if base is None:
        base = model
    encoder = getattr(base, "encoder", None)
    if encoder is None or not hasattr(encoder, "layer"):
        raise RuntimeError(
            f"Could not find encoder.layer on model class {type(model).__name__}"
        )
    return base.encoder.layer[0]


def _self_attention_components(model):
    """Return (self_attention_module, output_dense_module) for the first layer."""
    layer = _first_encoder_layer(model)
    return layer.attention.self, layer.attention.output.dense


def _extract_linear(module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pull a ``torch.nn.Linear`` weight into the project's ``[d_in, d_out]`` convention.

    PyTorch stores ``nn.Linear`` weights as ``[out_features, in_features]`` and
    computes ``y = x @ W.T + b``. This helper transposes the weight so the
    obfuscated matmuls in ``SimulatedTEE`` / ``UntrustedGPUExecutor`` (which
    expect ``Y = X @ W``) consume them directly.
    """
    weight = module.weight.detach().clone().T.contiguous()
    bias = None if module.bias is None else module.bias.detach().clone()
    return weight, bias


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------


def _binary_to_additive_mask(
    binary_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a ``[B, S]`` 0/1 mask to BERT's additive ``[B, 1, 1, S]`` form."""
    extended = binary_mask[:, None, None, :].to(dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


def _all_ones_mask(
    batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    binary = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return _binary_to_additive_mask(binary, dtype)


def _padding_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Per-row padding mask: 0 or more trailing tokens are padded out per row."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 7)
    binary = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    if seq_len > 1:
        for i in range(batch_size):
            valid_len = int(
                torch.randint(
                    low=max(seq_len // 2, 1),
                    high=seq_len + 1,
                    size=(1,),
                    generator=generator,
                ).item()
            )
            if valid_len < seq_len:
                binary[i, valid_len:] = 0
    return _binary_to_additive_mask(binary, dtype)


# ---------------------------------------------------------------------------
# Plain reference computation
# ---------------------------------------------------------------------------


def _plain_attention(
    x: torch.Tensor,
    W_Q: torch.Tensor,
    b_Q: torch.Tensor | None,
    W_K: torch.Tensor,
    b_K: torch.Tensor | None,
    W_V: torch.Tensor,
    b_V: torch.Tensor | None,
    W_O: torch.Tensor,
    b_O: torch.Tensor | None,
    num_heads: int,
    additive_mask: torch.Tensor,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    q = x @ W_Q + (b_Q if b_Q is not None else 0)
    k = x @ W_K + (b_K if b_K is not None else 0)
    v = x @ W_V + (b_V if b_V is not None else 0)
    q_heads = split_heads(q, num_heads)
    k_heads = split_heads(k, num_heads)
    v_heads = split_heads(v, num_heads)
    scores = q_heads @ k_heads.transpose(-2, -1) / math.sqrt(head_dim)
    scores_masked = scores + additive_mask
    probs = F.softmax(scores_masked, dim=-1)
    av = probs @ v_heads
    merged = merge_heads(av)
    attn_out = merged @ W_O + (b_O if b_O is not None else 0)
    return {
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "scores": scores,
        "scores_masked": scores_masked,
        "probs": probs,
        "av": av,
        "merged": merged,
        "attn_out": attn_out,
    }


# ---------------------------------------------------------------------------
# Obfuscated single-projection helper
# ---------------------------------------------------------------------------


def _obfuscated_linear(
    flat_input: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    n_out: torch.Tensor,
    n_out_inv: torch.Tensor,
    *,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool,
    pad_scale: float,
) -> tuple[torch.Tensor, MaskState]:
    """Apply one obfuscated linear with caller-supplied output mask.

    The input mask ``n_in`` is freshly sampled by ``create_linear_mask_state``,
    so each Q / K / V / O projection has its own independent input mask. Pad
    compensation is applied when ``use_pad`` is true.
    """
    state = tee.create_linear_mask_state(
        flat_input, n_out.shape[0], use_pad=use_pad, pad_scale=pad_scale
    )
    state.n_out = n_out
    state.n_out_inv = n_out_inv
    compensation = tee.make_linear_pad_compensation(W, state)
    out_tilde = executor.linear_forward(
        tee.obfuscate_input(flat_input, state),
        *tee.transform_linear_weight(W, b, state),
        compensation,
    )
    return out_tilde, state


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_encoder_attention_probe(
    config: EncoderAttentionProbeConfig,
) -> dict[str, Any]:
    """Run the encoder-only attention probe and return a structured report."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)

    # ---- 1. Load BERT (with skip-on-failure) ----
    candidates = (
        (config.model_id,)
        if config.model_id is not None
        else DEFAULT_ARCHITECTURE_MODELS["encoder_only"]
    )
    try:
        model_id, model = load_for_architecture(
            "encoder_only", candidates=candidates
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "config": asdict(config),
            "model_loading": {
                "status": "skipped",
                "candidates_tried": list(candidates),
                "reason": f"{type(exc).__name__}: {exc}",
            },
            "qkv_invariants": {},
            "results_per_mask": {},
            "pad_report": {"use_pad": config.use_pad, "per_mask": {}},
            "mask_structure": {},
        }
    model.eval()

    self_attn, output_dense = _self_attention_components(model)
    hf_cfg = model.config
    hidden_size = int(hf_cfg.hidden_size)
    num_heads = int(hf_cfg.num_attention_heads)
    head_dim = hidden_size // num_heads

    # ---- 2. Extract weights into row-vector convention ----
    W_Q, b_Q = _extract_linear(self_attn.query)
    W_K, b_K = _extract_linear(self_attn.key)
    W_V, b_V = _extract_linear(self_attn.value)
    W_O, b_O = _extract_linear(output_dense)
    for w in (W_Q, W_K, W_V, W_O):
        if tuple(w.shape) != (hidden_size, hidden_size):
            raise RuntimeError(
                f"Expected square [hidden, hidden] linears for BERT self-attn, "
                f"got {tuple(w.shape)} on model {model_id}"
            )
    W_Q = W_Q.to(dtype=dtype, device=device)
    W_K = W_K.to(dtype=dtype, device=device)
    W_V = W_V.to(dtype=dtype, device=device)
    W_O = W_O.to(dtype=dtype, device=device)
    b_Q = b_Q.to(dtype=dtype, device=device) if b_Q is not None else None
    b_K = b_K.to(dtype=dtype, device=device) if b_K is not None else None
    b_V = b_V.to(dtype=dtype, device=device) if b_V is not None else None
    b_O = b_O.to(dtype=dtype, device=device) if b_O is not None else None

    # ---- 3. Random plain hidden state ----
    hidden_states = torch.randn(
        config.batch_size, config.seq_len, hidden_size, dtype=dtype, device=device
    )
    flat = hidden_states.reshape(-1, hidden_size)

    # ---- 4. Trusted-side mask materials ----
    tee = SimulatedTEE(dtype=dtype, device=device)
    executor = UntrustedGPUExecutor()

    key_masks, key_mask_inverses = generate_head_masks(
        num_heads, head_dim, dtype, device
    )
    value_masks, value_mask_inverses = generate_head_masks(
        num_heads, head_dim, dtype, device
    )
    N_Q = head_masks_to_block_diag(key_mask_inverses.transpose(-2, -1))
    N_Q_inv = head_masks_to_block_diag(key_masks.transpose(-2, -1))
    N_K = head_masks_to_block_diag(key_masks)
    N_K_inv = head_masks_to_block_diag(key_mask_inverses)
    N_V = head_masks_to_block_diag(value_masks)
    N_V_inv = head_masks_to_block_diag(value_mask_inverses)

    qk_constraint_error = qk_head_mask_constraint_error(key_masks, key_mask_inverses)

    # ---- 5. Obfuscated Q / K / V projections ----
    q_tilde_flat, q_state = _obfuscated_linear(
        flat, W_Q, b_Q, N_Q, N_Q_inv,
        tee=tee, executor=executor, use_pad=config.use_pad, pad_scale=1.0,
    )
    k_tilde_flat, k_state = _obfuscated_linear(
        flat, W_K, b_K, N_K, N_K_inv,
        tee=tee, executor=executor, use_pad=config.use_pad, pad_scale=1.0,
    )
    v_tilde_flat, v_state = _obfuscated_linear(
        flat, W_V, b_V, N_V, N_V_inv,
        tee=tee, executor=executor, use_pad=config.use_pad, pad_scale=1.0,
    )
    q_tilde = q_tilde_flat.reshape(config.batch_size, config.seq_len, hidden_size)
    k_tilde = k_tilde_flat.reshape(config.batch_size, config.seq_len, hidden_size)
    v_tilde = v_tilde_flat.reshape(config.batch_size, config.seq_len, hidden_size)
    q_heads_tilde = split_heads(q_tilde, num_heads)
    k_heads_tilde = split_heads(k_tilde, num_heads)
    v_heads_tilde = split_heads(v_tilde, num_heads)

    # ---- 6. Plain reference attention (computed once per mask kind) ----
    # We need q_plain etc. for QKV invariant checks too.
    q_plain = flat @ W_Q + (b_Q if b_Q is not None else 0)
    k_plain = flat @ W_K + (b_K if b_K is not None else 0)
    v_plain = flat @ W_V + (b_V if b_V is not None else 0)
    q_plain = q_plain.reshape(config.batch_size, config.seq_len, hidden_size)
    k_plain = k_plain.reshape(config.batch_size, config.seq_len, hidden_size)
    v_plain = v_plain.reshape(config.batch_size, config.seq_len, hidden_size)
    q_heads_plain = split_heads(q_plain, num_heads)
    k_heads_plain = split_heads(k_plain, num_heads)
    v_heads_plain = split_heads(v_plain, num_heads)

    expected_q_tilde = apply_head_masks(
        q_heads_plain, key_mask_inverses.transpose(-2, -1)
    )
    expected_k_tilde = apply_head_masks(k_heads_plain, key_masks)
    expected_v_tilde = apply_head_masks(v_heads_plain, value_masks)
    q_metrics = compare(expected_q_tilde, q_heads_tilde, atol=atol, rtol=rtol)
    k_metrics = compare(expected_k_tilde, k_heads_tilde, atol=atol, rtol=rtol)
    v_metrics = compare(expected_v_tilde, v_heads_tilde, atol=atol, rtol=rtol)

    # ---- 7. Per-mask attention metrics ----
    # Residual mask = output mask space for Y_tilde = Y N_out.
    residual_n_out, residual_n_out_inv = generate_invertible_matrix(
        hidden_size, dtype, device
    )

    mask_results: dict[str, dict] = {}
    pad_audit_per_mask: dict[str, dict[str, bool]] = {}

    for mask_kind in ("all_ones", "padding"):
        if mask_kind == "all_ones":
            additive_mask = _all_ones_mask(
                config.batch_size, config.seq_len, dtype, device
            )
        else:
            additive_mask = _padding_mask(
                config.batch_size, config.seq_len, dtype, device, config.seed
            )

        plain = _plain_attention(
            hidden_states.reshape(config.batch_size, config.seq_len, hidden_size),
            W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O,
            num_heads, additive_mask, head_dim,
        )

        # Score / probability invariants on obfuscated Q/K
        scores_tilde = (
            q_heads_tilde @ k_heads_tilde.transpose(-2, -1) / math.sqrt(head_dim)
        )
        score_metrics = compare(plain["scores"], scores_tilde, atol=atol, rtol=rtol)
        scores_masked_tilde = scores_tilde + additive_mask
        probs_tilde = F.softmax(scores_masked_tilde, dim=-1)
        prob_metrics = compare(plain["probs"], probs_tilde, atol=atol, rtol=rtol)

        # Value aggregation invariant: A V_tilde = (A V) N_V per head.
        av_tilde = probs_tilde @ v_heads_tilde
        expected_av_tilde = apply_head_masks(plain["av"], value_masks)
        v_aggr_metrics = compare(expected_av_tilde, av_tilde, atol=atol, rtol=rtol)

        # Output projection: map from V-mask space → residual mask space.
        merged_tilde = merge_heads(av_tilde).reshape(-1, hidden_size)
        if config.use_pad:
            # Recover to plain, then re-mask with a fresh pad for the O input.
            merged_plain = merged_tilde @ N_V_inv
            o_state = tee.create_linear_mask_state(
                merged_plain,
                hidden_size,
                use_pad=True,
                pad_scale=1.0,
            )
            o_state.n_in = N_V
            o_state.n_in_inv = N_V_inv
            o_state.n_out = residual_n_out
            o_state.n_out_inv = residual_n_out_inv
            o_input_tilde = tee.obfuscate_input(merged_plain, o_state)
        else:
            o_state = MaskState(
                n_in=N_V,
                n_in_inv=N_V_inv,
                n_out=residual_n_out,
                n_out_inv=residual_n_out_inv,
            )
            o_input_tilde = merged_tilde
        o_compensation = tee.make_linear_pad_compensation(W_O, o_state)
        out_tilde_flat = executor.linear_forward(
            o_input_tilde,
            *tee.transform_linear_weight(W_O, b_O, o_state),
            o_compensation,
        )
        out_recovered = (out_tilde_flat @ residual_n_out_inv).reshape(
            config.batch_size, config.seq_len, hidden_size
        )
        output_metrics = compare(plain["attn_out"], out_recovered, atol=atol, rtol=rtol)

        mask_results[mask_kind] = {
            "score_metrics": score_metrics,
            "prob_metrics": prob_metrics,
            "v_aggr_metrics": v_aggr_metrics,
            "output_metrics": output_metrics,
            "allclose": bool(
                score_metrics.get("allclose", False)
                and prob_metrics.get("allclose", False)
                and v_aggr_metrics.get("allclose", False)
                and output_metrics.get("allclose", False)
            ),
        }
        pad_audit_per_mask[mask_kind] = {
            "q_pad": q_state.pad is not None,
            "k_pad": k_state.pad is not None,
            "v_pad": v_state.pad is not None,
            "o_pad": o_state.pad is not None,
        }

    return {
        "config": asdict(config),
        "model_loading": {
            "status": "loaded",
            "model_id": model_id,
            "candidates_tried": list(candidates),
            "model_class": type(model).__name__,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "head_dim": head_dim,
        },
        "qkv_invariants": {
            "q_metrics": q_metrics,
            "k_metrics": k_metrics,
            "v_metrics": v_metrics,
            "qk_constraint_error": qk_constraint_error,
            "qkv_allclose": bool(
                q_metrics.get("allclose", False)
                and k_metrics.get("allclose", False)
                and v_metrics.get("allclose", False)
            ),
        },
        "results_per_mask": mask_results,
        "pad_report": {
            "use_pad": config.use_pad,
            "per_mask": pad_audit_per_mask,
            "compensation_formula": "C = T W N_out",
        },
        "mask_structure": {
            "attention_kind": "bidirectional_self_attention",
            "right_multiply_mask": True,
            "qk_constraint": "N_Q N_K^T = I",
            "value_mask": "per-head block-diagonal",
            "cache_type": "none",
        },
    }
