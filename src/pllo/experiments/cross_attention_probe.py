"""Stage 6.2 — Encoder-decoder cross-attention probe.

Validates the same mask + pad invariants the Stage 6.1 encoder probe enforces,
but with Q coming from the decoder hidden state and K/V coming from encoder
memory — so the input mask spaces for Q vs K/V can differ. Adds a small,
probe-level ``EncoderMemoryCache`` data structure capturing the encoder-side
obfuscated K/V tensors and the masks that produced them.

Invariants checked, per ``(batch_size, dec_seq_len, enc_seq_len, use_pad,
encoder_attention_mask_kind)``:

* ``Q_dec_tilde = Q_dec N_Q_dec``  (per-head N_Q = N_K^{-T})
* ``K_enc_tilde = K_enc N_K_enc``
* ``V_enc_tilde = V_enc N_V_enc``
* ``N_Q_dec N_K_enc^T = I``         (per head)
* ``Q_dec_tilde K_enc_tilde^T = Q_dec K_enc^T``
* ``softmax(Q_dec_tilde K_enc_tilde^T / sqrt(d) + M_enc)
    = softmax(Q_dec K_enc^T / sqrt(d) + M_enc)``
  for both all-ones and padding encoder masks
* ``AttnProb V_enc_tilde = (AttnProb V_enc) N_V_enc`` (per head)
* ``W_O`` projects from V-mask space → decoder residual mask space, with
  ``Y_dec_tilde = Y_dec N_dec_out`` and use-pad compensation when enabled.
* ``EncoderMemoryCache`` invariants ``K_enc_tilde ≈ K_enc N_K_enc`` and
  ``V_enc_tilde ≈ V_enc N_V_enc``.

Stage 6.2 deliberately does not implement full T5/BART obfuscated forward,
decoder self-attention cache, encoder-decoder generation, LayerNorm/FFN/
activation obfuscation, LM head, or relative position bias. Those exclusions
are listed in the report's Limitations section.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
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
class CrossAttentionProbeConfig:
    """One probe cell — covers both ``all_ones`` and ``padding`` mask kinds."""

    model_id: str | None = None  # None ⇒ try registry candidates in order
    batch_size: int = 2
    dec_seq_len: int = 4
    enc_seq_len: int = 8
    use_pad: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Probe-level EncoderMemoryCache
# ---------------------------------------------------------------------------


@dataclass
class EncoderMemoryCache:
    """Probe-only cache of encoder-side K/V (plain + obfuscated) and masks.

    This is a Stage-6.2 *probe* structure — it is **not** a generation-runtime
    cache. It captures just enough to validate ``K_enc_tilde = K_enc N_K_enc``
    and ``V_enc_tilde = V_enc N_V_enc`` for the cross-attention probe.
    """

    key_tilde: torch.Tensor       # [B, heads, T_enc, head_dim]
    value_tilde: torch.Tensor     # [B, heads, T_enc, head_dim]
    key_plain: torch.Tensor       # [B, heads, T_enc, head_dim]
    value_plain: torch.Tensor     # [B, heads, T_enc, head_dim]
    n_k: torch.Tensor             # [inner_dim, inner_dim]   block-diag N_K_enc
    n_v: torch.Tensor             # [inner_dim, inner_dim]   block-diag N_V_enc
    encoder_seq_len: int
    batch_size: int

    def invariants(
        self,
        key_masks: torch.Tensor,
        value_masks: torch.Tensor,
        atol: float,
        rtol: float,
    ) -> dict[str, dict[str, Any]]:
        """Verify ``K_enc_tilde ≈ K_enc N_K_enc`` and ``V_enc_tilde ≈ V_enc N_V_enc``."""
        expected_k_tilde = apply_head_masks(self.key_plain, key_masks)
        expected_v_tilde = apply_head_masks(self.value_plain, value_masks)
        return {
            "key_metrics": compare(expected_k_tilde, self.key_tilde, atol=atol, rtol=rtol),
            "value_metrics": compare(expected_v_tilde, self.value_tilde, atol=atol, rtol=rtol),
        }


# ---------------------------------------------------------------------------
# T5 / BART cross-attention layer discovery + linear extraction
# ---------------------------------------------------------------------------


def _cross_attention_module(model) -> tuple[Any, dict[str, str], str]:
    """Return ``(cross_attn_module, projection_map, family)`` for the first decoder layer.

    ``projection_map`` keys are ``"q"``, ``"k"``, ``"v"``, ``"o"`` and the
    values name the attribute on ``cross_attn_module`` holding each linear.
    ``family`` is ``"t5"`` or ``"bart"``.
    """
    # T5: model.decoder.block[0].layer[1].EncDecAttention.{q,k,v,o}
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "block") and len(decoder.block) > 0:
        first_block = decoder.block[0]
        if hasattr(first_block, "layer") and len(first_block.layer) >= 2:
            layer_1 = first_block.layer[1]
            cross = getattr(layer_1, "EncDecAttention", None)
            if cross is not None:
                return cross, {"q": "q", "k": "k", "v": "v", "o": "o"}, "t5"
    # BART: model.model.decoder.layers[0].encoder_attn.{q_proj,k_proj,v_proj,out_proj}
    inner = getattr(model, "model", model)
    decoder = getattr(inner, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers") and len(decoder.layers) > 0:
        first_layer = decoder.layers[0]
        cross = getattr(first_layer, "encoder_attn", None)
        if cross is not None:
            return cross, {
                "q": "q_proj",
                "k": "k_proj",
                "v": "v_proj",
                "o": "out_proj",
            }, "bart"
    raise RuntimeError(
        f"Could not locate decoder cross-attention on model class "
        f"{type(model).__name__}"
    )


def _extract_linear(module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pull a ``torch.nn.Linear`` weight into the project's ``[d_in, d_out]`` convention.

    Supports ``bias=None`` (T5 attention projections carry no bias).
    """
    weight = module.weight.detach().clone().T.contiguous()
    bias = None if module.bias is None else module.bias.detach().clone()
    return weight, bias


def _resolve_head_dim(cfg, hidden_size: int, num_heads: int) -> int:
    """Resolve attention head dimension from a T5/BART config.

    T5 carries an explicit ``d_kv`` that can differ from ``d_model / num_heads``.
    BART uses ``d_model / num_heads``.
    """
    d_kv = getattr(cfg, "d_kv", None)
    if d_kv is not None:
        return int(d_kv)
    return hidden_size // num_heads


# ---------------------------------------------------------------------------
# Encoder padding mask construction
# ---------------------------------------------------------------------------


def _binary_to_additive_mask(
    binary_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a ``[B, T_enc]`` 0/1 mask to broadcastable ``[B, 1, 1, T_enc]`` additive form."""
    extended = binary_mask[:, None, None, :].to(dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


def _all_ones_encoder_mask(
    batch_size: int, enc_seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    binary = torch.ones(batch_size, enc_seq_len, dtype=torch.long, device=device)
    return _binary_to_additive_mask(binary, dtype)


def _padding_encoder_mask(
    batch_size: int,
    enc_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Per-row encoder padding mask: 0 or more trailing encoder tokens are padded out."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 13)
    binary = torch.ones(batch_size, enc_seq_len, dtype=torch.long, device=device)
    if enc_seq_len > 1:
        for i in range(batch_size):
            valid_len = int(
                torch.randint(
                    low=max(enc_seq_len // 2, 1),
                    high=enc_seq_len + 1,
                    size=(1,),
                    generator=generator,
                ).item()
            )
            if valid_len < enc_seq_len:
                binary[i, valid_len:] = 0
    return _binary_to_additive_mask(binary, dtype)


# ---------------------------------------------------------------------------
# Plain reference cross-attention
# ---------------------------------------------------------------------------


def _plain_cross_attention(
    hidden_dec: torch.Tensor,
    hidden_enc: torch.Tensor,
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
    """Reference cross-attention computation."""
    q = hidden_dec @ W_Q + (b_Q if b_Q is not None else 0)
    k = hidden_enc @ W_K + (b_K if b_K is not None else 0)
    v = hidden_enc @ W_V + (b_V if b_V is not None else 0)
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
# Obfuscated single-projection helper (mirrors encoder probe)
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

    Each projection gets its own freshly sampled input mask (and pad, when
    ``use_pad=True``). Supports ``b=None`` for bias-free projections (T5).
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


def run_cross_attention_probe(
    config: CrossAttentionProbeConfig,
) -> dict[str, Any]:
    """Run the encoder-decoder cross-attention probe and return a structured report."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)

    # ---- 1. Load encoder-decoder model (skip on failure) ----
    candidates = (
        (config.model_id,)
        if config.model_id is not None
        else DEFAULT_ARCHITECTURE_MODELS["encoder_decoder"]
    )
    try:
        model_id, model = load_for_architecture(
            "encoder_decoder", candidates=candidates
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
            "encoder_memory_cache": {
                "key_metrics": {},
                "value_metrics": {},
                "encoder_seq_len": config.enc_seq_len,
                "batch_size": config.batch_size,
                "allclose": None,
                "per_mask": {},
            },
            "mask_structure": {},
        }
    model.eval()

    cross_attn, projection_map, family = _cross_attention_module(model)
    hf_cfg = model.config
    hidden_size = int(getattr(hf_cfg, "d_model", None) or getattr(hf_cfg, "hidden_size"))
    num_heads = int(
        getattr(hf_cfg, "decoder_attention_heads", None)
        or getattr(hf_cfg, "num_heads", None)
        or getattr(hf_cfg, "num_attention_heads")
    )
    head_dim = _resolve_head_dim(hf_cfg, hidden_size, num_heads)
    inner_dim = num_heads * head_dim

    # ---- 2. Extract Q/K/V/O ----
    q_module = getattr(cross_attn, projection_map["q"])
    k_module = getattr(cross_attn, projection_map["k"])
    v_module = getattr(cross_attn, projection_map["v"])
    o_module = getattr(cross_attn, projection_map["o"])
    W_Q, b_Q = _extract_linear(q_module)
    W_K, b_K = _extract_linear(k_module)
    W_V, b_V = _extract_linear(v_module)
    W_O, b_O = _extract_linear(o_module)

    # Q/K/V map d_model -> inner_dim; O maps inner_dim -> d_model.
    for name, w, expected_shape in (
        ("W_Q", W_Q, (hidden_size, inner_dim)),
        ("W_K", W_K, (hidden_size, inner_dim)),
        ("W_V", W_V, (hidden_size, inner_dim)),
        ("W_O", W_O, (inner_dim, hidden_size)),
    ):
        if tuple(w.shape) != expected_shape:
            raise RuntimeError(
                f"{family} cross-attention {name} has shape {tuple(w.shape)}, "
                f"expected {expected_shape} on model {model_id}"
            )

    W_Q = W_Q.to(dtype=dtype, device=device)
    W_K = W_K.to(dtype=dtype, device=device)
    W_V = W_V.to(dtype=dtype, device=device)
    W_O = W_O.to(dtype=dtype, device=device)
    b_Q = b_Q.to(dtype=dtype, device=device) if b_Q is not None else None
    b_K = b_K.to(dtype=dtype, device=device) if b_K is not None else None
    b_V = b_V.to(dtype=dtype, device=device) if b_V is not None else None
    b_O = b_O.to(dtype=dtype, device=device) if b_O is not None else None

    bias_present = {
        "q": b_Q is not None,
        "k": b_K is not None,
        "v": b_V is not None,
        "o": b_O is not None,
    }

    # ---- 3. Random plain decoder + encoder hidden states ----
    hidden_dec = torch.randn(
        config.batch_size, config.dec_seq_len, hidden_size, dtype=dtype, device=device
    )
    hidden_enc = torch.randn(
        config.batch_size, config.enc_seq_len, hidden_size, dtype=dtype, device=device
    )
    flat_dec = hidden_dec.reshape(-1, hidden_size)
    flat_enc = hidden_enc.reshape(-1, hidden_size)

    # ---- 4. Trusted-side masks ----
    tee = SimulatedTEE(dtype=dtype, device=device)
    executor = UntrustedGPUExecutor()

    key_masks, key_mask_inverses = generate_head_masks(
        num_heads, head_dim, dtype, device
    )
    value_masks, value_mask_inverses = generate_head_masks(
        num_heads, head_dim, dtype, device
    )
    N_Q_dec = head_masks_to_block_diag(key_mask_inverses.transpose(-2, -1))
    N_Q_dec_inv = head_masks_to_block_diag(key_masks.transpose(-2, -1))
    N_K_enc = head_masks_to_block_diag(key_masks)
    N_K_enc_inv = head_masks_to_block_diag(key_mask_inverses)
    N_V_enc = head_masks_to_block_diag(value_masks)
    N_V_enc_inv = head_masks_to_block_diag(value_mask_inverses)

    qk_constraint_error = qk_head_mask_constraint_error(
        key_masks, key_mask_inverses
    )

    # ---- 5. Obfuscated Q (decoder side) ----
    q_tilde_flat, q_state = _obfuscated_linear(
        flat_dec, W_Q, b_Q, N_Q_dec, N_Q_dec_inv,
        tee=tee, executor=executor, use_pad=config.use_pad, pad_scale=1.0,
    )

    # ---- 6. Obfuscated K, V (encoder memory side) ----
    k_tilde_flat, k_state = _obfuscated_linear(
        flat_enc, W_K, b_K, N_K_enc, N_K_enc_inv,
        tee=tee, executor=executor, use_pad=config.use_pad, pad_scale=1.0,
    )
    v_tilde_flat, v_state = _obfuscated_linear(
        flat_enc, W_V, b_V, N_V_enc, N_V_enc_inv,
        tee=tee, executor=executor, use_pad=config.use_pad, pad_scale=1.0,
    )

    q_tilde = q_tilde_flat.reshape(
        config.batch_size, config.dec_seq_len, inner_dim
    )
    k_tilde = k_tilde_flat.reshape(
        config.batch_size, config.enc_seq_len, inner_dim
    )
    v_tilde = v_tilde_flat.reshape(
        config.batch_size, config.enc_seq_len, inner_dim
    )
    q_heads_tilde = split_heads(q_tilde, num_heads)
    k_heads_tilde = split_heads(k_tilde, num_heads)
    v_heads_tilde = split_heads(v_tilde, num_heads)

    # ---- 7. Plain reference Q/K/V (for QKV invariants) ----
    q_plain = (flat_dec @ W_Q + (b_Q if b_Q is not None else 0)).reshape(
        config.batch_size, config.dec_seq_len, inner_dim
    )
    k_plain = (flat_enc @ W_K + (b_K if b_K is not None else 0)).reshape(
        config.batch_size, config.enc_seq_len, inner_dim
    )
    v_plain = (flat_enc @ W_V + (b_V if b_V is not None else 0)).reshape(
        config.batch_size, config.enc_seq_len, inner_dim
    )
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

    # ---- 8. EncoderMemoryCache (probe-level) ----
    encoder_cache = EncoderMemoryCache(
        key_tilde=k_heads_tilde,
        value_tilde=v_heads_tilde,
        key_plain=k_heads_plain,
        value_plain=v_heads_plain,
        n_k=N_K_enc,
        n_v=N_V_enc,
        encoder_seq_len=config.enc_seq_len,
        batch_size=config.batch_size,
    )
    cache_invariants = encoder_cache.invariants(
        key_masks, value_masks, atol=atol, rtol=rtol
    )

    # ---- 9. Per-encoder-mask attention metrics ----
    residual_n_out, residual_n_out_inv = generate_invertible_matrix(
        hidden_size, dtype, device
    )

    mask_results: dict[str, dict[str, Any]] = {}
    pad_audit_per_mask: dict[str, dict[str, bool]] = {}

    for mask_kind in ("all_ones", "padding"):
        if mask_kind == "all_ones":
            additive_mask = _all_ones_encoder_mask(
                config.batch_size, config.enc_seq_len, dtype, device
            )
        else:
            additive_mask = _padding_encoder_mask(
                config.batch_size, config.enc_seq_len, dtype, device, config.seed
            )

        plain = _plain_cross_attention(
            hidden_dec, hidden_enc,
            W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O,
            num_heads, additive_mask, head_dim,
        )

        scores_tilde = (
            q_heads_tilde @ k_heads_tilde.transpose(-2, -1) / math.sqrt(head_dim)
        )
        score_metrics = compare(plain["scores"], scores_tilde, atol=atol, rtol=rtol)
        scores_masked_tilde = scores_tilde + additive_mask
        probs_tilde = F.softmax(scores_masked_tilde, dim=-1)
        prob_metrics = compare(plain["probs"], probs_tilde, atol=atol, rtol=rtol)

        av_tilde = probs_tilde @ v_heads_tilde
        expected_av_tilde = apply_head_masks(plain["av"], value_masks)
        v_aggr_metrics = compare(expected_av_tilde, av_tilde, atol=atol, rtol=rtol)

        # ---- 10. Output projection: V-mask space → decoder residual space ----
        merged_tilde = merge_heads(av_tilde).reshape(-1, inner_dim)
        if config.use_pad:
            merged_plain = merged_tilde @ N_V_enc_inv
            o_state = tee.create_linear_mask_state(
                merged_plain, hidden_size, use_pad=True, pad_scale=1.0,
            )
            o_state.n_in = N_V_enc
            o_state.n_in_inv = N_V_enc_inv
            o_state.n_out = residual_n_out
            o_state.n_out_inv = residual_n_out_inv
            o_input_tilde = tee.obfuscate_input(merged_plain, o_state)
        else:
            o_state = MaskState(
                n_in=N_V_enc,
                n_in_inv=N_V_enc_inv,
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
            config.batch_size, config.dec_seq_len, hidden_size
        )
        output_metrics = compare(
            plain["attn_out"], out_recovered, atol=atol, rtol=rtol
        )

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
            "family": family,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "head_dim": head_dim,
            "inner_dim": inner_dim,
            "bias_present": bias_present,
            "cross_attention_has_relative_bias": bool(
                getattr(cross_attn, "has_relative_attention_bias", False)
            ),
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
        "encoder_memory_cache": {
            "key_metrics": cache_invariants["key_metrics"],
            "value_metrics": cache_invariants["value_metrics"],
            "encoder_seq_len": encoder_cache.encoder_seq_len,
            "batch_size": encoder_cache.batch_size,
            "allclose": bool(
                cache_invariants["key_metrics"].get("allclose", False)
                and cache_invariants["value_metrics"].get("allclose", False)
            ),
        },
        "results_per_mask": mask_results,
        "pad_report": {
            "use_pad": config.use_pad,
            "per_mask": pad_audit_per_mask,
            "compensation_formula": "C = T W N_out",
        },
        "mask_structure": {
            "attention_kind": "encoder_decoder_cross_attention",
            "right_multiply_mask": True,
            "qk_constraint": "N_Q_dec N_K_enc^T = I",
            "value_mask": "per-head block-diagonal",
            "decoder_query_input_mask_independent_from_encoder_kv_input_mask": True,
            "cache_type": "encoder_memory_cache (probe-level)",
        },
    }


__all__ = [
    "CrossAttentionProbeConfig",
    "EncoderMemoryCache",
    "run_cross_attention_probe",
]
