"""Stage 6.9 -- real HF full-model / local tiny-checkpoint masked CausalLM
skeleton integration.

Composes three earlier stages into a *whole-model* masked pipeline driven by
weights extracted from a HuggingFace-style LLaMA / Qwen2 ``...ForCausalLM``:

* Stage 6.6 (:mod:`pllo.hf_wrappers.llama_qwen_single_block`) -- per-layer
  weight extraction, RoPE/GQA-compatible masks, affine+mask folding, and the
  masked single-decoder-layer forward (bias-aware);
* Stage 6.7 (:mod:`pllo.ops.causal_lm_boundaries`) -- the trusted embedding
  input boundary, the vocab-logit mask, and the masked-logits output boundary;
* Stage 6.8 (:mod:`pllo.ops.masked_causal_lm_skeleton`) -- per-layer residual
  masks ``N_0..N_L`` with an honest, explicit handoff at each layer boundary.

The goal is NOT production generation. It validates that a HF CausalLM can be
*decomposed* into ``embedding -> decoder layers -> final norm -> LM head`` and
that the masked pipeline reproduces our extracted-weight plaintext reference
to machine precision, including a bounded greedy decode loop.

The reference is our extracted-weight plaintext forward (adjacent-pair RoPE,
Stage 6.4), **not** ``model.forward`` / ``model.generate`` -- HF attention,
RoPE, and cache conventions vary across versions, so the masked-vs-plain
invariant is what we check. CPU-only; no network download (local-files-only);
transformers is optional. No formal, cryptographic, or semantic security is
claimed; attention scores remain GPU-visible and vocab permutation+scaling is
weaker than dense vocab masking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from pllo.hf_wrappers.llama_qwen_single_block import (
    HFSingleBlockConfig,
    HFSingleBlockWeights,
    _apply_heads,
    _finite_score_err,
    _linear,
    _masked_attention,
    _masked_mlp,
    _mx,
    _sdpa,
    extract_hf_single_block_weights,
    fold_hf_single_block_weights,
    generate_hf_single_block_masks,
    has_transformers,
    hf_single_block_plain_prefill,
    infer_config_from_hf_layer,
    require_transformers_or_skip,
)
from pllo.ops.causal_lm_boundaries import (
    CausalLMBoundaryWeights,
    VocabLogitMask,
    embedding_boundary_forward,
    final_norm_lm_head_masked,
    final_norm_lm_head_plain,
    greedy_sample,
    make_vocab_logit_mask,
    recover_vocab_logits,
    trusted_embedding_lookup,
)
from pllo.ops.gqa_attention import merge_heads, repeat_kv, split_heads
from pllo.ops.llama_synthetic_block import rmsnorm_plain
from pllo.ops.nonlinear_islands import rmsnorm_core, silu_reference
from pllo.ops.rope import apply_rope, build_rope_cache

__all__ = [
    "HFCausalLMMaskBundle",
    "HFCausalLMSkeletonConfig",
    "HFCausalLMSkeletonWeights",
    "extract_hf_causal_lm_skeleton_weights",
    "generate_hf_causal_lm_masks",
    "has_transformers",
    "hf_causal_lm_masked_greedy_decode",
    "hf_causal_lm_masked_prefill",
    "hf_causal_lm_plain_decode",
    "hf_causal_lm_plain_prefill",
    "make_random_tiny_hf_causal_lm",
    "require_transformers_or_skip",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HFCausalLMSkeletonConfig:
    model_family: str = "llama"          # "llama" or "qwen2"
    local_model_path: str | None = None
    batch_size: int = 1
    prefill_seq_len: int = 4
    decode_steps: int = 2
    max_layers: int | None = 2
    max_vocab_size: int | None = 512
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 2033
    mask_family: str = "pairwise_complex_scaling"
    # Default False for real-HF fidelity: an input pad shifts the modelled
    # sequence through RMSNorm (Stage 6.8), so true fidelity needs T_in = 0.
    # use_input_pad=True is allowed only as a synthetic stress option.
    use_input_pad: bool = False


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass
class HFCausalLMSkeletonWeights:
    embed_tokens_weight: torch.Tensor          # [vocab, hidden]
    layer_weights: list[HFSingleBlockWeights]
    final_norm_weight: torch.Tensor            # [hidden]
    lm_head_weight: torch.Tensor               # [hidden, vocab]


def _boundary_weights(
    weights: HFCausalLMSkeletonWeights,
) -> CausalLMBoundaryWeights:
    """Adapt extracted full-model weights to the Stage 6.7 boundary weights
    (so the embedding / LM-head boundary helpers can be reused verbatim)."""
    return CausalLMBoundaryWeights(
        embed_tokens_weight=weights.embed_tokens_weight,
        final_norm_weight=weights.final_norm_weight,
        lm_head_weight=weights.lm_head_weight,
    )


# ---------------------------------------------------------------------------
# Random tiny HF model construction (no checkpoints, no downloads)
# ---------------------------------------------------------------------------


def make_random_tiny_hf_causal_lm(
    config: HFCausalLMSkeletonConfig,
) -> tuple[Any, Any]:
    """Instantiate a tiny, randomly-initialised LLaMA / Qwen2 CausalLM from
    config. No checkpoint, no download. Returns ``(model, model_config)``."""
    require_transformers_or_skip()
    torch.manual_seed(config.seed)
    vocab = min(config.max_vocab_size or 512, 512)
    n_layers = config.max_layers or 2
    common = dict(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    family = config.model_family.lower()
    if family == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        mc = LlamaConfig(**common)
        model = LlamaForCausalLM(mc)
    elif family in ("qwen2", "qwen"):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        mc = Qwen2Config(**common)
        model = Qwen2ForCausalLM(mc)
    else:
        raise ValueError(f"unknown model_family {config.model_family!r}")
    model.eval()
    model.to("cpu")
    return model, mc


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    val = getattr(obj, name, default)
    return default if val is None else val


def extract_hf_causal_lm_skeleton_weights(
    model: Any, model_config: Any, max_layers: int | None = None,
    dtype: torch.dtype = torch.float64, device: str = "cpu",
    mask_family: str = "pairwise_complex_scaling",
) -> tuple[HFCausalLMSkeletonWeights, list[HFSingleBlockConfig], dict[str, Any]]:
    """Decompose a HF CausalLM into embedding / decoder layers / final norm /
    LM head, extracting weights in our row-vector convention.

    Returns ``(weights, layer_configs, metadata)``. ``lm_head`` HF weight is
    ``[vocab, hidden]`` and is transposed to ``[hidden, vocab]``; if the head
    is tied/absent we fall back to ``embed_tokens.weight.T``.
    """
    dev = torch.device(device)
    base = model.model  # LlamaModel / Qwen2Model
    embed = base.embed_tokens.weight.detach().to(
        device=dev, dtype=dtype).clone()           # [vocab, hidden]
    hf_layers = base.layers
    n_total = len(hf_layers)
    n = n_total if max_layers is None else min(int(max_layers), n_total)

    layer_weights = [
        extract_hf_single_block_weights(hf_layers[i], dtype, device)
        for i in range(n)
    ]
    layer_configs = [
        infer_config_from_hf_layer(hf_layers[i], model_config, dtype, device,
                                   mask_family)
        for i in range(n)
    ]
    final_norm = base.norm.weight.detach().to(
        device=dev, dtype=dtype).clone()           # [hidden]

    lm_head_module = getattr(model, "lm_head", None)
    if lm_head_module is not None and getattr(
            lm_head_module, "weight", None) is not None:
        lm_head = lm_head_module.weight.detach().to(
            device=dev, dtype=dtype).t().contiguous()   # [hidden, vocab]
    else:
        lm_head = embed.t().contiguous()                # tied fallback

    weights = HFCausalLMSkeletonWeights(
        embed_tokens_weight=embed, layer_weights=layer_weights,
        final_norm_weight=final_norm, lm_head_weight=lm_head)

    metadata = {
        "num_layers_extracted": n,
        "num_layers_total": n_total,
        "vocab_size": int(embed.shape[0]),
        "hidden_size": int(embed.shape[1]),
        "model_type": str(_get(model_config, "model_type", "unknown")),
        "tie_word_embeddings": bool(
            _get(model_config, "tie_word_embeddings", False)),
    }
    return weights, layer_configs, metadata


# ---------------------------------------------------------------------------
# Masks (residual N_0..N_L + per-layer block masks + vocab mask + pad)
# ---------------------------------------------------------------------------


@dataclass
class HFCausalLMMaskBundle:
    residual_masks: list[torch.Tensor]
    residual_mask_inverses: list[torch.Tensor]
    layer_block_masks: list[dict[str, Any]]
    vocab_mask: VocabLogitMask
    input_pad: torch.Tensor | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def handoff(self, ell: int) -> torch.Tensor:
        """``T_ell = N_ell^{-1} @ N_{ell+1}`` (orthogonal change-of-basis)."""
        return self.residual_mask_inverses[ell] @ self.residual_masks[ell + 1]


def _orthogonal(dim: int, dtype: torch.dtype, device: torch.device,
                g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype,
                                       device=device))
    return q


def generate_hf_causal_lm_masks(
    weights: HFCausalLMSkeletonWeights,
    layer_configs: list[HFSingleBlockConfig],
    config: HFCausalLMSkeletonConfig,
    seed: int | None = None, mask_family: str | None = None,
) -> HFCausalLMMaskBundle:
    """Per-layer residual masks ``N_0..N_L``, per-layer RoPE/GQA-compatible
    attention masks + SwiGLU permutation, a vocab-logit mask, and an optional
    input pad. For real fidelity (default) ``input_pad`` is ``None``."""
    seed = config.seed if seed is None else seed
    mask_family = config.mask_family if mask_family is None else mask_family
    cfg0 = layer_configs[0]
    dtype = cfg0.dtype
    device = torch.device(cfg0.device)
    hidden = cfg0.hidden_size
    n_layers = len(layer_configs)

    g = torch.Generator(device=device).manual_seed(seed)
    residual_masks = [_orthogonal(hidden, dtype, device, g)
                      for _ in range(n_layers + 1)]
    residual_mask_inverses = [m.transpose(-2, -1).contiguous()
                              for m in residual_masks]

    layer_block_masks: list[dict[str, Any]] = []
    for ell in range(n_layers):
        # Per-layer attention masks + SwiGLU permutation; its internal n_res
        # is discarded and replaced by this layer's residual mask N_ell.
        bm = generate_hf_single_block_masks(
            layer_configs[ell], seed=seed + 101 * (ell + 1))
        bm["n_res"] = residual_masks[ell]
        bm["n_res_inv"] = residual_mask_inverses[ell]
        layer_block_masks.append(bm)

    vocab_size = int(weights.lm_head_weight.shape[1])
    vocab_mask = make_vocab_logit_mask(vocab_size, dtype, device, g)

    input_pad = None
    if config.use_input_pad:
        input_pad = torch.randn(hidden, generator=g, dtype=dtype,
                                device=device)

    return HFCausalLMMaskBundle(
        residual_masks=residual_masks,
        residual_mask_inverses=residual_mask_inverses,
        layer_block_masks=layer_block_masks, vocab_mask=vocab_mask,
        input_pad=input_pad,
        metadata={
            "residual_mask_family": "orthogonal_per_layer",
            "mask_family": mask_family,
            "handoff": "N_ell_to_N_ell_plus_1",
            "handoff_transform": "orthogonal_change_of_basis_per_boundary",
            "handoff_skip_term_needs_gemm": True,
            "handoff_offline_fusable_except_skip": True,
            "used_input_pad": input_pad is not None,
        },
    )


# ---------------------------------------------------------------------------
# Per-layer block forwards (bias-aware, masked input -> masked output, N_ell)
# ---------------------------------------------------------------------------


def _hf_masked_block_prefill(
    x_tilde: torch.Tensor, folded: dict[str, Any],
    config: HFSingleBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
) -> dict[str, Any]:
    eps = config.rms_norm_eps
    r1_core_tilde = rmsnorm_core(x_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, config, cos, sin,
                          causal_offset=0)
    x1_tilde = x_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    mlp = _masked_mlp(r2_core_tilde, folded)
    y_tilde = x1_tilde + mlp["out"]
    return {
        "y_tilde": y_tilde, "attn_out": a["out"], "scores": a["scores"],
        "mlp_out": mlp["out"],
        "cache": {"key_rope_tilde": a["key_rope_full"],
                  "value_tilde": a["value_full"]},
    }


def _hf_masked_block_decode(
    x_next_tilde: torch.Tensor, cache: dict[str, Any], folded: dict[str, Any],
    config: HFSingleBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
    position: int,
) -> dict[str, Any]:
    eps = config.rms_norm_eps
    pid = torch.tensor([position], device=x_next_tilde.device)
    r1_core_tilde = rmsnorm_core(x_next_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, config, cos, sin,
                          causal_offset=None, position_ids=pid,
                          past_key_rope=cache["key_rope_tilde"],
                          past_value=cache["value_tilde"])
    x1_tilde = x_next_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    mlp = _masked_mlp(r2_core_tilde, folded)
    y_tilde = x1_tilde + mlp["out"]
    return {
        "y_tilde": y_tilde,
        "appended_key_tilde": a["k_rope_new"], "appended_value_tilde": a["v"],
        "cache": {"key_rope_tilde": a["key_rope_full"],
                  "value_tilde": a["value_full"]},
    }


def _hf_plain_block_decode(
    x_new: torch.Tensor, cache: dict[str, Any], weights: HFSingleBlockWeights,
    config: HFSingleBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
    position: int,
) -> dict[str, Any]:
    """Plain (bias-aware) one-token decode appending to a plain KV cache."""
    import math
    eps = config.rms_norm_eps
    nh, nkv, hd = config.num_heads, config.num_key_value_heads, config.head_dim
    scale = 1.0 / math.sqrt(hd)
    pid = torch.tensor([position], device=x_new.device)
    r1 = rmsnorm_plain(x_new, weights.input_layernorm_weight, eps)
    q = split_heads(_linear(r1, weights.q_proj_weight, weights.q_proj_bias), nh)
    k = split_heads(_linear(r1, weights.k_proj_weight, weights.k_proj_bias), nkv)
    v = split_heads(_linear(r1, weights.v_proj_weight, weights.v_proj_bias), nkv)
    qr = apply_rope(q, cos, sin, position_ids=pid)
    kr = apply_rope(k, cos, sin, position_ids=pid)
    kr_full = torch.cat([cache["key_rope"], kr], dim=2)
    v_full = torch.cat([cache["value"], v], dim=2)
    _, _, av = _sdpa(qr, repeat_kv(kr_full, nh, nkv),
                     repeat_kv(v_full, nh, nkv), scale, causal_offset=None)
    attn_out = _linear(merge_heads(av), weights.o_proj_weight,
                       weights.o_proj_bias)
    x1 = x_new + attn_out
    r2 = rmsnorm_plain(x1, weights.post_attention_layernorm_weight, eps)
    gate = _linear(r2, weights.gate_proj_weight, weights.gate_proj_bias)
    up = _linear(r2, weights.up_proj_weight, weights.up_proj_bias)
    hidden = silu_reference(gate) * up
    mlp_out = _linear(hidden, weights.down_proj_weight, weights.down_proj_bias)
    y = x1 + mlp_out
    return {
        "y": y, "appended_key": kr, "appended_value": v,
        "cache": {"key_rope": kr_full, "value": v_full},
    }


# ---------------------------------------------------------------------------
# Plain full-model reference (from extracted weights)
# ---------------------------------------------------------------------------


def _rope_cache(config: HFCausalLMSkeletonConfig,
                cfg0: HFSingleBlockConfig) -> tuple[torch.Tensor, torch.Tensor]:
    max_pos = config.prefill_seq_len + config.decode_steps + 1
    return build_rope_cache(max_pos, cfg0.head_dim, cfg0.rope_theta,
                            cfg0.dtype, torch.device(cfg0.device))


def hf_causal_lm_plain_prefill(
    input_ids: torch.Tensor, weights: HFCausalLMSkeletonWeights,
    layer_configs: list[HFSingleBlockConfig], masks: HFCausalLMMaskBundle,
    config: HFCausalLMSkeletonConfig, *,
    cos: torch.Tensor | None = None, sin: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Plain reference: embedding -> extracted decoder layers -> final norm ->
    LM head -> greedy next token. Uses the de-masked input ``X - T_in``."""
    cfg0 = layer_configs[0]
    eps = cfg0.rms_norm_eps
    if cos is None or sin is None:
        cos, sin = _rope_cache(config, cfg0)

    x_plain = trusted_embedding_lookup(input_ids, weights.embed_tokens_weight)
    x0 = x_plain if masks.input_pad is None else x_plain - masks.input_pad

    h = x0
    hidden_by_layer = [h]
    caches: list[dict[str, Any]] = []
    layer_refs: list[dict[str, Any]] = []
    for ell, cfg in enumerate(layer_configs):
        res = hf_single_block_plain_prefill(
            h, weights.layer_weights[ell], cfg, cos, sin)
        layer_refs.append(res)
        caches.append(res["cache_plain"])
        h = res["y"]
        hidden_by_layer.append(h)

    fn = final_norm_lm_head_plain(h, weights.final_norm_weight,
                                  weights.lm_head_weight, eps)
    next_token = greedy_sample(fn["logits"][:, -1, :])
    return {
        "input_ids": input_ids,
        "embeddings_plain": x_plain,
        "x0_plain": x0,
        "hidden_by_layer_plain": hidden_by_layer,
        "caches_plain": caches,
        "layer_refs": layer_refs,
        "logits_plain": fn["logits"],
        "next_token_plain": next_token,
        "cos": cos, "sin": sin,
    }


def hf_causal_lm_plain_decode(
    next_token: torch.Tensor, caches_plain: list[dict[str, Any]],
    weights: HFCausalLMSkeletonWeights,
    layer_configs: list[HFSingleBlockConfig], masks: HFCausalLMMaskBundle,
    config: HFCausalLMSkeletonConfig, position: int,
    cos: torch.Tensor, sin: torch.Tensor,
) -> dict[str, Any]:
    """One plain decode step (single token). No HF ``generate``."""
    cfg0 = layer_configs[0]
    eps = cfg0.rms_norm_eps
    x_plain = trusted_embedding_lookup(
        next_token, weights.embed_tokens_weight).unsqueeze(1)   # [B,1,H]
    h = x_plain if masks.input_pad is None else x_plain - masks.input_pad
    new_caches: list[dict[str, Any]] = []
    for ell, cfg in enumerate(layer_configs):
        dec = _hf_plain_block_decode(h, caches_plain[ell],
                                     weights.layer_weights[ell], cfg, cos, sin,
                                     position)
        new_caches.append(dec["cache"])
        h = dec["y"]
    fn = final_norm_lm_head_plain(h, weights.final_norm_weight,
                                  weights.lm_head_weight, eps)
    tok = greedy_sample(fn["logits"][:, -1, :])
    return {"hidden_plain": h, "caches_plain": new_caches,
            "logits_plain": fn["logits"], "next_token_plain": tok}


# ---------------------------------------------------------------------------
# Masked full-model prefill
# ---------------------------------------------------------------------------


def hf_causal_lm_masked_prefill(
    input_ids: torch.Tensor, weights: HFCausalLMSkeletonWeights,
    layer_configs: list[HFSingleBlockConfig], masks: HFCausalLMMaskBundle,
    config: HFCausalLMSkeletonConfig,
) -> dict[str, Any]:
    """Trusted embedding boundary -> masked extracted layers (per-layer mask
    handoff) -> masked-logits output boundary -> TEE recovery + greedy."""
    cfg0 = layer_configs[0]
    eps = cfg0.rms_norm_eps
    dtype = cfg0.dtype
    n_layers = len(layer_configs)
    cos, sin = _rope_cache(config, cfg0)
    bw = _boundary_weights(weights)

    plain = hf_causal_lm_plain_prefill(input_ids, weights, layer_configs, masks,
                                       config, cos=cos, sin=sin)

    # Input boundary: release only masked embeddings.
    emb = embedding_boundary_forward(input_ids, bw, masks.residual_masks[0],
                                     masks.input_pad)
    embedding_mask_err = _mx(emb["x_tilde"], emb["expected_x_tilde"])

    foldeds = [
        fold_hf_single_block_weights(weights.layer_weights[ell],
                                     layer_configs[ell],
                                     masks.layer_block_masks[ell])
        for ell in range(n_layers)
    ]

    h_tilde = emb["x_tilde"]                        # H0_tilde in N_0
    hidden_tilde_by_layer = [h_tilde]
    caches_tilde: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    for ell in range(n_layers):
        bm = masks.layer_block_masks[ell]
        attn_masks = bm["attn"]
        n_ell = masks.residual_masks[ell]
        blk = _hf_masked_block_prefill(h_tilde, foldeds[ell],
                                       layer_configs[ell], cos, sin)
        y_plain_ell = plain["hidden_by_layer_plain"][ell + 1]
        ref = plain["layer_refs"][ell]
        cache_plain = plain["caches_plain"][ell]
        per_layer.append({
            "layer": ell,
            "final_output_max_abs_error": _mx(blk["y_tilde"],
                                              y_plain_ell @ n_ell),
            "attention_score_max_abs_error": _finite_score_err(
                blk["scores"], ref["scores"]),
            "mlp_output_max_abs_error": _mx(blk["mlp_out"],
                                            ref["mlp_out"] @ n_ell),
            "cache_key_max_abs_error": _mx(
                blk["cache"]["key_rope_tilde"],
                _apply_heads(cache_plain["key_rope"], attn_masks["key_masks"])),
            "cache_value_max_abs_error": _mx(
                blk["cache"]["value_tilde"],
                _apply_heads(cache_plain["value"], attn_masks["value_masks"])),
        })
        caches_tilde.append(blk["cache"])
        # Handoff N_ell -> N_{ell+1} (one [H,H] GEMM on the masked state).
        h_tilde = blk["y_tilde"] @ masks.handoff(ell)
        hidden_tilde_by_layer.append(h_tilde)

    handoff_errs = [
        _mx(hidden_tilde_by_layer[ell],
            plain["hidden_by_layer_plain"][ell] @ masks.residual_masks[ell])
        for ell in range(n_layers + 1)
    ]

    h_L_plain = plain["hidden_by_layer_plain"][-1]
    h_L_tilde = hidden_tilde_by_layer[-1]
    final_hidden_err = _mx(h_L_tilde, h_L_plain @ masks.residual_masks[-1])

    out = final_norm_lm_head_masked(
        h_L_tilde, h_L_plain, bw, masks.residual_mask_inverses[-1],
        masks.vocab_mask, eps)
    next_token_from_masked = greedy_sample(
        recover_vocab_logits(out["logits_tilde"], masks.vocab_mask)[:, -1, :])
    greedy_match = float(
        (plain["next_token_plain"] == next_token_from_masked)
        .to(dtype).mean().item())

    metrics = {
        "embedding_mask_max_abs_error": embedding_mask_err,
        "per_layer": per_layer,
        "per_layer_handoff_max_abs_error": handoff_errs,
        "final_hidden_max_abs_error": final_hidden_err,
        "masked_logits_max_abs_error":
            out["metrics"]["masked_logits_max_abs_error"],
        "recovered_logits_max_abs_error":
            out["metrics"]["recovered_logits_max_abs_error"],
        "greedy_token_match_rate": greedy_match,
    }
    metrics["allclose"] = bool(
        embedding_mask_err <= 1e-8
        and all(e <= 1e-8 for e in handoff_errs)
        and final_hidden_err <= 1e-8
        and out["metrics"]["masked_logits_max_abs_error"] <= 1e-8
        and out["metrics"]["recovered_logits_max_abs_error"] <= 1e-8
        and all(v <= 1e-8 for layer in per_layer for k, v in layer.items()
                if k != "layer")
        and greedy_match == 1.0)

    return {
        "metrics": metrics,
        "caches_plain": plain["caches_plain"],
        "caches_tilde": caches_tilde,
        "next_token_plain": plain["next_token_plain"],
        "next_token_from_masked": next_token_from_masked,
        "cos": cos, "sin": sin, "foldeds": foldeds,
        "logits_plain": plain["logits_plain"],
        "metadata": {
            "gpu_visible_tensors": ["masked_embeddings", "masked_hidden_states",
                                    "masked_kv_caches", "masked_logits"],
            "trusted_only_tensors": ["input_ids", "plaintext_embeddings",
                                     "plaintext_logits", "sampled_token_ids"],
        },
    }


# ---------------------------------------------------------------------------
# Masked greedy decode loop
# ---------------------------------------------------------------------------


def hf_causal_lm_masked_greedy_decode(
    input_ids: torch.Tensor, weights: HFCausalLMSkeletonWeights,
    layer_configs: list[HFSingleBlockConfig], masks: HFCausalLMMaskBundle,
    config: HFCausalLMSkeletonConfig, decode_steps: int | None = None,
) -> dict[str, Any]:
    """Prefill + bounded greedy decode, masked vs extracted-weight plain."""
    decode_steps = config.decode_steps if decode_steps is None else decode_steps
    cfg0 = layer_configs[0]
    eps = cfg0.rms_norm_eps
    dtype = cfg0.dtype
    n_layers = len(layer_configs)
    bw = _boundary_weights(weights)

    pre = hf_causal_lm_masked_prefill(input_ids, weights, layer_configs, masks,
                                      config)
    cos, sin, foldeds = pre["cos"], pre["sin"], pre["foldeds"]
    caches_plain = [dict(c) for c in pre["caches_plain"]]
    caches_tilde = [dict(c) for c in pre["caches_tilde"]]

    next_plain = pre["next_token_plain"]            # [B]
    next_masked = pre["next_token_from_masked"]     # [B]
    gen_plain = [next_plain]
    gen_masked = [next_masked]
    step_metrics: list[dict[str, Any]] = []

    n0 = masks.residual_masks[0]
    pad = masks.input_pad
    for step in range(decode_steps):
        position = config.prefill_seq_len + step
        tok = next_masked  # drive the masked path with its own trusted token
        x_next_plain = trusted_embedding_lookup(
            tok, weights.embed_tokens_weight).unsqueeze(1)   # [B,1,H]
        x0 = x_next_plain if pad is None else x_next_plain - pad
        h_plain = x0
        h_tilde = x0 @ n0

        layer_out_errs: list[float] = []
        layer_key_errs: list[float] = []
        layer_value_errs: list[float] = []
        for ell in range(n_layers):
            bm = masks.layer_block_masks[ell]
            attn_masks = bm["attn"]
            n_ell = masks.residual_masks[ell]
            dec_p = _hf_plain_block_decode(h_plain, caches_plain[ell],
                                           weights.layer_weights[ell],
                                           layer_configs[ell], cos, sin,
                                           position)
            dec_m = _hf_masked_block_decode(h_tilde, caches_tilde[ell],
                                            foldeds[ell], layer_configs[ell],
                                            cos, sin, position)
            caches_plain[ell] = dec_p["cache"]
            caches_tilde[ell] = dec_m["cache"]
            layer_out_errs.append(_mx(dec_m["y_tilde"], dec_p["y"] @ n_ell))
            layer_key_errs.append(_mx(
                dec_m["appended_key_tilde"],
                _apply_heads(dec_p["appended_key"], attn_masks["key_masks"])))
            layer_value_errs.append(_mx(
                dec_m["appended_value_tilde"],
                _apply_heads(dec_p["appended_value"],
                             attn_masks["value_masks"])))
            h_plain = dec_p["y"]
            h_tilde = dec_m["y_tilde"] @ masks.handoff(ell)

        final_hidden_err = _mx(h_tilde, h_plain @ masks.residual_masks[-1])
        fn_plain = final_norm_lm_head_plain(h_plain, bw.final_norm_weight,
                                            bw.lm_head_weight, eps)
        out = final_norm_lm_head_masked(
            h_tilde, h_plain, bw, masks.residual_mask_inverses[-1],
            masks.vocab_mask, eps)
        masked_logits_err = out["metrics"]["masked_logits_max_abs_error"]
        recovered_logits_err = out["metrics"]["recovered_logits_max_abs_error"]
        tok_plain = greedy_sample(fn_plain["logits"][:, -1, :])
        tok_masked = greedy_sample(
            recover_vocab_logits(out["logits_tilde"],
                                 masks.vocab_mask)[:, -1, :])
        step_metrics.append({
            "step": step, "position": position,
            "final_hidden_error": final_hidden_err,
            "masked_logits_error": masked_logits_err,
            "recovered_logits_error": recovered_logits_err,
            "sampled_token_match": float(
                (tok_plain == tok_masked).to(dtype).mean().item()),
            "per_layer_output_error": max(layer_out_errs),
            "per_layer_cache_append_key_error": max(layer_key_errs),
            "per_layer_cache_append_value_error": max(layer_value_errs),
        })
        next_plain = tok_plain
        next_masked = tok_masked
        gen_plain.append(tok_plain)
        gen_masked.append(tok_masked)

    gen_plain_t = torch.stack(gen_plain, dim=1)     # [B, 1+decode_steps]
    gen_masked_t = torch.stack(gen_masked, dim=1)
    token_match_rate = float(
        (gen_plain_t == gen_masked_t).to(dtype).mean().item())

    return {
        "generated_plain_tokens": gen_plain_t,
        "generated_from_masked_tokens": gen_masked_t,
        "token_match_rate": token_match_rate,
        "prefill_metrics": pre["metrics"],
        "decode_step_metrics": step_metrics,
        "metadata": pre["metadata"],
    }
