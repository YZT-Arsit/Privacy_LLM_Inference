"""Stage 6.8 -- multi-layer mask handoff + full masked CausalLM skeleton.

Composes the Stage 6.7 trusted input boundary, ``num_layers`` Stage 6.5
synthetic LLaMA/Qwen-like decoder blocks, the Stage 6.7 masked-logits output
boundary, and a bounded greedy decode loop. CPU-only, correctness-first.

Per-layer residual masks
------------------------
Each layer ``ell`` carries a residual mask ``N_ell``; the stack uses
``N_0 .. N_L`` (``L = num_layers``). The verified invariant is, at every
layer input::

    H_ell_tilde == H_ell_plain @ N_ell

Mask handoff (honest note)
--------------------------
A pre-norm residual block's skip connection carries the input mask straight
to the output, so a residual block's output mask necessarily equals its
input mask: a single block CANNOT change the residual mask for free. We
therefore run each layer ``ell`` as a single-mask (``N_ell``) Stage-6.5
block and realise the handoff ``N_ell -> N_{ell+1}`` as ONE orthogonal
change-of-basis ``T_ell = N_ell^{-1} @ N_{ell+1}`` applied to the masked
hidden state at the layer boundary. ``T_ell`` is orthogonal (a product of
orthogonals) and is offline-fusable into the preceding block's projection
GEMMs for the non-skip terms; the skip term genuinely needs the transform,
so it is one ``[H,H]`` GEMM per boundary here (not zero). Sharing a single
residual mask across all layers removes the transition entirely; per-layer
masks are used here to exercise the handoff. No claim of zero handoff cost.

No HF/ModelScope checkpoints, no GPU, no transformers, no GPT-2 wrappers.
No formal, cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from pllo.ops.causal_lm_boundaries import (
    CausalLMBoundaryConfig,
    CausalLMBoundaryWeights,
    VocabLogitMask,
    embedding_boundary_forward,
    final_norm_lm_head_masked,
    final_norm_lm_head_plain,
    greedy_sample,
    init_causal_lm_boundary_weights,
    make_vocab_logit_mask,
    recover_vocab_logits,
    trusted_embedding_lookup,
)
from pllo.ops.gqa_attention import (
    merge_heads,
    repeat_kv,
    split_heads,
)
from pllo.ops.llama_synthetic_block import (
    SyntheticLlamaBlockConfig,
    SyntheticLlamaBlockWeights,
    _apply,
    _masked_attention,
    _sdpa,
    fold_block_weights,
    generate_block_masks,
    init_synthetic_llama_block_weights,
    llama_block_plain_prefill,
    rmsnorm_plain,
    swiglu_plain,
)
from pllo.ops.nonlinear_islands import rmsnorm_core, silu_reference
from pllo.ops.rope import apply_rope, build_rope_cache

__all__ = [
    "LayerMaskBundle",
    "MaskedCausalLMSkeletonConfig",
    "MaskedCausalLMSkeletonWeights",
    "SkeletonMaskBundle",
    "causal_lm_masked_greedy_decode",
    "causal_lm_masked_prefill",
    "causal_lm_plain_prefill",
    "generate_skeleton_masks",
    "init_masked_causal_lm_skeleton_weights",
]


# ---------------------------------------------------------------------------
# Config + weights
# ---------------------------------------------------------------------------


@dataclass
class MaskedCausalLMSkeletonConfig:
    batch_size: int = 2
    prefill_seq_len: int = 5
    decode_steps: int = 3
    vocab_size: int = 128
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 3
    num_heads: int = 4
    num_key_value_heads: int = 2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_input_pad: bool = True
    mask_family: str = "pairwise_complex_scaling"
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 2031

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (RoPE adjacent pairs)")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_key_value_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.decode_steps < 1:
            raise ValueError("decode_steps must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def block_config(self) -> SyntheticLlamaBlockConfig:
        return SyntheticLlamaBlockConfig(
            batch_size=self.batch_size, seq_len=self.prefill_seq_len,
            decode_steps=self.decode_steps, hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size, num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_base=self.rope_base, rms_norm_eps=self.rms_norm_eps,
            mask_family=self.mask_family, dtype=self.dtype, device=self.device,
            seed=self.seed,
        )

    def boundary_config(self) -> CausalLMBoundaryConfig:
        return CausalLMBoundaryConfig(
            batch_size=self.batch_size, seq_len=self.prefill_seq_len,
            vocab_size=self.vocab_size, hidden_size=self.hidden_size,
            tie_word_embeddings=self.tie_word_embeddings,
            use_input_pad=self.use_input_pad, rms_norm_eps=self.rms_norm_eps,
            dtype=self.dtype, device=self.device, seed=self.seed,
        )


@dataclass
class MaskedCausalLMSkeletonWeights:
    boundary_weights: CausalLMBoundaryWeights
    layer_weights: list[SyntheticLlamaBlockWeights]


def init_masked_causal_lm_skeleton_weights(
    config: MaskedCausalLMSkeletonConfig, generator: torch.Generator,
) -> MaskedCausalLMSkeletonWeights:
    """Boundary weights (Stage 6.7) + per-layer block weights (Stage 6.5)."""
    config.validate()
    boundary = init_causal_lm_boundary_weights(config.boundary_config(),
                                               generator)
    block_cfg = config.block_config()
    layers = [init_synthetic_llama_block_weights(block_cfg, generator)
              for _ in range(config.num_layers)]
    return MaskedCausalLMSkeletonWeights(boundary_weights=boundary,
                                         layer_weights=layers)


# ---------------------------------------------------------------------------
# Multi-layer masks
# ---------------------------------------------------------------------------


@dataclass
class LayerMaskBundle:
    n_in: torch.Tensor
    n_out: torch.Tensor
    n_in_inv: torch.Tensor
    n_out_inv: torch.Tensor
    block_masks: dict[str, Any]
    layer_index: int

    @property
    def handoff(self) -> torch.Tensor:
        """``T_ell = N_ell^{-1} @ N_{ell+1}`` (orthogonal change-of-basis)."""
        return self.n_in_inv @ self.n_out


@dataclass
class SkeletonMaskBundle:
    residual_masks: list[torch.Tensor]
    residual_mask_inverses: list[torch.Tensor]
    layer_masks: list[LayerMaskBundle]
    vocab_mask: VocabLogitMask
    input_pad: torch.Tensor | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _orthogonal(dim: int, dtype: torch.dtype, device: torch.device,
                g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype,
                                       device=device))
    return q


def generate_skeleton_masks(
    config: MaskedCausalLMSkeletonConfig, generator: torch.Generator,
) -> SkeletonMaskBundle:
    """``N_0..N_L`` orthogonal residual masks, per-layer attention/MLP masks,
    a vocab-logit mask, and an optional input pad."""
    config.validate()
    dtype = config.dtype
    device = torch.device(config.device)
    block_cfg = config.block_config()
    n_layers = config.num_layers

    residual_masks = [_orthogonal(config.hidden_size, dtype, device, generator)
                      for _ in range(n_layers + 1)]
    residual_mask_inverses = [m.transpose(-2, -1).contiguous()
                              for m in residual_masks]

    layer_masks: list[LayerMaskBundle] = []
    for ell in range(n_layers):
        # Per-layer attention masks + SwiGLU permutation (its internal n_res
        # is discarded; we substitute this layer's residual mask N_ell).
        bm = generate_block_masks(block_cfg, generator)
        bm["n_res"] = residual_masks[ell]
        bm["n_res_inv"] = residual_mask_inverses[ell]
        layer_masks.append(LayerMaskBundle(
            n_in=residual_masks[ell], n_out=residual_masks[ell + 1],
            n_in_inv=residual_mask_inverses[ell],
            n_out_inv=residual_mask_inverses[ell + 1],
            block_masks=bm, layer_index=ell,
        ))

    vocab_mask = make_vocab_logit_mask(config.vocab_size, dtype, device,
                                       generator)
    input_pad = None
    if config.use_input_pad:
        input_pad = torch.randn(config.hidden_size, generator=generator,
                                dtype=dtype, device=device)

    return SkeletonMaskBundle(
        residual_masks=residual_masks,
        residual_mask_inverses=residual_mask_inverses,
        layer_masks=layer_masks, vocab_mask=vocab_mask, input_pad=input_pad,
        metadata={
            "residual_mask_family": "orthogonal_per_layer",
            "handoff": "N_ell_to_N_ell_plus_1",
            "handoff_transform": "orthogonal_change_of_basis_per_boundary",
            "handoff_skip_term_needs_gemm": True,
            "handoff_offline_fusable_except_skip": True,
        },
    )


# ---------------------------------------------------------------------------
# Local single-mask block forward (masked input in -> masked output, N_ell)
# ---------------------------------------------------------------------------


def _masked_block_prefill(
    x_tilde: torch.Tensor, folded: dict[str, Any],
    block_cfg: SyntheticLlamaBlockConfig, cos: torch.Tensor,
    sin: torch.Tensor,
) -> dict[str, Any]:
    """Masked Stage-6.5 block on an already-masked input (single mask)."""
    eps = block_cfg.rms_norm_eps
    r1_core_tilde = rmsnorm_core(x_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, block_cfg, cos, sin,
                          causal_offset=0)
    x1_tilde = x_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    gate = r2_core_tilde @ folded["wgate_tilde"]
    up = r2_core_tilde @ folded["wup_tilde"]
    hidden = silu_reference(gate) * up
    mlp_out = hidden @ folded["wdown_tilde"]
    y_tilde = x1_tilde + mlp_out
    return {
        "y_tilde": y_tilde, "attn_out": a["out"], "scores": a["scores"],
        "mlp_out": mlp_out,
        "cache": {"key_rope_tilde": a["key_rope_full"],
                  "value_tilde": a["value_full"]},
    }


def _masked_block_decode(
    x_next_tilde: torch.Tensor, cache: dict[str, Any],
    folded: dict[str, Any], block_cfg: SyntheticLlamaBlockConfig,
    cos: torch.Tensor, sin: torch.Tensor, position: int,
) -> dict[str, Any]:
    eps = block_cfg.rms_norm_eps
    pid = torch.tensor([position], device=x_next_tilde.device)
    r1_core_tilde = rmsnorm_core(x_next_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, block_cfg, cos, sin,
                          causal_offset=None, position_ids=pid,
                          past_key_rope=cache["key_rope_tilde"],
                          past_value=cache["value_tilde"])
    x1_tilde = x_next_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    gate = r2_core_tilde @ folded["wgate_tilde"]
    up = r2_core_tilde @ folded["wup_tilde"]
    hidden = silu_reference(gate) * up
    mlp_out = hidden @ folded["wdown_tilde"]
    y_tilde = x1_tilde + mlp_out
    return {
        "y_tilde": y_tilde,
        "appended_key_tilde": a["k_rope_new"], "appended_value_tilde": a["v"],
        "cache": {"key_rope_tilde": a["key_rope_full"],
                  "value_tilde": a["value_full"]},
    }


def _plain_block_decode(
    x_new: torch.Tensor, cache: dict[str, Any],
    weights: SyntheticLlamaBlockWeights, block_cfg: SyntheticLlamaBlockConfig,
    cos: torch.Tensor, sin: torch.Tensor, position: int,
) -> dict[str, Any]:
    """Plain one-token decode reference appending to a plain KV cache."""
    eps = block_cfg.rms_norm_eps
    nh, nkv, hd = (block_cfg.num_heads, block_cfg.num_key_value_heads,
                   block_cfg.head_dim)
    scale = 1.0 / math.sqrt(hd)
    pid = torch.tensor([position], device=x_new.device)
    r1 = rmsnorm_plain(x_new, weights.rms1_weight, eps)
    q = split_heads(r1 @ weights.wq, nh)
    k = split_heads(r1 @ weights.wk, nkv)
    v = split_heads(r1 @ weights.wv, nkv)
    qr = apply_rope(q, cos, sin, position_ids=pid)
    kr = apply_rope(k, cos, sin, position_ids=pid)
    kr_full = torch.cat([cache["key_rope"], kr], dim=2)
    v_full = torch.cat([cache["value"], v], dim=2)
    _, _, av = _sdpa(qr, repeat_kv(kr_full, nh, nkv),
                     repeat_kv(v_full, nh, nkv), scale, None)
    attn_out = merge_heads(av) @ weights.wo
    x1 = x_new + attn_out
    r2 = rmsnorm_plain(x1, weights.rms2_weight, eps)
    mlp = swiglu_plain(r2, weights.w_gate, weights.w_up, weights.w_down)
    y = x1 + mlp["out"]
    return {
        "y": y, "appended_key": kr, "appended_value": v,
        "cache": {"key_rope": kr_full, "value": v_full},
    }


def _mx(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _finite_err(a: torch.Tensor, b: torch.Tensor) -> float:
    finite = torch.isfinite(a)
    return float((a[finite] - b[finite]).abs().max().item())


# ---------------------------------------------------------------------------
# Plain multi-layer prefill (reference)
# ---------------------------------------------------------------------------


def causal_lm_plain_prefill(
    input_ids: torch.Tensor, weights: MaskedCausalLMSkeletonWeights,
    masks: SkeletonMaskBundle, config: MaskedCausalLMSkeletonConfig,
    cos: torch.Tensor, sin: torch.Tensor,
) -> dict[str, Any]:
    """Plain reference over the de-masked input ``X - T_in``."""
    eps = config.rms_norm_eps
    bw = weights.boundary_weights
    x_plain = trusted_embedding_lookup(input_ids, bw.embed_tokens_weight)
    x0 = x_plain if masks.input_pad is None else x_plain - masks.input_pad

    block_cfg = config.block_config()
    h = x0
    hidden_by_layer = [h]
    caches: list[dict[str, Any]] = []
    for ell in range(config.num_layers):
        res = llama_block_plain_prefill(h, weights.layer_weights[ell],
                                        block_cfg, cos, sin)
        caches.append(res["cache_plain"])
        h = res["y"]
        hidden_by_layer.append(h)

    fn = final_norm_lm_head_plain(h, bw.final_norm_weight, bw.lm_head_weight,
                                  eps)
    next_token = greedy_sample(fn["logits"][:, -1, :])
    return {
        "input_ids": input_ids,
        "embeddings_plain": x_plain,
        "x0_plain": x0,
        "hidden_by_layer_plain": hidden_by_layer,
        "caches_plain": caches,
        "logits_plain": fn["logits"],
        "next_token_plain": next_token,
    }


# ---------------------------------------------------------------------------
# Masked multi-layer prefill
# ---------------------------------------------------------------------------


def causal_lm_masked_prefill(
    input_ids: torch.Tensor, weights: MaskedCausalLMSkeletonWeights,
    masks: SkeletonMaskBundle, config: MaskedCausalLMSkeletonConfig,
) -> dict[str, Any]:
    """Full masked prefill: trusted input boundary -> masked layers (with
    per-layer mask handoff) -> masked-logits output boundary -> recovery."""
    config.validate()
    eps = config.rms_norm_eps
    dtype, device = config.dtype, torch.device(config.device)
    bw = weights.boundary_weights
    block_cfg = config.block_config()
    max_pos = config.prefill_seq_len + config.decode_steps + 1
    cos, sin = build_rope_cache(max_pos, config.head_dim, config.rope_base,
                                dtype, device)

    plain = causal_lm_plain_prefill(input_ids, weights, masks, config, cos, sin)

    # input boundary: release only masked embeddings
    emb = embedding_boundary_forward(input_ids, bw, masks.residual_masks[0],
                                     masks.input_pad)
    embedding_mask_err = _mx(emb["x_tilde"], emb["expected_x_tilde"])

    foldeds = [fold_block_weights(weights.layer_weights[ell],
                                  masks.layer_masks[ell].block_masks, block_cfg)
               for ell in range(config.num_layers)]

    # masked layer chain with handoff
    h_tilde = emb["x_tilde"]                       # H0_tilde in N_0
    hidden_tilde_by_layer = [h_tilde]
    caches_tilde: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    for ell in range(config.num_layers):
        lm = masks.layer_masks[ell]
        n_ell = masks.residual_masks[ell]
        attn_masks = lm.block_masks["attn"]
        blk = _masked_block_prefill(h_tilde, foldeds[ell], block_cfg, cos, sin)
        y_plain_ell = plain["hidden_by_layer_plain"][ell + 1]
        plain_layer_in = plain["hidden_by_layer_plain"][ell]
        # plain intermediates for this layer (recompute scores / mlp ref)
        ref = llama_block_plain_prefill(plain_layer_in,
                                        weights.layer_weights[ell], block_cfg,
                                        cos, sin)
        cache_plain = plain["caches_plain"][ell]
        v_masks_qhead = attn_masks["value_masks"].index_select(
            0, attn_masks["kv_index"])  # noqa: F841 - parity with 6.5
        per_layer.append({
            "layer": ell,
            "final_output_max_abs_error": _mx(blk["y_tilde"],
                                              y_plain_ell @ n_ell),
            "attention_score_max_abs_error": _finite_err(blk["scores"],
                                                         ref["attn"]["scores"]),
            "mlp_output_max_abs_error": _mx(
                blk["mlp_out"], ref["mlp"]["out"] @ n_ell),
            "cache_key_max_abs_error": _mx(
                blk["cache"]["key_rope_tilde"],
                _apply(cache_plain["key_rope"], attn_masks["key_masks"])),
            "cache_value_max_abs_error": _mx(
                blk["cache"]["value_tilde"],
                _apply(cache_plain["value"], attn_masks["value_masks"])),
        })
        caches_tilde.append(blk["cache"])
        # handoff N_ell -> N_{ell+1}
        h_tilde = blk["y_tilde"] @ lm.handoff
        hidden_tilde_by_layer.append(h_tilde)

    # per-layer input handoff invariant: H_ell_tilde == H_ell_plain @ N_ell
    handoff_errs = [
        _mx(hidden_tilde_by_layer[ell],
            plain["hidden_by_layer_plain"][ell] @ masks.residual_masks[ell])
        for ell in range(config.num_layers + 1)
    ]

    h_L_plain = plain["hidden_by_layer_plain"][-1]
    h_L_tilde = hidden_tilde_by_layer[-1]
    final_hidden_err = _mx(h_L_tilde, h_L_plain @ masks.residual_masks[-1])

    # output boundary
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
        "masked_logits_max_abs_error": out["metrics"]["masked_logits_max_abs_error"],
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
                                     "plaintext_logits"],
        },
    }


# ---------------------------------------------------------------------------
# Masked greedy decode loop
# ---------------------------------------------------------------------------


def causal_lm_masked_greedy_decode(
    input_ids: torch.Tensor, weights: MaskedCausalLMSkeletonWeights,
    masks: SkeletonMaskBundle, config: MaskedCausalLMSkeletonConfig,
) -> dict[str, Any]:
    """Prefill + bounded greedy decode loop, masked vs plain reference."""
    config.validate()
    eps = config.rms_norm_eps
    dtype, device = config.dtype, torch.device(config.device)
    bw = weights.boundary_weights
    block_cfg = config.block_config()

    pre = causal_lm_masked_prefill(input_ids, weights, masks, config)
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
    for step in range(config.decode_steps):
        position = config.prefill_seq_len + step
        # TEE: next token (trusted) -> embedding -> mask
        tok = next_masked  # drive the masked path with its own trusted token
        x_next_plain = trusted_embedding_lookup(tok, bw.embed_tokens_weight)
        x_next_plain = x_next_plain.unsqueeze(1)    # [B,1,H]
        x0 = x_next_plain if pad is None else x_next_plain - pad
        h_plain = x0
        h_tilde = x0 @ n0

        layer_out_errs: list[float] = []
        layer_key_errs: list[float] = []
        layer_value_errs: list[float] = []
        for ell in range(config.num_layers):
            lm = masks.layer_masks[ell]
            n_ell = masks.residual_masks[ell]
            attn_masks = lm.block_masks["attn"]
            dec_p = _plain_block_decode(h_plain, caches_plain[ell],
                                        weights.layer_weights[ell], block_cfg,
                                        cos, sin, position)
            dec_m = _masked_block_decode(h_tilde, caches_tilde[ell],
                                         foldeds[ell], block_cfg, cos, sin,
                                         position)
            caches_plain[ell] = dec_p["cache"]
            caches_tilde[ell] = dec_m["cache"]
            layer_out_errs.append(_mx(dec_m["y_tilde"], dec_p["y"] @ n_ell))
            layer_key_errs.append(_mx(
                dec_m["appended_key_tilde"],
                _apply(dec_p["appended_key"], attn_masks["key_masks"])))
            layer_value_errs.append(_mx(
                dec_m["appended_value_tilde"],
                _apply(dec_p["appended_value"], attn_masks["value_masks"])))
            h_plain = dec_p["y"]
            h_tilde = dec_m["y_tilde"] @ lm.handoff

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

    gen_plain_t = torch.stack(gen_plain, dim=1)      # [B, 1+decode_steps]
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
