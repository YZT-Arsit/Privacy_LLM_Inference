"""Stage 6.4c — Modern decoder-only model-level obfuscated wrapper.

Builds on Stage 6.4b's ``ObfuscatedModernDecoderBlockWrapper`` to provide
a full LLaMA / TinyLlama / Qwen-style decoder stack with:

* embedding lookup + N transformer blocks + final RMSNorm + LM head,
* KV-cache-aware ``prefill`` / ``decode_step`` with masked K / V cache,
* hand-written greedy generation that compares against a plain
  reference (no HF generate, no beam, no sampling),
* optional trace hook reusing the Stage 5.5 inventory.

Scope is intentionally bounded:

* No real-TEE measurement.
* No LoRA training path.
* No beam search / top-k / top-p.
* No batched variable-length prompts (batch_size 1 or fixed length).
* Default mode for the wider system remains ``nonlinear_mode='trusted'``
  and the default mitigation bundle remains ``'fresh_perm_only'``.

The model wrapper computes the obfuscated attention path inline rather
than delegating to the block wrapper for prefill / decode_step — this
lets it cache per-layer ``N_K`` / ``N_V`` mask material so the masked KV
cache append invariant holds across many decode steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from pllo.cache.modern_decoder_kv_cache import (
    ModernDecoderKVCache,
    ModernDecoderLayerKVCache,
    init_empty_modern_decoder_kv_cache,
)
from pllo.experiments.gqa_probe import repeat_kv
from pllo.experiments.rope_probe import _rotate_half
from pllo.hf_wrappers.modern_decoder_block_wrapper import (
    ModernDecoderBlockWeights,
    ObfuscatedModernDecoderBlockWrapper,
    _causal_mask,
    _merge_heads,
    _reshape_heads,
    _rmsnorm_with_gamma,
)
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    VALID_NONLINEAR_MODES,
    normalize_nonlinear_mode,
)
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.model_zoo.modern_decoder_spec import (
    ModernDecoderBlockSpec,
    extract_linear_row_weights,
    extract_rmsnorm_params,
    inspect_modern_decoder_block,
)
from pllo.ops.compatible_masks import (
    generate_orthogonal,
    generate_permutation,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
    describe_mitigation_bundle,
    normalize_mitigation_bundle,
)
from pllo.ops.nonlinear_islands import rmsnorm_core, run_swiglu_mlp_island


# ---------------------------------------------------------------------------
# Model-level weights container
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderModelWeights:
    """All weights for a full modern decoder model.

    All Linear weights are stored in **row-vector convention** (``Y = X @ W + b``)
    — i.e. ``[in_features, out_features]``. Embedding stays in HF
    convention (``[vocab_size, hidden_size]``).
    """

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_base: float
    vocab_size: int

    embed_tokens_weight: torch.Tensor             # [vocab, hidden]
    layers: list[ModernDecoderBlockWeights] = field(default_factory=list)
    final_norm_weight: torch.Tensor = None        # type: ignore[assignment]
    final_norm_eps: float = 1e-6
    lm_head_weight: torch.Tensor = None           # type: ignore[assignment]  # [hidden, vocab]
    lm_head_bias: torch.Tensor | None = None
    tied_embeddings: bool = False

    @classmethod
    def from_synthetic(
        cls,
        *,
        vocab_size: int = 32,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 8,
        num_layers: int = 2,
        rope_base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> "ModernDecoderModelWeights":
        g = torch.Generator(device="cpu").manual_seed(seed)
        device_obj = torch.device(device)

        def randn(*shape: int, scale: float = 0.1) -> torch.Tensor:
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
                dtype=dtype, device=device_obj
            )

        layers = []
        for i in range(num_layers):
            layers.append(
                ModernDecoderBlockWeights.from_synthetic(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    rope_base=rope_base,
                    seed=seed + 1000 * (i + 1),
                )
            )
        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_base=rope_base,
            vocab_size=vocab_size,
            embed_tokens_weight=randn(vocab_size, hidden_size, scale=0.05),
            layers=layers,
            final_norm_weight=(
                0.9 + (torch.rand(hidden_size, generator=g) * 0.2)
            ).to(dtype=dtype, device=device_obj),
            final_norm_eps=1e-6,
            lm_head_weight=randn(hidden_size, vocab_size, scale=0.05),
            lm_head_bias=None,
            tied_embeddings=False,
        )

    @classmethod
    def from_hf_model(
        cls,
        hf_model,
        *,
        spec: ModernDecoderBlockSpec,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        max_layers: int | None = None,
    ) -> "ModernDecoderModelWeights":
        """Extract weights from a HuggingFace LLaMA / Qwen-style model."""
        device_obj = torch.device(device)
        # Locate the inner module.
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            base = hf_model.model
        else:
            base = hf_model
        if not (hasattr(base, "embed_tokens") and hasattr(base, "layers")):
            raise ValueError(
                "expected hf_model to expose `.model.embed_tokens` and"
                " `.model.layers` (LLaMA / Qwen2 convention)"
            )
        embed = base.embed_tokens.weight.detach().clone().to(
            dtype=dtype, device=device_obj
        )
        all_layers = list(base.layers)
        if max_layers is not None:
            all_layers = all_layers[:max_layers]
        block_weights: list[ModernDecoderBlockWeights] = []
        for idx, blk in enumerate(all_layers):
            block_weights.append(
                ModernDecoderBlockWeights.from_hf_block(blk, spec)
            )

        # Final RMSNorm — LLaMA / Qwen2: `base.norm`
        norm_w, norm_eps = extract_rmsnorm_params(getattr(base, "norm"))
        # LM head — may be tied to embed_tokens.
        lm_head_mod = getattr(hf_model, "lm_head", None)
        if lm_head_mod is not None:
            lm_w, lm_b = extract_linear_row_weights(lm_head_mod)
            tied = bool(getattr(hf_model.config, "tie_word_embeddings", False))
        else:
            # Tied embeddings: lm_head shares embed_tokens.
            lm_w = embed.t().contiguous()
            lm_b = None
            tied = True
        return cls(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            rope_base=float(spec.rope_base or 10000.0),
            vocab_size=int(embed.shape[0]),
            embed_tokens_weight=embed,
            layers=block_weights,
            final_norm_weight=norm_w.to(dtype=dtype, device=device_obj),
            final_norm_eps=norm_eps,
            lm_head_weight=lm_w.to(dtype=dtype, device=device_obj),
            lm_head_bias=(
                lm_b.to(dtype=dtype, device=device_obj) if lm_b is not None else None
            ),
            tied_embeddings=tied,
        )


# ---------------------------------------------------------------------------
# RoPE with position offset (decode_step needs to start from past_seq_len)
# ---------------------------------------------------------------------------


def _rope_freqs_at(
    head_dim: int,
    positions: torch.Tensor,
    base: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``cos``/``sin`` shaped ``[len(positions), head_dim]`` at given absolute positions."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    pos = positions.to(dtype=torch.float64, device=device)
    theta = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = theta.cos().repeat(1, 2).to(dtype)
    sin = theta.sin().repeat(1, 2).to(dtype)
    return cos, sin


def _apply_rope_at(
    x: torch.Tensor, *, position_offset: int, base: float,
) -> torch.Tensor:
    """RoPE that respects an absolute position offset.

    ``x`` shape: ``[..., seq_len, head_dim]``. Position indices are
    ``[offset, offset+1, ..., offset+seq_len-1]``.
    """
    head_dim = x.shape[-1]
    seq_len = x.shape[-2]
    positions = torch.arange(
        position_offset, position_offset + seq_len, device=x.device,
    )
    cos, sin = _rope_freqs_at(head_dim, positions, base, x.dtype, x.device)
    extra = x.ndim - 2
    for _ in range(extra):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Plain reference (no HF call — uses extracted weights only)
# ---------------------------------------------------------------------------


def _plain_block_step(
    x: torch.Tensor,
    weights: ModernDecoderBlockWeights,
    *,
    cached_k: torch.Tensor | None = None,
    cached_v: torch.Tensor | None = None,
    position_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """One plain block forward with optional KV history.

    Returns ``{"y": [B, S_new, H], "k_new": [...], "v_new": [...],
    "q_rope": [...], "h_mid": [...], "mlp_out": [...]}``.
    """
    B, S_new, H = x.shape
    head_dim = weights.head_dim
    num_q = weights.num_attention_heads
    num_kv = weights.num_key_value_heads
    group = num_q // num_kv

    h1 = _rmsnorm_with_gamma(x, weights.input_norm_weight, weights.input_norm_eps)
    q = h1 @ weights.w_q
    if weights.b_q is not None:
        q = q + weights.b_q
    k = h1 @ weights.w_k
    if weights.b_k is not None:
        k = k + weights.b_k
    v = h1 @ weights.w_v
    if weights.b_v is not None:
        v = v + weights.b_v
    q = _reshape_heads(q, num_q, head_dim)
    k = _reshape_heads(k, num_kv, head_dim)
    v = _reshape_heads(v, num_kv, head_dim)
    q_rope = _apply_rope_at(q, position_offset=position_offset, base=weights.rope_base)
    k_rope = _apply_rope_at(k, position_offset=position_offset, base=weights.rope_base)

    if cached_k is not None and cached_v is not None:
        k_total = torch.cat([cached_k, k_rope], dim=-2)
        v_total = torch.cat([cached_v, v], dim=-2)
    else:
        k_total = k_rope
        v_total = v

    k_rep = repeat_kv(k_total, group)
    v_rep = repeat_kv(v_total, group)

    S_total = k_total.shape[-2]
    scores = q_rope @ k_rep.transpose(-2, -1) / math.sqrt(head_dim)
    # Causal mask: query position p is in [position_offset, position_offset+S_new),
    # may attend to key positions [0, p].
    causal = torch.zeros(S_new, S_total, dtype=scores.dtype, device=scores.device)
    neg_inf = torch.finfo(scores.dtype).min
    q_idx = torch.arange(S_new, device=scores.device).unsqueeze(-1) + position_offset
    k_idx = torch.arange(S_total, device=scores.device).unsqueeze(0)
    causal = torch.where(k_idx > q_idx, torch.full_like(causal, neg_inf), causal)
    scores = scores + causal
    probs = F.softmax(scores, dim=-1)
    attn = probs @ v_rep
    attn_merged = _merge_heads(attn)
    attn_out = attn_merged @ weights.w_o
    if weights.b_o is not None:
        attn_out = attn_out + weights.b_o
    h_mid = x + attn_out
    h2 = _rmsnorm_with_gamma(
        h_mid, weights.post_attention_norm_weight, weights.post_attention_norm_eps,
    )
    gate = h2 @ weights.w_gate
    if weights.b_gate is not None:
        gate = gate + weights.b_gate
    up = h2 @ weights.w_up
    if weights.b_up is not None:
        up = up + weights.b_up
    hidden_swiglu = F.silu(gate) * up
    mlp_out = hidden_swiglu @ weights.w_down
    if weights.b_down is not None:
        mlp_out = mlp_out + weights.b_down
    y = h_mid + mlp_out
    return {
        "y": y, "k_new": k_rope, "v_new": v,
        "h_mid": h_mid, "mlp_out": mlp_out, "h1": h1, "h2": h2,
    }


def plain_model_forward(
    input_ids: torch.Tensor, weights: ModernDecoderModelWeights,
) -> torch.Tensor:
    """Plain reference forward — embeds, runs N blocks, final norm + lm_head."""
    x = F.embedding(input_ids, weights.embed_tokens_weight)
    for layer_weights in weights.layers:
        out = _plain_block_step(x, layer_weights, position_offset=0)
        x = out["y"]
    x = _rmsnorm_with_gamma(x, weights.final_norm_weight, weights.final_norm_eps)
    logits = x @ weights.lm_head_weight
    if weights.lm_head_bias is not None:
        logits = logits + weights.lm_head_bias
    return logits


def plain_prefill(
    input_ids: torch.Tensor, weights: ModernDecoderModelWeights,
) -> dict[str, Any]:
    """Plain prefill returning logits + per-layer cached K/V."""
    x = F.embedding(input_ids, weights.embed_tokens_weight)
    layer_caches: list[dict[str, torch.Tensor]] = []
    for layer_weights in weights.layers:
        out = _plain_block_step(x, layer_weights, position_offset=0)
        layer_caches.append({"k": out["k_new"], "v": out["v_new"]})
        x = out["y"]
    x_norm = _rmsnorm_with_gamma(x, weights.final_norm_weight, weights.final_norm_eps)
    logits = x_norm @ weights.lm_head_weight
    if weights.lm_head_bias is not None:
        logits = logits + weights.lm_head_bias
    return {"logits": logits, "layer_caches": layer_caches}


def plain_decode_step(
    next_ids: torch.Tensor,
    weights: ModernDecoderModelWeights,
    layer_caches: list[dict[str, torch.Tensor]],
    position: int,
) -> dict[str, Any]:
    """Plain decode_step that appends one new token."""
    x = F.embedding(next_ids, weights.embed_tokens_weight)
    new_caches: list[dict[str, torch.Tensor]] = []
    for layer_weights, cache in zip(weights.layers, layer_caches):
        out = _plain_block_step(
            x, layer_weights,
            cached_k=cache["k"], cached_v=cache["v"],
            position_offset=position,
        )
        new_caches.append({
            "k": torch.cat([cache["k"], out["k_new"]], dim=-2),
            "v": torch.cat([cache["v"], out["v_new"]], dim=-2),
        })
        x = out["y"]
    x_norm = _rmsnorm_with_gamma(x, weights.final_norm_weight, weights.final_norm_eps)
    logits = x_norm @ weights.lm_head_weight
    if weights.lm_head_bias is not None:
        logits = logits + weights.lm_head_bias
    return {"logits": logits, "layer_caches": new_caches}


# ---------------------------------------------------------------------------
# Obfuscated model wrapper
# ---------------------------------------------------------------------------


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)


def _allclose_metrics(
    plain: torch.Tensor, recovered: torch.Tensor, dtype: torch.dtype,
) -> dict[str, float]:
    atol, rtol = _atol_rtol(dtype)
    diff = (recovered - plain).abs()
    ref_norm = float(plain.norm().clamp_min(1e-30).item())
    rel_l2 = float(
        ((recovered - plain).norm() / max(ref_norm, 1e-30)).item()
    )
    cos = float(
        (recovered.flatten() @ plain.flatten()
         / (recovered.norm() * plain.norm()).clamp_min(1e-30)).item()
    )
    return {
        "max_abs_error": float(diff.max().item()),
        "relative_l2_error": rel_l2,
        "cosine_similarity": cos,
        "allclose": bool(torch.allclose(plain, recovered, atol=atol, rtol=rtol)),
    }


class ObfuscatedModernDecoderModelWrapper:
    """Multi-block obfuscated decoder with prefill / decode_step / greedy.

    Default mode ``nonlinear_mode='trusted'`` replays plain math for
    correctness (the report still labels the mode honestly). Compatible
    mode runs the Stage 6.4b obfuscated math for each block PLUS an
    optional dense mask around the LM head (``N_in_lm`` /
    ``N_vocab_lm``); when ``nonlinear_mode='trusted'`` the LM head stays
    plain.
    """

    def __init__(
        self,
        weights: ModernDecoderModelWeights,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        use_pad: bool = False,
        nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
        mitigation_bundle: str = DEFAULT_MITIGATION_BUNDLE,
        collect_traces: bool = False,
    ) -> None:
        self.weights = weights
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_pad = bool(use_pad)
        self.nonlinear_mode = normalize_nonlinear_mode(nonlinear_mode)
        self.mitigation_bundle = normalize_mitigation_bundle(mitigation_bundle)
        self.collect_traces = bool(collect_traces)
        self._desc = describe_mitigation_bundle(self.mitigation_bundle)
        self.num_layers = len(weights.layers)
        # Pre-build per-layer Stage 6.4b wrappers for ``full_forward`` ONLY.
        # prefill / decode_step use an inline path so per-layer N_K / N_V can
        # be cached.
        self.block_wrappers = [
            ObfuscatedModernDecoderBlockWrapper(
                w,
                dtype=dtype, device=device,
                use_pad=use_pad,
                nonlinear_mode=self.nonlinear_mode,
                mitigation_bundle=self.mitigation_bundle,
            )
            for w in weights.layers
        ]

    # -------------------------------------------------- helpers
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weights.embed_tokens_weight)

    def _lm_head_obfuscated(
        self, x_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Optionally mask the LM head with N_in (hidden) and N_vocab (vocab).

        Trusted mode: pass-through plain matmul. Compatible mode: sample
        fresh ``N_in_lm`` (dense invertible over hidden) and ``N_vocab_lm``
        (dense invertible over vocab) per call; recover at the trusted
        side.
        """
        W = self.weights.lm_head_weight
        b = self.weights.lm_head_bias
        if self.nonlinear_mode == "trusted":
            logits = x_norm @ W
            if b is not None:
                logits = logits + b
            return logits, {
                "lm_head_status": "trusted_shortcut",
                "lm_head_mask_in_fingerprint": None,
                "lm_head_mask_vocab_fingerprint": None,
            }
        H = W.shape[0]
        V = W.shape[1]
        n_in, n_in_inv = generate_invertible_matrix(
            H, dtype=x_norm.dtype, device=x_norm.device,
        )
        n_vocab, n_vocab_inv = generate_invertible_matrix(
            V, dtype=x_norm.dtype, device=x_norm.device,
        )
        x_tilde = x_norm @ n_in
        W_tilde = (n_in_inv @ W) @ n_vocab
        logits_tilde = x_tilde @ W_tilde
        if b is not None:
            logits_tilde = logits_tilde + b @ n_vocab
        logits_recovered = logits_tilde @ n_vocab_inv
        return logits_recovered, {
            "lm_head_status": "single_dense_mask_pair_with_vocab_mask",
            "lm_head_mask_in_fingerprint": None,   # never publish material
            "lm_head_mask_vocab_fingerprint": None,
        }

    # -------------------------------------------------- full forward
    def full_forward(
        self, input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Embed, run N blocks (via Stage 6.4b wrappers), final norm + LM head.

        Returns ``(recovered_logits, report)``. ``report['allclose']``
        compares against a plain reference built from the extracted weights.
        """
        input_ids = input_ids.to(device=self.device)
        plain_logits = plain_model_forward(input_ids, self.weights)
        x = self._embed(input_ids)
        per_layer_reports: list[dict[str, Any]] = []
        for blk in self.block_wrappers:
            y, blk_report = blk.forward(x)
            x = y
            per_layer_reports.append({
                "layer_index": int(blk_report.get("nonlinear_mode") and len(per_layer_reports)),
                "allclose": blk_report["allclose"],
                "max_abs_error": blk_report["max_abs_error"],
                "rmsnorm_status": blk_report["rmsnorm_status"],
                "rope_attention_status": blk_report["rope_attention_status"],
                "swiglu_status": blk_report["swiglu_status"],
            })
        x_norm = _rmsnorm_with_gamma(
            x, self.weights.final_norm_weight, self.weights.final_norm_eps,
        )
        recovered_logits, lm_head_info = self._lm_head_obfuscated(x_norm)
        m = _allclose_metrics(plain_logits, recovered_logits, self.dtype)
        m["top1_match_rate"] = float(
            (plain_logits.argmax(dim=-1) == recovered_logits.argmax(dim=-1))
            .to(torch.float64).mean().item()
        )
        bundle_meta = bundle_metadata(
            self.mitigation_bundle, use_pad=self.use_pad,
            online_extra_matmul_count=0,
        )
        report = {
            "nonlinear_mode": self.nonlinear_mode,
            "valid_nonlinear_modes": list(VALID_NONLINEAR_MODES),
            "mitigation_bundle": self.mitigation_bundle,
            "valid_mitigation_bundles": list(VALID_MITIGATION_BUNDLES),
            "use_pad": self.use_pad,
            "num_layers": self.num_layers,
            "vocab_size": int(self.weights.vocab_size),
            "logits_metrics": m,
            "per_layer_reports": per_layer_reports,
            "final_norm_status": (
                "trusted_shortcut"
                if self.nonlinear_mode == "trusted"
                else "trusted_final_rmsnorm"
            ),
            "lm_head_status": lm_head_info["lm_head_status"],
            "online_extra_matmul_count": 0,
            "mitigation_bundle_metadata": bundle_meta,
            "dense_sandwich_enabled": bundle_meta["dense_sandwich_enabled"],
            "boundary_pad_enabled": bundle_meta["boundary_pad_enabled"],
            "default_on_candidate_under_stage_5_4": bundle_meta[
                "default_on_candidate_under_stage_5_4"
            ],
            "caveats": [
                "Model-level integration; not a real TEE deployment.",
                "Real wall-time is not measured.",
                "Default mode remains 'trusted'; compatible_islands is opt-in.",
                "This is not formal security.",
            ],
        }
        return recovered_logits, report

    # -------------------------------------------------- prefill (inline)
    def _attention_inline(
        self,
        x_layer: torch.Tensor,
        weights: ModernDecoderBlockWeights,
        *,
        position_offset: int,
        layer_cache: ModernDecoderLayerKVCache | None,
        initialise_cache: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """Inline obfuscated attention with optional KV-cache reuse / init.

        Returns ``(h_mid_plain, debug_dict, info)``.
        ``debug_dict`` carries ``{"k_tilde_new", "v_tilde_new",
        "k_plain_new", "v_plain_new", "q_rope_plain"}`` so the caller can
        append to the cache.
        """
        B, S_new, H = x_layer.shape
        head_dim = weights.head_dim
        num_q = weights.num_attention_heads
        num_kv = weights.num_key_value_heads
        group = num_q // num_kv
        device = x_layer.device
        dtype = x_layer.dtype

        # Compatible: fold γ into qkv.  Always also produce plain projections.
        h1_plain = _rmsnorm_with_gamma(
            x_layer, weights.input_norm_weight, weights.input_norm_eps,
        )
        q_plain = h1_plain @ weights.w_q
        if weights.b_q is not None:
            q_plain = q_plain + weights.b_q
        k_plain = h1_plain @ weights.w_k
        if weights.b_k is not None:
            k_plain = k_plain + weights.b_k
        v_plain = h1_plain @ weights.w_v
        if weights.b_v is not None:
            v_plain = v_plain + weights.b_v
        q = _reshape_heads(q_plain, num_q, head_dim)
        k = _reshape_heads(k_plain, num_kv, head_dim)
        v = _reshape_heads(v_plain, num_kv, head_dim)
        q_rope = _apply_rope_at(q, position_offset=position_offset, base=weights.rope_base)
        k_rope = _apply_rope_at(k, position_offset=position_offset, base=weights.rope_base)

        # Decide whether to sample fresh masks or reuse the cache's masks.
        if self.nonlinear_mode == "compatible_islands":
            if initialise_cache:
                # Fresh per-kv-head N_K / N_V (and their inverses).
                N_K_list, N_K_inv_list = [], []
                N_V_list, N_V_inv_list = [], []
                for _ in range(num_kv):
                    nk, nki = generate_invertible_matrix(head_dim, dtype, device)
                    nv, nvi = generate_invertible_matrix(head_dim, dtype, device)
                    N_K_list.append(nk); N_K_inv_list.append(nki)
                    N_V_list.append(nv); N_V_inv_list.append(nvi)
                N_K_stack = torch.stack(N_K_list, dim=0)
                N_V_stack = torch.stack(N_V_list, dim=0)
                N_V_inv_stack = torch.stack(N_V_inv_list, dim=0)
                if layer_cache is not None:
                    layer_cache.n_k_stack = N_K_stack
                    layer_cache.n_v_stack = N_V_stack
                    layer_cache.n_v_inv_stack = N_V_inv_stack
            else:
                assert layer_cache is not None
                N_K_stack = layer_cache.n_k_stack
                N_V_stack = layer_cache.n_v_stack
                N_V_inv_stack = layer_cache.n_v_inv_stack
            # Per-q-head N_Q = N_K[group]^{-T}.
            N_Q_per_q = torch.stack(
                [
                    torch.linalg.inv(N_K_stack[i // group].to(torch.float64))
                    .to(dtype).transpose(-2, -1)
                    for i in range(num_q)
                ], dim=0,
            )
            q_tilde = q_rope @ N_Q_per_q.unsqueeze(0)
            k_tilde_new = k_rope @ N_K_stack.unsqueeze(0)
            v_tilde_new = v @ N_V_stack.unsqueeze(0)
            # Combine with cache.
            if layer_cache is not None and layer_cache.seq_len > 0:
                k_tilde_total = torch.cat([layer_cache.key_tilde, k_tilde_new], dim=-2)
                v_tilde_total = torch.cat([layer_cache.value_tilde, v_tilde_new], dim=-2)
            else:
                k_tilde_total = k_tilde_new
                v_tilde_total = v_tilde_new
            k_tilde_rep = repeat_kv(k_tilde_total, group)
            v_tilde_rep = repeat_kv(v_tilde_total, group)
            N_V_inv_per_q = torch.stack(
                [N_V_inv_stack[i // group] for i in range(num_q)], dim=0,
            )
            S_total = k_tilde_total.shape[-2]
            scores = (
                q_tilde @ k_tilde_rep.transpose(-2, -1) / math.sqrt(head_dim)
            )
            neg_inf = torch.finfo(scores.dtype).min
            q_idx = (
                torch.arange(S_new, device=device).unsqueeze(-1) + position_offset
            )
            k_idx = torch.arange(S_total, device=device).unsqueeze(0)
            causal = torch.where(
                k_idx > q_idx,
                torch.full((S_new, S_total), neg_inf, dtype=dtype, device=device),
                torch.zeros(S_new, S_total, dtype=dtype, device=device),
            )
            scores = scores + causal
            probs = F.softmax(scores, dim=-1)
            av_tilde = probs @ v_tilde_rep
            av_recovered = av_tilde @ N_V_inv_per_q.unsqueeze(0)
            attn_merged = _merge_heads(av_recovered)
            attn_out_plain = attn_merged @ weights.w_o
            if weights.b_o is not None:
                attn_out_plain = attn_out_plain + weights.b_o
            h_mid = x_layer + attn_out_plain
            return (
                h_mid,
                {
                    "k_tilde_new": k_tilde_new,
                    "v_tilde_new": v_tilde_new,
                    "k_plain_new": k_rope,
                    "v_plain_new": v,
                    "q_rope_plain": q_rope,
                },
                {
                    "attention_status": "rope_post_mask_only",
                    "gqa_status": "per_kv_head_mask_with_repeat_kv",
                },
            )

        # Trusted mode: plain attention with cache.
        if layer_cache is not None and layer_cache.seq_len > 0:
            # Trusted mode caches plain K/V as well (kept in layer_cache.key_plain).
            k_total = torch.cat([layer_cache.key_plain, k_rope], dim=-2)
            v_total = torch.cat([layer_cache.value_plain, v], dim=-2)
        else:
            k_total = k_rope
            v_total = v
        k_rep = repeat_kv(k_total, group)
        v_rep = repeat_kv(v_total, group)
        S_total = k_total.shape[-2]
        scores = q_rope @ k_rep.transpose(-2, -1) / math.sqrt(head_dim)
        neg_inf = torch.finfo(scores.dtype).min
        q_idx = (
            torch.arange(S_new, device=device).unsqueeze(-1) + position_offset
        )
        k_idx = torch.arange(S_total, device=device).unsqueeze(0)
        causal = torch.where(
            k_idx > q_idx,
            torch.full((S_new, S_total), neg_inf, dtype=dtype, device=device),
            torch.zeros(S_new, S_total, dtype=dtype, device=device),
        )
        scores = scores + causal
        probs = F.softmax(scores, dim=-1)
        attn = probs @ v_rep
        attn_merged = _merge_heads(attn)
        attn_out = attn_merged @ weights.w_o
        if weights.b_o is not None:
            attn_out = attn_out + weights.b_o
        h_mid = x_layer + attn_out
        return (
            h_mid,
            {
                "k_tilde_new": k_rope,        # in trusted mode N_K = I
                "v_tilde_new": v,
                "k_plain_new": k_rope,
                "v_plain_new": v,
                "q_rope_plain": q_rope,
            },
            {
                "attention_status": "trusted_shortcut",
                "gqa_status": "trusted_shortcut",
            },
        )

    def _mlp_inline(
        self,
        h_mid: torch.Tensor,
        weights: ModernDecoderBlockWeights,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Inline MLP path — compatible mode uses Stage 6.4b SwiGLU island."""
        B, S_new, H = h_mid.shape
        dtype = h_mid.dtype
        device = h_mid.device
        h2_plain = _rmsnorm_with_gamma(
            h_mid, weights.post_attention_norm_weight, weights.post_attention_norm_eps,
        )
        if self.nonlinear_mode == "trusted":
            gate = h2_plain @ weights.w_gate
            if weights.b_gate is not None:
                gate = gate + weights.b_gate
            up = h2_plain @ weights.w_up
            if weights.b_up is not None:
                up = up + weights.b_up
            mlp_hidden = F.silu(gate) * up
            mlp_out = mlp_hidden @ weights.w_down
            if weights.b_down is not None:
                mlp_out = mlp_out + weights.b_down
            return h_mid + mlp_out, {"swiglu_status": "trusted_shortcut"}
        # Compatible: route through Stage 6.4b SwiGLU island.
        w_gate_folded = weights.post_attention_norm_weight.unsqueeze(-1) * weights.w_gate
        w_up_folded = weights.post_attention_norm_weight.unsqueeze(-1) * weights.w_up
        n_in_island, n_in_island_inv = generate_invertible_matrix(H, dtype, device)
        n_out_island, n_out_island_inv = generate_invertible_matrix(H, dtype, device)
        perm = generate_permutation(
            weights.intermediate_size, dtype=dtype, device=device,
        )["perm"]
        # h2 (post-RMSNorm + γ) is what the plain SwiGLU expects.  γ has been
        # folded into the island's weights so the island input is the
        # γ-free core ``rmsnorm_core(h_mid)``.
        h2_core = rmsnorm_core(h_mid, eps=weights.post_attention_norm_eps)
        pad_in = None
        if self.use_pad:
            pad_in = torch.randn(
                h2_core.reshape(-1, H).shape, dtype=dtype, device=device,
            )
        island = run_swiglu_mlp_island(
            x=h2_core.reshape(-1, H),
            w_up=w_up_folded, b_up=weights.b_up,
            w_gate=w_gate_folded, b_gate=weights.b_gate,
            w_down=weights.w_down, b_down=weights.b_down,
            n_in=n_in_island, n_in_inv=n_in_island_inv,
            permutation=perm,
            n_out=n_out_island,
            pad_in=pad_in,
            mitigation_bundle=self.mitigation_bundle,
        )
        mlp_out_plain = island["y_tilde"].reshape(B, S_new, H) @ n_out_island_inv
        return h_mid + mlp_out_plain, {
            "swiglu_status": "compatible_island_paired_permutation",
        }

    # -------------------------------------------------- prefill
    def prefill(
        self, input_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Run prefill and return logits + initialised KV cache."""
        input_ids = input_ids.to(device=self.device)
        B, S = input_ids.shape
        # Plain reference for correctness checking.
        plain = plain_prefill(input_ids, self.weights)
        plain_logits = plain["logits"]
        plain_layer_caches = plain["layer_caches"]

        x = self._embed(input_ids)
        # Allocate empty cache shell with the per-layer mask material.
        cache = init_empty_modern_decoder_kv_cache(
            num_layers=self.num_layers,
            batch_size=B,
            num_kv_heads=self.weights.num_key_value_heads,
            head_dim=self.weights.head_dim,
            attention_variant=self._attention_variant(),
            dtype=self.dtype, device=self.device,
        )
        traces: dict[str, Any] = {} if self.collect_traces else {}
        for idx, weights in enumerate(self.weights.layers):
            layer_cache = cache.layers[idx]
            h_mid, dbg, attn_info = self._attention_inline(
                x, weights,
                position_offset=0,
                layer_cache=layer_cache,
                initialise_cache=True,
            )
            # Append to layer cache (one-shot for prefill).
            layer_cache.key_tilde = dbg["k_tilde_new"]
            layer_cache.value_tilde = dbg["v_tilde_new"]
            layer_cache.seq_len = int(dbg["k_tilde_new"].shape[-2])
            layer_cache.cache_status = "filled_after_prefill"
            x_layer_out, mlp_info = self._mlp_inline(h_mid, weights)
            x = x_layer_out
        cache.total_seq_len = int(S)

        x_norm = _rmsnorm_with_gamma(
            x, self.weights.final_norm_weight, self.weights.final_norm_eps,
        )
        recovered_logits, lm_head_info = self._lm_head_obfuscated(x_norm)
        metrics = _allclose_metrics(plain_logits, recovered_logits, self.dtype)
        metrics["top1_match_rate"] = float(
            (plain_logits.argmax(dim=-1) == recovered_logits.argmax(dim=-1))
            .to(torch.float64).mean().item()
        )
        report = {
            "stage": "prefill",
            "nonlinear_mode": self.nonlinear_mode,
            "mitigation_bundle": self.mitigation_bundle,
            "use_pad": self.use_pad,
            "prompt_length": int(S),
            "logits_metrics": metrics,
            "cache_summary": cache.summary_dict(),
            "lm_head_status": lm_head_info["lm_head_status"],
            "online_extra_matmul_count": 0,
            "caveats": [
                "Prefill is recoverable to plain; no real TEE isolation.",
                "Not formal security.",
            ],
        }
        return {
            "logits_tilde": recovered_logits,    # already recovered; kept name for spec parity
            "logits_recovered": recovered_logits,
            "logits_plain": plain_logits,
            "kv_cache": cache,
            "plain_layer_caches": plain_layer_caches,
            "report": report,
        }

    # -------------------------------------------------- decode_step
    def decode_step(
        self,
        next_ids: torch.Tensor,
        kv_cache: ModernDecoderKVCache,
        position: int,
        *,
        plain_layer_caches: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        """Process one new token using the cached masked K/V."""
        next_ids = next_ids.to(device=self.device)
        if next_ids.dim() == 1:
            next_ids = next_ids.unsqueeze(-1)   # [B] → [B, 1]
        B, S_new = next_ids.shape
        # Plain reference.
        if plain_layer_caches is not None:
            plain = plain_decode_step(
                next_ids, self.weights, plain_layer_caches, position=position,
            )
            plain_logits = plain["logits"]
            plain_updated_caches = plain["layer_caches"]
        else:
            plain_logits = None
            plain_updated_caches = None

        x = self._embed(next_ids)
        for idx, weights in enumerate(self.weights.layers):
            layer_cache = kv_cache.layers[idx]
            h_mid, dbg, attn_info = self._attention_inline(
                x, weights,
                position_offset=position,
                layer_cache=layer_cache,
                initialise_cache=False,
            )
            kv_cache.append_layer(
                idx,
                dbg["k_tilde_new"], dbg["v_tilde_new"],
            )
            x_out, mlp_info = self._mlp_inline(h_mid, weights)
            x = x_out
        kv_cache.bump_seq_len(S_new)

        x_norm = _rmsnorm_with_gamma(
            x, self.weights.final_norm_weight, self.weights.final_norm_eps,
        )
        recovered_logits, lm_head_info = self._lm_head_obfuscated(x_norm)
        if plain_logits is not None:
            metrics = _allclose_metrics(plain_logits, recovered_logits, self.dtype)
            metrics["top1_match_rate"] = float(
                (plain_logits.argmax(dim=-1) == recovered_logits.argmax(dim=-1))
                .to(torch.float64).mean().item()
            )
        else:
            metrics = None
        report = {
            "stage": "decode_step",
            "nonlinear_mode": self.nonlinear_mode,
            "mitigation_bundle": self.mitigation_bundle,
            "use_pad": self.use_pad,
            "position": int(position),
            "new_seq_len": int(S_new),
            "logits_metrics": metrics,
            "cache_summary": kv_cache.summary_dict(),
            "lm_head_status": lm_head_info["lm_head_status"],
            "rope_position_increment": True,
            "rope_position_used": int(position),
            "online_extra_matmul_count": 0,
            "caveats": [
                "Decode is recoverable to plain; no real TEE isolation.",
                "Not formal security.",
            ],
        }
        return {
            "next_logits_tilde": recovered_logits,    # already recovered
            "next_logits_recovered": recovered_logits,
            "next_logits_plain": plain_logits,
            "kv_cache": kv_cache,
            "plain_layer_caches": plain_updated_caches,
            "report": report,
        }

    # -------------------------------------------------- greedy generate
    def greedy_generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int = 3,
    ) -> dict[str, Any]:
        """Hand-written greedy loop, compared against plain reference."""
        prefill_out = self.prefill(input_ids)
        cache = prefill_out["kv_cache"]
        plain_caches = prefill_out["plain_layer_caches"]
        # First sampled token from prefill logits.
        next_token = prefill_out["logits_recovered"][:, -1, :].argmax(dim=-1)
        plain_next = prefill_out["logits_plain"][:, -1, :].argmax(dim=-1)
        obf_tokens: list[torch.Tensor] = [next_token]
        plain_tokens: list[torch.Tensor] = [plain_next]
        per_step_metrics: list[dict[str, Any]] = []
        cur_token = next_token
        position = int(input_ids.shape[-1])
        for step in range(max_new_tokens - 1):
            step_out = self.decode_step(
                cur_token.unsqueeze(-1), cache, position,
                plain_layer_caches=plain_caches,
            )
            cache = step_out["kv_cache"]
            plain_caches = step_out["plain_layer_caches"]
            cur_logits = step_out["next_logits_recovered"][:, -1, :]
            plain_logits = step_out["next_logits_plain"][:, -1, :]
            cur_token = cur_logits.argmax(dim=-1)
            plain_token = plain_logits.argmax(dim=-1)
            obf_tokens.append(cur_token)
            plain_tokens.append(plain_token)
            per_step_metrics.append(step_out["report"]["logits_metrics"])
            position += 1
        obf_seq = torch.cat([input_ids] + [t.unsqueeze(-1) for t in obf_tokens], dim=-1)
        plain_seq = torch.cat(
            [input_ids] + [t.unsqueeze(-1) for t in plain_tokens], dim=-1
        )
        new_slice = slice(input_ids.shape[-1], obf_seq.shape[-1])
        token_match_rate = float(
            (obf_seq[:, new_slice] == plain_seq[:, new_slice])
            .to(torch.float64).mean().item()
        )
        sequence_exact_match = bool(torch.equal(obf_seq, plain_seq))
        report = {
            "stage": "greedy_generate",
            "nonlinear_mode": self.nonlinear_mode,
            "mitigation_bundle": self.mitigation_bundle,
            "use_pad": self.use_pad,
            "max_new_tokens": int(max_new_tokens),
            "prompt_length": int(input_ids.shape[-1]),
            "token_match_rate": token_match_rate,
            "sequence_exact_match": sequence_exact_match,
            "top1_match_rate": token_match_rate,
            "per_step_logits_metrics": per_step_metrics,
            "prefill_logits_metrics": prefill_out["report"]["logits_metrics"],
            "online_extra_matmul_count": 0,
            "caveats": [
                "Greedy only; beam / top-k / top-p not implemented.",
                "Not a real TEE measurement.",
                "Not formal security.",
            ],
        }
        return {
            "obf_sequence": obf_seq,
            "plain_sequence": plain_seq,
            "report": report,
        }

    # -------------------------------------------------- misc helpers
    def _attention_variant(self) -> str:
        nq = self.weights.num_attention_heads
        nk = self.weights.num_key_value_heads
        if nq == nk:
            return "mha"
        if nk == 1:
            return "mqa"
        return "gqa"


__all__ = [
    "ModernDecoderModelWeights",
    "ObfuscatedModernDecoderModelWrapper",
    "plain_decode_step",
    "plain_model_forward",
    "plain_prefill",
]
