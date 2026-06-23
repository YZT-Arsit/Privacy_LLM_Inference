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
    fold_final_norm_lm_head_with_vocab_mask,
    greedy_sample,
    make_vocab_logit_mask,
    recover_vocab_logits,
    trusted_embedding_lookup,
)
from pllo.ops.gqa_attention import (
    generate_gqa_rope_masks,
    merge_heads,
    repeat_kv,
    split_heads,
)
from pllo.ops.llama_synthetic_block import rmsnorm_plain
from pllo.ops.nonlinear_islands import rmsnorm_core, silu_reference
from pllo.ops.rope import apply_rope, build_rope_cache

__all__ = [
    "MASK_SECURITY_NOTES",
    "NEGATIVE_CONTROLS",
    "HFCausalLMMaskBundle",
    "HFCausalLMSkeletonConfig",
    "HFCausalLMSkeletonWeights",
    "apply_signed_permutation",
    "extract_hf_causal_lm_skeleton_weights",
    "generate_hf_causal_lm_masks",
    "generate_hf_single_block_masks_scalable",
    "invert_signed_permutation",
    "make_block_orthogonal_mask",
    "make_dense_orthogonal_mask_cpu_float32",
    "make_residual_mask",
    "signed_permutation_components",
    "has_transformers",
    "hf_causal_lm_masked_greedy_decode",
    "hf_causal_lm_masked_only_decode",
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
    # Residual-mask scalability knobs (Stage 8.2). Defaults reproduce the
    # Stage 6.9 behavior (per-layer dense orthogonal masks) exactly.
    mask_mode: str = "dense_orthogonal"   # signed_permutation|block_orthogonal|dense_orthogonal
    residual_mask_strategy: str = "per_layer"   # "shared" | "per_layer"
    mask_block_size: int = 64
    allow_dense_large_mask: bool = False
    # Mixed-precision knobs (Stage 8.2 bf16). ``None`` -> identical to the
    # Stage 6.9 single-dtype behavior. ``folded_runtime_dtype`` casts the
    # folded weights + masked hidden states for the masked forward (e.g. bf16)
    # while the plain reference + folding stay at ``dtype``; ``recovery_dtype``
    # runs the final-norm/LM-head/vocab recovery + comparison in higher
    # precision (e.g. float32) to avoid bf16 inverse/scaling drift.
    folded_runtime_dtype: torch.dtype | None = None
    recovery_dtype: torch.dtype | None = None
    # Negative controls (Stage 8.2 audit). "none" -> the verified masked path.
    # "wrong_vocab_recovery" recovers masked logits with an intentionally wrong
    # vocab permutation/scale (expected: tokens mismatch, large recovered err).
    # "plaintext_weights_on_masked_hidden" feeds the masked hidden into the
    # plain (folding-disabled) output head (expected: mismatch). Both are used
    # to prove the masked path is actually exercised (and that breaking it
    # breaks the result).
    negative_control: str = "none"


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
    shared_residual_mask: bool = False

    def handoff(self, ell: int) -> torch.Tensor:
        """``T_ell = N_ell^{-1} @ N_{ell+1}`` (orthogonal change-of-basis)."""
        return self.residual_mask_inverses[ell] @ self.residual_masks[ell + 1]

    def needs_handoff(self, ell: int) -> bool:
        """A shared residual mask makes every handoff the identity, so the
        online ``[H,H]`` GEMM can be skipped entirely."""
        return not self.shared_residual_mask


def _orthogonal(dim: int, dtype: torch.dtype, device: torch.device,
                g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype,
                                       device=device))
    return q


# ---------------------------------------------------------------------------
# Scalable residual masks (Stage 8.2): orthogonal + RMSNorm-compatible, but
# cheaper to build than a dense HxH QR. All return a *dense* ``[H,H]`` matrix
# (so the verified Stage 6.9 weight-folding APIs are reused unchanged) plus
# its transpose-inverse; the win is in *generation* cost + the shared-mask
# strategy, not in the one-time fold.
# ---------------------------------------------------------------------------


_DENSE_MASK_HIDDEN_LIMIT = 1024

MASK_SECURITY_NOTES: dict[str, str] = {
    "signed_permutation":
        "signed_permutation is a (signed) permutation matrix: orthogonal and "
        "RMSNorm-compatible, scalable to large hidden sizes, but WEAKER than "
        "dense orthogonal masking (it permutes + flips coordinates without "
        "mixing them).",
    "block_orthogonal":
        "block_orthogonal mixes coordinates within blocks (default size "
        "{block}); stronger than signed_permutation, cheaper than a full "
        "dense QR, still weaker than dense orthogonal masking.",
    "dense_orthogonal":
        "dense_orthogonal (full HxH QR) is the strongest here but is "
        "memory/compute-heavy; intended for tiny / diagnostic models only.",
}


# CUDA QR ("geqrf_cuda") is NOT implemented for bf16/fp16, so any QR for a
# low-precision target must run on CPU in float32 and then be cast/moved.
_LOW_PRECISION = (torch.bfloat16, torch.float16)


def _cpu_generator(g: torch.Generator) -> torch.Generator:
    """A CPU generator with the same initial seed as ``g`` (so CPU-side QR /
    randn is deterministic even when ``g`` lives on CUDA)."""
    if g.device.type == "cpu":
        return g
    gc = torch.Generator(device="cpu")
    gc.manual_seed(g.initial_seed())
    return gc


def signed_permutation_components(
    hidden: int, g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cheap, QR-free residual-mask representation: ``(perm, inv_perm, signs)``
    on CPU (move as needed). ``signs`` are float32 ``+/-1``."""
    gc = _cpu_generator(g)
    perm = torch.randperm(hidden, generator=gc, device="cpu")
    inv_perm = torch.argsort(perm)
    signs = torch.where(
        torch.rand(hidden, generator=gc, device="cpu") < 0.5,
        torch.tensor(-1.0), torch.tensor(1.0)).to(torch.float32)
    return perm, inv_perm, signs


def apply_signed_permutation(
    x: torch.Tensor, perm: torch.Tensor, signs: torch.Tensor,
) -> torch.Tensor:
    """``X_tilde[..., k] = X[..., perm[k]] * signs[k]`` (no dense matrix)."""
    return x.index_select(-1, perm) * signs


def invert_signed_permutation(
    x_tilde: torch.Tensor, inv_perm: torch.Tensor, signs: torch.Tensor,
) -> torch.Tensor:
    """Undo :func:`apply_signed_permutation`."""
    return (x_tilde * signs).index_select(-1, inv_perm)


def make_dense_orthogonal_mask_cpu_float32(
    hidden: int, g: torch.Generator,
) -> torch.Tensor:
    """Full HxH orthogonal mask via QR on **CPU float32** (never bf16 CUDA)."""
    gc = _cpu_generator(g)
    q, _ = torch.linalg.qr(
        torch.randn(hidden, hidden, generator=gc, dtype=torch.float32,
                    device="cpu"))
    return q


def make_block_orthogonal_mask(
    hidden: int, block_size: int, target_dtype: torch.dtype,
    target_device: torch.device, g: torch.Generator,
) -> torch.Tensor:
    """Block-diagonal orthogonal mask. QR runs on CPU float32 for
    low-precision targets, then casts/moves; native otherwise."""
    bs = max(1, min(block_size, hidden))
    low = target_dtype in _LOW_PRECISION
    qr_dtype = torch.float32 if low else target_dtype
    qr_device = torch.device("cpu") if low else target_device
    gen = _cpu_generator(g) if low else g
    m = torch.zeros(hidden, hidden, dtype=qr_dtype, device=qr_device)
    start = 0
    while start < hidden:
        d = min(bs, hidden - start)
        q, _ = torch.linalg.qr(
            torch.randn(d, d, generator=gen, dtype=qr_dtype, device=qr_device))
        m[start:start + d, start:start + d] = q
        start += d
    return m.to(dtype=target_dtype, device=target_device)


def make_residual_mask(
    hidden: int, mode: str, dtype: torch.dtype, device: torch.device,
    g: torch.Generator, *, block_size: int = 64,
    allow_dense_large: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense orthogonal ``[H,H]`` residual mask + its inverse, never
    invoking bf16/fp16 CUDA QR (which is unimplemented).

    ``signed_permutation``: QR-free; built from ``(perm, signs)``.
    ``block_orthogonal`` / ``dense_orthogonal``: QR on CPU float32 for
    low-precision targets, then cast/move (native for fp32/fp64)."""
    device = torch.device(device)
    if mode == "signed_permutation":
        perm, _inv, signs = signed_permutation_components(hidden, g)
        # Materialize the dense signed-permutation matrix from perm/signs
        # (no QR): eye[:, perm] * signs -> M[perm[k], k] = signs[k].
        eye = torch.eye(hidden, dtype=torch.float32, device="cpu")
        m = (eye.index_select(1, perm) * signs).to(dtype=dtype, device=device)
        return m, m.transpose(0, 1).contiguous()
    if mode == "block_orthogonal":
        m = make_block_orthogonal_mask(hidden, block_size, dtype, device, g)
        return m, m.transpose(0, 1).contiguous()
    if mode == "dense_orthogonal":
        if hidden > _DENSE_MASK_HIDDEN_LIMIT and not allow_dense_large:
            raise ValueError(
                f"dense_orthogonal mask refused for hidden_size={hidden} > "
                f"{_DENSE_MASK_HIDDEN_LIMIT}; pass allow_dense_large=True "
                f"(or --allow-dense-large-mask) to override, or use "
                f"mask_mode='signed_permutation'/'block_orthogonal'.")
        if dtype in _LOW_PRECISION:
            m = make_dense_orthogonal_mask_cpu_float32(hidden, g).to(
                dtype=dtype, device=device)
        else:
            m = _orthogonal(hidden, dtype, device, g)
        return m, m.transpose(0, 1).contiguous()
    raise ValueError(f"unknown mask_mode {mode!r}")


def _attn_masks_to(attn: dict[str, Any], dtype: torch.dtype,
                   device: torch.device) -> dict[str, Any]:
    """Cast float mask tensors to ``(dtype, device)``; keep index/scalar
    fields. Used to move CPU-float32-built attention masks to the target."""
    float_keys = ("key_masks", "key_mask_inverses", "value_masks",
                  "value_mask_inverses", "q_masks")
    out: dict[str, Any] = {}
    for k, v in attn.items():
        if k in float_keys:
            out[k] = v.to(dtype=dtype, device=device)
        elif k == "kv_index":
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def generate_hf_single_block_masks_scalable(
    config: HFSingleBlockConfig, seed: int, n_res: torch.Tensor,
    n_res_inv: torch.Tensor,
) -> dict[str, Any]:
    """QR-free per-layer block masks for real checkpoints: RoPE/GQA attention
    masks (Stage 6.4.1, no QR) + SwiGLU permutation, with the residual mask
    supplied externally (so no dense bf16 CUDA QR is ever attempted).

    Attention masks are built on CPU float32 then cast/moved for low-precision
    targets (bf16/fp16); built natively for fp32/fp64 to preserve precision."""
    target_dtype = config.dtype
    target_device = torch.device(config.device)
    low = target_dtype in _LOW_PRECISION
    gen_dtype = torch.float32 if low else target_dtype
    gen_device = torch.device("cpu") if low else target_device
    g = torch.Generator(device=gen_device).manual_seed(seed)

    attn = generate_gqa_rope_masks(
        config.num_heads, config.num_key_value_heads, config.head_dim,
        gen_dtype, gen_device, g, mask_family=config.mask_family)
    perm = torch.randperm(config.intermediate_size, generator=g,
                          device=gen_device).to(target_device)
    if low:
        attn = _attn_masks_to(attn, target_dtype, target_device)

    return {
        "n_res": n_res, "n_res_inv": n_res_inv, "attn": attn, "perm": perm,
        "mask_family": config.mask_family,
        "residual_mask_family": "external_scalable",
    }


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
    mask_mode = getattr(config, "mask_mode", "dense_orthogonal")
    strategy = getattr(config, "residual_mask_strategy", "per_layer")
    block_size = getattr(config, "mask_block_size", 64)
    allow_dense_large = getattr(config, "allow_dense_large_mask", False)
    cfg0 = layer_configs[0]
    dtype = cfg0.dtype
    device = torch.device(cfg0.device)
    hidden = cfg0.hidden_size
    n_layers = len(layer_configs)
    shared = strategy == "shared"

    g = torch.Generator(device=device).manual_seed(seed)
    if shared:
        # One residual mask reused for every layer: each handoff is the
        # identity, so the online [H,H] GEMM disappears entirely.
        m, m_inv = make_residual_mask(
            hidden, mask_mode, dtype, device, g, block_size=block_size,
            allow_dense_large=allow_dense_large)
        residual_masks = [m] * (n_layers + 1)
        residual_mask_inverses = [m_inv] * (n_layers + 1)
    else:
        pairs = [make_residual_mask(
            hidden, mask_mode, dtype, device, g, block_size=block_size,
            allow_dense_large=allow_dense_large)
            for _ in range(n_layers + 1)]
        residual_masks = [p[0] for p in pairs]
        residual_mask_inverses = [p[1] for p in pairs]

    layer_block_masks: list[dict[str, Any]] = []
    for ell in range(n_layers):
        # QR-free per-layer attention masks + SwiGLU permutation; the residual
        # mask N_ell is supplied externally, so the old dense-QR n_res path
        # (which crashes on bf16 CUDA) is never entered.
        bm = generate_hf_single_block_masks_scalable(
            layer_configs[ell], seed + 101 * (ell + 1),
            residual_masks[ell], residual_mask_inverses[ell])
        layer_block_masks.append(bm)

    vocab_size = int(weights.lm_head_weight.shape[1])
    vocab_mask = make_vocab_logit_mask(vocab_size, dtype, device, g)

    input_pad = None
    if config.use_input_pad:
        input_pad = torch.randn(hidden, generator=g, dtype=dtype,
                                device=device)

    note = MASK_SECURITY_NOTES.get(mask_mode, "").format(block=block_size)
    return HFCausalLMMaskBundle(
        residual_masks=residual_masks,
        residual_mask_inverses=residual_mask_inverses,
        layer_block_masks=layer_block_masks, vocab_mask=vocab_mask,
        input_pad=input_pad, shared_residual_mask=shared,
        metadata={
            "residual_mask_family": f"{mask_mode}_{strategy}",
            "mask_mode": mask_mode,
            "residual_mask_strategy": strategy,
            "mask_block_size": block_size if mask_mode == "block_orthogonal"
            else None,
            "materialized_dense_from_signed_perm":
                mask_mode == "signed_permutation",
            "qr_free_residual_mask": mask_mode == "signed_permutation",
            "mask_security_note": note,
            "mask_family": mask_family,
            "handoff": "identity_shared" if shared else "N_ell_to_N_ell_plus_1",
            "handoff_transform": "none_shared_mask" if shared
            else "orthogonal_change_of_basis_per_boundary",
            "handoff_skip_term_needs_gemm": not shared,
            "handoff_offline_fusable_except_skip": not shared,
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
# Mixed-precision helpers (Stage 8.2)
# ---------------------------------------------------------------------------


def _cast_folded(folded: dict[str, Any],
                 dtype: torch.dtype) -> dict[str, Any]:
    """Cast every tensor entry of a folded-weight dict to ``dtype``."""
    return {k: (v.to(dtype=dtype) if isinstance(v, torch.Tensor) else v)
            for k, v in folded.items()}


def _cast_vocab_mask(vm: VocabLogitMask, dtype: torch.dtype) -> VocabLogitMask:
    """Vocab mask with scale/inverse_scale cast to ``dtype`` (perm stays
    integer). Used to keep the permutation+scaling recovery in fp32."""
    return VocabLogitMask(
        permutation=vm.permutation, inverse_permutation=vm.inverse_permutation,
        scale=vm.scale.to(dtype=dtype),
        inverse_scale=vm.inverse_scale.to(dtype=dtype))


def _logit_diagnostics(
    logits_plain: torch.Tensor, logits_recovered: torch.Tensor,
    max_abs_err: float,
) -> dict[str, Any]:
    """Scalar-only logit drift diagnostics (no tensor dumps). All in the dtype
    of the inputs (callers pass fp32)."""
    diff = (logits_recovered - logits_plain).abs()
    denom = float(logits_plain.float().pow(2).sum().sqrt().item()) or 1.0
    rel_l2 = float((logits_recovered - logits_plain).float().pow(2).sum()
                   .sqrt().item()) / denom
    # top-1 margin per position over [B, T]: top1 - top2 of the plain logits.
    flat = logits_plain.reshape(-1, logits_plain.shape[-1]).float()
    top2 = flat.topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1])
    below = int((margins < max_abs_err).sum().item())
    return {
        "recovered_logits_mean_abs_err": float(diff.float().mean().item()),
        "recovered_logits_relative_l2_err": rel_l2,
        "top1_margin_stats": {
            "min_margin": float(margins.min().item()),
            "mean_margin": float(margins.mean().item()),
            "num_positions_with_margin_below_error": below,
            "num_positions": int(margins.numel()),
        },
    }


def _masked_output_boundary(
    h_L_plain: torch.Tensor, h_L_tilde: torch.Tensor,
    bw: CausalLMBoundaryWeights, n_res_inv: torch.Tensor,
    vocab_mask: VocabLogitMask, eps: float,
    recovery_dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Final norm + masked LM head + TEE recovery, optionally upcast to
    ``recovery_dtype`` (e.g. float32) so the inverse/scaling recovery never
    runs in bf16. ``recovery_dtype is None`` reproduces the single-dtype path
    exactly."""
    if recovery_dtype is None or recovery_dtype == h_L_tilde.dtype:
        out = final_norm_lm_head_masked(
            h_L_tilde, h_L_plain, bw, n_res_inv, vocab_mask, eps)
        return out
    rb = CausalLMBoundaryWeights(
        embed_tokens_weight=bw.embed_tokens_weight,
        final_norm_weight=bw.final_norm_weight.to(recovery_dtype),
        lm_head_weight=bw.lm_head_weight.to(recovery_dtype))
    out = final_norm_lm_head_masked(
        h_L_tilde.to(recovery_dtype), h_L_plain.to(recovery_dtype), rb,
        n_res_inv.to(recovery_dtype), _cast_vocab_mask(vocab_mask, recovery_dtype),
        eps)
    return out


# ---------------------------------------------------------------------------
# Negative controls (Stage 8.2 audit)
# ---------------------------------------------------------------------------

NEGATIVE_CONTROLS = (
    "none", "wrong_vocab_recovery", "plaintext_weights_on_masked_hidden",
)


def _wrong_vocab_mask(vm: VocabLogitMask, seed: int) -> VocabLogitMask:
    """An intentionally WRONG vocab mask (independent random permutation +
    scale). Its inverse does not invert the real masking, so recovering masked
    logits with it scrambles the vocabulary -> token mismatch."""
    vocab = int(vm.permutation.shape[0])
    g = torch.Generator(device=vm.permutation.device).manual_seed(seed)
    return make_vocab_logit_mask(vocab, vm.scale.dtype, vm.permutation.device, g)


def _negative_control_recovered_logits(
    negative_control: str, logits_tilde: torch.Tensor, h_tilde: torch.Tensor,
    plain_logits: torch.Tensor, bw: CausalLMBoundaryWeights,
    vm_rec: VocabLogitMask, eps: float, seed: int,
) -> tuple[torch.Tensor, float]:
    """Recover masked logits under the requested negative control.

    Returns ``(recovered_logits, recovered_logits_max_abs_error_vs_plain)``.
    For ``none`` this is the verified TEE recovery (error ~ 0); the negative
    controls deliberately produce wrong logits (large error)."""
    nc = negative_control or "none"
    dt = plain_logits.dtype
    if nc == "wrong_vocab_recovery":
        wrong = _wrong_vocab_mask(vm_rec, seed)
        rec = recover_vocab_logits(logits_tilde, wrong)
    elif nc == "plaintext_weights_on_masked_hidden":
        # Folding-disabled output path: treat the *masked* hidden as if it were
        # plaintext and run the plain final-norm + LM head. No de-masking ->
        # wrong logits.
        fn = final_norm_lm_head_plain(
            h_tilde.to(dt), bw.final_norm_weight.to(dt),
            bw.lm_head_weight.to(dt), eps)
        rec = fn["logits"]
    elif nc == "none":
        rec = recover_vocab_logits(logits_tilde, vm_rec)
    else:
        raise ValueError(f"unknown negative_control {negative_control!r}; "
                         f"expected one of {NEGATIVE_CONTROLS}")
    rec = rec.to(dt)
    err = float((rec - plain_logits).abs().max().item())
    return rec, err


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

    # Fold in `dtype` (folding precision); optionally cast the folded weights
    # to a lower runtime precision (e.g. bf16) for the masked forward while the
    # plain reference stays in `dtype`.
    foldeds = [
        fold_hf_single_block_weights(weights.layer_weights[ell],
                                     layer_configs[ell],
                                     masks.layer_block_masks[ell])
        for ell in range(n_layers)
    ]
    rt_dtype = config.folded_runtime_dtype
    runtime_cast = rt_dtype is not None and rt_dtype != dtype
    cos_m, sin_m = (cos.to(rt_dtype), sin.to(rt_dtype)) if runtime_cast \
        else (cos, sin)
    if runtime_cast:
        foldeds = [_cast_folded(f, rt_dtype) for f in foldeds]

    h_tilde = emb["x_tilde"]                        # H0_tilde in N_0
    if runtime_cast:
        h_tilde = h_tilde.to(rt_dtype)
    hidden_tilde_by_layer = [h_tilde]
    caches_tilde: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    for ell in range(n_layers):
        bm = masks.layer_block_masks[ell]
        attn_masks = bm["attn"]
        n_ell = masks.residual_masks[ell]
        blk = _hf_masked_block_prefill(h_tilde, foldeds[ell],
                                       layer_configs[ell], cos_m, sin_m)
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
        # Skipped entirely for a shared residual mask (handoff == identity).
        if masks.needs_handoff(ell):
            h_tilde = blk["y_tilde"] @ masks.handoff(ell).to(
                blk["y_tilde"].dtype)
        else:
            h_tilde = blk["y_tilde"]
        hidden_tilde_by_layer.append(h_tilde)

    handoff_errs = [
        _mx(hidden_tilde_by_layer[ell],
            plain["hidden_by_layer_plain"][ell] @ masks.residual_masks[ell])
        for ell in range(n_layers + 1)
    ]

    h_L_plain = plain["hidden_by_layer_plain"][-1]
    h_L_tilde = hidden_tilde_by_layer[-1]
    final_hidden_err = _mx(h_L_tilde, h_L_plain @ masks.residual_masks[-1])

    rec_dtype = config.recovery_dtype
    out = _masked_output_boundary(
        h_L_plain, h_L_tilde, bw, masks.residual_mask_inverses[-1],
        masks.vocab_mask, eps, rec_dtype)
    vm_rec = (masks.vocab_mask if rec_dtype is None
              else _cast_vocab_mask(masks.vocab_mask, rec_dtype))
    nc = getattr(config, "negative_control", "none") or "none"
    recovered_logits, nc_recovered_err = _negative_control_recovered_logits(
        nc, out["logits_tilde"], h_L_tilde, out["logits_plain"], bw, vm_rec,
        eps, config.seed + 777)
    next_token_from_masked = greedy_sample(recovered_logits[:, -1, :])
    greedy_match = float(
        (plain["next_token_plain"] == next_token_from_masked)
        .to(torch.float32).mean().item())

    recovered_err = (out["metrics"]["recovered_logits_max_abs_error"]
                     if nc == "none" else nc_recovered_err)
    diagnostics = _logit_diagnostics(
        out["logits_plain"], recovered_logits, recovered_err)
    diagnostics["embedding_boundary_max_abs_err"] = embedding_mask_err
    diagnostics["layer_0_input_invariant_max_abs_err"] = handoff_errs[0]
    diagnostics["layer_0_output_invariant_max_abs_err"] = (
        per_layer[0]["final_output_max_abs_error"] if per_layer else 0.0)
    diagnostics["final_norm_core_max_abs_err"] = \
        out["metrics"]["final_norm_core_max_abs_error"]
    diagnostics["masked_logits_max_abs_err"] = \
        out["metrics"]["masked_logits_max_abs_error"]
    diagnostics["recovered_logits_max_abs_err"] = recovered_err
    diagnostics["greedy_token_match_rate"] = greedy_match

    metrics = {
        "embedding_mask_max_abs_error": embedding_mask_err,
        "per_layer": per_layer,
        "per_layer_handoff_max_abs_error": handoff_errs,
        "final_hidden_max_abs_error": final_hidden_err,
        "masked_logits_max_abs_error":
            out["metrics"]["masked_logits_max_abs_error"],
        "recovered_logits_max_abs_error": recovered_err,
        "greedy_token_match_rate": greedy_match,
        "negative_control": nc,
        "diagnostics": diagnostics,
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

    rt_dtype = config.folded_runtime_dtype
    rec_dtype = config.recovery_dtype
    runtime_cast = rt_dtype is not None and rt_dtype != dtype
    nc = getattr(config, "negative_control", "none") or "none"

    pre = hf_causal_lm_masked_prefill(input_ids, weights, layer_configs, masks,
                                      config)
    cos, sin, foldeds = pre["cos"], pre["sin"], pre["foldeds"]
    cos_m, sin_m = (cos.to(rt_dtype), sin.to(rt_dtype)) if runtime_cast \
        else (cos, sin)
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
        if runtime_cast:
            h_tilde = h_tilde.to(rt_dtype)

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
                                            cos_m, sin_m, position)
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
            h_tilde = (dec_m["y_tilde"] @ masks.handoff(ell).to(
                dec_m["y_tilde"].dtype)
                if masks.needs_handoff(ell) else dec_m["y_tilde"])

        final_hidden_err = _mx(h_tilde, h_plain @ masks.residual_masks[-1])
        fn_plain = final_norm_lm_head_plain(h_plain, bw.final_norm_weight,
                                            bw.lm_head_weight, eps)
        out = _masked_output_boundary(
            h_plain, h_tilde, bw, masks.residual_mask_inverses[-1],
            masks.vocab_mask, eps, rec_dtype)
        vm_rec = (masks.vocab_mask if rec_dtype is None
                  else _cast_vocab_mask(masks.vocab_mask, rec_dtype))
        masked_logits_err = out["metrics"]["masked_logits_max_abs_error"]
        recovered_logits, nc_rec_err = _negative_control_recovered_logits(
            nc, out["logits_tilde"], h_tilde, out["logits_plain"], bw, vm_rec,
            eps, config.seed + 777)
        recovered_logits_err = (
            out["metrics"]["recovered_logits_max_abs_error"]
            if nc == "none" else nc_rec_err)
        tok_plain = greedy_sample(fn_plain["logits"][:, -1, :])
        tok_masked = greedy_sample(recovered_logits[:, -1, :])
        step_metrics.append({
            "step": step, "position": position,
            "final_hidden_error": final_hidden_err,
            "masked_logits_error": masked_logits_err,
            "recovered_logits_error": recovered_logits_err,
            "sampled_token_match": float(
                (tok_plain == tok_masked).to(torch.float32).mean().item()),
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
        (gen_plain_t == gen_masked_t).to(torch.float32).mean().item())

    return {
        "generated_plain_tokens": gen_plain_t,
        "generated_from_masked_tokens": gen_masked_t,
        "token_match_rate": token_match_rate,
        "prefill_metrics": pre["metrics"],
        "decode_step_metrics": step_metrics,
        "metadata": pre["metadata"],
    }


def hf_causal_lm_masked_only_decode(
    input_ids: torch.Tensor, weights: HFCausalLMSkeletonWeights,
    layer_configs: list[HFSingleBlockConfig], masks: HFCausalLMMaskBundle,
    config: HFCausalLMSkeletonConfig, decode_steps: int | None = None,
) -> dict[str, Any]:
    """Masked runtime ONLY (no plaintext reference, no error metrics).

    This is the honest deployment cost: trusted embedding boundary -> masked
    decoder blocks (folded weights, masked KV cache) -> masked LM head ->
    trusted vocab recovery + greedy. Use it to time the masked path without
    the diagnostic plain-reference recompute that the verification decode does.
    """
    decode_steps = config.decode_steps if decode_steps is None else decode_steps
    cfg0 = layer_configs[0]
    eps = cfg0.rms_norm_eps
    dtype = cfg0.dtype
    n_layers = len(layer_configs)
    bw = _boundary_weights(weights)
    cos, sin = _rope_cache(config, cfg0)

    rt_dtype = config.folded_runtime_dtype
    rec_dtype = config.recovery_dtype
    runtime_cast = rt_dtype is not None and rt_dtype != dtype

    foldeds = [
        fold_hf_single_block_weights(weights.layer_weights[ell],
                                     layer_configs[ell],
                                     masks.layer_block_masks[ell])
        for ell in range(n_layers)
    ]
    cos_m, sin_m = (cos.to(rt_dtype), sin.to(rt_dtype)) if runtime_cast \
        else (cos, sin)
    if runtime_cast:
        foldeds = [_cast_folded(f, rt_dtype) for f in foldeds]

    # Offline-foldable masked LM head (final-norm affine + N^{-1} + vocab mask).
    n_res_inv_last = masks.residual_mask_inverses[-1]
    vm = masks.vocab_mask
    if rec_dtype is not None:
        head_fn_w = bw.final_norm_weight.to(rec_dtype)
        head_lm_w = bw.lm_head_weight.to(rec_dtype)
        head_ninv = n_res_inv_last.to(rec_dtype)
        vm_rec = _cast_vocab_mask(vm, rec_dtype)
    else:
        head_fn_w, head_lm_w = bw.final_norm_weight, bw.lm_head_weight
        head_ninv, vm_rec = n_res_inv_last, vm
    w_lm_tilde = fold_final_norm_lm_head_with_vocab_mask(
        head_fn_w, head_lm_w, head_ninv, vm_rec)

    def _masked_head(h_tilde: torch.Tensor) -> torch.Tensor:
        core_tilde = rmsnorm_core(
            h_tilde if rec_dtype is None else h_tilde.to(rec_dtype), eps)
        logits_tilde = core_tilde @ w_lm_tilde            # GPU sees this only
        return recover_vocab_logits(logits_tilde, vm_rec)  # TEE recovery

    # Prefill (masked) --------------------------------------------------
    emb = embedding_boundary_forward(input_ids, bw, masks.residual_masks[0],
                                     masks.input_pad)
    h_tilde = emb["x_tilde"]
    if runtime_cast:
        h_tilde = h_tilde.to(rt_dtype)
    caches_tilde: list[dict[str, Any]] = []
    for ell in range(n_layers):
        blk = _hf_masked_block_prefill(h_tilde, foldeds[ell],
                                       layer_configs[ell], cos_m, sin_m)
        caches_tilde.append(blk["cache"])
        h_tilde = (blk["y_tilde"] @ masks.handoff(ell).to(blk["y_tilde"].dtype)
                   if masks.needs_handoff(ell) else blk["y_tilde"])
    next_masked = greedy_sample(_masked_head(h_tilde)[:, -1, :])
    gen = [next_masked]

    # Decode (masked) ---------------------------------------------------
    n0 = masks.residual_masks[0]
    pad = masks.input_pad
    for step in range(decode_steps):
        position = config.prefill_seq_len + step
        x_next = trusted_embedding_lookup(
            next_masked, weights.embed_tokens_weight).unsqueeze(1)
        x0 = x_next if pad is None else x_next - pad
        h_tilde = x0 @ n0
        if runtime_cast:
            h_tilde = h_tilde.to(rt_dtype)
        for ell in range(n_layers):
            dec_m = _hf_masked_block_decode(h_tilde, caches_tilde[ell],
                                            foldeds[ell], layer_configs[ell],
                                            cos_m, sin_m, position)
            caches_tilde[ell] = dec_m["cache"]
            h_tilde = (dec_m["y_tilde"] @ masks.handoff(ell).to(
                dec_m["y_tilde"].dtype)
                if masks.needs_handoff(ell) else dec_m["y_tilde"])
        next_masked = greedy_sample(_masked_head(h_tilde)[:, -1, :])
        gen.append(next_masked)

    return {
        "generated_from_masked_tokens": torch.stack(gen, dim=1),
        "num_tokens": len(gen),
    }
