"""Stage 6.6 -- real HF/ModelScope LLaMA/Qwen single-decoder-layer adapter.

Maps the Stage 6.5 synthetic-block masking design onto *real* HuggingFace
LLaMA/Qwen2 decoder-layer weights. The flow is:

1. introspect a HF ``LlamaDecoderLayer`` / ``Qwen2DecoderLayer`` to a config;
2. extract its weights into our row-vector convention (``x @ W + b``,
   transposing HF ``[out, in]`` Linear weights, preserving optional biases);
3. compute a plain reference forward **from the extracted weights** (not by
   calling the HF layer forward -- HF attention/RoPE internals vary across
   versions);
4. fold RMSNorm affine + residual orthogonal mask + per-head RoPE-compatible
   masks + SwiGLU permutation into the weights/biases and verify the masked
   path reproduces ``y_tilde == y_plain @ n_res``.

This validates the *adapter*, not a full model. No tokenizer, embedding,
LM head, sampling, or generation loop. No network download (transformers is
an optional dependency; loading is local-files-only). Adjacent-pair RoPE
(Stage 6.4) is used for both plain and masked paths, so the masked-vs-plain
invariant holds regardless of HF's internal RoPE convention. No formal,
cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from pllo.ops.gqa_attention import (
    block_diag_from_head_masks,
    merge_heads,
    repeat_kv,
    split_heads,
)
from pllo.ops.llama_synthetic_block import (
    SyntheticLlamaBlockConfig,
    generate_block_masks,
    rmsnorm_plain,
)
from pllo.ops.nonlinear_islands import rmsnorm_core, silu_reference
from pllo.ops.rope import apply_rope, build_rope_cache

__all__ = [
    "HFSingleBlockConfig",
    "HFSingleBlockWeights",
    "extract_hf_single_block_weights",
    "fold_hf_single_block_weights",
    "generate_hf_single_block_masks",
    "has_transformers",
    "hf_single_block_masked_decode",
    "hf_single_block_masked_prefill",
    "hf_single_block_plain_prefill",
    "infer_config_from_hf_layer",
    "make_random_hf_decoder_layer",
    "require_transformers_or_skip",
]


# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------


def has_transformers() -> bool:
    """True if the ``transformers`` package is importable."""
    import importlib.util

    return importlib.util.find_spec("transformers") is not None


def require_transformers_or_skip() -> Any:
    """Import + return ``transformers`` or raise a clear error.

    Tests should prefer ``pytest.importorskip('transformers')``; this helper
    is for non-test callers that want an explicit message.
    """
    if not has_transformers():
        raise RuntimeError(
            "transformers is not installed; this is an optional dependency "
            "for Stage 6.6 (HF single-block adapter)."
        )
    import transformers

    return transformers


# ---------------------------------------------------------------------------
# Config + weights
# ---------------------------------------------------------------------------


@dataclass
class HFSingleBlockConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    mask_family: str = "pairwise_complex_scaling"
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def validate(self) -> None:
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (RoPE adjacent pairs)")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_key_value_heads")


@dataclass
class HFSingleBlockWeights:
    input_layernorm_weight: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    q_proj_weight: torch.Tensor
    k_proj_weight: torch.Tensor
    v_proj_weight: torch.Tensor
    o_proj_weight: torch.Tensor
    q_proj_bias: torch.Tensor | None
    k_proj_bias: torch.Tensor | None
    v_proj_bias: torch.Tensor | None
    o_proj_bias: torch.Tensor | None
    gate_proj_weight: torch.Tensor
    up_proj_weight: torch.Tensor
    down_proj_weight: torch.Tensor
    gate_proj_bias: torch.Tensor | None
    up_proj_bias: torch.Tensor | None
    down_proj_bias: torch.Tensor | None


# ---------------------------------------------------------------------------
# Introspection + extraction
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    val = getattr(obj, name, default)
    return default if val is None else val


def _read_rope_theta(mc: Any, default: float = 10000.0) -> float:
    """Read RoPE base/theta robustly across transformers versions.

    transformers >=5 moved ``rope_theta`` into the nested ``rope_parameters`` /
    ``rope_scaling`` dict (e.g. ``{"rope_theta": 1000000.0, "rope_type":
    "default"}``); older versions exposed it as a top-level attribute. Missing
    it silently defaults to 10000.0, which is WRONG for Qwen2.5 (1e6) and
    corrupts every position -- so check the nested dicts first."""
    if mc is None:
        return default
    for attr in ("rope_parameters", "rope_scaling"):
        d = getattr(mc, attr, None)
        if isinstance(d, dict) and d.get("rope_theta") is not None:
            return float(d["rope_theta"])
    v = getattr(mc, "rope_theta", None)
    return float(v) if v is not None else default


def infer_config_from_hf_layer(
    layer: Any, model_config: Any = None,
    dtype: torch.dtype = torch.float64, device: str = "cpu",
    mask_family: str = "pairwise_complex_scaling",
) -> HFSingleBlockConfig:
    """Infer an :class:`HFSingleBlockConfig` from a HF decoder layer.

    ``model_config`` is preferred for head counts; falls back to the
    config stored on the attention module (``layer.self_attn.config``).
    """
    attn = layer.self_attn
    mc = model_config if model_config is not None else getattr(
        attn, "config", None)

    q_proj = attn.q_proj
    k_proj = attn.k_proj
    gate_proj = layer.mlp.gate_proj

    hidden_size = q_proj.in_features
    q_out = q_proj.out_features
    k_out = k_proj.out_features
    intermediate_size = gate_proj.out_features

    num_heads = int(_get(mc, "num_attention_heads", 0)) if mc is not None else 0
    head_dim = int(_get(attn, "head_dim", 0))
    if head_dim == 0 and mc is not None:
        head_dim = int(_get(mc, "head_dim", 0))
    if num_heads == 0 and head_dim:
        num_heads = q_out // head_dim
    if head_dim == 0 and num_heads:
        head_dim = q_out // num_heads
    if num_heads == 0 or head_dim == 0:
        raise ValueError(
            "cannot infer num_heads/head_dim; pass model_config")

    num_kv = int(_get(mc, "num_key_value_heads", 0)) if mc is not None else 0
    if num_kv == 0:
        num_kv = k_out // head_dim

    model_type = str(_get(mc, "model_type", "unknown_llama_qwen_like")) \
        if mc is not None else "unknown_llama_qwen_like"
    rope_theta = _read_rope_theta(mc)
    rms_eps = float(_get(mc, "rms_norm_eps", 1e-5)) if mc is not None else 1e-5

    return HFSingleBlockConfig(
        model_type=model_type,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        rope_theta=rope_theta,
        rms_norm_eps=rms_eps,
        attention_bias=q_proj.bias is not None,
        mlp_bias=gate_proj.bias is not None,
        mask_family=mask_family,
        dtype=dtype,
        device=device,
    )


def _extract_linear(module: Any, dtype: torch.dtype, device: torch.device,
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """HF Linear ``[out, in]`` -> row-vector ``W [in, out]`` + optional bias."""
    w = module.weight.detach().to(device=device, dtype=dtype).t().contiguous()
    b = None
    if getattr(module, "bias", None) is not None:
        b = module.bias.detach().to(device=device, dtype=dtype).clone()
    return w, b


def extract_hf_single_block_weights(
    layer: Any, dtype: torch.dtype = torch.float64, device: str = "cpu",
) -> HFSingleBlockWeights:
    """Extract (clone, detach, transpose, cast) weights; never mutate layer."""
    dev = torch.device(device)
    attn = layer.self_attn
    wq, bq = _extract_linear(attn.q_proj, dtype, dev)
    wk, bk = _extract_linear(attn.k_proj, dtype, dev)
    wv, bv = _extract_linear(attn.v_proj, dtype, dev)
    wo, bo = _extract_linear(attn.o_proj, dtype, dev)
    wg, bg = _extract_linear(layer.mlp.gate_proj, dtype, dev)
    wu, bu = _extract_linear(layer.mlp.up_proj, dtype, dev)
    wd, bd = _extract_linear(layer.mlp.down_proj, dtype, dev)
    rms1 = layer.input_layernorm.weight.detach().to(
        device=dev, dtype=dtype).clone()
    rms2 = layer.post_attention_layernorm.weight.detach().to(
        device=dev, dtype=dtype).clone()
    return HFSingleBlockWeights(
        input_layernorm_weight=rms1,
        post_attention_layernorm_weight=rms2,
        q_proj_weight=wq, k_proj_weight=wk, v_proj_weight=wv, o_proj_weight=wo,
        q_proj_bias=bq, k_proj_bias=bk, v_proj_bias=bv, o_proj_bias=bo,
        gate_proj_weight=wg, up_proj_weight=wu, down_proj_weight=wd,
        gate_proj_bias=bg, up_proj_bias=bu, down_proj_bias=bd,
    )


# ---------------------------------------------------------------------------
# Plain reference forward (from extracted weights)
# ---------------------------------------------------------------------------


def _linear(x: torch.Tensor, w: torch.Tensor,
            b: torch.Tensor | None,
            xpad: torch.Tensor | None = None,
            cpad: torch.Tensor | None = None) -> torch.Tensor:
    """Folded Linear ``y = x @ w (+ b)`` with OPTIONAL Linear-boundary additive
    input padding. When ``xpad`` (= ``T N_in``) and ``cpad`` (= ``T W N_out``) are
    given, the GPU matmul operand becomes ``(x - xpad)`` -- i.e. the masked padded
    input view ``(X - T) N_in`` -- and ``cpad`` is added back so the output returns
    to the compatible masked basis ``Y N_out`` (algebraically unchanged:
    ``xpad @ w == cpad``). ``xpad``/``cpad`` are precomputed composed offsets; the
    runtime cost is a fused broadcast subtract + add (no extra matmul). Defaults
    (None) keep the historical mask-only path byte-for-byte."""
    xin = x if xpad is None else x - xpad
    out = xin @ w
    if b is not None:
        out = out + b
    if cpad is not None:
        out = out + cpad
    return out


def _pad(folded: dict[str, Any], weight_key: str) -> tuple:
    """Return ``(xpad, cpad)`` for a folded weight key from the layer dict (both
    None when the package was built mask-only)."""
    base = weight_key[:-len("_tilde")]
    return folded.get(base + "_xpad_tilde"), folded.get(base + "_cpad_tilde")


def _causal_bias(t_q: int, t_k: int, dtype: torch.dtype,
                 device: torch.device, offset: int) -> torch.Tensor:
    q_pos = torch.arange(offset, offset + t_q, device=device).unsqueeze(1)
    k_pos = torch.arange(t_k, device=device).unsqueeze(0)
    bias = torch.zeros(t_q, t_k, dtype=dtype, device=device)
    bias.masked_fill_(k_pos > q_pos, float("-inf"))
    return bias


def _sdpa(qr: torch.Tensor, kr_rep: torch.Tensor, v_rep: torch.Tensor,
          scale: float, causal_offset: int | None, runner: Any = None,
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = qr @ kr_rep.transpose(-2, -1) * scale
    if causal_offset is not None:
        scores = scores + _causal_bias(
            qr.shape[-2], kr_rep.shape[-2], scores.dtype, scores.device,
            causal_offset)
    # ``runner`` (optional) dispatches the softmax through the selected nonlinear
    # design (design B migrates it onto the accelerator with a trusted row-max
    # shortcut); default None keeps the historical ``torch.softmax`` path exactly.
    probs = runner.softmax(scores, dim=-1) if runner is not None \
        else torch.softmax(scores, dim=-1)
    # Align dtype at the attention matmul boundary only. The nonlinear ``runner``
    # softmax (design B) may return ``probs`` in float32 while ``v_rep`` stays
    # bf16, which raises "expected scalar type Float but found BFloat16". Casting
    # ``v_rep`` to ``probs.dtype`` is a no-op when they already match (same-dtype
    # ``.to`` returns the same tensor), so the float32 path stays bitwise-identical
    # and the pure-bf16 path is numerically unchanged; it only repairs the mixed
    # path. No change to softmax / masking / projections / output semantics.
    av = probs @ v_rep.to(probs.dtype)
    return scores, probs, av


def hf_single_block_plain_prefill(
    x: torch.Tensor, weights: HFSingleBlockWeights,
    config: HFSingleBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
) -> dict[str, Any]:
    """Plain decoder-block forward computed from extracted weights."""
    eps = config.rms_norm_eps
    nh, nkv, hd = config.num_heads, config.num_key_value_heads, config.head_dim
    scale = 1.0 / math.sqrt(hd)

    r1_core = rmsnorm_core(x, eps)
    r1 = r1_core * weights.input_layernorm_weight
    q = split_heads(_linear(r1, weights.q_proj_weight, weights.q_proj_bias), nh)
    k = split_heads(_linear(r1, weights.k_proj_weight, weights.k_proj_bias), nkv)
    v = split_heads(_linear(r1, weights.v_proj_weight, weights.v_proj_bias), nkv)
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    scores, probs, av = _sdpa(qr, repeat_kv(kr, nh, nkv),
                              repeat_kv(v, nh, nkv), scale, causal_offset=0)
    attn_out = _linear(merge_heads(av), weights.o_proj_weight,
                       weights.o_proj_bias)
    x1 = x + attn_out
    r2_core = rmsnorm_core(x1, eps)
    r2 = r2_core * weights.post_attention_layernorm_weight
    gate = _linear(r2, weights.gate_proj_weight, weights.gate_proj_bias)
    up = _linear(r2, weights.up_proj_weight, weights.up_proj_bias)
    hidden = silu_reference(gate) * up
    mlp_out = _linear(hidden, weights.down_proj_weight, weights.down_proj_bias)
    y = x1 + mlp_out
    return {
        "r1_core": r1_core, "q": q, "k": k, "v": v, "q_rope": qr, "k_rope": kr,
        "scores": scores, "probs": probs, "av": av, "attn_out": attn_out,
        "x1": x1, "r2_core": r2_core, "gate": gate, "up": up, "hidden": hidden,
        "mlp_out": mlp_out, "y": y,
        "cache_plain": {"key_rope": kr, "value": v},
    }


# ---------------------------------------------------------------------------
# Masks + folded weights
# ---------------------------------------------------------------------------


def _as_block_config(config: HFSingleBlockConfig) -> SyntheticLlamaBlockConfig:
    return SyntheticLlamaBlockConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_heads=config.num_heads,
        num_key_value_heads=config.num_key_value_heads,
        rope_base=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        mask_family=config.mask_family,
        dtype=config.dtype,
        device=config.device,
    )


def generate_hf_single_block_masks(
    config: HFSingleBlockConfig, seed: int = 2029,
) -> dict[str, Any]:
    """Reuse the Stage 6.5 block-mask generator (orthogonal ``n_res``,
    RoPE-compatible attention masks, SwiGLU permutation)."""
    config.validate()
    g = torch.Generator(device=torch.device(config.device)).manual_seed(seed)
    return generate_block_masks(_as_block_config(config), g)


def fold_hf_single_block_weights(
    weights: HFSingleBlockWeights, config: HFSingleBlockConfig,
    masks: dict[str, Any],
) -> dict[str, Any]:
    """Fold affine + residual mask + per-head masks + permutation (with bias)."""
    n_res = masks["n_res"]
    n_res_inv = masks["n_res_inv"]
    attn = masks["attn"]
    perm = masks["perm"]

    mq_block = block_diag_from_head_masks(attn["q_masks"])
    mk_block = block_diag_from_head_masks(attn["key_masks"])
    mv_block = block_diag_from_head_masks(attn["value_masks"])
    v_inv_qhead = attn["value_mask_inverses"].index_select(0, attn["kv_index"])
    sv_block_inv = block_diag_from_head_masks(v_inv_qhead)

    rms1 = weights.input_layernorm_weight.unsqueeze(1)
    rms2 = weights.post_attention_layernorm_weight.unsqueeze(1)

    def maybe(b: torch.Tensor | None, m: torch.Tensor) -> torch.Tensor | None:
        return None if b is None else b @ m

    return {
        "wq_tilde": n_res_inv @ (rms1 * weights.q_proj_weight) @ mq_block,
        "wk_tilde": n_res_inv @ (rms1 * weights.k_proj_weight) @ mk_block,
        "wv_tilde": n_res_inv @ (rms1 * weights.v_proj_weight) @ mv_block,
        "wo_tilde": sv_block_inv @ weights.o_proj_weight @ n_res,
        "bq_tilde": maybe(weights.q_proj_bias, mq_block),
        "bk_tilde": maybe(weights.k_proj_bias, mk_block),
        "bv_tilde": maybe(weights.v_proj_bias, mv_block),
        "bo_tilde": maybe(weights.o_proj_bias, n_res),
        "wgate_tilde": n_res_inv @ (rms2 * weights.gate_proj_weight).index_select(1, perm),
        "wup_tilde": n_res_inv @ (rms2 * weights.up_proj_weight).index_select(1, perm),
        "wdown_tilde": weights.down_proj_weight.index_select(0, perm) @ n_res,
        "bgate_tilde": None if weights.gate_proj_bias is None
        else weights.gate_proj_bias.index_select(0, perm),
        "bup_tilde": None if weights.up_proj_bias is None
        else weights.up_proj_bias.index_select(0, perm),
        "bdown_tilde": maybe(weights.down_proj_bias, n_res),
    }


# ---------------------------------------------------------------------------
# Masked forward
# ---------------------------------------------------------------------------


def _mx(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _finite_score_err(a: torch.Tensor, b: torch.Tensor) -> float:
    finite = torch.isfinite(a)
    return float((a[finite] - b[finite]).abs().max().item())


def _apply_heads(x_heads: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hde->bhte", x_heads, m)


def _masked_attention(
    r1_core_tilde: torch.Tensor, folded: dict[str, Any],
    config: HFSingleBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
    *, causal_offset: int | None, position_ids: torch.Tensor | None = None,
    past_key_rope: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None, runner: Any = None,
) -> dict[str, Any]:
    nh, nkv, hd = config.num_heads, config.num_key_value_heads, config.head_dim
    scale = 1.0 / math.sqrt(hd)
    q_t = split_heads(_linear(r1_core_tilde, folded["wq_tilde"],
                              folded["bq_tilde"], *_pad(folded, "wq_tilde")), nh)
    k_t = split_heads(_linear(r1_core_tilde, folded["wk_tilde"],
                              folded["bk_tilde"], *_pad(folded, "wk_tilde")), nkv)
    v_t = split_heads(_linear(r1_core_tilde, folded["wv_tilde"],
                              folded["bv_tilde"], *_pad(folded, "wv_tilde")), nkv)
    qr_t = apply_rope(q_t, cos, sin, position_ids=position_ids)
    kr_t_new = apply_rope(k_t, cos, sin, position_ids=position_ids)
    if past_key_rope is not None:
        kr_t = torch.cat([past_key_rope, kr_t_new], dim=2)
        v_full = torch.cat([past_value, v_t], dim=2)
    else:
        kr_t, v_full = kr_t_new, v_t
    scores, probs, av = _sdpa(qr_t, repeat_kv(kr_t, nh, nkv),
                              repeat_kv(v_full, nh, nkv), scale, causal_offset,
                              runner=runner)
    out = _linear(merge_heads(av), folded["wo_tilde"], folded["bo_tilde"],
                  *_pad(folded, "wo_tilde"))
    return {
        "out": out, "q_pre_rope": q_t, "k_pre_rope": k_t, "v": v_t,
        "k_rope_new": kr_t_new, "scores": scores, "probs": probs, "av": av,
        "key_rope_full": kr_t, "value_full": v_full,
    }


def _masked_mlp(r2_core_tilde: torch.Tensor, folded: dict[str, Any],
                runner: Any = None) -> dict[str, torch.Tensor]:
    gate = _linear(r2_core_tilde, folded["wgate_tilde"], folded["bgate_tilde"],
                   *_pad(folded, "wgate_tilde"))
    up = _linear(r2_core_tilde, folded["wup_tilde"], folded["bup_tilde"],
                 *_pad(folded, "wup_tilde"))
    # The SwiGLU activation is the MLP nonlinear island. ``runner`` (optional)
    # dispatches the SiLU through the selected design: design B *lifts* it onto
    # the accelerator (exact after the folded squeeze); default None keeps the
    # historical ``silu_reference`` inline path exactly. The additive pad does NOT
    # enter the activation: gate/up are de-padded to ``Y N_out`` BEFORE SiLU, and
    # the down-projection re-pads its OWN input view independently.
    act = runner.silu(gate) if runner is not None else silu_reference(gate)
    hidden = act * up
    out = _linear(hidden, folded["wdown_tilde"], folded["bdown_tilde"],
                  *_pad(folded, "wdown_tilde"))
    return {"gate": gate, "up": up, "hidden": hidden, "out": out}


def hf_single_block_masked_prefill(
    x: torch.Tensor, weights: HFSingleBlockWeights,
    config: HFSingleBlockConfig, masks: dict[str, Any],
    decode_steps: int = 2,
) -> dict[str, Any]:
    """Masked single-block prefill from extracted weights + invariants."""
    eps = config.rms_norm_eps
    dtype, device = config.dtype, torch.device(config.device)
    n_res, perm, attn_masks = masks["n_res"], masks["perm"], masks["attn"]
    folded = fold_hf_single_block_weights(weights, config, masks)

    seq_len = x.shape[1]
    max_pos = seq_len + decode_steps + 1
    cos, sin = build_rope_cache(max_pos, config.head_dim, config.rope_theta,
                                dtype, device)
    plain = hf_single_block_plain_prefill(x, weights, config, cos, sin)

    x_tilde = x @ n_res
    r1_core_tilde = rmsnorm_core(x_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, config, cos, sin,
                          causal_offset=0)
    x1_tilde = x_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    mlp = _masked_mlp(r2_core_tilde, folded)
    y_tilde = x1_tilde + mlp["out"]

    v_masks_qhead = attn_masks["value_masks"].index_select(
        0, attn_masks["kv_index"])
    exp_y = plain["y"] @ n_res
    metrics = {
        "rms1_core_max_abs_error": _mx(r1_core_tilde, plain["r1_core"] @ n_res),
        "q_mask_max_abs_error": _mx(a["q_pre_rope"],
                                    _apply_heads(plain["q"], attn_masks["q_masks"])),
        "k_mask_max_abs_error": _mx(a["k_pre_rope"],
                                    _apply_heads(plain["k"], attn_masks["key_masks"])),
        "v_mask_max_abs_error": _mx(a["v"],
                                    _apply_heads(plain["v"], attn_masks["value_masks"])),
        "attention_score_max_abs_error": _finite_score_err(plain["scores"],
                                                           a["scores"]),
        "attention_prob_max_abs_error": _mx(plain["probs"], a["probs"]),
        "attention_output_max_abs_error": _mx(a["out"], plain["attn_out"] @ n_res),
        "residual1_max_abs_error": _mx(x1_tilde, plain["x1"] @ n_res),
        "rms2_core_max_abs_error": _mx(r2_core_tilde, plain["r2_core"] @ n_res),
        "gate_max_abs_error": _mx(mlp["gate"], plain["gate"].index_select(-1, perm)),
        "up_max_abs_error": _mx(mlp["up"], plain["up"].index_select(-1, perm)),
        "swiglu_hidden_max_abs_error": _mx(mlp["hidden"],
                                           plain["hidden"].index_select(-1, perm)),
        "mlp_output_max_abs_error": _mx(mlp["out"], plain["mlp_out"] @ n_res),
        "final_output_max_abs_error": _mx(y_tilde, exp_y),
        "cache_key_max_abs_error": _mx(a["key_rope_full"],
                                       _apply_heads(plain["cache_plain"]["key_rope"],
                                                    attn_masks["key_masks"])),
        "cache_value_max_abs_error": _mx(a["value_full"],
                                         _apply_heads(plain["cache_plain"]["value"],
                                                      attn_masks["value_masks"])),
    }
    metrics["allclose"] = all(v <= 1e-8 for v in metrics.values()
                              if isinstance(v, float))

    cache_tilde = {"key_rope_tilde": a["key_rope_full"],
                   "value_tilde": a["value_full"], "folded": folded,
                   "seq_len": seq_len}
    cache_plain = {"key_rope": plain["cache_plain"]["key_rope"],
                   "value": plain["cache_plain"]["value"], "seq_len": seq_len}
    return {
        "y_plain": plain["y"], "y_tilde": y_tilde, "expected_y_tilde": exp_y,
        "cache_tilde": cache_tilde, "cache_plain": cache_plain,
        "metrics": metrics,
        "metadata": _security_metadata(config),
    }


def hf_single_block_masked_decode(
    x_new: torch.Tensor, cache_tilde: dict[str, Any],
    cache_plain: dict[str, Any], weights: HFSingleBlockWeights,
    config: HFSingleBlockConfig, masks: dict[str, Any], position: int,
) -> dict[str, Any]:
    """One masked decode step at absolute ``position`` (== past length)."""
    eps = config.rms_norm_eps
    dtype, device = config.dtype, torch.device(config.device)
    n_res, attn_masks = masks["n_res"], masks["attn"]
    folded = cache_tilde.get("folded") or fold_hf_single_block_weights(
        weights, config, masks)
    nh, nkv, hd = config.num_heads, config.num_key_value_heads, config.head_dim
    scale = 1.0 / math.sqrt(hd)
    cos, sin = build_rope_cache(position + 2, hd, config.rope_theta, dtype,
                               device)
    pid = torch.tensor([position], device=device)

    # plain decode
    r1 = rmsnorm_plain(x_new, weights.input_layernorm_weight, eps)
    q = split_heads(_linear(r1, weights.q_proj_weight, weights.q_proj_bias), nh)
    k = split_heads(_linear(r1, weights.k_proj_weight, weights.k_proj_bias), nkv)
    v = split_heads(_linear(r1, weights.v_proj_weight, weights.v_proj_bias), nkv)
    qr = apply_rope(q, cos, sin, position_ids=pid)
    kr = apply_rope(k, cos, sin, position_ids=pid)
    kr_full = torch.cat([cache_plain["key_rope"], kr], dim=2)
    v_full = torch.cat([cache_plain["value"], v], dim=2)
    _, _, av = _sdpa(qr, repeat_kv(kr_full, nh, nkv), repeat_kv(v_full, nh, nkv),
                     scale, causal_offset=None)
    attn_out = _linear(merge_heads(av), weights.o_proj_weight,
                       weights.o_proj_bias)
    x1 = x_new + attn_out
    r2 = rmsnorm_plain(x1, weights.post_attention_layernorm_weight, eps)
    gate = _linear(r2, weights.gate_proj_weight, weights.gate_proj_bias)
    up = _linear(r2, weights.up_proj_weight, weights.up_proj_bias)
    hidden = silu_reference(gate) * up
    mlp_out = _linear(hidden, weights.down_proj_weight, weights.down_proj_bias)
    y_new = x1 + mlp_out

    # masked decode
    x_new_tilde = x_new @ n_res
    r1_core_tilde = rmsnorm_core(x_new_tilde, eps)
    a = _masked_attention(r1_core_tilde, folded, config, cos, sin,
                          causal_offset=None, position_ids=pid,
                          past_key_rope=cache_tilde["key_rope_tilde"],
                          past_value=cache_tilde["value_tilde"])
    x1_tilde = x_new_tilde + a["out"]
    r2_core_tilde = rmsnorm_core(x1_tilde, eps)
    mlp = _masked_mlp(r2_core_tilde, folded)
    y_new_tilde = x1_tilde + mlp["out"]

    exp_key = _apply_heads(kr, attn_masks["key_masks"])
    exp_val = _apply_heads(v, attn_masks["value_masks"])
    metrics = {
        "output_max_abs_error": _mx(y_new_tilde, y_new @ n_res),
        "cache_append_key_max_abs_error": _mx(a["k_rope_new"], exp_key),
        "cache_append_value_max_abs_error": _mx(a["v"], exp_val),
    }
    metrics["allclose"] = all(v <= 1e-8 for v in metrics.values()
                              if isinstance(v, float))

    new_ct = dict(cache_tilde)
    new_ct["key_rope_tilde"] = a["key_rope_full"]
    new_ct["value_tilde"] = a["value_full"]
    new_ct["seq_len"] = cache_tilde["seq_len"] + 1
    new_cp = dict(cache_plain)
    new_cp["key_rope"] = kr_full
    new_cp["value"] = v_full
    new_cp["seq_len"] = cache_plain["seq_len"] + 1
    return {
        "y_new_plain": y_new, "y_new_tilde": y_new_tilde,
        "expected_y_new_tilde": y_new @ n_res,
        "appended_key_tilde": a["k_rope_new"], "appended_value_tilde": a["v"],
        "expected_appended_key_tilde": exp_key,
        "expected_appended_value_tilde": exp_val,
        "cache_tilde": new_ct, "cache_plain": new_cp, "metrics": metrics,
    }


def _security_metadata(config: HFSingleBlockConfig) -> dict[str, Any]:
    return {
        "security_status":
            "operator_compatible_leakage_reduction_not_semantic_security",
        "semantic_security_claimed": False,
        "formal_security_claimed": False,
        "cryptographic_security_claimed": False,
        "no_intermediate_tee": True,
        "no_network_download": True,
        "mask_family": config.mask_family,
        "residual_mask_family": "orthogonal",
        "model_type": config.model_type,
    }


# ---------------------------------------------------------------------------
# Random HF layer construction (no downloads)
# ---------------------------------------------------------------------------


def make_random_hf_decoder_layer(
    model_family: str = "llama", *, hidden_size: int = 32,
    intermediate_size: int = 64, num_attention_heads: int = 4,
    num_key_value_heads: int = 2, max_position_embeddings: int = 64,
    rms_norm_eps: float = 1e-5, rope_theta: float = 10000.0,
    seed: int = 2029,
) -> tuple[Any, Any]:
    """Instantiate one random HF decoder layer + its model config.

    No checkpoints, no downloads. Raises if transformers (or the requested
    family's layer class) is unavailable.
    """
    require_transformers_or_skip()
    torch.manual_seed(seed)
    family = model_family.lower()
    if family == "llama":
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        cfg = LlamaConfig(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps, rope_theta=rope_theta,
            num_hidden_layers=1)
        layer = LlamaDecoderLayer(cfg, layer_idx=0)
    elif family in ("qwen2", "qwen"):
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        cfg = Qwen2Config(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps, rope_theta=rope_theta,
            num_hidden_layers=1)
        layer = Qwen2DecoderLayer(cfg, layer_idx=0)
    else:
        raise ValueError(f"unknown model_family {model_family!r}")
    layer.eval()
    return layer, cfg
