"""Memory-optimized full-layer masked execution (Stage 8.4).

Untrusted-GPU masked decoder pipeline only. **No TEE** is imported or used
here; this module runs decoder blocks / attention / MLP / KV cache / LM head on
the untrusted side. It exists to let large checkpoints (e.g. Qwen2.5-7B) run
*all* decoder layers in masked mode without OOM.

Why the baseline OOMs: :func:`hf_causal_lm_masked_prefill` folds **every** layer
up front (``[fold(ell) for ell in range(n_layers)]``) and the folded
down-projection ``down[perm] @ n_res`` (``[intermediate, hidden]``) is the
single largest tensor. Keeping 28 of those resident blows past VRAM.

This module fixes that with:

1. **layerwise folded-weight streaming** -- fold ONE layer, run it, free it
   (and its temporaries), ``torch.cuda.empty_cache()`` between layers;
2. **chunked folded down-projection** -- accumulate the down-proj output in
   ``mlp_down_chunk_size`` chunks so the full ``[intermediate, hidden]`` folded
   down tensor is never materialized;
3. optional **CPU offload / JIT folding** -- prepare folded weights on CPU and
   move only the current layer / chunk to the GPU;
4. **aggressive memory instrumentation** -- peak allocated/reserved, per-layer
   before/after, and the OOM layer index on failure.

Correctness is checked against the extracted-weight plaintext reference (the
project's faithful plain forward): ``top1_match_rate`` over all positions,
``greedy_token_match`` over generated tokens, and ``max_abs_error`` of recovered
vs plain logits. KV cache is always on unless explicitly disabled.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.hf_wrappers.hf_causal_lm_skeleton import (
    HFCausalLMSkeletonConfig,
    HFCausalLMSkeletonWeights,
    _boundary_weights,
    _hf_plain_block_decode,
    generate_hf_causal_lm_masks,
)
from pllo.hf_wrappers.llama_qwen_single_block import (
    HFSingleBlockConfig,
    _linear,
    _masked_attention,
    extract_hf_single_block_weights,
    hf_single_block_plain_prefill,
    infer_config_from_hf_layer,
)
from pllo.ops.causal_lm_boundaries import (
    final_norm_lm_head_masked,
    final_norm_lm_head_plain,
    greedy_sample,
    recover_vocab_logits,
    trusted_embedding_lookup,
)
from pllo.ops.gqa_attention import block_diag_from_head_masks
from pllo.ops.nonlinear_islands import rmsnorm_core, silu_reference
from pllo.ops.rope import build_rope_cache

__all__ = [
    "MemoryInstrument",
    "MemoryOptimizedConfig",
    "align_qk_weights_to_hf_rope",
    "chunked_folded_down_projection",
    "fold_layer_attention_and_up",
    "hf_rope_interleave_index",
    "masked_prefill_full_logits",
    "run_memory_optimized_masked",
]


def hf_rope_interleave_index(num_heads: int, head_dim: int,
                             device=None) -> torch.Tensor:
    """Per-head gather index ``g`` converting HF half-split layout -> the
    adjacent-pair layout, so adjacent-pair RoPE reproduces HF half-split RoPE.

    Within a head: ``g[2i] = i`` and ``g[2i+1] = i + head_dim/2``."""
    half = head_dim // 2
    g = torch.empty(num_heads * head_dim, dtype=torch.long, device=device)
    for h in range(num_heads):
        b = h * head_dim
        for i in range(half):
            g[b + 2 * i] = b + i
            g[b + 2 * i + 1] = b + half + i
    return g


def align_qk_weights_to_hf_rope(weights, num_heads: int, num_kv_heads: int,
                                head_dim: int):
    """Permute q_proj / k_proj output features (+ biases) so the adjacent-pair
    RoPE used internally matches HuggingFace Qwen2's half-split RoPE.

    Only q and k are permuted (RoPE acts on them); v / o / MLP are untouched.
    The same permutation on q and k leaves attention scores invariant up to the
    RoPE convention, so this makes the plaintext reference HF-faithful while
    keeping the masked==plain invariant (masks are applied to the permuted q/k
    and still commute with adjacent-pair RoPE)."""
    from dataclasses import replace
    gq = hf_rope_interleave_index(num_heads, head_dim,
                                  weights.q_proj_weight.device)
    gk = hf_rope_interleave_index(num_kv_heads, head_dim,
                                  weights.k_proj_weight.device)
    new = {
        "q_proj_weight": weights.q_proj_weight.index_select(1, gq),
        "k_proj_weight": weights.k_proj_weight.index_select(1, gk),
        "q_proj_bias": None if weights.q_proj_bias is None
        else weights.q_proj_bias.index_select(0, gq),
        "k_proj_bias": None if weights.k_proj_bias is None
        else weights.k_proj_bias.index_select(0, gk),
    }
    return replace(weights, **new)


# ---------------------------------------------------------------------------
# Config + instrumentation
# ---------------------------------------------------------------------------


@dataclass
class MemoryOptimizedConfig:
    num_layers: int | None = None          # None -> all decoder layers
    batch_size: int = 1
    seq_len: int = 64
    max_new_tokens: int = 1
    device: str = "cuda"                    # compute device for activations
    dtype: str = "float16"                  # model load dtype name (diagnostic)
    folding_dtype: str = "float32"          # fold/compare precision
    folded_weight_device: str = "cuda"      # "cuda" | "cpu" (CPU offload)
    layerwise_folding: bool = True          # fold one layer at a time
    mlp_down_chunk_size: int = 1024         # 0/None -> single chunk
    use_kv_cache: bool = True
    mask_mode: str = "signed_permutation"
    residual_mask_strategy: str = "shared"
    mask_block_size: int = 64
    # Per-head attention Q/K/V mask family. The default keeps back-compat; the
    # A_rightmul paper-facing build forces "pairwise_rotation" (orthogonal +
    # score-preserving) so the masked attention state is in a compatible family
    # (pairwise_complex_scaling is NOT compatible -- it is non-orthogonal).
    attention_mask_family: str = "pairwise_complex_scaling"
    seed: int = 2035
    empty_cache_between_layers: bool = True
    # Permute q/k output features so the project's adjacent-pair RoPE reproduces
    # HuggingFace Qwen2's half-split RoPE exactly (required for HF parity).
    align_rope_to_hf: bool = True
    # Linear-boundary additive input padding (default OFF for back-compat). When
    # on, every folded Linear (q/k/v/o/gate/up/down/lm_head) carries a masked
    # input pad ``xpad = T N_in`` + compensation ``cpad = T W N_out`` so the GPU
    # matmul operand is ``(X - T) N_in`` while the output stays in the compatible
    # masked basis ``Y N_out`` (no persistent residual pad; not in nonlinear cores).
    use_linear_boundary_pad: bool = False
    linear_pad_scale: float = 0.1


_DTYPE = {"float16": torch.float16, "fp16": torch.float16,
          "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float32": torch.float32, "fp32": torch.float32}


def _resolve_dtype(name: str) -> torch.dtype:
    return _DTYPE.get((name or "").lower(), torch.float32)


class MemoryInstrument:
    """CUDA memory snapshots (no-ops + ``None`` on CPU)."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def reset_peak(self) -> None:
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()

    def snapshot(self) -> dict[str, float] | None:
        if not self.is_cuda:
            return None
        mb = 1 / 2 ** 20
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() * mb, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() * mb, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() * mb, 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() * mb, 2),
        }


# ---------------------------------------------------------------------------
# Chunked folded down projection (the key memory saver)
# ---------------------------------------------------------------------------


def chunked_folded_down_projection(
    hidden_act: torch.Tensor, down_proj_weight: torch.Tensor,
    perm: torch.Tensor, n_res: torch.Tensor,
    bdown_tilde: torch.Tensor | None = None, chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute ``hidden_act @ (down_proj_weight[perm] @ n_res)`` in chunks.

    Mathematically identical to materializing the full folded down weight
    ``wdown_tilde = down_proj_weight[perm] @ n_res`` and doing one matmul, but
    only a ``[chunk, hidden]`` slice of it ever exists at once. ``hidden_act``
    is already in the permuted intermediate basis (gate/up were permuted), so
    chunk ``k`` pairs ``hidden_act[..., k]`` with row ``down[perm[k]]``.

    The chunk weight is moved to ``hidden_act``'s device/dtype, so this also
    implements per-chunk CPU->GPU streaming when the weights live on CPU."""
    inter = int(down_proj_weight.shape[0])
    if not chunk_size or chunk_size <= 0:
        chunk_size = inter
    out: torch.Tensor | None = None
    for start in range(0, inter, chunk_size):
        idx = perm[start:start + chunk_size]
        w_chunk = (down_proj_weight.index_select(0, idx) @ n_res).to(
            device=hidden_act.device, dtype=hidden_act.dtype)
        contrib = hidden_act[..., start:start + chunk_size] @ w_chunk
        out = contrib if out is None else out + contrib
        del w_chunk, contrib
    assert out is not None
    if bdown_tilde is not None:
        out = out + bdown_tilde.to(device=out.device, dtype=out.dtype)
    return out


# ---------------------------------------------------------------------------
# Single-layer folding WITHOUT the down projection (down stays chunked)
# ---------------------------------------------------------------------------


def fold_layer_attention_and_up(
    weights: Any, bm: dict[str, Any],
) -> dict[str, Any]:
    """Fold attention (q/k/v/o) + MLP gate/up for ONE layer. The down
    projection is intentionally NOT folded here -- it is applied chunked via
    :func:`chunked_folded_down_projection` so its full folded tensor never
    materializes."""
    n_res = bm["n_res"]
    n_res_inv = bm["n_res_inv"]
    attn = bm["attn"]
    perm = bm["perm"]

    mq = block_diag_from_head_masks(attn["q_masks"])
    mk = block_diag_from_head_masks(attn["key_masks"])
    mv = block_diag_from_head_masks(attn["value_masks"])
    v_inv_qhead = attn["value_mask_inverses"].index_select(0, attn["kv_index"])
    sv_inv = block_diag_from_head_masks(v_inv_qhead)
    rms1 = weights.input_layernorm_weight.unsqueeze(1)
    rms2 = weights.post_attention_layernorm_weight.unsqueeze(1)

    def maybe(b, m):
        return None if b is None else b @ m

    return {
        "wq_tilde": n_res_inv @ (rms1 * weights.q_proj_weight) @ mq,
        "wk_tilde": n_res_inv @ (rms1 * weights.k_proj_weight) @ mk,
        "wv_tilde": n_res_inv @ (rms1 * weights.v_proj_weight) @ mv,
        "wo_tilde": sv_inv @ weights.o_proj_weight @ n_res,
        "bq_tilde": maybe(weights.q_proj_bias, mq),
        "bk_tilde": maybe(weights.k_proj_bias, mk),
        "bv_tilde": maybe(weights.v_proj_bias, mv),
        "bo_tilde": maybe(weights.o_proj_bias, n_res),
        "wgate_tilde": n_res_inv @ (rms2 * weights.gate_proj_weight)
        .index_select(1, perm),
        "wup_tilde": n_res_inv @ (rms2 * weights.up_proj_weight)
        .index_select(1, perm),
        "bgate_tilde": None if weights.gate_proj_bias is None
        else weights.gate_proj_bias.index_select(0, perm),
        "bup_tilde": None if weights.up_proj_bias is None
        else weights.up_proj_bias.index_select(0, perm),
    }


def _move_folded(folded: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in folded.items()}


# ---------------------------------------------------------------------------
# Masked block forwards (chunked MLP down)
# ---------------------------------------------------------------------------


def _masked_block_prefill_chunked(
    x_tilde: torch.Tensor, folded: dict[str, Any], down_info: tuple,
    config: HFSingleBlockConfig, cos: torch.Tensor, sin: torch.Tensor,
    chunk_size: int,
) -> dict[str, Any]:
    eps = config.rms_norm_eps
    r1 = rmsnorm_core(x_tilde, eps)
    a = _masked_attention(r1, folded, config, cos, sin, causal_offset=0)
    x1 = x_tilde + a["out"]
    r2 = rmsnorm_core(x1, eps)
    gate = _linear(r2, folded["wgate_tilde"], folded["bgate_tilde"])
    up = _linear(r2, folded["wup_tilde"], folded["bup_tilde"])
    hidden = silu_reference(gate) * up
    mlp_out = chunked_folded_down_projection(hidden, *down_info,
                                             chunk_size=chunk_size)
    y = x1 + mlp_out
    return {"y_tilde": y,
            "cache": {"key_rope_tilde": a["key_rope_full"],
                      "value_tilde": a["value_full"]}}


def _masked_block_decode_chunked(
    x_next_tilde: torch.Tensor, cache: dict[str, Any], folded: dict[str, Any],
    down_info: tuple, config: HFSingleBlockConfig, cos: torch.Tensor,
    sin: torch.Tensor, position: int, chunk_size: int,
) -> dict[str, Any]:
    eps = config.rms_norm_eps
    pid = torch.tensor([position], device=x_next_tilde.device)
    r1 = rmsnorm_core(x_next_tilde, eps)
    a = _masked_attention(r1, folded, config, cos, sin, causal_offset=None,
                          position_ids=pid,
                          past_key_rope=cache["key_rope_tilde"],
                          past_value=cache["value_tilde"])
    x1 = x_next_tilde + a["out"]
    r2 = rmsnorm_core(x1, eps)
    gate = _linear(r2, folded["wgate_tilde"], folded["bgate_tilde"])
    up = _linear(r2, folded["wup_tilde"], folded["bup_tilde"])
    hidden = silu_reference(gate) * up
    mlp_out = chunked_folded_down_projection(hidden, *down_info,
                                             chunk_size=chunk_size)
    y = x1 + mlp_out
    return {"y_tilde": y,
            "cache": {"key_rope_tilde": a["key_rope_full"],
                      "value_tilde": a["value_full"]}}


# ---------------------------------------------------------------------------
# Streaming driver
# ---------------------------------------------------------------------------


def _base_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _extract_boundary(model: Any, model_config: Any, dtype: torch.dtype,
                      device: torch.device) -> HFCausalLMSkeletonWeights:
    base = _base_model(model)
    embed = base.embed_tokens.weight.detach().to(device=device,
                                                 dtype=dtype).clone()
    final_norm = base.norm.weight.detach().to(device=device, dtype=dtype).clone()
    head = getattr(model, "lm_head", None)
    if head is not None and getattr(head, "weight", None) is not None:
        lm_head = head.weight.detach().to(device=device, dtype=dtype) \
            .t().contiguous()
    else:
        lm_head = embed.t().contiguous()
    return HFCausalLMSkeletonWeights(embed_tokens_weight=embed,
                                     layer_weights=[],
                                     final_norm_weight=final_norm,
                                     lm_head_weight=lm_head)


def run_memory_optimized_masked(
    model: Any, model_config: Any, input_ids: torch.Tensor,
    config: MemoryOptimizedConfig,
) -> dict[str, Any]:
    """Stream all decoder layers in masked mode (one folded layer at a time)."""
    compute_device = torch.device(config.device)
    fold_device = torch.device(config.folded_weight_device)
    fdtype = _resolve_dtype(config.folding_dtype)
    chunk = config.mlp_down_chunk_size
    instr = MemoryInstrument(config.device)
    instr.reset_peak()

    base = _base_model(model)
    total_layers = len(base.layers)
    n = total_layers if config.num_layers is None else min(int(config.num_layers),
                                                           total_layers)

    input_ids = input_ids.to(compute_device)
    boundary = _extract_boundary(model, model_config, fdtype, compute_device)
    bw = _boundary_weights(boundary)
    eps = float(getattr(model_config, "rms_norm_eps", 1e-5))

    # Per-layer configs (cheap) + masks (shared residual mask + per-layer block
    # masks + vocab mask). Folded *weights* are NOT built here.
    layer_configs = [
        infer_config_from_hf_layer(base.layers[i], model_config, fdtype,
                                   str(fold_device))
        for i in range(n)
    ]
    skel_cfg = HFCausalLMSkeletonConfig(
        model_family=str(getattr(model_config, "model_type", "qwen2")),
        prefill_seq_len=config.seq_len,
        decode_steps=max(0, config.max_new_tokens - 1),
        max_layers=n, dtype=fdtype, device=str(fold_device), seed=config.seed,
        mask_mode=config.mask_mode,
        residual_mask_strategy=config.residual_mask_strategy,
        mask_block_size=config.mask_block_size)
    masks = generate_hf_causal_lm_masks(boundary, layer_configs, skel_cfg)
    # residual masks on the compute device for the activation-side products
    n0 = masks.residual_masks[0].to(compute_device)

    head_dim = layer_configs[0].head_dim
    rope_theta = layer_configs[0].rope_theta
    max_pos = config.seq_len + max(0, config.max_new_tokens - 1) + 1
    cos, sin = build_rope_cache(max_pos, head_dim, rope_theta, fdtype,
                               compute_device)

    report: dict[str, Any] = {
        "stage": "8.4_qwen_full_layer_memory_optimized",
        "config": asdict(config),
        "model_type": str(getattr(model_config, "model_type", "unknown")),
        "total_layers": total_layers,
        "requested_layers": n,
        "hidden_size": int(boundary.embed_tokens_weight.shape[1]),
        "vocab_size": int(boundary.embed_tokens_weight.shape[0]),
        "tee_used": False,
        "per_layer_memory": [],
    }

    def _extract_layer(ell: int):
        w = extract_hf_single_block_weights(base.layers[ell], fdtype,
                                            str(fold_device))
        if config.align_rope_to_hf:
            cfg_l = layer_configs[ell]
            w = align_qk_weights_to_hf_rope(
                w, cfg_l.num_heads, cfg_l.num_key_value_heads, cfg_l.head_dim)
        return w

    def run_layer_prefill(ell: int, h_plain, h_tilde):
        cfg_l = layer_configs[ell]
        bm = masks.layer_block_masks[ell]
        w = _extract_layer(ell)
        # plain reference (untrusted) -- weights on compute device
        wc = w if fold_device == compute_device else _block_to(w, compute_device)
        cfg_c = _cfg_to(cfg_l, compute_device)
        plain = hf_single_block_plain_prefill(h_plain, wc, cfg_c, cos, sin)
        # masked: fold attn+up (on fold device), stream to compute device
        folded = fold_layer_attention_and_up(w, bm)
        if fold_device != compute_device:
            folded = _move_folded(folded, compute_device)
        down_info = (w.down_proj_weight, bm["perm"], bm["n_res"],
                     None if w.down_proj_bias is None
                     else (w.down_proj_bias @ bm["n_res"]))
        masked = _masked_block_prefill_chunked(h_tilde, folded, down_info,
                                               cfg_c, cos, sin, chunk)
        out = (plain["y"], masked["y_tilde"], plain["cache_plain"],
               masked["cache"])
        del w, wc, folded, down_info, plain, masked
        return out

    # ---- prefill: stream layers (plain + masked together) -------------
    status = "ok"
    oom_layer = None
    t0 = time.perf_counter()
    h_plain = trusted_embedding_lookup(input_ids, boundary.embed_tokens_weight)
    h_tilde = h_plain @ n0
    caches_plain: list[dict[str, Any]] = []
    caches_tilde: list[dict[str, Any]] = []
    try:
        for ell in range(n):
            before = instr.snapshot()
            yp, yt, cp, ct = run_layer_prefill(ell, h_plain, h_tilde)
            h_plain, h_tilde = yp, yt
            caches_plain.append(cp)
            caches_tilde.append(ct)
            if config.empty_cache_between_layers:
                instr.empty_cache()
            after = instr.snapshot()
            report["per_layer_memory"].append(
                {"layer": ell, "phase": "prefill", "before": before,
                 "after": after})
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            status, oom_layer = "stopped_oom", ell
            instr.empty_cache()
            report.update(status=status, oom_layer_index=oom_layer,
                          reason=f"{type(exc).__name__}: {exc}",
                          executed_layers=ell,
                          peak_memory=instr.snapshot())
            return report
        raise

    executed_layers = n

    # ---- output boundary: plain + masked recovered logits -------------
    fn_plain = final_norm_lm_head_plain(h_plain, bw.final_norm_weight,
                                        bw.lm_head_weight, eps)
    masked_out = final_norm_lm_head_masked(
        h_tilde, h_plain, bw, masks.residual_mask_inverses[-1].to(compute_device),
        _vocab_to(masks.vocab_mask, compute_device), eps)
    recovered = recover_vocab_logits(masked_out["logits_tilde"],
                                     _vocab_to(masks.vocab_mask, compute_device))
    plain_logits = fn_plain["logits"]

    top1_match = float((plain_logits.argmax(-1) == recovered.argmax(-1))
                       .to(torch.float32).mean().item())
    max_abs_err = float((plain_logits - recovered).abs().max().item())

    gen_plain = [greedy_sample(plain_logits[:, -1, :])]
    gen_masked = [greedy_sample(recovered[:, -1, :])]

    # ---- bounded greedy decode (re-stream layers each step) -----------
    for step in range(max(0, config.max_new_tokens - 1)):
        position = config.seq_len + step
        tok_p, tok_m = gen_plain[-1], gen_masked[-1]
        xp = trusted_embedding_lookup(
            tok_p, boundary.embed_tokens_weight).unsqueeze(1)
        xt = trusted_embedding_lookup(
            tok_m, boundary.embed_tokens_weight).unsqueeze(1) @ n0
        for ell in range(n):
            cfg_l = layer_configs[ell]
            bm = masks.layer_block_masks[ell]
            w = _extract_layer(ell)
            wc = w if fold_device == compute_device else _block_to(
                w, compute_device)
            cfg_c = _cfg_to(cfg_l, compute_device)
            dp = _hf_plain_block_decode(xp, caches_plain[ell], wc, cfg_c, cos,
                                        sin, position)
            folded = fold_layer_attention_and_up(w, bm)
            if fold_device != compute_device:
                folded = _move_folded(folded, compute_device)
            down_info = (w.down_proj_weight, bm["perm"], bm["n_res"],
                         None if w.down_proj_bias is None
                         else (w.down_proj_bias @ bm["n_res"]))
            dm = _masked_block_decode_chunked(xt, caches_tilde[ell], folded,
                                              down_info, cfg_c, cos, sin,
                                              position, chunk)
            caches_plain[ell] = dp["cache"]
            caches_tilde[ell] = dm["cache"]
            xp, xt = dp["y"], dm["y_tilde"]
            del w, wc, folded, down_info, dp, dm
            if config.empty_cache_between_layers:
                instr.empty_cache()
        fp = final_norm_lm_head_plain(xp, bw.final_norm_weight,
                                      bw.lm_head_weight, eps)
        mo = final_norm_lm_head_masked(
            xt, xp, bw, masks.residual_mask_inverses[-1].to(compute_device),
            _vocab_to(masks.vocab_mask, compute_device), eps)
        rec = recover_vocab_logits(mo["logits_tilde"],
                                   _vocab_to(masks.vocab_mask, compute_device))
        gen_plain.append(greedy_sample(fp["logits"][:, -1, :]))
        gen_masked.append(greedy_sample(rec[:, -1, :]))

    gp = torch.stack(gen_plain, dim=1)
    gm = torch.stack(gen_masked, dim=1)
    greedy_match = float((gp == gm).to(torch.float32).mean().item())

    report.update(
        status=status,
        executed_layers=executed_layers,
        oom_layer_index=oom_layer,
        latency_s=round(time.perf_counter() - t0, 4),
        top1_match_rate=top1_match,
        max_abs_error=max_abs_err,
        greedy_token_match=greedy_match,
        generated_plain_tokens=gp.tolist(),
        generated_masked_tokens=gm.tolist(),
        peak_memory=instr.snapshot(),
    )
    instr.empty_cache()
    return report


def masked_prefill_full_logits(
    model: Any, model_config: Any, input_ids: torch.Tensor,
    config: MemoryOptimizedConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single streaming masked prefill over ``input_ids``; returns the
    full-sequence ``(plain_logits, recovered_logits)`` of shape ``[B, T, V]``.

    Used for teacher-forced parity (logits at every position under a fixed
    prefix), so a single near-tie token cannot cascade as it does in
    free-running greedy decode. No decode loop, no token sampling."""
    compute_device = torch.device(config.device)
    fold_device = torch.device(config.folded_weight_device)
    fdtype = _resolve_dtype(config.folding_dtype)
    chunk = config.mlp_down_chunk_size
    base = _base_model(model)
    n = len(base.layers) if config.num_layers is None else min(
        int(config.num_layers), len(base.layers))
    input_ids = input_ids.to(compute_device)
    boundary = _extract_boundary(model, model_config, fdtype, compute_device)
    bw = _boundary_weights(boundary)
    eps = float(getattr(model_config, "rms_norm_eps", 1e-5))
    layer_configs = [
        infer_config_from_hf_layer(base.layers[i], model_config, fdtype,
                                   str(fold_device)) for i in range(n)]
    T = int(input_ids.shape[1])
    skel_cfg = HFCausalLMSkeletonConfig(
        model_family=str(getattr(model_config, "model_type", "qwen2")),
        prefill_seq_len=T, decode_steps=0, max_layers=n, dtype=fdtype,
        device=str(fold_device), seed=config.seed, mask_mode=config.mask_mode,
        residual_mask_strategy=config.residual_mask_strategy,
        mask_block_size=config.mask_block_size)
    masks = generate_hf_causal_lm_masks(boundary, layer_configs, skel_cfg)
    n0 = masks.residual_masks[0].to(compute_device)
    cos, sin = build_rope_cache(T + 1, layer_configs[0].head_dim,
                               layer_configs[0].rope_theta, fdtype,
                               compute_device)
    instr = MemoryInstrument(config.device)

    h_plain = trusted_embedding_lookup(input_ids, boundary.embed_tokens_weight)
    h_tilde = h_plain @ n0
    for ell in range(n):
        cfg_l = layer_configs[ell]
        bm = masks.layer_block_masks[ell]
        w = extract_hf_single_block_weights(base.layers[ell], fdtype,
                                            str(fold_device))
        if config.align_rope_to_hf:
            w = align_qk_weights_to_hf_rope(w, cfg_l.num_heads,
                                            cfg_l.num_key_value_heads,
                                            cfg_l.head_dim)
        wc = w if fold_device == compute_device else _block_to(w, compute_device)
        cfg_c = _cfg_to(cfg_l, compute_device)
        plain = hf_single_block_plain_prefill(h_plain, wc, cfg_c, cos, sin)
        folded = fold_layer_attention_and_up(w, bm)
        if fold_device != compute_device:
            folded = _move_folded(folded, compute_device)
        down_info = (w.down_proj_weight, bm["perm"], bm["n_res"],
                     None if w.down_proj_bias is None
                     else (w.down_proj_bias @ bm["n_res"]))
        masked = _masked_block_prefill_chunked(h_tilde, folded, down_info,
                                               cfg_c, cos, sin, chunk)
        h_plain, h_tilde = plain["y"], masked["y_tilde"]
        del w, wc, folded, down_info, plain, masked
        if config.empty_cache_between_layers:
            instr.empty_cache()
    fn_plain = final_norm_lm_head_plain(h_plain, bw.final_norm_weight,
                                        bw.lm_head_weight, eps)
    mo = final_norm_lm_head_masked(
        h_tilde, h_plain, bw, masks.residual_mask_inverses[-1].to(compute_device),
        _vocab_to(masks.vocab_mask, compute_device), eps)
    recovered = recover_vocab_logits(mo["logits_tilde"],
                                     _vocab_to(masks.vocab_mask, compute_device))
    return fn_plain["logits"], recovered


# ---------------------------------------------------------------------------
# Small device helpers for CPU-offload mode
# ---------------------------------------------------------------------------


def _block_to(w: Any, device: torch.device) -> Any:
    from dataclasses import replace
    flds = {f: getattr(w, f) for f in w.__dataclass_fields__}
    flds = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in flds.items()}
    return replace(w, **flds)


def _cfg_to(cfg: HFSingleBlockConfig, device: torch.device) -> HFSingleBlockConfig:
    from dataclasses import replace
    return replace(cfg, device=str(device))


def _vocab_to(vm: Any, device: torch.device) -> Any:
    from pllo.ops.causal_lm_boundaries import VocabLogitMask
    return VocabLogitMask(
        permutation=vm.permutation.to(device),
        inverse_permutation=vm.inverse_permutation.to(device),
        scale=vm.scale.to(device), inverse_scale=vm.inverse_scale.to(device))
