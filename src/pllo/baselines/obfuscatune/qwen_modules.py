"""ObfuscaTune-Qwen block modules: RMSNorm/RoPE/GQA attention + SwiGLU MLP.

TEE / outside-TEE split (simulated; see docs/obfuscatune_qwen_integration.md):

  * simulated TEE (plaintext domain): RMSNorm, RoPE application, SiLU + SwiGLU
    elementwise product, softmax, de-/re-obfuscation, residual adds, bias.
  * outside TEE (obfuscated matmuls): q/k/v/o projections and the SwiGLU
    gate/up/down projections.

Because the input obfuscation cancels (X R)(R^{-1} W) = X W, the untrusted side
sees the TRUE plaintext Q/K/V and the MLP gate/up intermediates -- exactly the
paper's exposure. RMSNorm, RoPE and SiLU are NEVER executed in the obfuscated
domain (they are non-linear / norm ops); they run on de-obfuscated values.

Correctness is validated against Qwen2's own math (HF ``apply_rotary_pos_emb`` /
``repeat_kv`` are reused so GQA + RoPE match bit-for-bit at fp64).
"""

from __future__ import annotations

from typing import Any

import torch

from .config import ObfuscaTuneConfig
from .modules import ObfuscaTuneLinearSimulator, PhaseMetrics
from .qwen_config import (
    discover_attention_projections,
    discover_layer_norms,
    discover_mlp_projections,
)


def _rope_helpers():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    return apply_rotary_pos_emb, repeat_kv


def qwen_attention_obfuscated(
    hidden_normed: torch.Tensor,     # (B, S, H) post-RMSNorm (TEE plaintext)
    attn: torch.nn.Module,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ObfuscaTuneConfig,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seed: int | None = None,
    scaling: float | None = None,
    past_k: torch.Tensor | None = None,   # cached (B, n_kv, past_len, hd) post-RoPE
    past_v: torch.Tensor | None = None,
    q_pos_start: int = 0,                 # absolute position of the first query
) -> tuple[torch.Tensor, dict[str, Any], PhaseMetrics, tuple[torch.Tensor, torch.Tensor]]:
    """Obfuscated Qwen attention with optional KV cache.

    Returns ``(attn_output (B,S,H), audit, metrics, (new_k, new_v))`` where
    ``new_k/new_v`` are the post-RoPE, pre-GQA-repeat key/value for THIS chunk
    (to be appended to the cache).
    """
    apply_rotary_pos_emb, repeat_kv = _rope_helpers()
    cfg = config
    seed = cfg.seed if seed is None else seed
    proj = discover_attention_projections(attn)
    B, S, H = hidden_normed.shape
    scaling = (head_dim ** -0.5) if scaling is None else scaling
    metrics = PhaseMetrics()

    # q/k/v share the SAME input obfuscation R_attn_in (paper: X* = X R_a)
    q_sim = ObfuscaTuneLinearSimulator(proj["q_proj"], cfg, direction="input", seed=seed)
    k_sim = ObfuscaTuneLinearSimulator(proj["k_proj"], cfg, direction="input", seed=seed)
    v_sim = ObfuscaTuneLinearSimulator(proj["v_proj"], cfg, direction="input", seed=seed)
    q, mq = q_sim.forward(hidden_normed)   # TRUE plaintext Q on the untrusted side
    k, mk = k_sim.forward(hidden_normed)
    v, mv = v_sim.forward(hidden_normed)
    for mm in (mq, mk, mv):
        metrics.merge(mm)

    q = q.view(B, S, n_heads, head_dim).transpose(1, 2)      # (B, nh, S, hd)
    k = k.view(B, S, n_kv_heads, head_dim).transpose(1, 2)   # (B, nkv, S, hd)
    v = v.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)              # TEE: RoPE (plaintext)
    metrics.boundary_calls += 1
    new_k, new_v = k, v                                      # cache stores post-RoPE k/v
    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    total_k = k.shape[2]
    n_rep = n_heads // n_kv_heads
    kr = repeat_kv(k, n_rep)                                 # GQA expand
    vr = repeat_kv(v, n_rep)
    scores = (q @ kr.transpose(-1, -2)) * scaling           # TEE: attention scores
    # causal mask: query row i (abs pos q_pos_start+i) attends to key j <= abs pos
    qpos = torch.arange(S, device=q.device).unsqueeze(1) + q_pos_start
    kpos = torch.arange(total_k, device=q.device).unsqueeze(0)
    allow = kpos <= qpos                                     # (S, total_k)
    scores = scores.masked_fill(~allow, float("-inf"))
    att = torch.softmax(scores, dim=-1)                     # TEE: softmax (plaintext)
    metrics.boundary_calls += 1
    ctx = (att @ vr).transpose(1, 2).reshape(B, S, n_heads * head_dim)
    o_sim = ObfuscaTuneLinearSimulator(proj["o_proj"], cfg, direction="output", seed=seed + 3)
    out, mo = o_sim.forward(ctx)
    metrics.merge(mo)
    audit = {
        "exposed_plaintext_tensors": ["Q_plaintext", "K_plaintext", "V_plaintext",
                                      "attention_scores_plaintext"],
        "rope_plain_domain": True,
        "softmax_in_tee": True,
        "public_qkv": True,
        "public_attention_scores": True,
        "is_gqa": n_kv_heads != n_heads,
    }
    return out, audit, metrics, (new_k, new_v)


def qwen_mlp_obfuscated(
    hidden_normed: torch.Tensor,     # (B, S, H) post-RMSNorm (TEE plaintext)
    mlp: torch.nn.Module,
    config: ObfuscaTuneConfig,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], PhaseMetrics]:
    """Obfuscated Qwen SwiGLU MLP: silu(gate(x)) * up(x) -> down. (B,S,H) out."""
    cfg = config
    seed = cfg.seed if seed is None else seed
    proj = discover_mlp_projections(mlp)
    metrics = PhaseMetrics()
    act = getattr(mlp, "act_fn", None) or torch.nn.functional.silu

    # gate/up share the same input obfuscation R_mlp_in
    gate_sim = ObfuscaTuneLinearSimulator(proj["gate_proj"], cfg, direction="input", seed=seed)
    up_sim = ObfuscaTuneLinearSimulator(proj["up_proj"], cfg, direction="input", seed=seed)
    gate, mg = gate_sim.forward(hidden_normed)   # TRUE plaintext gate
    up, mu = up_sim.forward(hidden_normed)        # TRUE plaintext up
    metrics.merge(mg)
    metrics.merge(mu)
    inter = act(gate) * up                        # TEE: SiLU + product (plaintext)
    metrics.boundary_calls += 1
    down_sim = ObfuscaTuneLinearSimulator(proj["down_proj"], cfg, direction="output", seed=seed + 1)
    out, md = down_sim.forward(inter)
    metrics.merge(md)
    audit = {
        "exposed_plaintext_tensors": ["mlp_gate_plaintext", "mlp_up_plaintext"],
        "silu_in_tee": True,
    }
    return out, audit, metrics


def qwen_rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm matching Qwen2RMSNorm (compute in fp32, cast back)."""
    dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (weight * x32.to(dtype))


def qwen_decoder_block_obfuscated(
    layer: torch.nn.Module,
    hidden: torch.Tensor,            # (B, S, H)
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ObfuscaTuneConfig,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seed: int | None = None,
    scaling: float | None = None,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
    q_pos_start: int = 0,
) -> tuple[torch.Tensor, dict[str, Any], PhaseMetrics, tuple[torch.Tensor, torch.Tensor]]:
    """One full Qwen2 decoder layer under ObfuscaTune (cache-aware).

    Returns ``(hidden, audit, metrics, (new_k, new_v))``.
    """
    cfg = config
    dtype = cfg.torch_dtype()
    hidden = hidden.to(dtype)
    norms = discover_layer_norms(layer)
    metrics = PhaseMetrics()
    exposed: list[str] = []

    residual = hidden
    h = norms["input_layernorm"](hidden)                    # TEE: RMSNorm
    metrics.boundary_calls += 1
    attn_out, a_audit, a_m, new_kv = qwen_attention_obfuscated(
        h, layer.self_attn, cos, sin, cfg, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, seed=seed, scaling=scaling,
        past_k=past_k, past_v=past_v, q_pos_start=q_pos_start)
    metrics.merge(a_m)
    exposed += a_audit["exposed_plaintext_tensors"]
    hidden = residual + attn_out                            # TEE: residual

    residual = hidden
    h2 = norms["post_attention_layernorm"](hidden)          # TEE: RMSNorm
    metrics.boundary_calls += 1
    mlp_out, m_audit, m_m = qwen_mlp_obfuscated(h2, layer.mlp, cfg, seed=seed)
    metrics.merge(m_m)
    exposed += m_audit["exposed_plaintext_tensors"]
    hidden = residual + mlp_out

    audit = {
        "exposed_plaintext_tensors": exposed,
        "tee_nonlinear_ops": ["rmsnorm", "rope", "softmax", "rmsnorm", "silu_swiglu"],
        "rope_plain_domain": True,
        "public_qkv": True,
        "public_attention_scores": True,
        "is_gqa": n_kv_heads != n_heads,
    }
    return hidden, audit, metrics, new_kv


__all__ = [
    "qwen_attention_obfuscated",
    "qwen_mlp_obfuscated",
    "qwen_rmsnorm_reference",
    "qwen_decoder_block_obfuscated",
]
