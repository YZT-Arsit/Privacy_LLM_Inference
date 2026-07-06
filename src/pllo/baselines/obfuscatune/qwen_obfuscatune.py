"""Full ObfuscaTune-Qwen forward: prefill, decode, greedy generation.

Reproduces Qwen2's exact forward (embeddings -> N obfuscated decoder blocks ->
final RMSNorm -> LM head) while routing every large linear through the
obfuscated primitive and keeping all non-linear / norm / RoPE ops in the
simulated TEE. The plaintext reference is the model's own ``forward``.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import ObfuscaTuneConfig, torch_dtype_from_str
from .metrics import argmax_match_rate, correctness_metrics, nan_or_inf_count
from .modules import PhaseMetrics
from .qwen_cache import QwenObfCache
from .qwen_config import extract_arch
from .qwen_modules import qwen_decoder_block_obfuscated


def _cos_sin(model, hidden: torch.Tensor, positions: torch.Tensor):
    """Compute RoPE cos/sin for the given absolute positions (TEE, plaintext)."""
    pos = positions.unsqueeze(0)                       # (1, S)
    return model.model.rotary_emb(hidden, pos)


def qwen_obfuscated_forward(
    model,
    input_ids: torch.Tensor,          # (B, S)
    config: ObfuscaTuneConfig,
    *,
    cache: QwenObfCache | None = None,
    position_offset: int = 0,
) -> tuple[torch.Tensor, QwenObfCache, dict[str, Any]]:
    """One obfuscated forward chunk (prefill if position_offset==0, else decode).

    Returns ``(logits (B, S, vocab), cache, audit)``. The cache is created if not
    provided and updated in place with the new K/V.
    """
    dtype = config.torch_dtype()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, S = input_ids.shape
    arch = extract_arch(model)
    m = model.model
    scaling = getattr(model.model.layers[0].self_attn, "scaling", arch.head_dim ** -0.5)

    hidden = m.embed_tokens(input_ids).to(dtype)       # TEE: embeddings
    positions = torch.arange(position_offset, position_offset + S, device=hidden.device)
    cos, sin = _cos_sin(model, hidden, positions)
    cos, sin = cos.to(dtype), sin.to(dtype)

    if cache is None:
        cache = QwenObfCache(arch.num_layers)
    total = PhaseMetrics()
    exposed: set[str] = set()
    for i, layer in enumerate(m.layers):
        pk, pv = cache.get(i)
        hidden, audit, met, (nk, nv) = qwen_decoder_block_obfuscated(
            layer, hidden, cos, sin, config,
            n_heads=arch.num_attention_heads, n_kv_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim, seed=config.seed, scaling=scaling,
            past_k=pk, past_v=pv, q_pos_start=position_offset)
        cache.append(i, nk, nv)
        total.merge(met)
        exposed.update(audit["exposed_plaintext_tensors"])
    hidden = m.norm(hidden)                            # TEE: final RMSNorm
    logits = hidden @ model.lm_head.weight.to(dtype).transpose(0, 1)  # TEE: LM head
    audit = {
        "exposed_plaintext_tensors": sorted(exposed),
        "tee_nonlinear_ops": ["rmsnorm", "rope", "softmax", "silu_swiglu"],
        "phase_metrics": total.to_dict(),
        "num_layers": arch.num_layers,
        "public_qkv": True,
        "public_attention_scores": True,
        "is_gqa": arch.is_gqa,
    }
    return logits, cache, audit


def qwen_plain_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Reference plaintext logits (B, S, vocab) from the model's own forward."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids)
    return out.logits


def _make_config(mode: str, dtype: torch.dtype, seed: int, condition_number: float) -> ObfuscaTuneConfig:
    dtype_str = {torch.float64: "float64", torch.float32: "float32"}.get(dtype, "float32")
    if mode in ("orthogonal", "obfuscatune_qwen_orthogonal"):
        return ObfuscaTuneConfig(dtype=dtype_str, seed=seed, random_matrix_type="orthogonal")
    if mode in ("random", "obfuscatune_qwen_random"):
        return ObfuscaTuneConfig(dtype=dtype_str, seed=seed, random_matrix_type="random")
    if mode in ("cond", "obfuscatune_qwen_cond"):
        return ObfuscaTuneConfig(dtype=dtype_str, seed=seed, random_matrix_type="cond",
                                 condition_number=condition_number)
    raise ValueError(f"unknown obfuscation mode {mode!r}")


def run_prefill(
    model, input_ids: torch.Tensor, mode: str, *,
    dtype: torch.dtype = torch.float32, seed: int = 0, condition_number: float = 1.0,
) -> dict[str, Any]:
    """Prefill correctness for one mode ('unprotected' or an obfuscation mode)."""
    plain = qwen_plain_logits(model, input_ids)
    if mode == "unprotected":
        return {"mode": mode, "logits": plain, "cache": None,
                "correctness": correctness_metrics(plain, plain),
                "argmax_match_rate": 1.0, "nan_or_inf_count": nan_or_inf_count(plain),
                "audit": {"exposed_plaintext_tensors": []}}
    cfg = _make_config(mode, dtype, seed, condition_number)
    logits, cache, audit = qwen_obfuscated_forward(model, input_ids, cfg)
    return {
        "mode": mode, "logits": logits, "cache": cache,
        "correctness": correctness_metrics(logits, plain),
        "argmax_match_rate": argmax_match_rate(logits, plain),
        "nan_or_inf_count": nan_or_inf_count(logits),
        "audit": audit,
    }


def run_decode_step(
    model, prompt_ids: torch.Tensor, next_id: torch.Tensor, mode: str, *,
    dtype: torch.dtype = torch.float32, seed: int = 0, condition_number: float = 1.0,
) -> dict[str, Any]:
    """Prefill the prompt, then decode one more token; compare to full recompute."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if next_id.dim() == 0:
        next_id = next_id.view(1, 1)
    elif next_id.dim() == 1:
        next_id = next_id.unsqueeze(0)
    full = torch.cat([prompt_ids, next_id], dim=1)
    ref_full = qwen_plain_logits(model, full)
    ref_last = ref_full[:, -1, :]

    if mode == "unprotected":
        with torch.no_grad():
            out1 = model(prompt_ids, use_cache=True)
            prefill_len = prompt_ids.shape[1]
            out2 = model(next_id, past_key_values=out1.past_key_values, use_cache=True)
        dec_last = out2.logits[:, -1, :]
        return {
            "mode": mode, "logits": dec_last, "cache": None,
            "prefill_cache_len": prefill_len, "decode_cache_len": prefill_len + 1,
            "correctness": correctness_metrics(dec_last, ref_last),
            "argmax_match_rate": argmax_match_rate(dec_last, ref_last),
            "nan_or_inf_count": nan_or_inf_count(dec_last),
            "audit": {"exposed_plaintext_tensors": []},
        }

    cfg = _make_config(mode, dtype, seed, condition_number)
    _, cache, _ = qwen_obfuscated_forward(model, prompt_ids, cfg)
    prefill_len = cache.length()
    logits, cache, audit = qwen_obfuscated_forward(
        model, next_id, cfg, cache=cache, position_offset=prefill_len)
    dec_last = logits[:, -1, :]
    return {
        "mode": mode, "logits": dec_last, "cache": cache,
        "prefill_cache_len": prefill_len, "decode_cache_len": cache.length(),
        "correctness": correctness_metrics(dec_last, ref_last),
        "argmax_match_rate": argmax_match_rate(dec_last, ref_last),
        "nan_or_inf_count": nan_or_inf_count(dec_last),
        "audit": audit,
    }


def generate_greedy(
    model, prompt_ids: torch.Tensor, max_new_tokens: int, mode: str, *,
    dtype: torch.dtype = torch.float32, seed: int = 0, condition_number: float = 1.0,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Greedy decode ``max_new_tokens`` tokens under one mode. Returns tokens + audit."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if mode == "unprotected":
        gen = prompt_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                nxt = model(gen).logits[:, -1, :].argmax(-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=1)
        return {"mode": mode, "tokens": gen[:, prompt_ids.shape[1]:],
                "full": gen, "audit": {"exposed_plaintext_tensors": []}}

    cfg = _make_config(mode, dtype, seed, condition_number)
    if use_cache:
        logits, cache, audit = qwen_obfuscated_forward(model, prompt_ids, cfg)
        offset = cache.length()
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        toks = [nxt]
        for _ in range(max_new_tokens - 1):
            logits, cache, audit = qwen_obfuscated_forward(
                model, nxt, cfg, cache=cache, position_offset=offset)
            offset = cache.length()
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(nxt)
        gen_new = torch.cat(toks, dim=1)
    else:
        gen = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits, _, audit = qwen_obfuscated_forward(model, gen, cfg)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        gen_new = gen[:, prompt_ids.shape[1]:]
    return {"mode": mode, "tokens": gen_new,
            "full": torch.cat([prompt_ids, gen_new], dim=1), "audit": audit}


__all__ = [
    "qwen_obfuscated_forward",
    "qwen_plain_logits",
    "run_prefill",
    "run_decode_step",
    "generate_greedy",
]
