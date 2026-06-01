"""Stage 5.5b — real-token-prompted real-activation trace collection.

Drives the Stage 6.4c modern decoder model-level wrapper end-to-end with
real ``input_ids`` from either a HuggingFace tokenizer (opt-in) or a
deterministic synthetic-token fallback, then collects per-layer
(plain, visible) tensor pairs at the same attacker boundaries Stage 5.5
already evaluated at the block level. Output feeds
:mod:`pllo.experiments.real_token_activation_attacker`.

Scope:

* **Model-level wrapper, prefill + decode_step.** Trace hooks were added
  in Stage 5.5b as an *additive* feature flag on
  :class:`pllo.hf_wrappers.modern_decoder_model_wrapper.ObfuscatedModernDecoderModelWrapper`
  (``prefill(..., return_traces=True)`` / ``decode_step(..., return_traces=True)``).
  Correctness math is untouched.
* **Real tokenizer is opt-in.** ``attempt_tokenizer_load=False`` (default)
  hits a deterministic synthetic token-ID fallback so pytest never reads
  from the network. ``attempt_real_model_load`` is similarly opt-in.
* **No tensor leakage in JSON.** ``traces`` are held in memory as torch
  tensors for the in-process attacker; the JSON-safe ``trace_summary``
  carries only shapes, sample counts, fingerprints and scalar statistics.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.experiments.modern_decoder_model_probe import (
    ModernDecoderModelWrapperConfig,
    _resolve_weights,
)
from pllo.hf_wrappers.modern_decoder_model_wrapper import (
    ObfuscatedModernDecoderModelWrapper,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
    normalize_mitigation_bundle,
)


DEFAULT_TARGET_TENSORS: tuple[str, ...] = (
    "boundary_input",
    "q",
    "k",
    "v",
    "gate",
    "up",
    "swiglu_intermediate",
    "post_island",
    "final",
)


DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "In a secure system, the user",
    "Machine learning models can",
    "Privacy preserving inference requires",
    "The next token should be",
    "A simple example of encryption is",
    "The transformer architecture uses",
    "Large language models generate",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RealTokenTraceConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    model_id: str | None = None
    attempt_real_model_load: bool = False
    attempt_tokenizer_load: bool = False
    local_files_only: bool = False
    allow_synthetic_fallback: bool = True
    max_layers: int = 2
    max_new_tokens: int = 3
    prompt_max_length: int = 16
    num_prompts: int = 8
    batch_size: int = 1
    use_pad: bool = True
    nonlinear_mode: str = "compatible_islands"
    mitigation_bundle: str = "fresh_perm_plus_sandwich_plus_pad"
    collect_prefill_traces: bool = True
    collect_decode_traces: bool = True
    collect_generation_traces: bool = True
    target_tensors: tuple[str, ...] = DEFAULT_TARGET_TENSORS
    dtype: str = "float32"
    device: str = "cpu"
    # Synthetic-fallback shape.
    synthetic_vocab_size: int = 256
    synthetic_hidden_size: int = 32
    synthetic_intermediate_size: int = 64
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 8


# ---------------------------------------------------------------------------
# Tokenizer / prompt-id helpers
# ---------------------------------------------------------------------------


def _try_load_tokenizer(config: RealTokenTraceConfig) -> dict[str, Any]:
    """Best-effort tokenizer load. Returns metadata + the tokenizer (or None).

    Never raises; on any failure the synthetic-token fallback is honoured.
    """
    if not config.attempt_tokenizer_load:
        return {
            "tokenizer_status": "not_requested",
            "tokenizer_id": None,
            "tokenizer_error": None,
            "tokenizer": None,
        }
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        tok = AutoTokenizer.from_pretrained(
            config.model_id, local_files_only=config.local_files_only,
        )
        if getattr(tok, "pad_token", None) is None and getattr(
            tok, "eos_token", None,
        ) is not None:
            tok.pad_token = tok.eos_token
        return {
            "tokenizer_status": "loaded",
            "tokenizer_id": config.model_id,
            "tokenizer_error": None,
            "tokenizer": tok,
        }
    except Exception as exc:  # noqa: BLE001 — best-effort
        return {
            "tokenizer_status": "unavailable",
            "tokenizer_id": config.model_id,
            "tokenizer_error": str(exc),
            "tokenizer": None,
        }


def _synthetic_input_ids(
    config: RealTokenTraceConfig, vocab_size: int,
) -> torch.Tensor:
    """Deterministic synthetic input_ids in ``[0, vocab_size)``."""
    gen = torch.Generator(device="cpu").manual_seed(config.seed + 11)
    L = max(1, int(config.prompt_max_length))
    N = max(1, int(config.num_prompts))
    safe_vocab = max(2, int(vocab_size))
    return torch.randint(0, safe_vocab, (N, L), generator=gen)


def _encode_prompts(
    tokenizer, prompts: list[str], *, max_length: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Encode prompts to a left-aligned ``[N, L]`` tensor padded with pad_token_id."""
    encs: list[list[int]] = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=True, truncation=True,
                               max_length=max_length)
        if len(ids) < 1:
            ids = [0]
        encs.append(ids[:max_length])
    L = min(max_length, max(len(e) for e in encs))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    padded = [e + [int(pad_id)] * (L - len(e)) for e in encs]
    return torch.tensor(padded, dtype=torch.long), encs


def _build_prompt_input_ids(
    config: RealTokenTraceConfig,
    tokenizer_info: dict[str, Any],
    vocab_size: int,
) -> dict[str, Any]:
    """Return ``{"input_ids", "token_source", "prompts_used", "encoded_lengths"}``."""
    tokenizer = tokenizer_info.get("tokenizer")
    prompts = list(DEFAULT_PROMPTS[: config.num_prompts])
    if tokenizer is None:
        ids = _synthetic_input_ids(config, vocab_size)
        return {
            "input_ids": ids,
            "token_source": "synthetic_token_ids",
            "prompts_used": prompts,
            "encoded_lengths": [int(ids.shape[-1])] * int(ids.shape[0]),
            "vocab_size_used": int(vocab_size),
            "tokenizer_status": tokenizer_info["tokenizer_status"],
        }
    try:
        ids, raw = _encode_prompts(
            tokenizer, prompts, max_length=int(config.prompt_max_length),
        )
        # Clamp into the model's vocab if it doesn't match the tokenizer's vocab.
        if int(ids.max().item()) >= int(vocab_size):
            ids = ids % int(vocab_size)
        return {
            "input_ids": ids,
            "token_source": "real_tokenizer",
            "prompts_used": prompts,
            "encoded_lengths": [len(r) for r in raw],
            "vocab_size_used": int(vocab_size),
            "tokenizer_status": tokenizer_info["tokenizer_status"],
        }
    except Exception as exc:  # noqa: BLE001
        ids = _synthetic_input_ids(config, vocab_size)
        return {
            "input_ids": ids,
            "token_source": "synthetic_token_ids",
            "prompts_used": prompts,
            "encoded_lengths": [int(ids.shape[-1])] * int(ids.shape[0]),
            "vocab_size_used": int(vocab_size),
            "tokenizer_status": "encode_failed",
            "tokenizer_encode_error": str(exc),
        }


# ---------------------------------------------------------------------------
# Flattening / fingerprinting
# ---------------------------------------------------------------------------


def _flatten_pair(
    name: str, plain: torch.Tensor, visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reshape a (plain, visible) pair into a 2D ``[N, D]`` dataset.

    Per-head tensors ``[B, H, S, D]`` flatten heads into rows; per-token
    tensors ``[B, S, F]`` flatten ``B × S`` into rows. ``[B, 1, F]``
    decode-step shapes are handled by the per-token branch as well.
    """
    if plain.ndim == 4:
        B, H, S, D = plain.shape
        plain_flat = plain.permute(0, 2, 1, 3).reshape(B * S * H, D)
        visible_flat = visible.permute(0, 2, 1, 3).reshape(B * S * H, D)
    elif plain.ndim == 3:
        B, S, F = plain.shape
        plain_flat = plain.reshape(B * S, F)
        visible_flat = visible.reshape(B * S, F)
    elif plain.ndim == 2:
        plain_flat = plain
        visible_flat = visible
    else:
        raise ValueError(f"unexpected ndim {plain.ndim} for tensor {name!r}")
    return {"plain": plain_flat, "visible": visible_flat}


def _fingerprint(t: torch.Tensor) -> str:
    buf = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def _tensor_statistics(t: torch.Tensor) -> dict[str, float]:
    flat = t.detach().to(torch.float32).reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "abs_max": float(flat.abs().max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "fingerprint_sha256_prefix": _fingerprint(t),
    }


# ---------------------------------------------------------------------------
# Trace accumulation across (layer, prompt, step)
# ---------------------------------------------------------------------------


def _accumulate(
    accum: dict[str, dict[str, list[torch.Tensor]]],
    layer_traces: dict[str, torch.Tensor],
    target_tensors: tuple[str, ...],
) -> None:
    """Append plain/visible halves of each known target into ``accum``."""
    for name in target_tensors:
        plain_key = f"{name}_plain"
        visible_key = f"{name}_visible"
        if plain_key not in layer_traces or visible_key not in layer_traces:
            continue
        pair = _flatten_pair(
            name, layer_traces[plain_key], layer_traces[visible_key],
        )
        accum[name]["plain"].append(pair["plain"])
        accum[name]["visible"].append(pair["visible"])


def _stitch_summary(
    accum: dict[str, dict[str, list[torch.Tensor]]],
    target_tensors: tuple[str, ...],
    source: str,
    bundle: str,
    use_pad: bool,
    scope: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    stitched: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, Any] = {}
    for name in target_tensors:
        if not accum[name]["plain"]:
            continue
        plain_t = torch.cat(accum[name]["plain"], dim=0)
        visible_t = torch.cat(accum[name]["visible"], dim=0)
        stitched[name] = {"plain": plain_t, "visible": visible_t}
        summary[name] = {
            "tensor_name": name,
            "scope": scope,
            "num_samples": int(plain_t.shape[0]),
            "feature_dim": int(plain_t.shape[-1]),
            "plain_shape": list(plain_t.shape),
            "visible_shape": list(visible_t.shape),
            "flattened_feature_dim": int(plain_t.shape[-1]),
            "source": source,
            "mitigation_bundle": bundle,
            "use_pad": bool(use_pad),
            "plain_statistics": _tensor_statistics(plain_t),
            "visible_statistics": _tensor_statistics(visible_t),
        }
    return stitched, summary


# ---------------------------------------------------------------------------
# Top-level collection entry point
# ---------------------------------------------------------------------------


def collect_real_token_traces(
    config: RealTokenTraceConfig,
) -> dict[str, Any]:
    """Run prefill + decode_step + greedy_generate on real-token prompts.

    Returned dict shape:

    .. code-block:: text

       {
         "config":            <RealTokenTraceConfig dict>,
         "model_loading":     <_try_load_real_block dict>,
         "tokenizer_loading": <tokenizer status dict>,
         "source":            "synthetic_block" | "tiny_random_llama_model" | ...,
         "prompt_summary":    <prompt set metadata>,
         "trace_summary":     <per-scope per-tensor JSON-safe summary>,
         "traces":            {scope: {tensor_name: {"plain": Tensor, "visible": Tensor}}},
         "generation_summary":<greedy token match summary>,
         "metadata":          <run-level dict>,
       }
    """
    bundle = normalize_mitigation_bundle(config.mitigation_bundle)
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)

    # Reuse Stage 6.4c probe weight loader.
    probe_cfg = ModernDecoderModelWrapperConfig(
        model_id=config.model_id,
        attempt_real_model_load=config.attempt_real_model_load,
        allow_synthetic_fallback=config.allow_synthetic_fallback,
        local_files_only=config.local_files_only,
        nonlinear_mode=config.nonlinear_mode,
        mitigation_bundle=bundle,
        use_pad=config.use_pad,
        max_layers=config.max_layers,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        synthetic_vocab_size=config.synthetic_vocab_size,
        synthetic_hidden_size=config.synthetic_hidden_size,
        synthetic_intermediate_size=config.synthetic_intermediate_size,
        synthetic_num_attention_heads=config.synthetic_num_attention_heads,
        synthetic_num_key_value_heads=config.synthetic_num_key_value_heads,
        synthetic_head_dim=config.synthetic_head_dim,
    )
    spec, weights, load_record, source = _resolve_weights(
        probe_cfg, dtype, device,
    )

    tokenizer_info = _try_load_tokenizer(config)
    prompt_pkg = _build_prompt_input_ids(
        config, tokenizer_info, vocab_size=int(weights.vocab_size),
    )
    input_ids = prompt_pkg["input_ids"].to(device=device)

    wrapper = ObfuscatedModernDecoderModelWrapper(
        weights, dtype=dtype, device=device,
        use_pad=config.use_pad,
        nonlinear_mode=config.nonlinear_mode,
        mitigation_bundle=bundle,
        collect_traces=True,
    )

    prefill_accum: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"plain": [], "visible": []} for name in config.target_tensors
    }
    decode_accum: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"plain": [], "visible": []} for name in config.target_tensors
    }

    per_prompt_metadata: list[dict[str, Any]] = []
    generation_token_matches: list[float] = []
    prefill_allclose: list[bool] = []
    decode_allclose: list[bool] = []
    decode_step_log: list[dict[str, Any]] = []
    generation_step_log: list[dict[str, Any]] = []
    rope_positions_seen: list[int] = []

    N = int(input_ids.shape[0])
    for prompt_idx in range(N):
        prompt_ids = input_ids[prompt_idx : prompt_idx + 1]  # [1, L]
        # -------- prefill --------
        pf_out = wrapper.prefill(prompt_ids, return_traces=True)
        if config.collect_prefill_traces:
            for layer_traces in pf_out["per_layer_traces"]:
                _accumulate(prefill_accum, layer_traces, config.target_tensors)
        prefill_allclose.append(
            bool(pf_out["report"]["logits_metrics"]["allclose"])
        )

        # -------- greedy + decode --------
        # Greedy generate against plain reference; capture per-step traces by
        # running decode_step manually with return_traces=True.
        cache = pf_out["kv_cache"]
        plain_caches = pf_out["plain_layer_caches"]
        next_token_obf = pf_out["logits_recovered"][:, -1, :].argmax(dim=-1)
        next_token_plain = pf_out["logits_plain"][:, -1, :].argmax(dim=-1)
        position = int(prompt_ids.shape[-1])
        obf_tokens: list[int] = [int(next_token_obf.item())]
        plain_tokens: list[int] = [int(next_token_plain.item())]
        rope_positions_seen.append(position)
        cur_token = next_token_obf
        decode_local_steps: list[dict[str, Any]] = []
        for step in range(max(0, int(config.max_new_tokens) - 1)):
            cache_seq_len_before = cache.total_seq_len
            ds_out = wrapper.decode_step(
                cur_token.unsqueeze(-1), cache, position,
                plain_layer_caches=plain_caches,
                return_traces=True,
            )
            cache = ds_out["kv_cache"]
            plain_caches = ds_out["plain_layer_caches"]
            if config.collect_decode_traces:
                for layer_traces in ds_out["per_layer_traces"]:
                    _accumulate(decode_accum, layer_traces, config.target_tensors)
            decode_allclose.append(
                bool(ds_out["report"]["logits_metrics"]["allclose"])
            )
            decode_local_steps.append({
                "decode_step_index": int(step),
                "position": int(position),
                "cache_seq_len_before": int(cache_seq_len_before),
                "cache_seq_len_after": int(cache.total_seq_len),
                "top1_match_rate": float(
                    ds_out["report"]["logits_metrics"]["top1_match_rate"]
                ),
            })
            cur_token = ds_out["next_logits_recovered"][:, -1, :].argmax(dim=-1)
            plain_token = ds_out["next_logits_plain"][:, -1, :].argmax(dim=-1)
            obf_tokens.append(int(cur_token.item()))
            plain_tokens.append(int(plain_token.item()))
            rope_positions_seen.append(position)
            position += 1
        if decode_local_steps:
            decode_step_log.append({
                "prompt_index": int(prompt_idx),
                "steps": decode_local_steps,
            })
        token_match = float(
            sum(1 for a, b in zip(obf_tokens, plain_tokens) if a == b)
            / max(1, len(obf_tokens))
        )
        generation_token_matches.append(token_match)
        generation_step_log.append({
            "prompt_index": int(prompt_idx),
            "obf_tokens": obf_tokens,
            "plain_tokens": plain_tokens,
            "token_match_rate": token_match,
            "sequence_exact_match": (obf_tokens == plain_tokens),
        })
        per_prompt_metadata.append({
            "prompt_index": int(prompt_idx),
            "prompt_text": (
                prompt_pkg["prompts_used"][prompt_idx]
                if prompt_idx < len(prompt_pkg["prompts_used"])
                else None
            ),
            "prompt_length": int(prompt_ids.shape[-1]),
        })

    prefill_traces, prefill_summary = _stitch_summary(
        prefill_accum, config.target_tensors, source, bundle,
        config.use_pad, scope="prefill",
    )
    decode_traces, decode_summary = _stitch_summary(
        decode_accum, config.target_tensors, source, bundle,
        config.use_pad, scope="decode",
    )

    generation_summary = {
        "max_new_tokens": int(config.max_new_tokens),
        "num_prompts": int(N),
        "mean_token_match_rate": float(
            sum(generation_token_matches) / max(1, len(generation_token_matches))
        ),
        "all_sequences_exact_match": bool(
            all(g["sequence_exact_match"] for g in generation_step_log)
        ),
        "per_prompt": generation_step_log,
    }

    metadata = {
        "mitigation_bundle": bundle,
        "mitigation_bundle_metadata": bundle_metadata(
            bundle, use_pad=config.use_pad, online_extra_matmul_count=0,
        ),
        "use_pad": bool(config.use_pad),
        "nonlinear_mode": config.nonlinear_mode,
        "num_prompts": int(N),
        "max_new_tokens": int(config.max_new_tokens),
        "prompt_max_length": int(config.prompt_max_length),
        "num_layers_used": int(len(weights.layers)),
        "all_prefill_allclose": bool(all(prefill_allclose)) if prefill_allclose else False,
        "all_decode_allclose": (
            bool(all(decode_allclose)) if decode_allclose else None
        ),
        "rope_positions_seen_min": (
            int(min(rope_positions_seen)) if rope_positions_seen else None
        ),
        "rope_positions_seen_max": (
            int(max(rope_positions_seen)) if rope_positions_seen else None
        ),
        "rope_position_increment": True,
    }

    return {
        "config": asdict(config),
        "model_loading": load_record,
        "tokenizer_loading": {
            k: v for k, v in tokenizer_info.items() if k != "tokenizer"
        },
        "source": source,
        "prompt_summary": {
            "token_source": prompt_pkg["token_source"],
            "tokenizer_status": prompt_pkg.get(
                "tokenizer_status", tokenizer_info["tokenizer_status"]
            ),
            "num_prompts": int(N),
            "prompt_max_length": int(config.prompt_max_length),
            "prompts_used": prompt_pkg["prompts_used"],
            "encoded_lengths": prompt_pkg["encoded_lengths"],
            "vocab_size_used": int(prompt_pkg["vocab_size_used"]),
        },
        "block_spec_summary": {
            "model_family": spec.model_family,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_attention_heads": spec.num_attention_heads,
            "num_key_value_heads": spec.num_key_value_heads,
            "head_dim": spec.head_dim,
            "attention_variant": spec.attention_variant,
            "rope_base": spec.rope_base,
        },
        "traces": {
            "prefill": prefill_traces,
            "decode": decode_traces,
        },
        "trace_summary": {
            "prefill": prefill_summary,
            "decode": decode_summary,
        },
        "decode_step_log": decode_step_log,
        "generation_summary": generation_summary,
        "per_prompt_metadata": per_prompt_metadata,
        "metadata": metadata,
    }


__all__ = [
    "DEFAULT_PROMPTS",
    "DEFAULT_TARGET_TENSORS",
    "RealTokenTraceConfig",
    "collect_real_token_traces",
]
