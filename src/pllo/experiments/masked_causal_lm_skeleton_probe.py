"""Stage 6.8 -- full masked CausalLM skeleton probe.

Runs prefill (trusted input boundary -> masked decoder layers with per-layer
mask handoff -> masked-logits output boundary) and a bounded greedy decode
loop, comparing the masked path against a plain reference. CPU-only,
correctness-first; synthetic weights; no HF checkpoints, no transformers, no
GPU. No formal, cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.ops.masked_causal_lm_skeleton import (
    MaskedCausalLMSkeletonConfig,
    causal_lm_masked_greedy_decode,
    generate_skeleton_masks,
    init_masked_causal_lm_skeleton_weights,
)

_REQUIRED_STATEMENT = (
    "This stage validates a bounded full masked CausalLM skeleton with "
    "trusted input embedding and trusted masked-logits recovery. It does not "
    "validate production generation or claim semantic security."
)

_CAVEATS = [
    "Synthetic CausalLM skeleton, not a real HF full model",
    "No tokenizer or chat template",
    "Greedy decode only in this stage",
    "No full-vocab LM-head optimization",
    "Vocab permutation+scaling is weaker than dense vocab masking",
    "Attention scores remain visible in current masked attention design",
    "RoPE-compatible masks preserve pair partition",
    "KV cache masks are reused within a generation session",
    "Final output text semantics are not protected once returned to the user",
]


@dataclass
class MaskedCausalLMSkeletonProbeConfig:
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
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 2031


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _to_skeleton_config(
    cfg: MaskedCausalLMSkeletonProbeConfig,
) -> MaskedCausalLMSkeletonConfig:
    return MaskedCausalLMSkeletonConfig(
        batch_size=cfg.batch_size, prefill_seq_len=cfg.prefill_seq_len,
        decode_steps=cfg.decode_steps, vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_layers=cfg.num_layers, num_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_key_value_heads, rope_base=cfg.rope_base,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=cfg.tie_word_embeddings,
        use_input_pad=cfg.use_input_pad, mask_family=cfg.mask_family,
        dtype=_dtype(cfg.dtype), device=cfg.device, seed=cfg.seed,
    )


def run_masked_causal_lm_skeleton_probe(
    config: MaskedCausalLMSkeletonProbeConfig,
) -> dict[str, Any]:
    scfg = _to_skeleton_config(config)
    scfg.validate()
    device = torch.device(scfg.device)
    g = torch.Generator(device=device).manual_seed(scfg.seed)

    weights = init_masked_causal_lm_skeleton_weights(scfg, g)
    masks = generate_skeleton_masks(scfg, g)
    input_ids = torch.randint(0, scfg.vocab_size,
                              (scfg.batch_size, scfg.prefill_seq_len),
                              generator=g, device=device)

    run = causal_lm_masked_greedy_decode(input_ids, weights, masks, scfg)
    prefill = run["prefill_metrics"]
    decode = run["decode_step_metrics"]
    decode_allclose = all(
        s["sampled_token_match"] == 1.0
        and s["final_hidden_error"] <= 1e-8
        and s["masked_logits_error"] <= 1e-8
        and s["recovered_logits_error"] <= 1e-8
        and s["per_layer_output_error"] <= 1e-8
        and s["per_layer_cache_append_key_error"] <= 1e-8
        and s["per_layer_cache_append_value_error"] <= 1e-8
        for s in decode)
    all_allclose = bool(prefill["allclose"] and decode_allclose
                        and run["token_match_rate"] == 1.0)

    return {
        "stage": "6.8_masked_causal_lm_skeleton",
        "experiment": "masked_causal_lm_skeleton_probe",
        "status": "ok",
        "statement": _REQUIRED_STATEMENT,
        "config": asdict(config),
        "prefill_metrics": prefill,
        "decode_step_metrics": decode,
        "token_match_rate": run["token_match_rate"],
        "all_allclose": all_allclose,
        "mask_metadata": masks.metadata,
        "metadata": {
            "stage": "6.8_masked_causal_lm_skeleton",
            "no_intermediate_tee": True,
            "input_ids_visible_to_gpu": False,
            "plaintext_embedding_visible_to_gpu": False,
            "plaintext_logits_visible_to_gpu": False,
            "masked_logits_visible_to_gpu": True,
            "logits_recovered_in_tee": True,
            "sampling_boundary": "trusted_side",
            "decoder_runs_on_gpu_assumption": "masked_tensors_only",
            "security_status":
                "operator_compatible_leakage_reduction_not_semantic_security",
            "semantic_security_claimed": False,
            "formal_security_claimed": False,
            "cryptographic_security_claimed": False,
            "num_layers": scfg.num_layers,
            "residual_mask_handoff": "per_layer_orthogonal_masks",
            "caveats": _CAVEATS,
        },
        "limitations": _CAVEATS,
    }


__all__ = [
    "MaskedCausalLMSkeletonProbeConfig",
    "run_masked_causal_lm_skeleton_probe",
]
