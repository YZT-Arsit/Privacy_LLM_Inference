"""Stage 6.9 -- HF full-model / local tiny-checkpoint masked CausalLM probe.

Builds (or locally loads) a tiny LLaMA / Qwen2 ``...ForCausalLM``, decomposes
it into embedding / decoder layers / final norm / LM head, and runs the masked
pipeline (Stage 6.6 + 6.7 + 6.8) over the extracted weights, comparing against
our extracted-weight plaintext reference. Compact summary metrics only; no
tensor dumps. CPU-only, no network, no HF ``generate``. No formal,
cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.hf_wrappers.hf_causal_lm_skeleton import (
    HFCausalLMSkeletonConfig,
    extract_hf_causal_lm_skeleton_weights,
    generate_hf_causal_lm_masks,
    has_transformers,
    hf_causal_lm_masked_greedy_decode,
    make_random_tiny_hf_causal_lm,
)

__all__ = [
    "HFCausalLMSkeletonProbeConfig",
    "run_hf_causal_lm_skeleton_probe",
]


_DTYPES = {"float64": torch.float64, "float32": torch.float32}

REQUIRED_STATEMENT = (
    "This stage validates a local HuggingFace-style full CausalLM skeleton "
    "using extracted weights and trusted input/output boundaries. It does "
    "not validate production generation or claim semantic security."
)

LIMITATIONS = [
    "HF extracted-weight reference, not HF generate / forward.",
    "tiny/random model by default.",
    "no tokenizer / chat template.",
    "greedy decode only.",
    "no large-checkpoint benchmark.",
    "no production inference.",
    "no semantic / formal / cryptographic security claim.",
    "attention scores remain GPU-visible.",
    "vocab permutation+scaling is weaker than dense vocab masking.",
]


@dataclass
class HFCausalLMSkeletonProbeConfig:
    model_family: str = "llama"
    local_model_path: str | None = None
    use_random_tiny_if_no_path: bool = True
    batch_size: int = 1
    prefill_seq_len: int = 4
    decode_steps: int = 2
    max_layers: int = 2
    max_vocab_size: int = 512
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 2033
    mask_family: str = "pairwise_complex_scaling"
    use_input_pad: bool = False


def _skeleton_config(
    pc: HFCausalLMSkeletonProbeConfig,
) -> HFCausalLMSkeletonConfig:
    return HFCausalLMSkeletonConfig(
        model_family=pc.model_family, local_model_path=pc.local_model_path,
        batch_size=pc.batch_size, prefill_seq_len=pc.prefill_seq_len,
        decode_steps=pc.decode_steps, max_layers=pc.max_layers,
        max_vocab_size=pc.max_vocab_size, dtype=_DTYPES[pc.dtype],
        device=pc.device, seed=pc.seed, mask_family=pc.mask_family,
        use_input_pad=pc.use_input_pad)


def _security_metadata(pc: HFCausalLMSkeletonProbeConfig, source: str,
                       ) -> dict[str, Any]:
    return {
        "stage": "6.9_hf_causal_lm_skeleton",
        "model_family": pc.model_family,
        "source": source,
        "local_files_only": True,
        "no_network_download": True,
        "no_gpu_required": True,
        "input_ids_visible_to_gpu": False,
        "plaintext_embedding_visible_to_gpu": False,
        "plaintext_logits_visible_to_gpu": False,
        "masked_logits_visible_to_gpu": True,
        "logits_recovered_in_tee": True,
        "sampling_boundary": "trusted_side",
        "security_status":
            "operator_compatible_leakage_reduction_not_semantic_security",
        "semantic_security_claimed": False,
        "formal_security_claimed": False,
        "cryptographic_security_claimed": False,
    }


def _load_local_model(pc: HFCausalLMSkeletonProbeConfig) -> tuple[Any, Any]:
    """Load a local checkpoint with ``local_files_only=True``. Never downloads.
    Returns ``(model, model_config)`` or raises (caller turns it into skip)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        pc.local_model_path, local_files_only=True, dtype=torch.float32)
    model.eval()
    model.to("cpu")
    return model, model.config


def run_hf_causal_lm_skeleton_probe(
    config: HFCausalLMSkeletonProbeConfig,
) -> dict[str, Any]:
    """Run the Stage 6.9 masked full-CausalLM skeleton probe."""
    if not has_transformers():
        return {
            "status": "skipped_transformers_unavailable",
            "metadata": _security_metadata(config, source="none"),
            "required_statement": REQUIRED_STATEMENT,
            "limitations": LIMITATIONS,
        }

    skel_cfg = _skeleton_config(config)
    dtype = _DTYPES[config.dtype]

    # --- model source ---------------------------------------------------
    if config.local_model_path:
        try:
            model, model_config = _load_local_model(config)
        except Exception as exc:  # missing path / not a model / load failure
            return {
                "status": "skipped_local_model_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "metadata": _security_metadata(config, source="local_checkpoint"),
                "required_statement": REQUIRED_STATEMENT,
                "limitations": LIMITATIONS,
            }
        vocab = int(getattr(model_config, "vocab_size", 0))
        hidden = int(getattr(model_config, "hidden_size", 0))
        if vocab > config.max_vocab_size or hidden > 4096:
            return {
                "status": "skipped_local_model_too_large",
                "reason": (f"vocab_size={vocab} hidden_size={hidden} exceeds "
                           f"bounds (max_vocab_size={config.max_vocab_size})"),
                "metadata": _security_metadata(config, source="local_checkpoint"),
                "required_statement": REQUIRED_STATEMENT,
                "limitations": LIMITATIONS,
            }
        source = "local_checkpoint"
    else:
        if not config.use_random_tiny_if_no_path:
            return {
                "status": "skipped_no_model_source",
                "metadata": _security_metadata(config, source="none"),
                "required_statement": REQUIRED_STATEMENT,
                "limitations": LIMITATIONS,
            }
        model, model_config = make_random_tiny_hf_causal_lm(skel_cfg)
        source = "random_tiny_hf_model"

    # --- decompose + run ------------------------------------------------
    weights, layer_configs, extract_meta = \
        extract_hf_causal_lm_skeleton_weights(
            model, model_config, max_layers=config.max_layers, dtype=dtype,
            device=config.device, mask_family=config.mask_family)
    masks = generate_hf_causal_lm_masks(weights, layer_configs, skel_cfg)

    g = torch.Generator(device=torch.device(config.device)).manual_seed(
        config.seed)
    vocab = extract_meta["vocab_size"]
    input_ids = torch.randint(
        0, vocab, (config.batch_size, config.prefill_seq_len),
        generator=g, device=torch.device(config.device))

    result = hf_causal_lm_masked_greedy_decode(
        input_ids, weights, layer_configs, masks, skel_cfg)

    pre = result["prefill_metrics"]
    per_layer_max = max(
        (v for layer in pre["per_layer"] for k, v in layer.items()
         if k != "layer"), default=0.0)

    metadata = _security_metadata(config, source=source)
    metadata.update({
        "model_type": extract_meta["model_type"],
        "num_layers_extracted": extract_meta["num_layers_extracted"],
        "hidden_size": extract_meta["hidden_size"],
        "vocab_size": extract_meta["vocab_size"],
        "tie_word_embeddings": extract_meta["tie_word_embeddings"],
        "handoff_skip_term_needs_gemm":
            masks.metadata["handoff_skip_term_needs_gemm"],
        "use_input_pad": config.use_input_pad,
    })

    return {
        "status": "ok",
        "metadata": metadata,
        "config": {
            "model_family": config.model_family,
            "batch_size": config.batch_size,
            "prefill_seq_len": config.prefill_seq_len,
            "decode_steps": config.decode_steps,
            "max_layers": config.max_layers,
            "max_vocab_size": config.max_vocab_size,
            "mask_family": config.mask_family,
            "dtype": config.dtype,
        },
        "prefill_metrics": {
            "embedding_mask_max_abs_error":
                pre["embedding_mask_max_abs_error"],
            "per_layer_max_abs_error": per_layer_max,
            "per_layer_handoff_max_abs_error":
                max(pre["per_layer_handoff_max_abs_error"]),
            "final_hidden_max_abs_error": pre["final_hidden_max_abs_error"],
            "masked_logits_max_abs_error": pre["masked_logits_max_abs_error"],
            "recovered_logits_max_abs_error":
                pre["recovered_logits_max_abs_error"],
            "greedy_token_match_rate": pre["greedy_token_match_rate"],
            "allclose": pre["allclose"],
        },
        "decode_metrics": {
            "token_match_rate": result["token_match_rate"],
            "num_steps": len(result["decode_step_metrics"]),
            "max_per_step_output_error": max(
                (s["per_layer_output_error"]
                 for s in result["decode_step_metrics"]), default=0.0),
            "max_per_step_logit_error": max(
                (s["recovered_logits_error"]
                 for s in result["decode_step_metrics"]), default=0.0),
            "per_step": result["decode_step_metrics"],
        },
        "generated_plain_tokens":
            result["generated_plain_tokens"].tolist(),
        "generated_from_masked_tokens":
            result["generated_from_masked_tokens"].tolist(),
        "required_statement": REQUIRED_STATEMENT,
        "limitations": LIMITATIONS,
    }
