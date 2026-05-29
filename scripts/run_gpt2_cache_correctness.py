#!/usr/bin/env python
"""Run Stage 4.8 GPT-2 prefill/decode/KV cache correctness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import top1_match_rate
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper, gpt2_cache_invariant_metrics
from pllo.model_zoo import ExternalModelConfig, get_model_loader, torch_dtype_from_string


def parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--use-pad", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gpt2_cache_correctness.json")
    )
    return parser.parse_args()


def _summarize_logits_metrics(
    plain: torch.Tensor,
    recovered: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    metrics = compute_correctness_metrics(plain, recovered, atol=atol, rtol=rtol)
    metrics["top1_match_rate"] = top1_match_rate(plain, recovered)
    return metrics


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_string(args.dtype, args.device)
    device = torch.device(args.device)
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)

    config = ExternalModelConfig(
        source="huggingface",
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    _, model = get_model_loader("huggingface").load(config)

    vocab_size = model.config.vocab_size
    prompt_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.prompt_len), device=device
    )
    decode_token_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.decode_steps), device=device
    )

    c_attn_class_before = type(model.transformer.h[0].attn.c_attn).__name__

    wrapper = ObfuscatedGPT2ModelWrapper(
        model=model,
        dtype=dtype,
        device=device,
        use_pad=args.use_pad,
    )

    with torch.no_grad():
        # ---- Plain reference path ----
        plain_prefill_out = model(prompt_ids, use_cache=True)
        plain_prefill_logits = plain_prefill_out.logits
        plain_past = plain_prefill_out.past_key_values

        plain_decode_logits_list: list[torch.Tensor] = []
        for step in range(args.decode_steps):
            step_ids = decode_token_ids[:, step : step + 1]
            plain_step_out = model(
                step_ids, past_key_values=plain_past, use_cache=True
            )
            plain_decode_logits_list.append(plain_step_out.logits)
            plain_past = plain_step_out.past_key_values

        # ---- Obfuscated path ----
        recovered_prefill_logits, obf_cache = wrapper.prefill(prompt_ids)
        cache_after_prefill_seq_len = obf_cache.seq_len

        # Capture session-level mask identities at prefill time, so we can
        # confirm they do not get re-sampled during subsequent decode steps.
        prefill_layer_mask_ids = [
            (
                id(layer.key_masks),
                id(layer.value_masks),
                id(layer.key_mask_inverses),
                id(layer.value_mask_inverses),
            )
            for layer in obf_cache.layers
        ]

        recovered_decode_logits_list: list[torch.Tensor] = []
        for step in range(args.decode_steps):
            step_ids = decode_token_ids[:, step : step + 1]
            step_logits, obf_cache = wrapper.decode_step(step_ids, obf_cache)
            recovered_decode_logits_list.append(step_logits)

        post_decode_layer_mask_ids = [
            (
                id(layer.key_masks),
                id(layer.value_masks),
                id(layer.key_mask_inverses),
                id(layer.value_mask_inverses),
            )
            for layer in obf_cache.layers
        ]

    c_attn_class_after = type(model.transformer.h[0].attn.c_attn).__name__

    prefill_metrics = _summarize_logits_metrics(
        plain_prefill_logits, recovered_prefill_logits, atol, rtol
    )

    per_step_metrics: list[dict] = []
    max_abs_error_max = 0.0
    top1_match_rate_min = 1.0
    allclose_all = True
    for step, (plain_step, rec_step) in enumerate(
        zip(plain_decode_logits_list, recovered_decode_logits_list)
    ):
        m = _summarize_logits_metrics(plain_step, rec_step, atol, rtol)
        m["step"] = step
        per_step_metrics.append(m)
        max_abs_error_max = max(max_abs_error_max, float(m["max_abs_error"]))
        top1_match_rate_min = min(top1_match_rate_min, float(m["top1_match_rate"]))
        allclose_all = allclose_all and bool(m["allclose"])

    cache_metrics = gpt2_cache_invariant_metrics(obf_cache, atol=atol, rtol=rtol)

    masks_unchanged = prefill_layer_mask_ids == post_decode_layer_mask_ids

    result = {
        "config": {
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "decode_steps": args.decode_steps,
            "device": args.device,
            "dtype": args.dtype,
            "use_pad": args.use_pad,
        },
        "prefill_logits_metrics": prefill_metrics,
        "decode_logits_metrics": {
            "per_step": per_step_metrics,
            "max_abs_error_max": max_abs_error_max,
            "top1_match_rate_min": top1_match_rate_min,
            "allclose_all": allclose_all,
        },
        "cache_invariant_metrics": cache_metrics,
        "scope": {
            "gpt2_prefill_decode": True,
            "kv_cache": True,
            "generation": False,
            "sampling": False,
            "hf_generate": False,
            "hf_module_replacement": False,
            "trusted_layernorm": True,
            "trusted_activation": True,
            "modelscope": False,
            "real_tee": False,
        },
        "cache_design": {
            "uses_internal_obfuscated_cache": True,
            "uses_hf_past_key_values_directly": False,
            "cache_shape": "[batch, heads, seq, head_dim]",
            "same_mask_within_session": masks_unchanged,
            "session_seq_len_after_prefill": cache_after_prefill_seq_len,
            "session_seq_len_after_decode": obf_cache.seq_len,
        },
        "hf_model_integrity": {
            "c_attn_class_before": c_attn_class_before,
            "c_attn_class_after": c_attn_class_after,
            "c_attn_class_unchanged": c_attn_class_before == c_attn_class_after,
            "tied_embedding": model.lm_head.weight is model.transformer.wte.weight,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "use_pad": args.use_pad,
                "prefill_max_abs_error": prefill_metrics["max_abs_error"],
                "prefill_allclose": prefill_metrics["allclose"],
                "prefill_top1": prefill_metrics["top1_match_rate"],
                "decode_max_abs_error_max": max_abs_error_max,
                "decode_top1_min": top1_match_rate_min,
                "decode_allclose_all": allclose_all,
                "cache_max_key_error": cache_metrics["max_key_error"],
                "cache_max_value_error": cache_metrics["max_value_error"],
                "cache_allclose": cache_metrics["allclose"],
                "same_mask_within_session": masks_unchanged,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
