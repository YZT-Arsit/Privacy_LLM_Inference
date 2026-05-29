#!/usr/bin/env python
"""Run Stage 4.9 GPT-2 greedy generation correctness."""

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
from pllo.evaluation.correctness import (
    sequence_exact_match,
    token_match_rate,
    top1_match_rate,
)
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
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--use-pad", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gpt2_generation_correctness.json")
    )
    return parser.parse_args()


def plain_greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Hand-written HF greedy loop returning (generated_ids, per_step_last_logits).

    Intentionally avoids ``model.generate()`` so the per-step logits used for
    next-token selection can be captured for direct comparison.
    """
    step_logits: list[torch.Tensor] = []
    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=True)
        last_logits = prefill_out.logits[:, -1:, :]
        step_logits.append(last_logits)
        next_token = last_logits[:, -1, :].argmax(dim=-1)
        new_tokens = [next_token]
        past = prefill_out.past_key_values
        for _ in range(max_new_tokens - 1):
            step_out = model(
                next_token.unsqueeze(-1),
                past_key_values=past,
                use_cache=True,
            )
            past = step_out.past_key_values
            step_logits.append(step_out.logits)
            next_token = step_out.logits[:, -1, :].argmax(dim=-1)
            new_tokens.append(next_token)
    generated_ids = torch.cat([input_ids, torch.stack(new_tokens, dim=1)], dim=1)
    return generated_ids, step_logits


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

    c_attn_class_before = type(model.transformer.h[0].attn.c_attn).__name__
    tied_before = model.lm_head.weight is model.transformer.wte.weight

    wrapper = ObfuscatedGPT2ModelWrapper(
        model=model,
        dtype=dtype,
        device=device,
        use_pad=args.use_pad,
    )

    plain_generated, plain_step_logits = plain_greedy_generate(
        model, prompt_ids, args.max_new_tokens
    )
    with torch.no_grad():
        obf_generated, trace = wrapper.generate_greedy(prompt_ids, args.max_new_tokens)
    obf_step_logits = trace["step_logits"]
    obf_cache = trace["final_cache"]

    c_attn_class_after = type(model.transformer.h[0].attn.c_attn).__name__
    tied_after = model.lm_head.weight is model.transformer.wte.weight

    # ---- Generation metrics ----
    new_token_slice = slice(args.prompt_len, args.prompt_len + args.max_new_tokens)
    gen_metrics = {
        "token_match_rate": token_match_rate(
            plain_generated[:, new_token_slice], obf_generated[:, new_token_slice]
        ),
        "sequence_exact_match": sequence_exact_match(
            plain_generated[:, new_token_slice], obf_generated[:, new_token_slice]
        ),
        "top1_match_rate": top1_match_rate(plain_generated, obf_generated),
        "full_sequence_token_match_rate": token_match_rate(plain_generated, obf_generated),
    }

    # ---- Per-step logits metrics ----
    per_step: list[dict] = []
    max_abs_error_max = 0.0
    top1_min = 1.0
    allclose_all = True
    for step, (plain_step, obf_step) in enumerate(zip(plain_step_logits, obf_step_logits)):
        m = compute_correctness_metrics(plain_step, obf_step, atol=atol, rtol=rtol)
        m["top1_match_rate"] = top1_match_rate(plain_step, obf_step)
        m["step"] = step
        m["source"] = "prefill_last" if step == 0 else "decode"
        per_step.append(m)
        max_abs_error_max = max(max_abs_error_max, float(m["max_abs_error"]))
        top1_min = min(top1_min, float(m["top1_match_rate"]))
        allclose_all = allclose_all and bool(m["allclose"])

    cache_metrics = gpt2_cache_invariant_metrics(obf_cache, atol=atol, rtol=rtol)

    result = {
        "config": {
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "dtype": args.dtype,
            "use_pad": args.use_pad,
        },
        "generation_metrics": gen_metrics,
        "logits_metrics": {
            "per_step": per_step,
            "max_abs_error_max": max_abs_error_max,
            "allclose_all": allclose_all,
            "top1_match_rate_min": top1_min,
        },
        "cache_invariant_metrics": cache_metrics,
        "cache_design": {
            "uses_internal_obfuscated_cache": True,
            "uses_hf_past_key_values_directly": False,
            "uses_hf_generate": False,
            "cache_shape": "[batch, heads, seq, head_dim]",
            "session_seq_len_after_generation": int(trace["cache_seq_len"]),
        },
        "scope": {
            "gpt2_greedy_generation": True,
            "sampling": False,
            "beam_search": False,
            "top_k_top_p": False,
            "hf_generate": False,
            "hf_module_replacement": False,
            "trusted_layernorm": True,
            "trusted_activation": True,
            "modelscope": False,
            "real_tee": False,
        },
        "hf_model_integrity": {
            "c_attn_class_before": c_attn_class_before,
            "c_attn_class_after": c_attn_class_after,
            "c_attn_class_unchanged": c_attn_class_before == c_attn_class_after,
            "tied_embedding_before": tied_before,
            "tied_embedding_after": tied_after,
            "tied_embedding_status_unchanged": tied_before == tied_after,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "use_pad": args.use_pad,
                "token_match_rate": gen_metrics["token_match_rate"],
                "sequence_exact_match": gen_metrics["sequence_exact_match"],
                "top1_match_rate": gen_metrics["top1_match_rate"],
                "logits_max_abs_error_max": max_abs_error_max,
                "logits_top1_match_rate_min": top1_min,
                "logits_allclose_all": allclose_all,
                "cache_max_key_error": cache_metrics["max_key_error"],
                "cache_max_value_error": cache_metrics["max_value_error"],
                "cache_allclose": cache_metrics["allclose"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
