#!/usr/bin/env python
"""Run Stage 4.7 GPT-2 model-level obfuscated wrapper logits correctness."""

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
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper
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
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "float16"])
    parser.add_argument("--use-pad", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gpt2_model_correctness.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_string(args.dtype, args.device)
    device = torch.device(args.device)

    config = ExternalModelConfig(
        source="huggingface",
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    _, model = get_model_loader("huggingface").load(config)

    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

    with torch.no_grad():
        plain_logits = model(input_ids).logits

        wrapper = ObfuscatedGPT2ModelWrapper(
            model=model,
            dtype=dtype,
            device=device,
            use_pad=args.use_pad,
        )
        recovered_logits = wrapper.forward(input_ids)

    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)
    metrics = compute_correctness_metrics(plain_logits, recovered_logits, atol=atol, rtol=rtol)
    metrics["top1_match_rate"] = top1_match_rate(plain_logits, recovered_logits)

    result = {
        "config": {
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "device": args.device,
            "dtype": args.dtype,
            "use_pad": args.use_pad,
        },
        "metrics": metrics,
        "scope": {
            "multi_block": True,
            "full_forward_logits": True,
            "right_multiply_mask": True,
            "trusted_layernorm": True,
            "trusted_activation": True,
            "kv_cache": False,
            "generation": False,
            "hf_module_replacement": False,
            "modelscope": False,
            "real_tee": False,
        },
        "pad_report": {
            "use_pad": args.use_pad,
            "block_conv1d_pad": args.use_pad,
            "lm_head_pad": False,
            "lm_head_pad_reason": (
                "vocab dimension is large; Stage 4.7 uses vocab output mask only"
            ),
            "vocab_mask_type": "diagonal (scaling vector, memory-efficient)",
        },
        "lm_head_report": {
            "tied_embedding": model.lm_head.weight is model.transformer.wte.weight,
            "uses_runtime_weight_transform": True,
            "modifies_lm_head_weight": False,
        },
        "hf_model_integrity": {
            "c_attn_class": type(model.transformer.h[0].attn.c_attn).__name__,
            "tied_embedding_preserved": (
                model.lm_head.weight is model.transformer.wte.weight
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "use_pad": args.use_pad,
                "max_abs_error": metrics["max_abs_error"],
                "mean_abs_error": metrics["mean_abs_error"],
                "relative_l2_error": metrics["relative_l2_error"],
                "cosine_similarity": metrics["cosine_similarity"],
                "allclose": metrics["allclose"],
                "top1_match_rate": metrics["top1_match_rate"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
