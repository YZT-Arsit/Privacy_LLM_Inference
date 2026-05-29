#!/usr/bin/env python
"""Run Stage 4.6 GPT-2 single-block obfuscated wrapper correctness."""

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
from pllo.hf_wrappers import ObfuscatedGPT2BlockWrapper
from pllo.model_zoo import ExternalModelConfig, get_model_loader, torch_dtype_from_string


def parse_bool(value: str | None) -> bool:
    """Parse bool CLI values while supporting flag-only usage."""
    if value is None:
        return True
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "float16"])
    parser.add_argument("--use-pad", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpt2_block_correctness.json"))
    return parser.parse_args()


def _first_hidden(output):
    """Handle transformers versions that return Tensor or tuple."""
    return output[0] if isinstance(output, tuple) else output


def main() -> None:
    """Run block correctness and write JSON."""
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
    block = model.transformer.h[args.block_index]
    hidden_size = int(model.config.n_embd)
    hidden_states = torch.randn(args.batch_size, args.seq_len, hidden_size, dtype=dtype, device=device)

    with torch.no_grad():
        plain_output = _first_hidden(block(hidden_states, attention_mask=None, use_cache=False))
        wrapper = ObfuscatedGPT2BlockWrapper(
            block,
            model.config,
            dtype=dtype,
            device=device,
            use_pad=args.use_pad,
        )
        recovered_output = wrapper.forward(hidden_states, attention_mask=None)

    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)
    metrics = compute_correctness_metrics(plain_output, recovered_output, atol=atol, rtol=rtol)
    result = {
        "config": {
            "model_id": args.model_id,
            "block_index": args.block_index,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "device": args.device,
            "dtype": args.dtype,
            "use_pad": args.use_pad,
            "atol": atol,
            "rtol": rtol,
        },
        "metrics": metrics,
        "wrapper_scope": {
            "trusted_layernorm": True,
            "trusted_activation": True,
            "kv_cache": False,
            "generation": False,
            "module_replacement": False,
            "modelscope": False,
            "real_tee": False,
        },
        "attention_strategy": {
            "c_attn": "fused_qkv_block_diagonal_mask",
            "qk_constraint": "N_Q N_K^T = I",
            "c_proj": "obfuscated_conv1d_as_linear",
        },
        "pad_report": wrapper.pad_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "block_index": args.block_index,
                "max_abs_error": metrics["max_abs_error"],
                "relative_l2_error": metrics["relative_l2_error"],
                "allclose": metrics["allclose"],
                "pad_report": result["pad_report"],
                "attention_strategy": result["attention_strategy"]["c_attn"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
