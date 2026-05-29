#!/usr/bin/env python
"""Validate GPT-2 Conv1D-to-linear mapping and fused c_attn splitting."""

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

from pllo.model_zoo import (
    ExternalModelConfig,
    build_gpt2_linear_mapping_report,
    compare_c_attn_split_equivalence,
    compare_conv1d_equivalence,
    get_model_loader,
)
from pllo.model_zoo.base import torch_dtype_from_string


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "float16"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpt2_conv1d_mapping.json"))
    return parser.parse_args()


def main() -> None:
    """Run mapping validation and write a JSON report."""
    args = parse_args()
    dtype = torch_dtype_from_string(args.dtype, args.device)
    config = ExternalModelConfig(
        source="huggingface",
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    _, model = get_model_loader("huggingface").load(config)
    hidden_size = int(model.config.n_embd)
    x = torch.randn(args.batch_size, args.seq_len, hidden_size, dtype=dtype, device=torch.device(args.device))

    layer_checks = []
    for layer_idx, block in enumerate(model.transformer.h):
        checks = {
            "layer": layer_idx,
            "attn_c_attn": compare_conv1d_equivalence(block.attn.c_attn, x),
            "attn_c_attn_split": compare_c_attn_split_equivalence(block.attn.c_attn, x, hidden_size),
            "attn_c_proj": compare_conv1d_equivalence(block.attn.c_proj, x),
            "mlp_c_fc": compare_conv1d_equivalence(block.mlp.c_fc, x),
            "mlp_c_proj": compare_conv1d_equivalence(
                block.mlp.c_proj,
                torch.randn(
                    args.batch_size,
                    args.seq_len,
                    int(block.mlp.c_fc.nf),
                    dtype=dtype,
                    device=torch.device(args.device),
                ),
            ),
        }
        layer_checks.append(checks)

    conv_checks = [
        check
        for layer in layer_checks
        for name, check in layer.items()
        if isinstance(check, dict) and name != "attn_c_attn_split"
    ]
    split_checks = [layer["attn_c_attn_split"] for layer in layer_checks]
    max_abs_error_max = max(float(check["max_abs_error"]) for check in conv_checks + split_checks)
    split_max_abs_error = max(float(check["max_abs_error"]) for check in split_checks)
    result = {
        "config": {
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
        },
        "mapping_report": build_gpt2_linear_mapping_report(model),
        "equivalence_checks": {
            "layers": layer_checks,
            "all_conv1d_equivalent": all(bool(check["allclose"]) for check in conv_checks),
            "all_c_attn_split_equivalent": all(bool(check["allclose"]) for check in split_checks),
            "max_abs_error_max": max_abs_error_max,
            "c_attn_split_max_abs_error": split_max_abs_error,
        },
        "stage4_5_scope": {
            "conv1d_adapter": True,
            "c_attn_split": True,
            "obfuscated_gpt2": False,
            "module_replacement": False,
            "modelscope": False,
            "real_tee": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "num_layers": result["mapping_report"]["num_layers"],
                "all_conv1d_equivalent": result["equivalence_checks"]["all_conv1d_equivalent"],
                "max_abs_error_max": max_abs_error_max,
                "all_c_attn_split_equivalent": result["equivalence_checks"]["all_c_attn_split_equivalent"],
                "c_attn_split_max_abs_error": split_max_abs_error,
                "tied_embedding": result["mapping_report"]["tied_embedding"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
