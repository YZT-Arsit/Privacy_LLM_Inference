#!/usr/bin/env python3
"""Prepare the external-only transformed Qwen package inside TDX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from pllo.baselines.obfuscatune_lora_v2.transforms import orthogonal_matrix, transform_base, transform_factors

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def projections(layer):
    return {"q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj}


def rotation_seed(seed: int, layer: int, target_index: int) -> int:
    return 100000 + seed + 100 * layer + target_index


def factor_seed(seed: int, layer: int, target_index: int) -> int:
    return 7000 + seed + 100 * layer + target_index


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rank", type=int, default=8); parser.add_argument("--alpha", type=float, default=16)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32).cpu().eval()
    tensors = {}; inventory = {}
    for layer_index, layer in enumerate(model.model.layers):
        for target_index, (name, module) in enumerate(projections(layer).items()):
            weight = module.weight.detach().transpose(0, 1).contiguous()
            direction = "output" if name in {"o_proj", "down_proj"} else "input"
            dim = weight.shape[1] if direction == "output" else weight.shape[0]
            rotation = orthogonal_matrix(dim, rotation_seed(args.seed, layer_index, target_index), dtype=weight.dtype)
            generator = torch.Generator().manual_seed(factor_seed(args.seed, layer_index, target_index))
            a = torch.randn(weight.shape[0], args.rank, generator=generator) * .02
            b = torch.randn(args.rank, weight.shape[1], generator=generator) * 1e-4
            a_star, b_star = transform_factors(a, b, rotation, direction)
            prefix = f"layers.{layer_index}.{name}"
            tensors[f"{prefix}.weight_star"] = transform_base(weight, rotation, direction).contiguous()
            tensors[f"{prefix}.a_star"] = a_star.contiguous(); tensors[f"{prefix}.b_star"] = b_star.contiguous()
            inventory[prefix] = {"direction": direction, "weight_shape": list(weight.shape),
                                 "rotation_seed": rotation_seed(args.seed, layer_index, target_index),
                                 "factor_seed": factor_seed(args.seed, layer_index, target_index)}
    tensor_path = args.output / "transformed_package.safetensors"; save_file(tensors, tensor_path)
    manifest = {"schema": "obfuscatune_style_adapted_qwen_package_v1",
                "paper_facing_name": "OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE",
                "contract_id": "OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1",
                "seed": args.seed, "rank": args.rank, "alpha": args.alpha, "targets": list(TARGETS),
                "model_config": model.config.to_dict(), "inventory": inventory,
                "contains_plaintext_weight": False, "contains_rotation": False,
                "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest()}
    (args.output / "package_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
