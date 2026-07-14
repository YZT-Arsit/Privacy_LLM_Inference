#!/usr/bin/env python3
"""Isolated, deterministic shadow-LoRA trainer for the frozen MIA-v2 pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

IGNORE = -100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def batch(records: list[dict], device: str) -> tuple[torch.Tensor, ...]:
    max_len = max(len(r["input_ids"]) for r in records)
    ids = torch.zeros((len(records), max_len), dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, IGNORE)
    for row, record in enumerate(records):
        tokens = torch.tensor(record["input_ids"], dtype=torch.long)
        n = tokens.numel()
        ids[row, :n] = tokens
        mask[row, :n] = 1
        labels[row, record["sup_start"] + 1:n] = tokens[record["sup_start"] + 1:n]
    return ids.to(device), mask.to(device), labels.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-config", type=Path, required=True)
    ap.add_argument("--train-members", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.shadow_config.read_text())
    tc = cfg["training"]
    records = [json.loads(x) for x in args.train_members.read_text().splitlines() if x.strip()]
    if len(records) != 500 or len({x["sample_id"] for x in records}) != 500:
        raise RuntimeError("training input is not exactly 500 unique frozen members")
    expected = {"rank": 8, "alpha": 16, "lr": 2e-4, "steps": 750,
                "batch_size": 16, "dtype": "fp32", "members": 500}
    for key, value in expected.items():
        if tc[key] != value:
            raise RuntimeError(f"frozen config mismatch {key}: {tc[key]} != {value}")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if tc["targets"] != targets:
        raise RuntimeError("frozen target list mismatch")

    seed = int(cfg["shadow_seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0,
                                  betas=(0.9, 0.999), eps=1e-8)
    steps, batch_size, warmup = 750, 16, max(1, int(0.03 * 750))

    def lr_at(step: int) -> float:
        if step < warmup:
            return 2e-4 * step / warmup
        return 2e-4 * max(0.0, 1.0 - (step - warmup) / max(1, steps - warmup))

    losses, grad_norms = [], []
    t0 = time.time()
    for step in range(steps):
        # Frozen cyclic ordering: contiguous groups of 16 in the exact JSONL order, wrapping at 500.
        idx = [(step * batch_size + j) % len(records) for j in range(batch_size)]
        ids, mask, labels = batch([records[j] for j in idx], device)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu()))
        if step % 20 == 0 or step == steps - 1:
            print(json.dumps({"step": step + 1, "steps": steps, "lr": lr_at(step),
                              "loss": losses[-1], "mean_last20": sum(losses[-20:]) / len(losses[-20:]),
                              "wall_sec": round(time.time() - t0, 1)}), flush=True)

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    from safetensors.torch import load_file
    tensors = load_file(str(adapter_path), device="cpu")
    if not tensors or any(not torch.isfinite(x).all() for x in tensors.values()):
        raise RuntimeError("saved adapter contains missing or non-finite tensors")
    expected_tensor_count = 24 * len(targets) * 2
    if len(tensors) != expected_tensor_count:
        raise RuntimeError(f"adapter tensor count {len(tensors)} != {expected_tensor_count}")
    manifest = {
        "schema": "mia_v3_shadow_training_manifest", "status": "PASS",
        "shadow_seed": seed, "base_model": str(args.base_model),
        "base_package_root_hash": cfg["base_package_root_hash"],
        "shadow_config_sha256": sha256(args.shadow_config),
        "train_members_sha256": sha256(args.train_members),
        "membership_sha256": cfg["membership_sha256"],
        "pool_manifest_sha256": cfg["pool_manifest_sha256"],
        "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 0.0,
                      "betas": [0.9, 0.999], "eps": 1e-8, "warmup_fraction": 0.03,
                      "schedule": "linear_decay", "grad_clip_norm": 1.0},
        "training": {**tc, "ordering": "frozen_jsonl_cyclic_contiguous_wrap"},
        "adapter": {"path": str(adapter_path), "sha256": sha256(adapter_path),
                    "tensor_count": len(tensors),
                    "tensor_shapes": {k: list(v.shape) for k, v in sorted(tensors.items())},
                    "all_finite": True},
        "metrics": {"initial_loss": losses[0], "final_loss": losses[-1],
                    "mean_last20": sum(losses[-20:]) / 20,
                    "max_grad_norm_preclip": max(grad_norms), "all_finite": True,
                    "wall_sec": round(time.time() - t0, 1)},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "dtype": "fp32"},
        "warm_started": False,
    }
    write_new(args.output_dir / "training_manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_new(args.output_dir / "loss_history.json", json.dumps({"loss": losses, "grad_norm": grad_norms}) + "\n")
    print(json.dumps({"status": "PASS", "adapter_sha256": manifest["adapter"]["sha256"],
                      "wall_sec": manifest["metrics"]["wall_sec"]}), flush=True)


if __name__ == "__main__":
    main()
