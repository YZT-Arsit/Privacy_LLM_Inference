#!/usr/bin/env python3
"""Local, deterministic, second-process smoke for the adapted baseline.

This is a correctness gate only.  It never reports local trusted-runtime calls
as real A10/TDX measurements and it is not a paper-facing quality run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from pllo.baselines.obfuscatune.qwen_config import make_tiny_qwen_config
from pllo.baselines.obfuscatune_lora_v2.artifacts import (
    apply_adapter, export_adapter, load_adapter, load_training_checkpoint,
    save_training_checkpoint,
)
from pllo.baselines.obfuscatune_lora_v2.optimizer import AdamWHyperparameters, TransformedAdamW
from pllo.baselines.obfuscatune_lora_v2.qwen_model import AdaptedQwenCausalLM
from pllo.baselines.obfuscatune_lora_v2.runtime import TrustedRuntime


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(seed: int):
    from transformers import Qwen2ForCausalLM
    torch.manual_seed(seed)
    config = make_tiny_qwen_config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8)
    reference = Qwen2ForCausalLM(config).to(torch.float64).eval()
    trusted = TrustedRuntime(real_transport=False)
    model = AdaptedQwenCausalLM(reference, trusted=trusted, rank=8, alpha=16, transform_seed=seed)
    optimizer = TransformedAdamW(model.transformed_parameters(), AdamWHyperparameters(
        lr=2e-4, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01))
    return model, optimizer, trusted


def binding(seed: int) -> dict:
    return {"run_kind": "LOCAL_CORRECTNESS_ONLY", "seed": seed, "rank": 8, "alpha": 16,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "contract_id": "OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1"}


def verify(args) -> None:
    model, optimizer, _ = build(args.seed)
    load_training_checkpoint(args.checkpoint, parameters=model.transformed_parameters(), optimizer=optimizer,
                             expected_binding=binding(args.seed))
    tensors = load_adapter(args.adapter, expected_binding=binding(args.seed))
    apply_adapter(model.transformed_parameters(), tensors)
    prompt = torch.tensor([[1, 2, 3, 4]])
    tokens = model.generate_greedy(prompt, max_new_tokens=8)
    raw = tokens.cpu().contiguous().numpy().tobytes()
    print(json.dumps({"pid": os.getpid(), "checkpoint_step": optimizer.step_index,
                      "tokens": tokens.tolist(), "token_sha256": hashlib.sha256(raw).hexdigest()}))


def run(args) -> None:
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite append-only output: {output}")
    output.mkdir(parents=True)
    model, optimizer, trusted = build(args.seed)
    losses = []
    started = time.time()
    for step in range(args.steps):
        torch.manual_seed(args.seed + step + 1)
        ids = torch.randint(0, 64, (2, 12))
        labels = ids.clone()
        _, _, loss = model(ids, labels=labels)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite local loss at step {step}")
        loss.backward()
        gradients = {}
        for name, parameter in model.transformed_parameters().items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"invalid gradient: {name}")
            gradients[name] = parameter.grad.detach().clone()
        optimizer.step(gradients)
        trusted.counters.optimizer_updates += 1
        for parameter in model.transformed_parameters().values():
            parameter.grad = None
        losses.append(float(loss.detach()))
    checkpoint = output / "checkpoint.pt"
    checkpoint_hash = save_training_checkpoint(checkpoint, parameters=model.transformed_parameters(), optimizer=optimizer,
                                                binding=binding(args.seed), step=args.steps,
                                                counters=trusted.counters.to_dict())
    adapter = output / "adapter"
    export_adapter(adapter, parameters=model.transformed_parameters(), binding=binding(args.seed), final_step=args.steps)
    command = [sys.executable, str(Path(__file__).resolve()), "verify", "--seed", str(args.seed),
               "--checkpoint", str(checkpoint), "--adapter", str(adapter)]
    checks = []
    for _ in range(2):
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        checks.append(json.loads(result.stdout.strip().splitlines()[-1]))
    if checks[0]["pid"] == os.getpid() or checks[1]["pid"] == os.getpid() or checks[0]["token_sha256"] != checks[1]["token_sha256"]:
        raise RuntimeError("fresh-process restart agreement failed")
    report = {
        "schema": "obfuscatune_v2_local_tiny_smoke_v1",
        "status": "LOCAL_CORRECTNESS_ONLY_NOT_REAL_A10_TDX",
        "seed": args.seed, "steps": args.steps, "finite_steps": len(losses), "losses": losses,
        "wall_sec": time.time() - started, "checkpoint_sha256": checkpoint_hash,
        "adapter_manifest_sha256": sha256(adapter / "adapter_manifest.json"),
        "adapter_tensors_sha256": sha256(adapter / "adapter_tensors.safetensors"),
        "fresh_process_checks": checks, "token_hash_agreement": True,
        "runtime_counters": trusted.counters.to_dict(),
        "counter_provenance": "LOCAL_ORACLE_NOT_MEASURED_REMOTE",
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    local = sub.add_parser("run"); local.add_argument("--output-dir", type=Path, required=True)
    local.add_argument("--seed", type=int, default=1234); local.add_argument("--steps", type=int, default=10)
    check = sub.add_parser("verify"); check.add_argument("--seed", type=int, required=True)
    check.add_argument("--checkpoint", type=Path, required=True); check.add_argument("--adapter", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    verify(arguments) if arguments.mode == "verify" else run(arguments)
