#!/usr/bin/env python3
"""Plaintext BF16-runtime LoRA with FP32 authoritative AdamW state.

This is the G1_BF16_MATCHED control. It contains no transformed tensors, TDX
traffic, masks, corrections, or protected serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch

TARGET_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE = -100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_batch(records: list[dict], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in records)
    ids = torch.zeros(len(records), max_len, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, IGNORE)
    for index, row in enumerate(records):
        tokens = row["input_ids"]; length = len(tokens)
        ids[index, :length] = torch.tensor(tokens)
        attention[index, :length] = 1
        labels[index, row["sup_start"] + 1:length] = torch.tensor(tokens[row["sup_start"] + 1:length])
    return ids.to(device), attention.to(device), labels.to(device)


def module_index(name: str) -> tuple[int, str]:
    match = re.search(r"model\.layers\.(\d+)\.", name)
    if not match:
        raise ValueError(f"cannot determine layer from PEFT module {name}")
    return int(match.group(1)), name.rsplit(".", 1)[-1]


def initialize_exact_g2_plain_coordinates(model, seed: int, rank: int, targets: list[str]):
    runtime = {}; master = {}; moments = {}; seed_base = 7000 + seed
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        layer, projection = module_index(name)
        if projection not in targets:
            continue
        aw = module.lora_A["default"].weight
        bw = module.lora_B["default"].weight
        generator = torch.Generator().manual_seed(seed_base + layer * 10 + TARGET_ORDER.index(projection))
        a0 = torch.randn(rank, aw.shape[1], generator=generator, dtype=torch.float64) * 0.02
        b0 = torch.randn(bw.shape[0], rank, generator=generator, dtype=torch.float64) * 1e-4
        key_a, key_b = f"A.{layer}.{projection}", f"B.{layer}.{projection}"
        master[key_a] = a0.float().cuda(); master[key_b] = b0.float().cuda()
        moments[key_a] = [torch.zeros_like(master[key_a]), torch.zeros_like(master[key_a])]
        moments[key_b] = [torch.zeros_like(master[key_b]), torch.zeros_like(master[key_b])]
        aw.data = master[key_a].to(torch.bfloat16); bw.data = master[key_b].to(torch.bfloat16)
        runtime[key_a] = aw; runtime[key_b] = bw
    expected = 24 * len(targets) * 2
    if len(runtime) != expected:
        raise RuntimeError(f"expected {expected} LoRA factors, found {len(runtime)}")
    return runtime, master, moments


def adamw(master, moments, gradients, step: int, lr: float, weight_decay: float) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    with torch.no_grad():
        for key, parameter in master.items():
            gradient = gradients[key]
            m, v = moments[key]
            m.mul_(beta1).add_(gradient, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            mhat = m / (1 - beta1 ** step); vhat = v / (1 - beta2 ** step)
            parameter.add_(mhat / (vhat.sqrt() + eps) + weight_decay * parameter, alpha=-lr)


def copy_runtime(runtime, master) -> None:
    with torch.no_grad():
        for key in runtime:
            runtime[key].data.copy_(master[key].to(torch.bfloat16))


def lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for left in a:
        previous = 0
        for index, right in enumerate(b, 1):
            old = dp[index]
            dp[index] = previous + 1 if left == right else max(dp[index], dp[index - 1])
            previous = old
    return dp[-1]


def rouge_l(text: str, references: list[str]) -> float:
    hypothesis = text.split(); best = 0.0
    for reference in references:
        target = reference.split()
        if not hypothesis or not target:
            continue
        common = lcs(hypothesis, target)
        if common:
            precision, recall = common / len(hypothesis), common / len(target)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def repetition(tokens: list[int]) -> float:
    if len(tokens) < 2:
        return 0.0
    pairs = list(zip(tokens, tokens[1:]))
    return round(1.0 - len(set(pairs)) / len(pairs), 3)


def metrics(rows: list[dict]) -> dict:
    hypotheses = [row["generated_text"] for row in rows]
    max_refs = max(len(row["references"]) for row in rows)
    references = [[row["references"][i] if i < len(row["references"]) else row["references"][0]
                   for row in rows] for i in range(max_refs)]
    return {
        "BLEU": sacrebleu.corpus_bleu(hypotheses, references).score,
        "chrF": sacrebleu.corpus_chrf(hypotheses, references).score,
        "ROUGE_L": 100 * float(np.mean([rouge_l(row["generated_text"], row["references"]) for row in rows])),
        "invalid_rate_pct": 100 * float(np.mean([bool(row["invalid_output"]) for row in rows])),
        "repetition_pct": 100 * float(np.mean([row["repetition_bigram_frac"] for row in rows])),
        "average_generated_tokens": float(np.mean([row["generated_token_count"] for row in rows])),
        "records": len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("train-and-generate")
    run.add_argument("--seed", type=int, required=True); run.add_argument("--schedule", type=Path, required=True)
    run.add_argument("--train-data", type=Path, required=True); run.add_argument("--generation-input", type=Path, required=True)
    run.add_argument("--base-checkpoint", type=Path, required=True); run.add_argument("--tokenizer", type=Path, required=True)
    run.add_argument("--targets", required=True); run.add_argument("--rank", type=int, default=8)
    run.add_argument("--alpha", type=int, default=16); run.add_argument("--dropout", type=float, default=0.0)
    run.add_argument("--lr", type=float, default=2e-4); run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--max-steps", type=int, default=750); run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--grad-accum", type=int, default=1); run.add_argument("--grad-clip", default="none")
    run.add_argument("--generation-dtype", choices=["fp32", "bf16"], default="fp32")
    run.add_argument("--generation-max", type=int, default=500)
    run.add_argument("--max-new", type=int, default=96); run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if targets != TARGET_ORDER:
        raise ValueError(f"targets must use frozen order: {','.join(TARGET_ORDER)}")
    if args.batch_size != 16 or args.grad_accum != 1 or args.grad_clip != "none":
        raise ValueError("this matched profile requires batch=16, grad_accum=1, grad_clip=none")
    output = args.output_dir.resolve(); checkpoint = output / "resume_checkpoint.pt"
    if output.exists() and not args.resume:
        raise RuntimeError(f"refusing to overwrite append-only output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "terminal_manifest.json").exists():
        raise RuntimeError("terminal manifest already exists")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    source = Path(__file__).resolve()
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
        "schedule": args.schedule, "train_data": args.train_data, "generation_input": args.generation_input,
        "base_config": args.base_checkpoint / "config.json", "base_weights": args.base_checkpoint / "model.safetensors",
        "tokenizer": args.tokenizer / "tokenizer.json",
    }.items()}
    config = {
        "schema": "g1_bf16_matched_config", "version": "1.0", "seed": args.seed,
        "targets": targets, "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
        "steps": args.max_steps, "batch_size": args.batch_size, "gradient_accumulation": args.grad_accum,
        "learning_rate": args.lr, "learning_rate_schedule": "constant_actual_G2_L12",
        "adamw": {"implementation": "explicit_unfused_G2_formula", "betas": [0.9, 0.999],
                  "epsilon": 1e-8, "weight_decay": args.weight_decay, "state_dtype": "fp32"},
        "forward_dtype": "bf16", "backward_runtime_dtype": "bf16", "master_parameter_dtype": "fp32",
        "loss_scaling": "none", "gradient_clipping": "none_actual_G2_L12",
        "generation_dtype": args.generation_dtype, "generation_records": args.generation_max,
        "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new}, "inputs": inputs,
        "no_transformed_package": True, "no_tdx": True, "no_masks_or_corrections": True,
        "known_mismatches": [
            "HF plaintext batched forward replaces G2 transformed per-example forward",
            "plaintext CE/loss is local instead of TDX CE/dlogits",
            "frozen_config declares weight_decay=0 and grad_clip=1, but this control matches actual G2 L12 code: weight_decay=0.01 and no gradient clipping",
        ],
    }
    config_hash = canonical_hash(config); config["config_sha256"] = config_hash
    atomic_json(output / "config.json", config)
    atomic_json(output / "precision_profile.json", {
        "forward": "bf16", "autocast": "disabled", "lora_runtime": "bf16",
        "authoritative_lora": "fp32", "gradient_observed": "recorded_at_runtime",
        "adamw_m_v": "fp32", "optimizer": "explicit_unfused_G2_formula",
        "loss_scaling": "none", "generation": args.generation_dtype,
    })
    atomic_json(output / "pid.json", {"pid": os.getpid(), "host": os.uname().nodename,
                                      "started_at": time.time(), "config_sha256": config_hash})
    (output / "training_schedule.json").write_text(json.dumps(
        json.loads(args.schedule.read_text())["schedule"][:args.max_steps], sort_keys=True) + "\n")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(str(args.base_checkpoint), torch_dtype=torch.bfloat16).cuda()
    base.config.use_cache = False
    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=targets,
                             lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
                             init_lora_weights=False)
    model = get_peft_model(base, peft_config); model.train()
    runtime, master, moments = initialize_exact_g2_plain_coordinates(model, args.seed, args.rank, targets)
    trajectory = []; start_step = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        if state["config_sha256"] != config_hash:
            raise RuntimeError("resume config hash mismatch")
        start_step = int(state["next_step"]); trajectory = state["trajectory"]
        for key in master:
            master[key].copy_(state["master"][key]); moments[key][0].copy_(state["moments"][key][0]); moments[key][1].copy_(state["moments"][key][1])
        copy_runtime(runtime, master)
    data = torch.load(args.train_data, map_location="cpu", weights_only=False)
    by_id = {int(row["sample_id"]): row for row in data}
    schedule = json.loads(args.schedule.read_text())["schedule"][:args.max_steps]
    if len(schedule) != args.max_steps:
        raise RuntimeError("schedule length mismatch")
    train_start = time.time(); gradient_dtypes = set()
    for index in range(start_step, args.max_steps):
        step_start = time.time(); entry = schedule[index]
        records = [by_id[int(sample)] for sample in entry["sample_ids"]]
        if len(records) != args.batch_size:
            raise RuntimeError("batch size mismatch")
        ids, attention, labels = build_batch(records, "cuda")
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {index}")
        loss.backward(); gradients = {}
        for key, parameter in runtime.items():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient {key}")
            gradient_dtypes.add(str(parameter.grad.dtype)); gradients[key] = parameter.grad.detach().float()
        adamw(master, moments, gradients, index + 1, args.lr, args.weight_decay)
        copy_runtime(runtime, master)
        finite = all(bool(torch.isfinite(value).all()) for value in master.values())
        if not finite:
            raise FloatingPointError(f"non-finite master at step {index}")
        loss_value = float(loss.detach())
        row = {"step": index, "global_batch_index": entry["global_batch_index"],
               "optimizer_step": entry["optimizer_step"], "loss": loss_value, "finite": finite,
               "wall_sec": time.time() - step_start}
        trajectory.append(row)
        atomic_json(output / "heartbeat.json", {"timestamp": time.time(), "pid": os.getpid(),
                    "completed_steps": index + 1, "total_steps": args.max_steps,
                    "last_loss": loss_value, "all_finite": True})
        if index % 20 == 0 or index == args.max_steps - 1:
            print(json.dumps(row), flush=True)
        if (index + 1) % 50 == 0 or index == args.max_steps - 1:
            torch.save({"config_sha256": config_hash, "next_step": index + 1,
                        "master": {key: value.detach().cpu() for key, value in master.items()},
                        "moments": {key: [value[0].detach().cpu(), value[1].detach().cpu()] for key, value in moments.items()},
                        "trajectory": trajectory}, checkpoint)
    train_wall = time.time() - train_start
    adapter = output / "adapter"; adapter.mkdir(exist_ok=False)
    with torch.no_grad():
        for key, parameter in runtime.items():
            parameter.data = master[key].detach().clone()
    model.save_pretrained(adapter, safe_serialization=True)
    training = {"steps": len(trajectory), "all_finite": all(row["finite"] for row in trajectory),
                "trajectory": trajectory, "wall_sec": train_wall,
                "runtime_gradient_dtypes": sorted(gradient_dtypes),
                "trainable_parameters": sum(value.numel() for value in master.values()),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(output / "training.json", training)
    del model, base, runtime, master, moments; torch.cuda.empty_cache()

    generation_dtype = torch.float32 if args.generation_dtype == "fp32" else torch.bfloat16
    generation_base = AutoModelForCausalLM.from_pretrained(str(args.base_checkpoint), torch_dtype=generation_dtype).cuda()
    generation_model = PeftModel.from_pretrained(generation_base, adapter).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer)); eos = tokenizer.eos_token_id
    prompts = json.loads(args.generation_input.read_text())[:args.generation_max]
    if len(prompts) != args.generation_max or not (1 <= args.generation_max <= 500):
        raise RuntimeError("invalid frozen generation subset")
    (output / "generation_sample_ids.json").write_text(
        json.dumps([prompt["sample_id"] for prompt in prompts]) + "\n")
    rows = []; generation_start = time.time(); total_tokens = 0
    with torch.no_grad():
        for index, prompt in enumerate(prompts):
            prompt_ids = torch.tensor([prompt["prompt_ids"]], device="cuda")
            prompt_attention = torch.ones_like(prompt_ids)
            generated = generation_model.generate(input_ids=prompt_ids, attention_mask=prompt_attention,
                max_new_tokens=args.max_new,
                do_sample=False, num_beams=1, pad_token_id=eos, eos_token_id=eos)
            tokens = generated[0, prompt_ids.shape[1]:].tolist(); finish = "length"
            if eos in tokens:
                finish = "eos"; tokens = tokens[:tokens.index(eos)]
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip(); total_tokens += len(tokens)
            rows.append({"sample_id": prompt["sample_id"],
                "input_hash": hashlib.sha256(json.dumps(prompt["prompt_ids"]).encode()).hexdigest()[:16],
                "meaning_representation": prompt.get("meaning_representation", ""),
                "references": prompt.get("references", []), "generated_text": text,
                "token_ids": tokens, "finish_reason": finish, "generated_token_count": len(tokens),
                "invalid_output": not bool(text), "repetition_bigram_frac": repetition(tokens),
                "cell": "G1_BF16_MATCHED", "seed": args.seed})
            if index % 50 == 0:
                print(json.dumps({"generated": index + 1, "total": len(prompts)}), flush=True)
    generation_wall = time.time() - generation_start
    generation_path = output / "generations.jsonl"
    generation_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    generation_profile = {"records": len(rows), "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "total_tokens": total_tokens, "wall_sec": generation_wall,
        "tokens_per_sec": total_tokens / generation_wall, "dtype": args.generation_dtype,
        "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new, "eos": eos}}
    atomic_json(output / "generation_profile.json", generation_profile)
    atomic_json(output / "quality_metrics.json", metrics(rows))
    if checkpoint.exists():
        checkpoint.unlink()
    artifacts = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "terminal_manifest.json"):
        artifacts[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"schema": "g1_bf16_matched_terminal_manifest", "version": "1.0",
        "status": "complete", "seed": args.seed, "config_sha256": config_hash,
        "source_sha256": sha256(source), "steps": len(trajectory), "all_finite": True,
        "generations": len(rows), "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "generation_sha256": sha256(generation_path), "artifacts": artifacts,
        "resume_command": "same command with --resume (only before terminal completion)",
        "completed_at": time.time()}
    atomic_json(output / "terminal_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
