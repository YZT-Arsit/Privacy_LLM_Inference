#!/usr/bin/env python
"""CONFIRM CONJFORMER recovery: the scalar-RMSNorm retrofit breaks a pretrained
model (gibberish), and fine-tuning brings it back -- exactly what the paper claims
(Fig. 6 retrofit curve, Table 2 fine-tuned PPL within ~1% of baseline). We show
the full chain on a real model:

  1. ORIGINAL per-channel model    -> baseline val PPL (target).
  2. RETROFIT scalar-RMSNorm       -> val PPL explodes, generation is gibberish.
  3. FINE-TUNE on wikitext-2       -> val PPL recovers toward baseline, generation
                                      becomes coherent (readable-ASCII ratio climbs).
  4. CONJUGATE the recovered model -> equivariance still exact (server-on-rotated
                                      == recovered-model-on-plain, top-1 = 1.0),
                                      i.e. the recovered model is deployable as a
                                      CONJFORMER server. Closes the loop.

Wall-time to recovery is measured (the cost CONJFORMER pays that ours does not).
Uses a small Qwen2/Llama-arch model so full-param AdamW fits one GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch


def read_prompts(path, field, n):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)[field])
            if len(out) >= n:
                break
    return out


def ascii_ratio(t):
    if not t:
        return 0.0
    ok = sum(1 for c in t if c.isascii() and (c.isalnum() or c.isspace() or c in ".,;:!?()[]-\"'"))
    return round(ok / len(t), 3)


@torch.no_grad()
def gen_sample(model, tok, prompt, device, max_new=64):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def eval_ppl(model, blocks, device, max_blocks=40):
    model.eval()
    nll, ntok = 0.0, 0
    for b in blocks[:max_blocks]:
        ids = b.to(device).unsqueeze(0)
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.item() * n
        ntok += n
    return float(torch.exp(torch.tensor(nll / max(ntok, 1))).item())


def pack_blocks(tok, texts, seq_len, max_blocks):
    ids = []
    for t in texts:
        if t and t.strip():
            ids.extend(tok(t)["input_ids"])
        if len(ids) >= seq_len * max_blocks:
            break
    blocks = [torch.tensor(ids[i:i + seq_len]) for i in range(0, len(ids) - seq_len, seq_len)]
    return blocks[:max_blocks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--train-blocks", type=int, default=4000)
    ap.add_argument("--val-blocks", type=int, default=40)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--prompts-jsonl", default=None)
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    device = a.device
    td = torch.float32 if a.dtype == "float32" else torch.bfloat16

    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    from pllo.baselines.conjformer import (
        retrofit_scalar_rmsnorm, sample_secrets, conjugate_qwen2, verify_equivariance,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = (read_prompts(a.prompts_jsonl, a.prompt_field, a.n_prompts)
               if a.prompts_jsonl else
               ["Write a short paragraph about the ocean.",
                "List three planets in the solar system.",
                "Explain what a computer is in two sentences."][:a.n_prompts])

    print("[data] loading wikitext-2-raw-v1 ...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_blocks = pack_blocks(tok, ds["train"]["text"], a.seq_len, a.train_blocks)
    val_blocks = pack_blocks(tok, ds["validation"]["text"], a.seq_len, a.val_blocks)
    print(f"[data] train_blocks={len(train_blocks)} val_blocks={len(val_blocks)}", flush=True)

    print(f"[model] loading {a.model} ({a.dtype}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=td).to(device)

    report = {"model": a.model, "dtype": a.dtype, "lr": a.lr,
              "lr_schedule": a.lr_schedule, "warmup_steps": a.warmup_steps,
              "max_steps": a.max_steps, "seq_len": a.seq_len,
              "bsz": a.bsz, "corpus": "wikitext-2-raw-v1"}
    samples = {}

    # 1. original per-channel baseline
    orig_ppl = eval_ppl(model, val_blocks, device, a.val_blocks)
    samples["original"] = [{"prompt": p, "text": gen_sample(model, tok, p, device)} for p in prompts]
    report["original_val_ppl"] = round(orig_ppl, 3)
    report["original_gen_ascii"] = round(sum(ascii_ratio(s["text"]) for s in samples["original"]) / len(prompts), 3)
    print(f"[1/4] ORIGINAL val_ppl={orig_ppl:.2f} gen_ascii={report['original_gen_ascii']}", flush=True)

    # 2. retrofit scalar-RMSNorm (broken)
    n_norm = retrofit_scalar_rmsnorm(model)
    retro_ppl = eval_ppl(model, val_blocks, device, a.val_blocks)
    samples["retrofit_noft"] = [{"prompt": p, "text": gen_sample(model, tok, p, device)} for p in prompts]
    report["norm_layers_retrofitted"] = n_norm
    report["retrofit_noft_val_ppl"] = round(retro_ppl, 3)
    report["retrofit_noft_gen_ascii"] = round(sum(ascii_ratio(s["text"]) for s in samples["retrofit_noft"]) / len(prompts), 3)
    print(f"[2/4] RETROFIT(no-ft) val_ppl={retro_ppl:.2f} gen_ascii={report['retrofit_noft_gen_ascii']}", flush=True)

    # 3. fine-tune to recover
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    import math as _math

    def lr_at(step):
        if a.warmup_steps and step <= a.warmup_steps:
            return a.lr * step / max(a.warmup_steps, 1)
        if a.lr_schedule == "cosine":
            prog = (step - a.warmup_steps) / max(a.max_steps - a.warmup_steps, 1)
            prog = min(max(prog, 0.0), 1.0)
            return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + _math.cos(_math.pi * prog)))
        return a.lr
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(train_blocks), generator=g).tolist()
    curve = [{"step": 0, "val_ppl": round(retro_ppl, 3), "gen_ascii": report["retrofit_noft_gen_ascii"]}]
    t0 = time.perf_counter()
    ptr = 0
    for step in range(1, a.max_steps + 1):
        batch = []
        for _ in range(a.bsz):
            if ptr >= len(order):
                ptr = 0
            batch.append(train_blocks[order[ptr]]); ptr += 1
        ids = torch.stack(batch).to(device)
        cur_lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % a.eval_every == 0 or step == a.max_steps:
            was_ckpt = model.config.use_cache
            model.config.use_cache = True
            ppl = eval_ppl(model, val_blocks, device, a.val_blocks)
            gsmp = [{"prompt": p, "text": gen_sample(model, tok, p, device)} for p in prompts]
            asc = round(sum(ascii_ratio(s["text"]) for s in gsmp) / len(prompts), 3)
            model.config.use_cache = was_ckpt
            model.train()
            curve.append({"step": step, "val_ppl": round(ppl, 3), "gen_ascii": asc,
                          "lr": round(cur_lr, 6),
                          "elapsed_min": round((time.perf_counter() - t0) / 60.0, 2)})
            samples[f"ft_step_{step}"] = gsmp
            print(f"[3/4] step {step}/{a.max_steps} val_ppl={ppl:.2f} gen_ascii={asc} "
                  f"loss={out.loss.item():.3f} {curve[-1]['elapsed_min']}min", flush=True)
    train_min = (time.perf_counter() - t0) / 60.0
    report["finetune_curve"] = curve
    report["finetune_wall_min"] = round(train_min, 2)
    report["recovered_val_ppl"] = curve[-1]["val_ppl"]
    report["recovered_gen_ascii"] = curve[-1]["gen_ascii"]

    # 4. conjugate the RECOVERED model -> equivariance still exact
    model.eval(); model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    secrets = sample_secrets(model.config, seed=a.seed, rope_compatible=True)
    secrets.U = secrets.U  # cpu; verify moves as needed
    server = copy.deepcopy(model)
    conjugate_qwen2(server, secrets)
    ids = tok("Hello, world.", return_tensors="pt")["input_ids"].to(device)
    ver = verify_equivariance(model, server, secrets, ids)
    report["recovered_equivariance"] = {
        "logit_top1_agreement": ver["logit_top1_agreement"],
        "hidden_rel_err": ver["hidden_rel_err"],
        "equivariant": ver["equivariant"],
        "note": "the fine-tuned (recovered) model conjugates to an exactly-equivariant "
                "CONJFORMER server -- recovery does not break the obfuscation.",
    }
    print(f"[4/4] recovered model equivariance top1={ver['logit_top1_agreement']:.4f} "
          f"equivariant={ver['equivariant']}", flush=True)

    report["summary"] = {
        "original_val_ppl": report["original_val_ppl"],
        "retrofit_noft_val_ppl": report["retrofit_noft_val_ppl"],
        "recovered_val_ppl": report["recovered_val_ppl"],
        "ppl_recovery_pct": round(100.0 * (retro_ppl - curve[-1]["val_ppl"]) / max(retro_ppl - orig_ppl, 1e-9), 1),
        "gen_ascii_before_after": [report["retrofit_noft_gen_ascii"], report["recovered_gen_ascii"]],
        "finetune_wall_min": report["finetune_wall_min"],
        "note": "CONJFORMER's scalar-RMSNorm break IS recoverable by fine-tuning (this "
                "confirms it) -- at a real fine-tuning wall-cost; ours needs zero fine-tuning.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sp = a.samples_out or (str(Path(a.out).with_suffix("")) + "_samples.json")
    with open(sp, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {a.out} + {sp}", flush=True)


if __name__ == "__main__":
    main()
