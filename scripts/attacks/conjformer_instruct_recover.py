#!/usr/bin/env python
"""CONJFORMER (b): can the scalar-RMSNorm retrofit relearn INSTRUCTION-FOLLOWING,
and how fast? Part (a) showed plain-corpus fine-tuning restores language-modeling
fluency but NOT instruction-following (the recovered model ignores the prompt).

The client holds the public original instruct model, so the realistic recovery is
self-distillation SFT: generate (prompt -> response) pairs from the ORIGINAL model,
then fine-tune the retrofit (scalar-RMSNorm) model to imitate them with a
response-masked loss. We track, on a held-out prompt split:
  - distill PPL: retrofit model's NLL on the original model's held-out responses,
  - instruction relevance: fraction of prompt content-words echoed in the response
    (generic off-topic text -> low; on-topic answer -> higher; original = target),
  - readable-ASCII + samples (off-topic -> on-topic).
Then conjugate the recovered model and re-check exact equivariance.

This measures the *instruction-following* recovery cost that plain-corpus fine-
tuning does not buy -- the cost CONJFORMER pays on an off-the-shelf instruct model
that ours (zero fine-tuning) does not.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from pathlib import Path

import torch

STOP = set("the a an of to and or in on for with is are be this that it as at by from "
           "your you please write list explain give make sure using use into their his "
           "her its not do does can will would about over under than then so if".split())


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


def content_words(t):
    return {w for w in re.findall(r"[a-zA-Z]{4,}", t.lower()) if w not in STOP}


def relevance(prompt, resp):
    cw = content_words(prompt)
    if not cw:
        return 0.0
    rw = content_words(resp)
    return round(len(cw & rw) / len(cw), 3)


def ascii_ratio(t):
    if not t:
        return 0.0
    return round(sum(1 for c in t if c.isascii() and (c.isalnum() or c.isspace() or c in ".,;:!?()[]-\"'")) / len(t), 3)


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=200):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def sft_example(tok, prompt, response, max_len):
    """chat-template prompt + response; labels masked to the response only."""
    msgs = [{"role": "user", "content": prompt}]
    pref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pref_ids = tok(pref, add_special_tokens=False)["input_ids"]
    resp_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (pref_ids + resp_ids)[:max_len]
    labels = ([-100] * len(pref_ids) + resp_ids)[:max_len]
    return ids, labels


@torch.no_grad()
def eval_distill_ppl(model, tok, pairs, device, max_len, max_n=40):
    model.eval()
    nll, ntok = 0.0, 0
    for prompt, resp in pairs[:max_n]:
        ids, labels = sft_example(tok, prompt, resp, max_len)
        if sum(1 for x in labels if x != -100) == 0:
            continue
        ii = torch.tensor([ids], device=device)
        ll = torch.tensor([labels], device=device)
        out = model(input_ids=ii, labels=ll)
        n = int((ll != -100).sum().item())
        nll += out.loss.item() * n
        ntok += n
    return float(math.exp(nll / max(ntok, 1)))


@torch.no_grad()
def eval_gen(model, tok, prompts, device):
    smp, rel, asc = [], [], []
    for p in prompts:
        t = generate(model, tok, p, device)
        smp.append({"prompt": p, "text": t})
        rel.append(relevance(p, t)); asc.append(ascii_ratio(t))
    return smp, round(sum(rel) / len(rel), 3), round(sum(asc) / len(asc), 3)


def collate(batch, pad_id):
    m = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--prompts-jsonl", required=True)
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--distill-max-new", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=100)
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
        retrofit_scalar_rmsnorm, sample_secrets, conjugate_qwen2, verify_equivariance)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    all_prompts = read_prompts(a.prompts_jsonl, a.prompt_field, a.n_train + a.n_eval)
    train_prompts = all_prompts[:a.n_train]
    eval_prompts = all_prompts[a.n_train:a.n_train + a.n_eval]
    print(f"[data] train={len(train_prompts)} eval={len(eval_prompts)}", flush=True)

    print(f"[model] loading original {a.model} ({a.dtype}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=td).to(device).eval()

    # 1. distill: original model's responses (client has the public instruct model)
    print("[distill] generating teacher responses ...", flush=True)
    t0 = time.perf_counter()
    train_pairs = [(p, generate(model, tok, p, device, a.distill_max_new)) for p in train_prompts]
    eval_pairs = [(p, generate(model, tok, p, device, a.distill_max_new)) for p in eval_prompts]
    distill_gen_min = (time.perf_counter() - t0) / 60.0
    orig_smp, orig_rel, orig_asc = eval_gen(model, tok, eval_prompts, device)

    report = {"model": a.model, "n_train": a.n_train, "n_eval": a.n_eval, "lr": a.lr,
              "epochs": a.epochs, "teacher_gen_min": round(distill_gen_min, 2),
              "original_relevance": orig_rel, "original_ascii": orig_asc}
    samples = {"original": orig_smp}
    print(f"[distill] teacher gen {distill_gen_min:.1f}min | original relevance={orig_rel} ascii={orig_asc}", flush=True)

    # 2. retrofit (broken)
    n_norm = retrofit_scalar_rmsnorm(model)
    r_ppl = eval_distill_ppl(model, tok, eval_pairs, device, a.max_len)
    r_smp, r_rel, r_asc = eval_gen(model, tok, eval_prompts, device)
    report.update({"norm_layers_retrofitted": n_norm, "retrofit_noft_distill_ppl": round(r_ppl, 2),
                   "retrofit_noft_relevance": r_rel, "retrofit_noft_ascii": r_asc})
    samples["retrofit_noft"] = r_smp
    print(f"[retrofit] distill_ppl={r_ppl:.1f} relevance={r_rel} ascii={r_asc}", flush=True)

    # 3. self-distillation SFT (response-masked)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    exs = [sft_example(tok, p, r, a.max_len) for p, r in train_pairs]
    exs = [e for e in exs if sum(1 for x in e[1] if x != -100) > 0]
    g = torch.Generator().manual_seed(a.seed)
    curve = [{"step": 0, "distill_ppl": round(r_ppl, 2), "relevance": r_rel, "ascii": r_asc}]
    pad_id = tok.eos_token_id
    step = 0
    tft = time.perf_counter()
    total_steps = a.epochs * math.ceil(len(exs) / a.bsz)
    for ep in range(a.epochs):
        order = torch.randperm(len(exs), generator=g).tolist()
        for i in range(0, len(exs), a.bsz):
            batch = [exs[j] for j in order[i:i + a.bsz]]
            ids, lab, att = collate(batch, pad_id)
            ids, lab, att = ids.to(device), lab.to(device), att.to(device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.eval_every == 0 or step == total_steps:
                was = model.config.use_cache; model.config.use_cache = True
                ppl = eval_distill_ppl(model, tok, eval_pairs, device, a.max_len)
                smp, rel, asc = eval_gen(model, tok, eval_prompts, device)
                model.config.use_cache = was; model.train()
                curve.append({"step": step, "distill_ppl": round(ppl, 2), "relevance": rel,
                              "ascii": asc, "elapsed_min": round((time.perf_counter() - tft) / 60.0, 2)})
                samples[f"sft_step_{step}"] = smp
                print(f"[sft] step {step}/{total_steps} distill_ppl={ppl:.2f} relevance={rel} "
                      f"ascii={asc} loss={out.loss.item():.3f} {curve[-1]['elapsed_min']}min", flush=True)
    sft_min = (time.perf_counter() - tft) / 60.0

    # 4. conjugate recovered -> equivariance
    model.eval(); model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    secrets = sample_secrets(model.config, seed=a.seed, rope_compatible=True)
    server = copy.deepcopy(model); conjugate_qwen2(server, secrets)
    ids = tok("Hello, world.", return_tensors="pt")["input_ids"].to(device)
    ver = verify_equivariance(model, server, secrets, ids)

    report.update({
        "sft_curve": curve, "sft_wall_min": round(sft_min, 2),
        "recovered_distill_ppl": curve[-1]["distill_ppl"],
        "recovered_relevance": curve[-1]["relevance"], "recovered_ascii": curve[-1]["ascii"],
        "recovered_equivariance": {"logit_top1_agreement": ver["logit_top1_agreement"],
                                   "equivariant": ver["equivariant"]},
        "summary": {
            "relevance_original_retrofit_recovered": [orig_rel, r_rel, curve[-1]["relevance"]],
            "distill_ppl_retrofit_recovered": [round(r_ppl, 2), curve[-1]["distill_ppl"]],
            "teacher_gen_min": round(distill_gen_min, 2), "sft_wall_min": round(sft_min, 2),
            "total_recovery_min": round(distill_gen_min + sft_min, 2),
            "note": "instruction-following recovery via self-distillation SFT; total wall-cost = "
                    "teacher generation + SFT. Ours needs ZERO of this (serves the unmodified "
                    "instruct model). Recovered model still conjugates to an exact CONJFORMER server.",
        },
    })
    print(f"[done] relevance {orig_rel}(orig) {r_rel}(retrofit) -> {curve[-1]['relevance']}(recovered); "
          f"equivariant={ver['equivariant']}; total {distill_gen_min + sft_min:.1f}min", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sp = a.samples_out or (str(Path(a.out).with_suffix("")) + "_samples.json")
    with open(sp, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {a.out} + {sp}", flush=True)


if __name__ == "__main__":
    main()
