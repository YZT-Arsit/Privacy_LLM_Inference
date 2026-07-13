"""G0/G1 — plaintext-coordinate LoRA reference on the REAL A10 GPU (plain torch + peft).

NOT protected, NOT TDX. Loads the plaintext Qwen2.5-0.5B checkpoint and trains ordinary LoRA with
standard AdamW, then runs deterministic greedy generation. This is the utility reference (G1) and the
base cell (G0, --no-lora). Same template / tokenizer / data / order as the protected cell.

Modes:
  train     : train LoRA on e2e_train_a10.pt using a fixed schedule; save adapter.
  generate  : greedy-decode prompts from a *_gen.json; write generations jsonl.

Runs on A10 (/root/qwen25_05b). Determinism: fixed seeds, greedy decode, prompt-masked CLM labels
identical to the protected supervision (labels[t]=input_ids[t+1], prompt=IGNORE, sup_start).
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch

IGNORE = -100


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def load_model(ckpt, targets, rank, alpha, dtype, no_lora, adapter_path=""):
    from transformers import AutoModelForCausalLM
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dt).cuda()
    if not no_lora:
        from peft import LoraConfig, get_peft_model, PeftModel
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=0.0,
                             bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg)
    return model


def build_batch(recs, dev):
    maxlen = max(len(r["input_ids"]) for r in recs)
    ids = torch.zeros(len(recs), maxlen, dtype=torch.long)
    lab = torch.full((len(recs), maxlen), IGNORE, dtype=torch.long)
    att = torch.zeros(len(recs), maxlen, dtype=torch.long)
    for i, r in enumerate(recs):
        x = r["input_ids"]; n = len(x)
        ids[i, :n] = torch.tensor(x); att[i, :n] = 1
        # HF CausalLM shifts labels internally (logits[:-1] vs labels[1:]); pass UNSHIFTED token ids at
        # the answer positions (prompt = IGNORE). Position t's label = x[t]; HF makes logit_{t-1} predict it.
        for t in range(r["sup_start"] + 1, n):     # answer token positions (incl. eos)
            lab[i, t] = x[t]
    return ids.to(dev), att.to(dev), lab.to(dev)


def train(args):
    dev = "cuda"
    torch.manual_seed(args.seed)
    model = load_model(args.ckpt, args.targets.split(","), args.rank, args.alpha, args.dtype, False)
    model.train()
    data = torch.load(args.train_data, map_location="cpu", weights_only=False)
    by_id = {r["sample_id"]: r for r in data}
    sched = json.loads(Path(args.schedule).read_text())["schedule"]
    if args.max_steps > 0:
        sched = sched[:args.max_steps]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    total = len(sched)
    warm = max(1, int(args.warmup * total))
    def lr_at(s):
        if s < warm: return args.lr * s / warm
        prog = (s - warm) / max(1, total - warm)
        return args.lr * max(0.0, 1.0 - prog)      # linear decay
    t0 = time.time(); losses = []
    for si, b in enumerate(sched):
        for g in opt.param_groups: g["lr"] = lr_at(si)
        recs = [by_id[i] for i in b["sample_ids"] if i in by_id]
        ids, att, lab = build_batch(recs, dev)
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        loss = out.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss))
        if si % 20 == 0 or si == total - 1:
            print(f"  step {si}/{total} lr {lr_at(si):.2e} loss {float(loss):.4f} "
                  f"avg20 {sum(losses[-20:])/len(losses[-20:]):.4f} {time.time()-t0:.0f}s", flush=True)
    model.save_pretrained(args.out_adapter)
    meta = {"mode": "train", "ckpt": args.ckpt, "targets": args.targets, "rank": args.rank,
            "alpha": args.alpha, "dtype": args.dtype, "lr": args.lr, "seed": args.seed,
            "steps": total, "schedule": Path(args.schedule).name, "warmup": args.warmup,
            "final_loss": losses[-1], "mean_last20": sum(losses[-20:])/len(losses[-20:]),
            "wall_sec": round(time.time()-t0, 1), "adapter_dir": args.out_adapter}
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print("[train] done", json.dumps({k: meta[k] for k in ["final_loss","mean_last20","steps","wall_sec"]}))


def rep_stats(toks):
    if len(toks) < 2: return 0.0
    bigrams = list(zip(toks, toks[1:]))
    uniq = len(set(bigrams))
    return round(1.0 - uniq / max(1, len(bigrams)), 3)   # fraction repeated bigrams


def generate(args):
    dev = "cuda"
    model = load_model(args.ckpt, args.targets.split(","), args.rank, args.alpha, args.dtype,
                       args.no_lora, adapter_path=args.adapter if args.adapter else "")
    model.eval()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.tok)
    eos = tk.eos_token_id
    prompts = json.loads(Path(args.gen_in).read_text())
    if args.gen_max > 0: prompts = prompts[:args.gen_max]
    adapter_hash = "none"
    if args.adapter:
        st = Path(args.adapter) / "adapter_model.safetensors"
        adapter_hash = sha16(st.read_bytes()) if st.exists() else "missing"
    decode_cfg = {"strategy": "greedy", "max_new_tokens": args.max_new, "eos": eos}
    out_lines = []; t0 = time.time(); n_tokens = 0
    for j, p in enumerate(prompts):
        pid = torch.tensor([p["prompt_ids"]], device=dev)
        with torch.no_grad():
            gen = model.generate(input_ids=pid, max_new_tokens=args.max_new, do_sample=False,
                                 num_beams=1, pad_token_id=eos, eos_token_id=eos)
        new = gen[0, pid.shape[1]:].tolist()
        if eos in new:
            fr = "eos"; new = new[:new.index(eos)]
        else:
            fr = "length"
        text = tk.decode(new, skip_special_tokens=True).strip()
        n_tokens += len(new)
        rec = {"sample_id": p["sample_id"], "input_hash": sha16(json.dumps(p["prompt_ids"]).encode()),
               "meaning_representation": p.get("meaning_representation", ""),
               "references": p.get("references", []), "generated_text": text, "token_ids": new,
               "finish_reason": fr, "generated_token_count": len(new),
               "invalid_output": len(text) == 0, "repetition_bigram_frac": rep_stats(new),
               "adapter_hash": adapter_hash, "base_ckpt_sha16": args.base_hash,
               "decoding": decode_cfg}
        out_lines.append(rec)
        if j % 50 == 0: print(f"  gen {j}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    Path(args.gen_out).write_text("".join(json.dumps(r) + "\n" for r in out_lines))
    prof = {"mode": "generate", "cell": args.cell, "n": len(out_lines), "no_lora": args.no_lora,
            "adapter": args.adapter, "adapter_hash": adapter_hash, "decoding": decode_cfg,
            "total_new_tokens": n_tokens, "wall_sec": round(time.time()-t0, 1),
            "tokens_per_sec": round(n_tokens/max(1e-9, time.time()-t0), 1),
            "gpu": torch.cuda.get_device_name(0), "dtype": args.dtype}
    Path(args.gen_out + ".profile.json").write_text(json.dumps(prof, indent=2))
    print("[generate] done", json.dumps({k: prof[k] for k in ["cell","n","total_new_tokens","wall_sec","tokens_per_sec"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default="/root/qwen25_05b")
    common.add_argument("--targets", default="q_proj,v_proj")
    common.add_argument("--rank", type=int, default=8)
    common.add_argument("--alpha", type=int, default=16)
    common.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    common.add_argument("--seed", type=int, default=1234)

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--train-data", required=True); t.add_argument("--schedule", required=True)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--warmup", type=float, default=0.03)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--out-adapter", required=True); t.add_argument("--out-meta", required=True)

    g = sub.add_parser("generate", parents=[common])
    g.add_argument("--adapter", default=""); g.add_argument("--no-lora", action="store_true")
    g.add_argument("--tok", default="/root/genlora_tok")
    g.add_argument("--gen-in", required=True); g.add_argument("--gen-out", required=True)
    g.add_argument("--gen-max", type=int, default=0); g.add_argument("--max-new", type=int, default=96)
    g.add_argument("--cell", default=""); g.add_argument("--base-hash", default="88c142557820ccad")

    args = ap.parse_args()
    if args.mode == "train": train(args)
    else: generate(args)


if __name__ == "__main__":
    main()
