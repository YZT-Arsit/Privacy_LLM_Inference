"""PHASE 2 — deterministic E2E NLG data pipeline for the Real Generative LoRA stage.

Mac control-plane (trusted). Tokenizes the official E2E NLG split (HF auto-converted parquet) with the
pinned Qwen2.5 tokenizer (tokenizer.json sha256 c0382117...), and emits the SAME strict A10/TDX split
used by the existing protected clm path (see scripts/prepare_utility_data.py):

  * A10 artifact  (untrusted side): per-sample {sample_id, input_ids, sup_start}. For teacher-forced
                  CLM the target text is inherently part of input_ids the model forwards (documented in
                  the Phase-11 security audit); the supervision/labels/CE live only in TDX.
  * TDX label table (trusted side): sample_id -> {target (shifted next-token ids, IGNORE on prompt),
                  sup_positions}.

Plus generation artifacts (prompt-only) and the multi-reference test set for BLEU:
  * *_gen.json: {sample_id, prompt_text, prompt_ids, meaning_representation, references[]}.

Determinism: fixed template, fixed max lengths, fixed subset selection seed. Full config recorded in
dataset_manifest.json + split_hashes.sha256. Identical artifacts feed G0/G1/G2 (same template, tokenizer,
lengths, order, splits).

Template (frozen this stage):
    User:
    Generate a natural-language description for the following restaurant attributes:
    <meaning_representation>

    Assistant:
    <human_reference>
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import OrderedDict
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/aaai_private_base/generative_lora/data"
TOK_DIR = "/private/tmp/claude-501/-Users-Hoshino-Desktop-privacy-llm-obfuscation/03c8cc57-b3de-441b-b9fe-3686deb94965/scratchpad/tok"
TOK_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
IGNORE = -100

PROMPT_TMPL = ("User:\nGenerate a natural-language description for the following restaurant "
               "attributes:\n{mr}\n\nAssistant:\n")
MAX_INPUT = 128        # prompt tokens (MR is short; truncation stats recorded)
MAX_TARGET = 96        # reference tokens (+eos)
MAX_SEQ = MAX_INPUT + MAX_TARGET + 4
SUBSET_SEED = 20260713


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def batch_schedule(n, seed, epochs, batch_size, accum=1):
    sched, gstep, micro = [], 0, 0
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 100003 + ep)
        order = torch.randperm(n, generator=g).tolist()
        for b0 in range(0, n, batch_size):
            ids = order[b0:b0 + batch_size]
            sched.append({"epoch": ep, "global_batch_index": gstep,
                          "microbatch_index": micro % accum, "optimizer_step": gstep // accum,
                          "sample_ids": ids, "size": len(ids)})
            micro += 1; gstep += 1
    return sched, sha_obj(sched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000, help="capped train subset (deterministic sample)")
    ap.add_argument("--n-val-gen", type=int, default=300, help="val generation subset (pilot selection)")
    ap.add_argument("--n-test-gen", type=int, default=500, help="test generation subset (unique MRs)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", default="1234,2025,7")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha_file(Path(TOK_DIR) / "tokenizer.json") == TOK_SHA, "tokenizer hash mismatch"
    tk = AutoTokenizer.from_pretrained(TOK_DIR)
    eos = tk.eos_token_id

    ds = load_dataset("e2e_nlg", revision="refs/convert/parquet")
    official = {k: len(v) for k, v in ds.items()}
    report = {"dataset": "e2e_nlg", "revision": "refs/convert/parquet",
              "official_split_sizes": official, "tokenizer_sha256": TOK_SHA,
              "prompt_template": PROMPT_TMPL, "target_template": "{human_reference} <eos>",
              "max_input_len": MAX_INPUT, "max_target_len": MAX_TARGET, "ignore_index": IGNORE,
              "subset_seed": SUBSET_SEED, "eos": eos, "splits": {}}
    manifest = {"dataset_name": "e2e_nlg", "dataset_config": "default",
                "source": "huggingface: e2e_nlg @ refs/convert/parquet",
                "official_split_sizes": official, "tokenizer_name": "Qwen2.5-0.5B",
                "tokenizer_sha256": TOK_SHA, "prompt_template": PROMPT_TMPL,
                "target_template": "{human_reference} <eos>", "max_input_length": MAX_INPUT,
                "max_target_length": MAX_TARGET, "random_seed": SUBSET_SEED,
                "n_train": args.n_train, "n_val_gen": args.n_val_gen, "n_test_gen": args.n_test_gen,
                "epochs": args.epochs, "batch_size": args.batch_size, "artifacts": {}}
    split_hashes = {}

    def tokenize_clm(mr, ref):
        prompt = PROMPT_TMPL.format(mr=mr.strip())
        p_ids = tk(prompt, add_special_tokens=False)["input_ids"][:MAX_INPUT]
        a_ids = tk(" " + ref.strip(), add_special_tokens=False)["input_ids"][:MAX_TARGET] + [eos]
        ids = p_ids + a_ids
        tgt = [IGNORE] * len(ids)
        for t in range(len(ids) - 1):
            if t >= len(p_ids) - 1:
                tgt[t] = ids[t + 1]
        sup_positions = [t for t in range(len(ids) - 1) if tgt[t] != IGNORE]
        return p_ids, ids, tgt, sup_positions

    # ---------- TRAIN (capped deterministic subset) ----------
    n_all = official["train"]
    g = torch.Generator().manual_seed(SUBSET_SEED)
    perm = torch.randperm(n_all, generator=g).tolist()[:args.n_train]
    tr = ds["train"]
    a10, labels, seqlens, suplens, trunc = [], {}, [], [], 0
    for new_id, orig in enumerate(perm):
        row = tr[orig]
        p_ids, ids, tgt, sup = tokenize_clm(row["meaning_representation"], row["human_reference"])
        if len(ids) > MAX_SEQ:
            ids = ids[:MAX_SEQ]; tgt = tgt[:MAX_SEQ]; sup = [s for s in sup if s < MAX_SEQ - 1]; trunc += 1
        a10.append({"sample_id": new_id, "input_ids": ids, "sup_start": len(p_ids) - 1})
        labels[new_id] = {"target": tgt, "sup_positions": sup}
        seqlens.append(len(ids)); suplens.append(len(sup))
    ap10 = OUT / "e2e_train_a10.pt"; alab = OUT / "e2e_train_labels.pt"
    torch.save(a10, ap10); torch.save(labels, alab)
    for nm, p in [("e2e_train_a10", ap10), ("e2e_train_labels", alab)]:
        split_hashes[p.name] = sha_file(p); manifest["artifacts"][nm] = {"rows": len(a10), "sha256": sha_file(p)}
    report["splits"]["train"] = {"rows": len(a10), "orig_indices_sha": sha_obj(perm),
                                 "max_seq": max(seqlens), "mean_seq": round(sum(seqlens)/len(seqlens), 1),
                                 "mean_sup": round(sum(suplens)/len(suplens), 1), "truncated": trunc}

    # ---------- VALIDATION (full teacher-forced eval .pt + generation subset) ----------
    va = ds["validation"]
    va_a10, va_lab, vseq = [], {}, []
    for i in range(len(va)):
        row = va[i]
        p_ids, ids, tgt, sup = tokenize_clm(row["meaning_representation"], row["human_reference"])
        if len(ids) > MAX_SEQ:
            ids = ids[:MAX_SEQ]; tgt = tgt[:MAX_SEQ]; sup = [s for s in sup if s < MAX_SEQ - 1]
        va_a10.append({"sample_id": i, "input_ids": ids, "sup_start": len(p_ids) - 1})
        va_lab[i] = {"target": tgt, "sup_positions": sup}; vseq.append(len(ids))
    vp10 = OUT / "e2e_val_a10.pt"; vlab = OUT / "e2e_val_labels.pt"
    torch.save(va_a10, vp10); torch.save(va_lab, vlab)
    for nm, p in [("e2e_val_a10", vp10), ("e2e_val_labels", vlab)]:
        split_hashes[p.name] = sha_file(p); manifest["artifacts"][nm] = {"rows": len(va_a10), "sha256": sha_file(p)}

    # validation generation subset: unique MRs -> references[]
    def grouped(hf, n_unique):
        by_mr = OrderedDict()
        for i in range(len(hf)):
            row = hf[i]; mr = row["meaning_representation"]
            by_mr.setdefault(mr, []).append(row["human_reference"])
        items = list(by_mr.items())[:n_unique]
        out = []
        for j, (mr, refs) in enumerate(items):
            prompt = PROMPT_TMPL.format(mr=mr.strip())
            pid = tk(prompt, add_special_tokens=False)["input_ids"][:MAX_INPUT]
            out.append({"sample_id": j, "meaning_representation": mr, "prompt_text": prompt,
                        "prompt_ids": pid, "references": refs})
        return out

    vgen = grouped(va, args.n_val_gen)
    vg = OUT / "e2e_val_gen.json"; vg.write_text(json.dumps(vgen))
    split_hashes[vg.name] = sha_file(vg); manifest["artifacts"]["e2e_val_gen"] = {"unique_mr": len(vgen), "sha256": sha_file(vg)}
    report["splits"]["validation"] = {"rows_tf": len(va_a10), "gen_unique_mr": len(vgen),
                                      "max_seq": max(vseq), "mean_seq": round(sum(vseq)/len(vseq), 1)}

    # ---------- TEST (generation only, multi-reference) ----------
    te = ds["test"]
    tgen = grouped(te, args.n_test_gen)
    tp = OUT / "e2e_test_gen.json"; tp.write_text(json.dumps(tgen))
    split_hashes[tp.name] = sha_file(tp); manifest["artifacts"]["e2e_test_gen"] = {"unique_mr": len(tgen), "sha256": sha_file(tp)}
    total_te_unique = len(OrderedDict((te[i]["meaning_representation"], 1) for i in range(len(te))))
    report["splits"]["test"] = {"gen_unique_mr": len(tgen), "total_unique_mr_available": total_te_unique,
                                "cap_applied": len(tgen) < total_te_unique,
                                "mean_refs_per_mr": round(sum(len(x["references"]) for x in tgen)/len(tgen), 2)}

    # ---------- schedules ----------
    for seed in [int(s) for s in args.seeds.split(",")]:
        sched, h = batch_schedule(args.n_train, seed, args.epochs, args.batch_size)
        sp = OUT / f"e2e_schedule_s{seed}.json"; sp.write_text(json.dumps({"hash": h, "schedule": sched}))
        split_hashes[sp.name] = sha_file(sp)
        manifest["artifacts"][f"e2e_schedule_s{seed}"] = {"batches": len(sched), "opt_steps": sched[-1]["optimizer_step"]+1, "hash": h}

    # jsonl mirrors for auditing (train/val/test)
    (OUT / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a10))
    (OUT / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in vgen))
    (OUT / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tgen))
    for nm in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        split_hashes[nm] = sha_file(OUT / nm)

    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (OUT / "preprocessing_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (OUT / "split_hashes.sha256").write_text("\n".join(f"{h}  {n}" for n, h in sorted(split_hashes.items())) + "\n")
    print("[phase2] train", len(a10), "val_tf", len(va_a10), "val_gen", len(vgen), "test_gen", len(tgen))
    print("[phase2] train mean_seq", report["splits"]["train"]["mean_seq"], "max", report["splits"]["train"]["max_seq"],
          "truncated", trunc)
    print("[phase2] wrote dataset_manifest.json + split_hashes.sha256")


if __name__ == "__main__":
    main()
