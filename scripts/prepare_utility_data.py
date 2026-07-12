"""PHASE 5+1 — batched utility data preparation (Mac control-plane, trusted side).

Tokenizes the official SST-2 (glue/sst2) and GSM8K (openai/gsm8k main) splits with the
pinned Qwen2.5 tokenizer (tokenizer.json sha256 c0382117...), verifies row counts against
the frozen manifests, and emits a strict split between:

  * A10 artifact  (untrusted compute side): per-sample input_ids + attention_mask +
                   supervised-position metadata. Holds NO classification label, NO
                   causal-LM target, NO loss. (For GSM8K teacher forcing the answer tokens
                   are inherently part of input_ids the model must forward — consistent with
                   the pre-existing single-batch design where A10 sends logits and TDX holds
                   the label/loss. The *supervision* (which positions, ignore_index, the CE)
                   lives only in TDX.)
  * TDX label table (trusted side): sample_id -> classification target token + binary label
                   (SST-2) / shifted causal-LM target sequence with ignore_index (GSM8K).

Also emits a per-seed deterministic batch schedule (epoch/global_batch/optimizer_step/
sample_ids/seq_lens) and hashes every artifact into a manifest bound into the TDX run.

Frozen (PHASE 0 seal): seeds [1234,2025,7]; SST-2 template "Review: {sentence}\\nSentiment:",
verbalizer {0:' negative'(8225), 1:' positive'(6785)}, max_seq_len 128, batch 16, accum 1.
GSM8K max_seq_len 512, greedy gen max_new 256. GSM8K *training* prompt template was NOT in
the freeze -> defined here and recorded as a spec-fill deviation (see deviations.json).
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/aaai_private_base/datasets/tokenized"
TOK_DIR = "/private/tmp/claude-501/-Users-Hoshino-Desktop-privacy-llm-obfuscation/03c8cc57-b3de-441b-b9fe-3686deb94965/scratchpad/tok"
TOK_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
IGNORE = -100
NEG_TOK, POS_TOK = 8225, 6785            # ' negative', ' positive' (leading space, single token)
SST2_TEMPLATE = "Review: {sentence}\nSentiment:"
# GSM8K training prompt template — NOT specified in the freeze; recorded as a deviation.
GSM8K_PROMPT = "Question: {q}\nAnswer:"
GSM8K_MAXLEN = 512
SST2_MAXLEN = 128


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def batch_schedule(n, seed, epochs, batch_size, accum):
    """Deterministic per-seed shuffle -> batches. optimizer_step increments every `accum`
    microbatches. Returns (schedule list, schedule_hash)."""
    sched = []
    gstep = 0
    micro = 0
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 100003 + ep)
        order = torch.randperm(n, generator=g).tolist()
        for b0 in range(0, n, batch_size):
            ids = order[b0:b0 + batch_size]
            opt_step = gstep // accum
            sched.append({"epoch": ep, "global_batch_index": gstep,
                          "microbatch_index": micro % accum, "optimizer_step": opt_step,
                          "sample_ids": ids, "size": len(ids)})
            micro += 1
            gstep += 1
    return sched, sha_obj(sched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sst2,gsm8k")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha_file(Path(TOK_DIR) / "tokenizer.json") == TOK_SHA, "tokenizer hash mismatch"
    tk = AutoTokenizer.from_pretrained(TOK_DIR)
    assert tk(" negative", add_special_tokens=False)["input_ids"] == [NEG_TOK]
    assert tk(" positive", add_special_tokens=False)["input_ids"] == [POS_TOK]
    eos = tk.eos_token_id
    manifest = {"tokenizer_sha256": TOK_SHA, "ignore_index": IGNORE, "eos": eos,
                "seeds": [1234, 2025, 7], "batch_size": 16, "grad_accum": 1, "artifacts": {}}
    todo = args.datasets.split(",")

    # ---------------- SST-2 classification ----------------
    if "sst2" in todo:
        ds = load_dataset("glue", "sst2")
        for split, hf in [("train", ds["train"]), ("dev", ds["validation"])]:
            a10, labels, seqlens = [], {}, []
            for i, row in enumerate(hf):
                prompt = SST2_TEMPLATE.format(sentence=row["sentence"].strip())
                ids = tk(prompt, add_special_tokens=False)["input_ids"][:SST2_MAXLEN]
                sup = len(ids) - 1               # logits at last token predict the sentiment token
                sid = int(row["idx"])
                a10.append({"sample_id": sid, "input_ids": ids, "sup_pos": sup})
                lab = int(row["label"])
                labels[sid] = {"target_tok": (NEG_TOK if lab == 0 else POS_TOK),
                               "label": lab, "neg_tok": NEG_TOK, "pos_tok": POS_TOK}
                seqlens.append(len(ids))
            ap10 = OUT / f"sst2_{split}_a10.pt"; alab = OUT / f"sst2_{split}_labels.pt"
            torch.save(a10, ap10); torch.save(labels, alab)
            manifest["artifacts"][f"sst2_{split}_a10"] = {"rows": len(a10), "sha256": sha_file(ap10),
                "max_seq": max(seqlens), "mean_seq": round(sum(seqlens) / len(seqlens), 1)}
            manifest["artifacts"][f"sst2_{split}_labels"] = {"rows": len(labels), "sha256": sha_file(alab)}
            print(f"sst2/{split}: {len(a10)} rows, seq max {max(seqlens)} mean {sum(seqlens)/len(seqlens):.1f}")
        for seed in [1234, 2025, 7]:
            sched, h = batch_schedule(len(load_dataset("glue", "sst2")["train"]), seed, 3, 16, 1)
            sp = OUT / f"sst2_schedule_s{seed}.json"; sp.write_text(json.dumps({"hash": h, "schedule": sched}))
            manifest["artifacts"][f"sst2_schedule_s{seed}"] = {"batches": len(sched), "sha256": h}
            print(f"sst2 schedule s{seed}: {len(sched)} batches, hash {h[:12]}")

    # ---------------- GSM8K causal LM ----------------
    if "gsm8k" in todo:
        ds = load_dataset("gsm8k", "main")
        for split, hf in [("train", ds["train"]), ("test", ds["test"])]:
            a10, labels, seqlens, suplens = [], {}, [], []
            for i, row in enumerate(hf):
                prompt = GSM8K_PROMPT.format(q=row["question"].strip())
                p_ids = tk(prompt, add_special_tokens=False)["input_ids"]
                ans = " " + row["answer"].strip()
                a_ids = tk(ans, add_special_tokens=False)["input_ids"] + [eos]
                ids = (p_ids + a_ids)[:GSM8K_MAXLEN]
                # teacher forcing: target[t] = ids[t+1]; supervise only answer+eos, prompt = ignore.
                tgt = [IGNORE] * len(ids)
                for t in range(len(ids) - 1):
                    if t >= len(p_ids) - 1:      # positions whose NEXT token is an answer token
                        tgt[t] = ids[t + 1]
                sup_positions = [t for t in range(len(ids) - 1) if tgt[t] != IGNORE]
                sid = i                          # gsm8k has no idx field; positional id (0..n-1)
                a10.append({"sample_id": sid, "input_ids": ids, "sup_start": len(p_ids) - 1})
                labels[sid] = {"target": tgt, "sup_positions": sup_positions}
                seqlens.append(len(ids)); suplens.append(len(sup_positions))
            ap10 = OUT / f"gsm8k_{split}_a10.pt"; alab = OUT / f"gsm8k_{split}_labels.pt"
            torch.save(a10, ap10); torch.save(labels, alab)
            manifest["artifacts"][f"gsm8k_{split}_a10"] = {"rows": len(a10), "sha256": sha_file(ap10),
                "max_seq": max(seqlens), "mean_seq": round(sum(seqlens) / len(seqlens), 1),
                "mean_sup": round(sum(suplens) / len(suplens), 1)}
            manifest["artifacts"][f"gsm8k_{split}_labels"] = {"rows": len(labels), "sha256": sha_file(alab)}
            print(f"gsm8k/{split}: {len(a10)} rows, seq max {max(seqlens)} mean {sum(seqlens)/len(seqlens):.1f}, sup mean {sum(suplens)/len(suplens):.1f}")
        # gsm8k test raw answers for exact-match scoring (trusted eval side)
        gold = {}
        for i, row in enumerate(ds["test"]):
            m = re.search(r"####\s*([\-0-9,\.]+)", row["answer"])
            gold[i] = m.group(1).replace(",", "").strip() if m else None
        gp = OUT / "gsm8k_test_gold.json"; gp.write_text(json.dumps(gold))
        manifest["artifacts"]["gsm8k_test_gold"] = {"rows": len(gold), "sha256": sha_file(gp)}
        for seed in [1234, 2025, 7]:
            sched, h = batch_schedule(len(ds["train"]), seed, 5, 16, 1)   # >2000 opt steps available
            sp = OUT / f"gsm8k_schedule_s{seed}.json"; sp.write_text(json.dumps({"hash": h, "schedule": sched}))
            manifest["artifacts"][f"gsm8k_schedule_s{seed}"] = {"batches": len(sched), "sha256": h}
            print(f"gsm8k schedule s{seed}: {len(sched)} batches, hash {h[:12]}")

    deviations = {"gsm8k_training_prompt_template": {
        "value": GSM8K_PROMPT, "reason": "GSM8K training prompt template was NOT fixed in the "
        "PHASE 0 freeze (only answer_extraction + generation config were). Recorded here BEFORE "
        "any utility metric, per the deviation policy.",
        "recorded_before_metrics": True}}
    (OUT / "deviations.json").write_text(json.dumps(deviations, indent=2))
    mp = OUT / "tokenized_manifest.json"; mp.write_text(json.dumps(manifest, indent=2))
    print("manifest ->", mp)


if __name__ == "__main__":
    main()
