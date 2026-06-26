"""Real LoRA SFT on Dolly-15k for Qwen2.5-7B-Instruct (PEFT, assistant-only loss).

Minimal, stable supervised fine-tuning: format each Dolly example with the Qwen
chat template (user=prompt, assistant=response), mask the loss to the assistant
RESPONSE tokens only, and train a PEFT LoRA adapter. Saves a standard HF PEFT
adapter (raw A/B) -- the FOLDED, mask-protected package is produced separately by
the E6 pipeline; this trainer never touches the protocol/folding path.

The data + formatting helpers are import-safe (no torch) so they are unit-tested
without loading the 7B model.

Example (H800)::

    python scripts/train_qwen7b_lora_dolly.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --train-jsonl /root/autodl-tmp/datasets/dolly/dolly_train.jsonl \\
        --valid-jsonl /root/autodl-tmp/datasets/dolly/dolly_valid.jsonl \\
        --output-adapter-dir /root/autodl-tmp/privacy_llm_packages/qwen7b_lora_dolly_r16 \\
        --max-seq-len 1024 --epochs 1 --lora-rank 16 --lora-alpha 32 \\
        --dtype bfloat16 --device cuda \\
        --output-json outputs/lora_dolly/train_qwen7b_lora_dolly.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


# ---- import-safe data + formatting helpers (unit-tested) -------------------

def build_prompt(row: dict) -> str:
    """Prompt for a Dolly row: explicit ``prompt`` field if present, else
    ``instruction`` (+ ``context`` when non-empty)."""
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()
    instr = str(row.get("instruction", "") or "").strip()
    ctx = str(row.get("context", "") or "").strip()
    return ("%s\n\n%s" % (instr, ctx)).strip() if ctx else instr


def load_dolly_jsonl(path, max_examples=None):
    """Parse a Dolly JSONL -> (rows, skipped). Each kept row is
    {id, prompt, response, category}; rows without a usable prompt+response are
    skipped (malformed / empty lines tolerated)."""
    rows, skipped = [], 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                # noqa: BLE001
                skipped += 1
                continue
            if not isinstance(rec, dict):
                skipped += 1
                continue
            prompt = build_prompt(rec)
            resp = rec.get("response")
            if not (prompt and isinstance(resp, str) and resp.strip()):
                skipped += 1
                continue
            rows.append({"id": rec.get("id", "ex-%d" % i), "prompt": prompt,
                         "response": resp, "category": rec.get("category")})
            if max_examples and len(rows) >= int(max_examples):
                break
    return rows, skipped


def format_training_example(tokenizer, prompt, response, max_seq_len):
    """Chat-templated SFT example with ASSISTANT-ONLY loss.

    Builds the user-turn prefix (with the assistant generation prefix) and the
    full user+assistant text via the chat template, tokenizes both, and labels
    only the assistant-response tokens (prompt + scaffold -> -100). Truncates to
    ``max_seq_len``. Returns {input_ids, attention_mask, labels}."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}], tokenize=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    full_ids = list(full_ids[:int(max_seq_len)])
    n_prefix = len(prefix_ids)
    labels = [(-100 if i < n_prefix else full_ids[i])
              for i in range(len(full_ids))]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels}


def trainable_param_summary(model):
    """(trainable, total, ratio) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total, (trainable / total if total else 0.0)


# ---- training driver (loads the model; not unit-tested) --------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--valid-jsonl", default=None)
    ap.add_argument("--output-adapter-dir", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-train-examples", type=int, default=0)
    ap.add_argument("--max-valid-examples", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=_DEFAULT_TARGETS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(args.dtype, torch.bfloat16)
    torch.manual_seed(int(args.seed))

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_rows, train_skip = load_dolly_jsonl(
        args.train_jsonl, args.max_train_examples or None)
    valid_rows, valid_skip = ([], 0)
    if args.valid_jsonl:
        valid_rows, valid_skip = load_dolly_jsonl(
            args.valid_jsonl, args.max_valid_examples or None)

    def _featurize(rows):
        return [format_training_example(tok, r["prompt"], r["response"],
                                        args.max_seq_len) for r in rows]

    train_feats = _featurize(train_rows)
    valid_feats = _featurize(valid_rows)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, feats):
            self.feats = feats

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, i):
            return self.feats[i]

    def _collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids, attn, lab = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * n)
            attn.append(b["attention_mask"] + [0] * n)
            lab.append(b["labels"] + [-100] * n)
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(lab)}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, device_map=args.device,
        trust_remote_code=True, local_files_only=True)
    model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=int(args.lora_rank), lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=[m.strip() for m in args.target_modules.split(",")
                        if m.strip()],
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable, total, ratio = trainable_param_summary(model)

    out_dir = Path(args.output_adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.learning_rate), logging_steps=10,
        save_strategy="no", report_to=[], seed=int(args.seed),
        bf16=(args.dtype == "bfloat16"), fp16=(args.dtype == "float16"))
    trainer = Trainer(model=model, args=targs, train_dataset=_DS(train_feats),
                      eval_dataset=(_DS(valid_feats) if valid_feats else None),
                      data_collator=_collate)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_out = trainer.train()
    train_runtime_s = time.perf_counter() - t0
    train_loss_final = float(getattr(train_out, "training_loss", None) or 0.0)
    valid_loss_final = None
    if valid_feats:
        ev = trainer.evaluate()
        valid_loss_final = float(ev.get("eval_loss")) if "eval_loss" in ev else None
    peak_mb = (round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
               if (args.device == "cuda" and torch.cuda.is_available()) else None)

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    adapter_param_count = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n)

    report = {
        "stage": "train_qwen7b_lora_dolly",
        "dataset": "databricks-dolly-15k",
        "model_path": args.model_path, "output_adapter_dir": str(out_dir),
        "train_jsonl": args.train_jsonl, "valid_jsonl": args.valid_jsonl,
        "num_train_examples": len(train_rows),
        "num_valid_examples": len(valid_rows),
        "num_train_skipped": train_skip, "num_valid_skipped": valid_skip,
        "max_seq_len": int(args.max_seq_len), "epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "lora_rank": int(args.lora_rank), "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "target_modules": args.target_modules,
        "train_loss_final": train_loss_final,
        "valid_loss_final": valid_loss_final,
        "train_runtime_s": round(train_runtime_s, 3),
        "peak_gpu_memory_mb": peak_mb,
        "adapter_param_count": int(adapter_param_count),
        "trainable_param_count": int(trainable),
        "total_param_count": int(total),
        "trainable_param_ratio": round(ratio, 8),
        "adapter_is_raw_peft": True,
        "paper_ready": True,
    }
    pj = Path(args.output_json)
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=== Dolly LoRA SFT done ===")
    print("train=%d valid=%d train_loss=%.4f valid_loss=%s adapter_params=%d "
          "(%.4f%% trainable)" % (len(train_rows), len(valid_rows),
                                  train_loss_final, valid_loss_final,
                                  adapter_param_count, 100.0 * ratio))
    print("adapter saved -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
