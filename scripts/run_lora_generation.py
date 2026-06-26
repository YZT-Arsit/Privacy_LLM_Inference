"""Plaintext generation for base vs LoRA (PEFT adapter) Qwen2.5-7B-Instruct.

Greedy decode with the SAME chat-template + generation conventions as
``run_ifeval_generation.py`` (shared via ``build_predictor`` /
``format_prompt_for_generation``), for two backends:

* ``base_plaintext`` -- base model;
* ``lora_plaintext`` -- base model + a PEFT LoRA adapter (``--adapter-path``).

This is the NON-private plaintext baseline for the LoRA experiment; the folded
remote LoRA path is produced/evaluated by the E6 pipeline. Emits a responses JSONL
+ a report JSON (with per-example prompt-format + token-count metadata).

Example::

    python scripts/run_lora_generation.py --backend lora_plaintext \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --adapter-path /root/autodl-tmp/privacy_llm_packages/qwen7b_lora_dolly_r16 \\
        --input-jsonl /root/autodl-tmp/datasets/dolly/dolly_test.jsonl \\
        --use-chat-template --seq-len 1024 --max-new-tokens 256 \\
        --dtype bfloat16 --device cuda \\
        --output-response-jsonl outputs/lora_dolly/lora_plaintext_responses.jsonl \\
        --output-report-json    outputs/lora_dolly/lora_plaintext_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_BACKENDS = ("base_plaintext", "lora_plaintext")


# ---- import-safe helpers (unit-tested) -------------------------------------

def load_prompts(path, max_examples=None):
    """Load a prompt JSONL (Dolly/IFEval style). Returns list of dicts with at
    least ``prompt``; derives prompt from instruction(+context) when absent."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            if not isinstance(rec, dict):
                continue
            prompt = rec.get("prompt")
            if not (isinstance(prompt, str) and prompt.strip()):
                instr = str(rec.get("instruction", "") or "").strip()
                ctx = str(rec.get("context", "") or "").strip()
                prompt = ("%s\n\n%s" % (instr, ctx)).strip() if ctx else instr
            if not prompt:
                continue
            rec["prompt"] = prompt
            rec.setdefault("id", "ex-%d" % i)
            rows.append(rec)
            if max_examples and len(rows) >= int(max_examples):
                break
    return rows


def finish_reason(token_ids, eos_ids, max_new_tokens):
    """'eos' if the generation ended on an eos token before the budget, else
    'length'."""
    if token_ids and eos_ids and int(token_ids[-1]) in set(eos_ids) \
            and len(token_ids) < int(max_new_tokens):
        return "eos"
    return "length"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--backend", required=True, choices=list(_BACKENDS))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-chat-template", action="store_true", default=False)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--output-response-jsonl", required=True)
    ap.add_argument("--output-report-json", required=True)
    args = ap.parse_args()

    if args.backend == "lora_plaintext" and not args.adapter_path:
        print("ERROR: lora_plaintext requires --adapter-path", file=sys.stderr)
        return 3

    from pllo.benchmarks.real_predictors import (
        _normalize_eos_ids, build_predictor)

    rows = load_prompts(args.input_jsonl, args.max_examples or None)
    if not rows:
        print("ERROR: no prompts in %s" % args.input_jsonl, file=sys.stderr)
        return 3

    predictor = build_predictor(
        "plaintext_local", model_path=args.model_path,
        model_name=args.model_name, seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens, dtype=args.dtype, device=args.device,
        use_chat_template=bool(args.use_chat_template),
        adapter_path=(args.adapter_path if args.backend == "lora_plaintext"
                      else None))

    eos_ids = _normalize_eos_ids(getattr(predictor._tok, "eos_token_id", None))
    pad_id = getattr(predictor._tok, "pad_token_id", None)

    responses, fmt_sha, p_tok_counts, gen_counts, finishes = [], [], [], [], []
    t0 = time.perf_counter()
    for r in rows:
        g = predictor.generate(r["prompt"])
        toks = g.get("token_ids") or []
        fr = finish_reason(toks, eos_ids, args.max_new_tokens)
        responses.append({
            "id": r.get("id"), "prompt": r["prompt"],
            "response": g.get("text", ""), "category": r.get("category"),
            "generated_tokens": len(toks), "finish_reason": fr,
            "prompt_format": g.get("prompt_format"),
            "formatted_prompt_sha256": g.get("formatted_prompt_sha256"),
            "prompt_token_count": g.get("prompt_token_count")})
        fmt_sha.append(g.get("formatted_prompt_sha256"))
        p_tok_counts.append(g.get("prompt_token_count"))
        gen_counts.append(len(toks))
        finishes.append(fr)
    online_s = time.perf_counter() - t0
    total_gen = sum(gen_counts) or None

    report = {
        "stage": "lora_generation", "backend": args.backend,
        "model_name": args.model_name, "dataset": "databricks-dolly-15k",
        "num_examples": len(rows), "adapter_path": args.adapter_path,
        "use_chat_template": bool(args.use_chat_template),
        "prompt_format": ("chat" if args.use_chat_template else "raw"),
        "seq_len": int(args.seq_len), "max_new_tokens": int(args.max_new_tokens),
        "dtype": args.dtype, "device": args.device,
        "prompt_token_count_per_example": p_tok_counts,
        "formatted_prompt_sha256_per_example": fmt_sha,
        "generated_tokens_per_example": gen_counts,
        "finish_reason_per_example": finishes,
        "online_generation_latency_s": round(online_s, 4),
        "latency_per_generated_token_s": (round(online_s / total_gen, 6)
                                          if total_gen else None),
        "eos_token_id": (sorted(eos_ids) if eos_ids else None),
        "pad_token_id": pad_id,
        "stop_on_eos": True,
        # plaintext baseline: no private GPU boundary (security flags N/A)
        "tee_used_on_gpu": False, "worker_has_mask_secrets": False,
        "worker_has_raw_lora": False, "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": [],
        "paper_ready": True,
    }
    rp = Path(args.output_response_jsonl)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as fh:
        for r in responses:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    jp = Path(args.output_report_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== LoRA generation (%s) ===" % args.backend)
    print("num_examples=%d adapter=%s use_chat_template=%s"
          % (len(rows), args.adapter_path, args.use_chat_template))
    print("avg_generated_tokens=%.1f online_latency_s=%.2f"
          % (sum(gen_counts) / max(1, len(gen_counts)), online_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
