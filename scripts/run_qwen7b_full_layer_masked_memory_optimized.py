"""Qwen2.5-7B full-layer masked execution -- memory-optimized + HF-aligned.

Untrusted-GPU masked decoder pipeline. **No TEE is imported or used** here and
no TEE execution is claimed; this only runs the untrusted masked
decoder/attention/MLP/KV/LM-head path with layerwise folded-weight streaming +
chunked folded down-projection so all 28 decoder layers fit in memory.

HF alignment (this revision): ``--seq-len`` is the MAX prompt length, not a
forced length. The prompt is built with ``tokenizer.apply_chat_template`` (when
``--use-chat-template true``), tokenized once, and the SAME ``input_ids`` /
``attention_mask`` drive (a) the official ``AutoModelForCausalLM.generate``
baseline, (b) the extracted-weight plaintext reference, and (c) the masked path.
Decode starts at the real prompt length (``attention_mask.sum()``), never at a
padded length. ModelScope cache only; never Hugging Face remote download.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    masked_prefill_full_logits,
    run_memory_optimized_masked,
)


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


# ---------------------------------------------------------------------------
# Prompt / input building (chat template, real length, no forced padding)
# ---------------------------------------------------------------------------


def load_raw_prompts(path: str | None, prompt: str | None) -> list[str]:
    if path:
        out = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(str(json.loads(line)["prompt"]))
        return out
    if prompt:
        return [prompt]
    return ["Hello, please answer briefly:"]


def build_chat_inputs(
    tokenizer, raw_prompts: list[str], max_prompt_len: int,
    use_chat_template: bool, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build ``input_ids`` / ``attention_mask`` from real prompts.

    ``max_prompt_len`` is the MAXIMUM prompt length. Short prompts keep their
    real length (no forced padding); long prompts are deterministically
    truncated and flagged. ``decode_start_index`` is the real (effective) prompt
    length, never a padded length. Batch>1 prompts of differing length are
    right-padded to the common max with an explicit ``attention_mask``."""
    chat_texts: list[str] = []
    id_lists: list[list[int]] = []
    for p in raw_prompts:
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
        else:
            chat = p
        chat_texts.append(chat)
        id_lists.append(list(tokenizer(chat)["input_ids"]))

    real_lens = [len(x) for x in id_lists]
    eff_lists, trunc_flags = [], []
    for ids in id_lists:
        if len(ids) > max_prompt_len:
            eff_lists.append(ids[:max_prompt_len])
            trunc_flags.append(True)
        else:
            eff_lists.append(ids)
            trunc_flags.append(False)
    effective_prompt_len = max(len(x) for x in eff_lists)

    pad_id = int(getattr(tokenizer, "pad_token_id", None)
                 or getattr(tokenizer, "eos_token_id", None) or 0)
    rows, masks = [], []
    for ids in eff_lists:
        pad = effective_prompt_len - len(ids)
        rows.append(ids + [pad_id] * pad)
        masks.append([1] * len(ids) + [0] * pad)

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    meta = {
        "raw_prompt": raw_prompts[0] if len(raw_prompts) == 1 else raw_prompts,
        "chat_template_input": chat_texts[0] if len(chat_texts) == 1
        else chat_texts,
        "requested_seq_len": int(max_prompt_len),
        "real_prompt_len": int(real_lens[0]) if len(real_lens) == 1
        else real_lens,
        "effective_prompt_len": int(effective_prompt_len),
        "decode_start_index": int(effective_prompt_len),
        "truncated": bool(any(trunc_flags)),
        "truncated_flags": trunc_flags,
        "used_chat_template": bool(use_chat_template),
        "pad_token_id": pad_id,
    }
    return input_ids, attention_mask, meta


# ---------------------------------------------------------------------------
# Generation comparison + status
# ---------------------------------------------------------------------------


def compare_generations(hf_ids, plain_ids, masked_ids) -> dict:
    """Token-match metrics + status across HF / plain / masked generations."""
    def rate(a, b):
        n = min(len(a), len(b))
        return (sum(1 for i in range(n) if a[i] == b[i]) / n) if n else 0.0

    seq_hp = list(hf_ids) == list(plain_ids)
    seq_hm = list(hf_ids) == list(masked_ids)
    seq_pm = list(plain_ids) == list(masked_ids)
    if seq_hp and seq_hm and seq_pm:
        status = "ok"
    elif seq_pm and not seq_hm:
        status = "hf_mismatch"
    else:
        status = "internal_mismatch"
    return {
        "hf_vs_plain_token_match_rate": round(rate(hf_ids, plain_ids), 6),
        "hf_vs_masked_token_match_rate": round(rate(hf_ids, masked_ids), 6),
        "plain_vs_masked_token_match_rate": round(rate(plain_ids, masked_ids), 6),
        "sequence_exact_match_hf_plain": seq_hp,
        "sequence_exact_match_hf_masked": seq_hm,
        "sequence_exact_match_plain_masked": seq_pm,
        "status": status,
    }


def teacher_forced_eval(model, model_config, input_ids, hf_ref_ids, cfg,
                        debug_topk: bool, topk: int) -> dict:
    """Per-step next-token parity under HF's reference prefix.

    Builds the teacher sequence ``prompt + HF_reference_tokens`` and compares,
    at each generated position, the next-token logits of HF / internal-plain /
    masked using the SAME prefix. One forward each (causal => position p sees
    exactly ``prompt + ref[:t]``), so a single near-tie token cannot cascade as
    in free-running greedy decode."""
    import torch as _t
    from dataclasses import replace
    device = input_ids.device
    L = int(input_ids.shape[1])
    N = len(hf_ref_ids)
    if N == 0:
        return {"teacher_forced_steps_evaluated": 0}
    ref = _t.tensor([list(hf_ref_ids)], dtype=_t.long, device=device)
    teacher = _t.cat([input_ids, ref], dim=1)            # [1, L+N]
    am = _t.ones_like(teacher)
    with _t.no_grad():
        hf_full = model(input_ids=teacher, attention_mask=am, use_cache=False,
                        return_dict=True).logits          # [1, L+N, V]
    tf_cfg = replace(cfg, seq_len=int(teacher.shape[1]), max_new_tokens=1)
    plain_full, masked_full = masked_prefill_full_logits(
        model, model_config, teacher, tf_cfg)

    hp = hm = pm = 0
    ep, em = [], []
    rows = []
    for t in range(N):
        pos = L - 1 + t                                   # predicts ref[t]
        hl, pl, ml = hf_full[0, pos], plain_full[0, pos], masked_full[0, pos]
        h1, p1, m1 = int(hl.argmax()), int(pl.argmax()), int(ml.argmax())
        hp += (h1 == p1); hm += (h1 == m1); pm += (p1 == m1)
        ep.append(float((pl.float() - hl.float()).abs().max().item()))
        em.append(float((ml.float() - hl.float()).abs().max().item()))
        if debug_topk:
            def _tk(x):
                v, i = x.float().topk(min(topk, x.numel()))
                margin = float((v[0] - v[1]).item()) if v.numel() > 1 else 0.0
                return i.tolist(), round(margin, 5)
            hti, hmar = _tk(hl); pti, pmar = _tk(pl); mti, mmar = _tk(ml)
            rows.append({"step": t, "hf_top1": h1, "plain_top1": p1,
                         "masked_top1": m1, "hf_topk": hti, "plain_topk": pti,
                         "masked_topk": mti, "hf_top1_top2_margin": hmar,
                         "plain_top1_top2_margin": pmar,
                         "masked_top1_top2_margin": mmar,
                         "hf_eq_masked": h1 == m1})
    res = {
        "teacher_forced_steps_evaluated": N,
        "teacher_forced_top1_match_rate_hf_plain": round(hp / N, 6),
        "teacher_forced_top1_match_rate_hf_masked": round(hm / N, 6),
        "teacher_forced_top1_match_rate_plain_masked": round(pm / N, 6),
        "teacher_forced_avg_logit_error_hf_plain": round(sum(ep) / N, 6),
        "teacher_forced_max_logit_error_hf_plain": round(max(ep), 6),
        "teacher_forced_avg_logit_error_hf_masked": round(sum(em) / N, 6),
        "teacher_forced_max_logit_error_hf_masked": round(max(em), 6),
    }
    if debug_topk:
        res["teacher_forced_topk_debug"] = rows
    return res


def _hf_generate(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    """Official HF greedy generate on the SAME input_ids/attention_mask."""
    eos = getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=eos)
    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_ids, text


def _safe_decode(tokenizer, ids):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        args.dtype, torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, dtype=dtype,
        trust_remote_code=True, device_map=None)
    model.eval()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model.to(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                        trust_remote_code=True)
    return model, model.config, tok


def _build_dry_run(args):
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        HFCausalLMSkeletonConfig, make_random_tiny_hf_causal_lm)
    skel = HFCausalLMSkeletonConfig(
        model_family="qwen2", max_layers=args.num_layers, max_vocab_size=256,
        dtype=torch.float32, device=args.device, seed=args.seed)
    model, mc = make_random_tiny_hf_causal_lm(skel)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model.to(args.device)
    return model, mc, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="MAX prompt length (not a forced length)")
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--layerwise-folding", type=_str2bool, nargs="?",
                    const=True, default=True)
    ap.add_argument("--folded-weight-device", default="cuda",
                    choices=["cpu", "cuda"])
    ap.add_argument("--mlp-down-chunk-size", type=int, default=1024)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16"])
    ap.add_argument("--folding-dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-chat-template", type=_str2bool, nargs="?",
                    const=True, default=True)
    ap.add_argument("--compare-hf-generate", type=_str2bool, nargs="?",
                    const=True, default=True)
    ap.add_argument("--teacher-force-hf-prefix", type=_str2bool, nargs="?",
                    const=True, default=False,
                    help="per-step next-token parity under HF's reference "
                         "prefix (avoids free-running cascade after a near-tie)")
    ap.add_argument("--debug-topk-margins", type=_str2bool, nargs="?",
                    const=True, default=False)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_full_layer_memory_optimized.json")
    ap.add_argument("--output-md",
                    default="outputs/qwen7b_full_layer_memory_optimized.md")
    ap.add_argument("--output-csv",
                    default="outputs/qwen7b_full_layer_memory_optimized.csv")
    args = ap.parse_args()

    if args.dry_run:
        model, model_config, tok = _build_dry_run(args)
    else:
        if not args.model_path:
            ap.error("--model-path is required unless --dry-run is set")
        model, model_config, tok = _load_model(args)

    vocab = int(getattr(model_config, "vocab_size", 256))

    # ---- inputs ----------------------------------------------------------
    if args.dry_run or tok is None:
        g = torch.Generator().manual_seed(args.seed)
        input_ids = torch.randint(0, vocab, (args.batch_size, args.seq_len),
                                  generator=g).to(args.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_meta = {
            "raw_prompt": None, "chat_template_input": None,
            "requested_seq_len": args.seq_len,
            "real_prompt_len": args.seq_len,
            "effective_prompt_len": args.seq_len,
            "decode_start_index": args.seq_len, "truncated": False,
            "used_chat_template": False,
        }
        effective_len = args.seq_len
    else:
        raw = load_raw_prompts(args.prompt_file, args.prompt)[:args.batch_size]
        input_ids, attention_mask, prompt_meta = build_chat_inputs(
            tok, raw, args.seq_len, args.use_chat_template, args.device)
        effective_len = prompt_meta["effective_prompt_len"]

    # ---- masked + plaintext-reference run (decode starts at real length) --
    cfg = MemoryOptimizedConfig(
        num_layers=args.num_layers, batch_size=input_ids.shape[0],
        seq_len=effective_len,                       # effective, NOT padded
        max_new_tokens=args.max_new_tokens, device=args.device,
        dtype=args.dtype, folding_dtype=args.folding_dtype,
        folded_weight_device=args.folded_weight_device,
        layerwise_folding=args.layerwise_folding,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=args.seed)
    report = run_memory_optimized_masked(model, model_config, input_ids, cfg)

    report["dry_run"] = bool(args.dry_run)
    report["tee_used"] = False
    report.update({f"prompt_{k}" if k in ("status",) else k: v
                   for k, v in prompt_meta.items()})
    report["input_ids"] = input_ids.tolist()
    report["attention_mask"] = attention_mask.tolist()
    report["input_ids_shape"] = list(input_ids.shape)

    plain_ids = (report.get("generated_plain_tokens") or [[]])[0]
    masked_ids = (report.get("generated_masked_tokens") or [[]])[0]
    report["plain_generated_token_ids"] = plain_ids
    report["masked_generated_token_ids"] = masked_ids
    report["plain_generated_text"] = _safe_decode(tok, plain_ids)
    report["masked_generated_text"] = _safe_decode(tok, masked_ids)

    # ---- official HF baseline on the EXACT same inputs -------------------
    run_status = report.get("status")
    if (not args.dry_run and args.compare_hf_generate and tok is not None
            and run_status == "ok"):
        hf_ids, hf_text = _hf_generate(model, tok, input_ids, attention_mask,
                                       args.max_new_tokens)
        report["hf_generated_token_ids"] = hf_ids
        report["hf_generated_text"] = hf_text
        cmp = compare_generations(hf_ids, plain_ids, masked_ids)
        report.update(cmp)                           # sets metrics + status
        report["execution_status"] = run_status      # preserve exec status
        # free-running (cascade-prone) metrics, kept separate from teacher-forced
        report["free_running_sequence_exact_match"] = \
            cmp["sequence_exact_match_hf_masked"]
        report["free_running_token_match_rate"] = \
            cmp["hf_vs_masked_token_match_rate"]

        # ---- teacher-forced HF-prefix parity (per-step, no cascade) ------
        if args.teacher_force_hf_prefix or args.debug_topk_margins:
            tf = teacher_forced_eval(model, model_config, input_ids, hf_ids,
                                     cfg, args.debug_topk_margins, args.topk)
            report.update(tf)
    else:
        report["execution_status"] = run_status

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    _write_md(Path(args.output_md), report)
    _write_csv(Path(args.output_csv), report)

    print(f"status={report['status']} "
          f"executed_layers={report.get('executed_layers')}/"
          f"{report['total_layers']} "
          f"effective_prompt_len={report.get('effective_prompt_len')} "
          f"decode_start_index={report.get('decode_start_index')}")
    if "hf_vs_masked_token_match_rate" in report:
        print(f"hf_vs_plain={report['hf_vs_plain_token_match_rate']} "
              f"hf_vs_masked={report['hf_vs_masked_token_match_rate']} "
              f"plain_vs_masked={report['plain_vs_masked_token_match_rate']}")
    print(f"Wrote: {out_json}\nWrote: {args.output_md}\nWrote: {args.output_csv}")
    return 0 if report["status"] == "ok" else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Qwen full-layer masked execution (memory-optimized, HF-aligned)", ""]
    L.append(f"- model_type: **{r.get('model_type')}** | dry_run: "
             f"**{r.get('dry_run')}** | TEE used: **{r.get('tee_used')}**")
    L.append(f"- executed_layers: **{r.get('executed_layers')}** / "
             f"total_layers: **{r.get('total_layers')}** | exec status: "
             f"**{r.get('execution_status', r.get('status'))}**")
    L.append(f"- requested_seq_len: **{r.get('requested_seq_len')}** | "
             f"real_prompt_len: **{r.get('real_prompt_len')}** | "
             f"effective_prompt_len: **{r.get('effective_prompt_len')}** | "
             f"decode_start_index: **{r.get('decode_start_index')}** | "
             f"truncated: **{r.get('truncated')}**")
    if "hf_vs_masked_token_match_rate" in r:
        L.append(f"- **comparison status: {r['status']}**")
        L.append(f"- hf_vs_plain: **{r['hf_vs_plain_token_match_rate']}** | "
                 f"hf_vs_masked: **{r['hf_vs_masked_token_match_rate']}** | "
                 f"plain_vs_masked: **{r['plain_vs_masked_token_match_rate']}**")
        L.append(f"- exact: hf==plain **{r['sequence_exact_match_hf_plain']}** | "
                 f"hf==masked **{r['sequence_exact_match_hf_masked']}** | "
                 f"plain==masked **{r['sequence_exact_match_plain_masked']}**")
        L.append(f"- hf_text: `{(r.get('hf_generated_text') or '')[:160]!r}`")
        L.append(f"- masked_text: `{(r.get('masked_generated_text') or '')[:160]!r}`")
    if "teacher_forced_steps_evaluated" in r:
        L.append("")
        L.append("## Teacher-forced HF-prefix parity (per-step, no cascade)")
        L.append(f"- steps_evaluated: **{r.get('teacher_forced_steps_evaluated')}**")
        L.append(f"- top1 match: hf_plain **"
                 f"{r.get('teacher_forced_top1_match_rate_hf_plain')}** | "
                 f"hf_masked **{r.get('teacher_forced_top1_match_rate_hf_masked')}**"
                 f" | plain_masked **"
                 f"{r.get('teacher_forced_top1_match_rate_plain_masked')}**")
        L.append(f"- logit error (hf vs masked): avg **"
                 f"{r.get('teacher_forced_avg_logit_error_hf_masked')}** max **"
                 f"{r.get('teacher_forced_max_logit_error_hf_masked')}**")
        L.append(f"- free-running: seq_exact **"
                 f"{r.get('free_running_sequence_exact_match')}** token_rate **"
                 f"{r.get('free_running_token_match_rate')}**")
    if r.get("status") == "stopped_oom":
        L.append(f"- OOM layer index: **{r.get('oom_layer_index')}**")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _write_csv(path: Path, r: dict) -> None:
    row = {
        "model_type": r.get("model_type"),
        "executed_layers": r.get("executed_layers"),
        "total_layers": r.get("total_layers"),
        "requested_seq_len": r.get("requested_seq_len"),
        "real_prompt_len": r.get("real_prompt_len"),
        "effective_prompt_len": r.get("effective_prompt_len"),
        "decode_start_index": r.get("decode_start_index"),
        "truncated": r.get("truncated"),
        "status": r.get("status"),
        "hf_vs_plain_token_match_rate": r.get("hf_vs_plain_token_match_rate"),
        "hf_vs_masked_token_match_rate": r.get("hf_vs_masked_token_match_rate"),
        "plain_vs_masked_token_match_rate":
            r.get("plain_vs_masked_token_match_rate"),
        "sequence_exact_match_hf_masked": r.get("sequence_exact_match_hf_masked"),
        "free_running_token_match_rate": r.get("free_running_token_match_rate"),
        "teacher_forced_steps_evaluated":
            r.get("teacher_forced_steps_evaluated"),
        "teacher_forced_top1_match_rate_hf_masked":
            r.get("teacher_forced_top1_match_rate_hf_masked"),
        "teacher_forced_top1_match_rate_plain_masked":
            r.get("teacher_forced_top1_match_rate_plain_masked"),
        "teacher_forced_max_logit_error_hf_masked":
            r.get("teacher_forced_max_logit_error_hf_masked"),
        "peak_memory_max_allocated_mb": (r.get("peak_memory") or {})
            .get("max_allocated_mb"),
    }
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
