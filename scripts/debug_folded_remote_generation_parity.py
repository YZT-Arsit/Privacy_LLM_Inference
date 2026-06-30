"""Minimal stepwise debug of folded_remote generation (repeat-token root cause).

Runs ONE prompt through the folded_remote masked prefill+decode (TDX boundary
client + H800 worker) and, optionally, the plaintext trusted greedy loop on the
SAME formatted prompt, and dumps a per-step diagnostic so the IFEval id=1005
"same token repeated to max_new_tokens" failure can be localised WITHOUT running a
full benchmark.

Per-step (prefill = step 0, then decode steps) it records:
  position, selected token id + text, recovered-logits top-10 (id/value/text),
  top1, top2, top1-top2 margin, top1 softmax probability (concentration), entropy,
  whether the selected token repeats the previous one, and the current
  same-token run length. Header records: formatted-prompt sha256, raw/chat/used
  token counts, truncation side + truncated flag, eos/pad token ids. With
  ``--with-plaintext`` it also reports per-step top1 / top5 agreement between the
  plaintext logits and the folded recovered logits.

Triage (printed in the report's ``suspect`` field):
  * prefill (step 0) top1 already wrong/degenerate  -> folded LM head / final
    RMSNorm / vocab recovery / linear pad compensation;
  * prefill OK but decode diverges                  -> KV cache / position id /
    RoPE / decode mask_token_embedding;
  * logits top1 OK but output still repeats         -> sampling/argmax / EOS stop /
    finish_reason propagation.

Privacy: the GPU/worker never receives the raw prompt or token ids; this trusted-
side debug JSON stores only the formatted-prompt SHA, output token ids/text, and
the TOP-10 recovered logits per step (never the full recovered-logits matrix, no
masks / N_inv / pad). stdlib + pllo predictors only; no downloads.

Example::

    python scripts/debug_folded_remote_generation_parity.py \\
      --model-path <MODEL> --embedding-path <EMB> \\
      --gpu-worker-url http://127.0.0.1:18082 --nonlinear-backend A_rightmul \\
      --seq-len 1024 --max-debug-steps 16 --dtype bfloat16 --use-chat-template \\
      --prompt "当然了解面试吗？请用三句话回答。" \\
      --with-plaintext --output-json <OUT>/parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _softmax_stats(values):
    """Top-1 probability (concentration) + Shannon entropy (nats) over a top-k
    logit slice (cheap proxy; full-vocab entropy not needed for triage)."""
    if not values:
        return None, None
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]
    top1_p = max(probs)
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return round(top1_p, 6), round(ent, 6)


def _decorate(topk, tok):
    out = []
    for e in topk:
        try:
            text = tok.decode([int(e["id"])])
        except Exception:                                        # noqa: BLE001
            text = None
        out.append({"id": e["id"], "value": e["value"], "text": text})
    return out


def _steps_from_logits(captured, tok, plain_steps=None):
    from pllo.benchmarks.logits_parity import compare_step, topk
    steps = []
    prev_sel = None
    run = 0
    for i, logits in enumerate(captured):
        tk = topk(logits, 10)
        sel = tk[0]["id"]
        top1 = tk[0]["value"]
        top2 = tk[1]["value"] if len(tk) > 1 else None
        margin = (round(top1 - top2, 6) if top2 is not None else None)
        top1_p, ent = _softmax_stats([e["value"] for e in tk])
        repeats = bool(prev_sel is not None and sel == prev_sel)
        run = run + 1 if repeats else 1
        prev_sel = sel
        rec = {
            "step": i, "phase": "prefill" if i == 0 else "decode",
            "selected_token_id": sel,
            "selected_token_text": (tok.decode([sel]) if tok else None),
            "top1_logit": top1, "top2_logit": top2, "top1_top2_margin": margin,
            "top1_softmax_prob": top1_p, "topk_entropy": ent,
            "repeats_previous": repeats, "same_token_run_length": run,
            "recovered_top10": _decorate(tk, tok) if tok else tk,
        }
        if plain_steps is not None and i < len(plain_steps):
            cmp = compare_step(plain_steps[i], logits, k=10)
            rec["plaintext_top1_agree"] = cmp["top1_agree"]
            rec["plaintext_top5_overlap"] = cmp["top5_overlap"]
            rec["plaintext_vs_ours_max_abs_error"] = cmp["max_abs_error"]
            rec["plaintext_top1_id"] = cmp["plain_top1"]
        steps.append(rec)
    return steps


def _suspect(steps, *, with_plaintext: bool) -> str:
    if not steps:
        return "no steps captured (worker/boundary returned nothing)"
    s0 = steps[0]
    # prefill degenerate / disagreeing
    if with_plaintext and s0.get("plaintext_top1_agree") is False:
        return ("prefill top1 disagrees with plaintext -> check folded LM head / "
                "final RMSNorm / vocab recovery / linear pad compensation")
    if (s0.get("top1_softmax_prob") or 0) > 0.99 and (
            s0.get("top1_top2_margin") or 0) > 50:
        return ("prefill logits already near-degenerate -> check folded LM head / "
                "final RMSNorm / vocab recovery / linear pad compensation")
    # decode divergence
    decode = steps[1:]
    if with_plaintext:
        for st in decode:
            if st.get("plaintext_top1_agree") is False:
                return ("prefill OK but decode step %d diverges -> check KV cache / "
                        "position id / RoPE / decode mask_token_embedding"
                        % st["step"])
    # repetition with otherwise-fine logits
    long_run = max((st["same_token_run_length"] for st in steps), default=0)
    if long_run >= 8:
        return ("repeated-token run length %d -> if logits top1 looks correct, "
                "check sampling/argmax / EOS stop / finish_reason; else check "
                "KV/position/RoPE" % long_run)
    return "no obvious degeneration in the captured window"


def _build(args):
    from pllo.benchmarks.real_predictors import build_predictor
    evidence = None
    if args.attestation_evidence_json:
        evidence = json.loads(
            Path(args.attestation_evidence_json).read_text(encoding="utf-8"))
    ours = build_predictor(
        "folded_remote", model_path=args.model_path, model_name=args.model_name,
        gpu_worker_url=args.gpu_worker_url, embedding_path=args.embedding_path,
        attestation_evidence=evidence, expected_mr_td=args.expected_mr_td,
        seq_len=args.seq_len, max_new_tokens=args.max_debug_steps,
        dtype=args.dtype, device=args.device, audit=False,
        nonlinear_backend=args.nonlinear_backend,
        use_chat_template=bool(args.use_chat_template),
        align_generation_config=bool(args.align_generation_config),
        repetition_penalty=args.repetition_penalty)
    plain = None
    if args.with_plaintext:
        plain = build_predictor(
            "plaintext_local", model_path=args.model_path,
            model_name=args.model_name, seq_len=args.seq_len,
            max_new_tokens=args.max_debug_steps, dtype=args.dtype,
            device=args.device, use_chat_template=bool(args.use_chat_template),
            align_generation_config=bool(args.align_generation_config),
            repetition_penalty=args.repetition_penalty)
    return ours, plain


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--embedding-path", required=True)
    ap.add_argument("--gpu-worker-url", required=True)
    ap.add_argument("--nonlinear-backend", default="A_rightmul")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-debug-steps", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-chat-template", action="store_true", default=False)
    ap.add_argument("--align-generation-config", action="store_true",
                    default=False)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--attestation-evidence-json", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--with-plaintext", action="store_true", default=False,
                    help="also run the plaintext greedy loop on the SAME host and "
                    "report per-step top1/top5 agreement (loads full weights; do "
                    "NOT use inside the TDX guest)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    ours, plain = _build(args)
    tok = getattr(ours, "_tok", None)

    info = ours.prompt_info(args.prompt)
    eos_ids = sorted(getattr(ours, "_eos_ids", set()) or [])
    pad_id = getattr(ours, "_pad_token_id", None)

    # folded_remote capture
    ours.enable_logits_capture(max_steps=args.max_debug_steps)
    gen = ours.generate(args.prompt)
    captured = ours.captured_logits()

    # optional plaintext per-step logits on the same formatted prompt
    plain_steps = None
    if plain is not None:
        _, plain_steps = plain.greedy_with_logits(
            args.prompt, max_steps=args.max_debug_steps)

    steps = _steps_from_logits(captured, tok, plain_steps=plain_steps)
    report = {
        "stage": "folded_remote_generation_parity_debug",
        "nonlinear_backend": args.nonlinear_backend,
        "seq_len": int(args.seq_len),
        "max_debug_steps": int(args.max_debug_steps),
        "formatted_prompt_sha256": info.get("formatted_prompt_sha256"),
        "raw_prompt_token_count": info.get("raw_prompt_token_count"),
        "chat_prompt_token_count": info.get("chat_prompt_token_count"),
        "formatted_prompt_token_count": info.get("formatted_prompt_token_count"),
        "prompt_token_count": info.get("prompt_token_count"),
        "truncation_side": info.get("truncation_side"),
        "truncated": info.get("truncated"),
        "eos_token_id": eos_ids,
        "pad_token_id": pad_id,
        "finish_reason": gen.get("finish_reason"),
        "num_generated_tokens": len(gen.get("token_ids") or []),
        "with_plaintext": bool(args.with_plaintext),
        "num_steps_captured": len(steps),
        "suspect": _suspect(steps, with_plaintext=bool(args.with_plaintext)),
        "steps": steps,
    }
    if plain_steps is not None:
        agree = [s.get("plaintext_top1_agree") for s in steps
                 if "plaintext_top1_agree" in s]
        report["plaintext_top1_agreement_rate"] = (
            round(sum(1 for a in agree if a) / len(agree), 4) if agree else None)

    for pr in (ours, plain):
        if pr is not None and hasattr(pr, "close"):
            try:
                pr.close()
            except Exception:                                    # noqa: BLE001
                pass

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2,
                                                 ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "finish_reason", "num_generated_tokens", "truncated", "truncation_side",
        "num_steps_captured", "suspect", "plaintext_top1_agreement_rate")
        if k in report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
