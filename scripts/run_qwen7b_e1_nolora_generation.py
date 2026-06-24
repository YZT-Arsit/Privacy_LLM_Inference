"""E1: no-LoRA Qwen2.5-7B masked-generation main experiment (fixed paper config).

Fixed configuration: bs=1, seq_len=128 (max prompt), max_new_tokens=64,
num_layers=28/28, dtype=bf16 (H800). Decoding modes: greedy, teacher-forced,
sampling (temp=0.7, top_p=0.9, fixed seed). Reports correctness (teacher-forced
top-1 hf/plain/masked, top-k overlap, logits errors, plain-vs-masked token match,
sequence exact match where meaningful), latency, peak GPU memory, and the
protocol fields (trusted/gpu bytes, boundary calls, tee_used_on_gpu=false,
gpu_visible_plaintext_fields=[], leaked_secret_fields=[]).

Real Qwen2.5-7B needs CUDA + a local ModelScope checkpoint; ``--dry-run`` runs a
tiny random Qwen2 on CPU to validate the runner (never a paper result).

Example (H800):
    python scripts/run_qwen7b_e1_nolora_generation.py \\
        --model-path /models/Qwen2.5-7B-Instruct --use-chat-template true \\
        --seq-len 128 --max-new-tokens 64 --num-layers 28 --dtype bfloat16 \\
        --device cuda --folded-weight-device cuda --mlp-down-chunk-size 512 \\
        --modes greedy,teacher_forced,sampling \\
        --output-json outputs/e1_nolora_qwen.json \\
        --output-md outputs/e1_nolora_qwen.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pllo.experiments.qwen_generation_experiments import (  # noqa: E402
    build_context_fields,
    run_e1_nolora,
)
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _pad_id(tok, mc) -> int:
    """Resolve a padding token id (tokenizer pad -> eos -> 0)."""
    if tok is not None and getattr(tok, "pad_token_id", None) is not None:
        return int(tok.pad_token_id)
    eos = getattr(mc, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return int(eos) if eos is not None else 0


def _build_context_inputs(args, real_ids, tok, mc, prompt_text, device, dtype,
                          dry_run):
    """Build the real attention mask, optional left-padded inputs, and the
    unambiguous context dict for E1/E2. The masked/plain path always consumes the
    REAL (unpadded) tokens; padding lives only on the HF reference paths."""
    pad_to = _bool(args.pad_to_seq_len)
    eff_len = int(real_ids.shape[1])
    attn_mask = torch.ones((real_ids.shape[0], eff_len), dtype=torch.long)
    padded_ids = padded_mask = None
    padded_seq_len = None
    if pad_to:
        target = int(args.seq_len)
        if eff_len < target:
            pad_n = target - eff_len
            pad_block = torch.full((real_ids.shape[0], pad_n), _pad_id(tok, mc),
                                   dtype=real_ids.dtype, device=real_ids.device)
            padded_ids = torch.cat([pad_block, real_ids], dim=1)   # left pad
            padded_mask = torch.cat(
                [torch.zeros((real_ids.shape[0], pad_n), dtype=torch.long),
                 attn_mask], dim=1)
            padded_seq_len = target
        else:                                  # prompt already >= budget
            padded_ids, padded_mask, padded_seq_len = real_ids, attn_mask, eff_len
    context = build_context_fields(
        seq_len_requested=int(args.seq_len), effective_prompt_len=eff_len,
        pad_to_seq_len=pad_to, padded_seq_len=padded_seq_len,
        decode_start_index=eff_len, attention_mask_used=True, dtype=dtype,
        model_name=args.model_name, model_path=args.model_path,
        prompt_text=prompt_text, dry_run=dry_run)
    return attn_mask, padded_ids, padded_mask, context


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=4, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _load_real(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(args.dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, device_map=args.device,
        trust_remote_code=True, local_files_only=True).eval()
    prompt = args.prompt
    if args.prompt_file:
        prompt = json.loads(Path(args.prompt_file).read_text(
            encoding="utf-8").splitlines()[0])["prompt"]
    if _bool(args.use_chat_template):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"][:, :args.seq_len].to(args.device)
    return model, model.config, input_ids, tok, prompt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct",
                    help="public model name stamped into the report")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="seq_len_requested: max prompt budget (natural mode) or "
                         "fixed context length (with --pad-to-seq-len true)")
    ap.add_argument("--pad-to-seq-len", default="false",
                    help="false: seq_len is a max prompt budget (natural mode); "
                         "true: left-pad the prompt to seq_len (fixed padded "
                         "context) and validate padding-invariance")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--modes", default="greedy,teacher_forced,sampling")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json", default="outputs/e1_nolora_qwen.json")
    ap.add_argument("--output-md", default="outputs/e1_nolora_qwen.md")
    args = ap.parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    dry_run = bool(args.dry_run or not args.model_path)
    if dry_run:
        if not args.dry_run:
            print("NOTE: no --model-path; running --dry-run tiny model (NOT a "
                  "paper result).")
        model, mc = _tiny_model()
        input_ids = torch.randint(0, 256, (1, min(args.seq_len, 8)))
        tok, prompt_text = None, args.prompt
        device, dtype = "cpu", "float32"
        num_layers = mc.num_hidden_layers
    else:
        model, mc, input_ids, tok, prompt_text = _load_real(args)
        device, dtype = args.device, args.dtype
        num_layers = min(args.num_layers, mc.num_hidden_layers)

    attn_mask, padded_ids, padded_mask, context = _build_context_inputs(
        args, input_ids, tok, mc, prompt_text, device, dtype, dry_run)

    cfg = MemoryOptimizedConfig(
        num_layers=num_layers, batch_size=1, seq_len=int(input_ids.shape[1]),
        max_new_tokens=args.max_new_tokens, device=device, dtype=dtype,
        folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size)

    report = run_e1_nolora(model, mc, input_ids, cfg, modes=modes, topk=args.topk,
                           temperature=args.temperature, top_p=args.top_p,
                           sample_seed=args.sample_seed, attention_mask=attn_mask,
                           context=context, padded_input_ids=padded_ids,
                           padded_attention_mask=padded_mask)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print(f"=== E1 no-LoRA Qwen generation (dry_run={report['dry_run']}) ===")
    print(f"model_name={report['model_name']} context_mode={report['context_mode']} "
          f"prompt_sha256_16={report['prompt_sha256_16']}")
    print(f"seq_len_requested={report['seq_len_requested']} "
          f"effective_prompt_len={report['effective_prompt_len']} "
          f"padded_seq_len={report['padded_seq_len']} "
          f"decode_start_index={report['decode_start_index']} "
          f"attention_mask_used={report['attention_mask_used']}")
    print(f"max_new_tokens={report['max_new_tokens']} "
          f"num_layers={report['num_layers']} dtype={report['dtype']} "
          f"tee_used_on_gpu={report['tee_used_on_gpu']}")
    if "padding_invariance_hf_token_match_rate" in report:
        print(f"padding_invariance_hf_token_match_rate="
              f"{report['padding_invariance_hf_token_match_rate']} "
              f"(ok={report.get('padding_invariance_ok')})")
    if "teacher_forced" in report["modes"]:
        tf = report["modes"]["teacher_forced"]
        print(f"teacher_forced: hf_plain={tf['teacher_forced_top1_match_rate_hf_plain']:.4f} "
              f"hf_masked={tf['teacher_forced_top1_match_rate_hf_masked']:.4f} "
              f"plain_masked={tf['teacher_forced_top1_match_rate_plain_masked']:.4f} "
              f"topk_overlap={tf['topk_overlap']:.4f} "
              f"logits_max_abs_err={tf['logits_max_abs_error']:.3e}")
    if "greedy" in report["modes"]:
        g = report["modes"]["greedy"]
        print(f"greedy: plain_vs_masked_token_match="
              f"{g['plain_vs_masked_token_match_rate']} "
              f"seq_exact_hf_masked={g['sequence_exact_match_hf_masked']} "
              f"latency_s={g['latency_s']:.3f}")
    print(f"peak_gpu_memory_mb={report['peak_gpu_memory_mb']} "
          f"trusted_bytes={report['trusted_bytes']} gpu_bytes={report['gpu_bytes']}")
    return 0


def _write_md(path: Path, r: dict) -> None:
    L = [f"# E1 no-LoRA Qwen2.5-7B masked generation (dry_run={r.get('dry_run')})",
         "",
         f"- model_name=`{r.get('model_name')}`  model_path=`{r.get('model_path')}`",
         f"- context_mode=**{r.get('context_mode')}**  "
         f"prompt_sha256_16=`{r.get('prompt_sha256_16')}`",
         f"- seq_len_requested={r.get('seq_len_requested')}  "
         f"effective_prompt_len={r.get('effective_prompt_len')}  "
         f"padded_seq_len={r.get('padded_seq_len')}  "
         f"decode_start_index={r.get('decode_start_index')}  "
         f"attention_mask_used={r.get('attention_mask_used')}",
         f"- max_new_tokens={r['max_new_tokens']} "
         f"num_layers={r['num_layers']} dtype={r['dtype']}"]
    if "padding_invariance_hf_token_match_rate" in r:
        L.append(f"- padding_invariance_hf_token_match_rate="
                 f"{r['padding_invariance_hf_token_match_rate']} "
                 f"(ok={r.get('padding_invariance_ok')})")
    L += [f"- **tee_used_on_gpu={r['tee_used_on_gpu']}**  "
         f"gpu_visible_plaintext_fields={r['gpu_visible_plaintext_fields'] or '[]'}  "
         f"leaked_secret_fields={r['leaked_secret_fields'] or '[]'}",
         f"- trusted_bytes={r['trusted_bytes']:,}  gpu_bytes={r['gpu_bytes']:,}  "
         f"peak_gpu_memory_mb={r['peak_gpu_memory_mb']}",
         f"- boundary_calls=`{r['boundary_calls']}`", ""]
    # flattened paper-critical metrics (same field names as the JSON top level)
    L += ["## Paper-critical metrics (flattened top-level)", "",
          "| field | value |", "|---|---|"]
    for k in ("teacher_forced_top1_match_rate_hf_plain",
              "teacher_forced_top1_match_rate_hf_masked",
              "teacher_forced_top1_match_rate_plain_masked",
              "plain_vs_masked_token_match_rate", "topk_overlap",
              "logits_max_abs_error", "logits_mean_abs_error",
              "logits_relative_l2_error", "latency_s", "greedy_latency_s",
              "teacher_forced_latency_s", "peak_gpu_memory_mb", "trusted_bytes",
              "gpu_bytes", "tee_used_on_gpu"):
        L.append(f"| {k} | {r.get(k)} |")
    L.append("")

    tf = r["modes"].get("teacher_forced")
    if tf:
        L += ["## Teacher-forced (main long-horizon correctness)", "",
              f"- steps={tf['teacher_forced_steps_evaluated']}",
              f"- top1 hf_plain={tf['teacher_forced_top1_match_rate_hf_plain']:.4f}",
              f"- top1 hf_masked={tf['teacher_forced_top1_match_rate_hf_masked']:.4f}",
              f"- top1 plain_masked={tf['teacher_forced_top1_match_rate_plain_masked']:.4f}",
              f"- top{tf['topk']}_overlap={tf['topk_overlap']:.4f}",
              f"- logits max/mean/rel_l2 = {tf['logits_max_abs_error']:.3e} / "
              f"{tf['logits_mean_abs_error']:.3e} / {tf['logits_relative_l2_error']:.3e}",
              ""]
    g = r["modes"].get("greedy")
    if g:
        L += ["## Greedy (deterministic)", "",
              f"- plain_vs_masked_token_match_rate="
              f"{g['plain_vs_masked_token_match_rate']}",
              f"- sequence_exact_match_hf_masked={g['sequence_exact_match_hf_masked']}",
              f"- logits_max_abs_error={g['logits_max_abs_error']}",
              f"- latency_s={g['latency_s']:.3f}", ""]
    s = r["modes"].get("sampling")
    if s:
        L += ["## Sampling (temp/top_p, fixed seed)", "",
              f"- completion_tokens={s['generation_completion_tokens']} "
              f"temperature={s['temperature']} top_p={s['top_p']} seed={s['seed']}",
              f"- topk_overlap={s['topk_overlap']:.4f}",
              f"- logits_max_abs_error={s['logits_max_abs_error']:.3e}",
              f"- latency_s={s['latency_s']:.3f}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
