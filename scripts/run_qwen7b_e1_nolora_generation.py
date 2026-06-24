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

from pllo.experiments.qwen_generation_experiments import run_e1_nolora  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    return model, model.config, input_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--seq-len", type=int, default=128)
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

    if args.dry_run or not args.model_path:
        if not args.dry_run:
            print("NOTE: no --model-path; running --dry-run tiny model (NOT a "
                  "paper result).")
        model, mc, input_ids = (*_tiny_model(),
                                torch.randint(0, 256, (1, min(args.seq_len, 8))))
        device = "cpu"
        num_layers = mc.num_hidden_layers
        dtype = "float32"
    else:
        model, mc, input_ids = _load_real(args)
        device = args.device
        num_layers = min(args.num_layers, mc.num_hidden_layers)
        dtype = args.dtype

    cfg = MemoryOptimizedConfig(
        num_layers=num_layers, batch_size=1, seq_len=int(input_ids.shape[1]),
        max_new_tokens=args.max_new_tokens, device=device, dtype=dtype,
        folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size)

    report = run_e1_nolora(model, mc, input_ids, cfg, modes=modes, topk=args.topk,
                           temperature=args.temperature, top_p=args.top_p,
                           sample_seed=args.sample_seed)
    report["dry_run"] = bool(args.dry_run or not args.model_path)
    report["model_path"] = args.model_path

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print(f"=== E1 no-LoRA Qwen generation (dry_run={report['dry_run']}) ===")
    print(f"seq_len={report['seq_len']} max_new_tokens={report['max_new_tokens']} "
          f"num_layers={report['num_layers']} dtype={report['dtype']} "
          f"tee_used_on_gpu={report['tee_used_on_gpu']}")
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
         "", f"- seq_len={r['seq_len']} max_new_tokens={r['max_new_tokens']} "
         f"num_layers={r['num_layers']} dtype={r['dtype']}",
         f"- **tee_used_on_gpu={r['tee_used_on_gpu']}**  "
         f"gpu_visible_plaintext_fields={r['gpu_visible_plaintext_fields'] or '[]'}  "
         f"leaked_secret_fields={r['leaked_secret_fields'] or '[]'}",
         f"- trusted_bytes={r['trusted_bytes']:,}  gpu_bytes={r['gpu_bytes']:,}  "
         f"peak_gpu_memory_mb={r['peak_gpu_memory_mb']}",
         f"- boundary_calls=`{r['boundary_calls']}`", ""]
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
