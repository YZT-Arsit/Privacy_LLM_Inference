"""E2: no-LoRA Qwen2.5-7B token-scaling experiment.

Fixed model / prompt / seq_len=128 / bs=1; sweep max_new_tokens over
{1, 8, 16, 32, 64}. Reports correctness (teacher-forced top-1 agreement,
plain-vs-masked token match), latency, peak GPU memory, trusted/gpu bytes,
boundary calls, tee_used_on_gpu=false per token budget.

Real Qwen2.5-7B needs CUDA + a local checkpoint; ``--dry-run`` validates the
runner with a tiny random Qwen2 on CPU (never a paper result).

Example (H800):
    python scripts/run_qwen7b_e2_token_scaling.py \\
        --model-path /models/Qwen2.5-7B-Instruct --use-chat-template true \\
        --seq-len 128 --token-grid 1,8,16,32,64 --num-layers 28 \\
        --dtype bfloat16 --device cuda --folded-weight-device cuda \\
        --mlp-down-chunk-size 512 \\
        --output-json outputs/e2_token_scaling_qwen.json \\
        --output-md outputs/e2_token_scaling_qwen.md
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
    run_e2_token_scaling,
)
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
    return model, model.config, enc["input_ids"][:, :args.seq_len].to(args.device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--token-grid", default="1,8,16,32,64")
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--modes", default="greedy,teacher_forced")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json", default="outputs/e2_token_scaling_qwen.json")
    ap.add_argument("--output-md", default="outputs/e2_token_scaling_qwen.md")
    args = ap.parse_args()
    grid = tuple(int(x) for x in args.token_grid.split(",") if x.strip())
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    if args.dry_run or not args.model_path:
        if not args.dry_run:
            print("NOTE: no --model-path; running --dry-run tiny model (NOT a "
                  "paper result).")
        model, mc = _tiny_model()
        input_ids = torch.randint(0, 256, (1, min(args.seq_len, 8)))
        device, dtype = "cpu", "float32"
        num_layers = mc.num_hidden_layers
    else:
        model, mc, input_ids = _load_real(args)
        device, dtype = args.device, args.dtype
        num_layers = min(args.num_layers, mc.num_hidden_layers)

    cfg = MemoryOptimizedConfig(
        num_layers=num_layers, batch_size=1, seq_len=int(input_ids.shape[1]),
        max_new_tokens=grid[0], device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size)

    out = run_e2_token_scaling(model, mc, input_ids, cfg, token_grid=grid,
                               modes=modes, topk=args.topk)
    out["dry_run"] = bool(args.dry_run or not args.model_path)
    out["seq_len"] = int(input_ids.shape[1])

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), out)

    print(f"=== E2 token scaling (dry_run={out['dry_run']}) grid={grid} ===")
    for row in out["rows"]:
        print(f"  N={row['max_new_tokens']:>3} tf_top1_plain_masked="
              f"{row.get('tf_top1_plain_masked')} tf_top1_hf_masked="
              f"{row.get('tf_top1_hf_masked')} "
              f"greedy_match={row.get('greedy_plain_vs_masked_token_match_rate')} "
              f"peak_mb={row['peak_gpu_memory_mb']} gpu_bytes={row['gpu_bytes']}")
    return 0


def _write_md(path: Path, out: dict) -> None:
    L = [f"# E2 no-LoRA Qwen2.5-7B token scaling (dry_run={out.get('dry_run')})",
         "", f"seq_len={out.get('seq_len')} grid={out['token_grid']} "
         f"tee_used_on_gpu={out['tee_used_on_gpu']}", "",
         "| max_new_tokens | tf_top1_hf_masked | tf_top1_plain_masked | "
         "greedy match | tf_logits_max_abs | tf_topk_overlap | latency_s | "
         "peak_gpu_mb | trusted_bytes | gpu_bytes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        L.append(
            f"| {r['max_new_tokens']} | {r.get('tf_top1_hf_masked')} | "
            f"{r.get('tf_top1_plain_masked')} | "
            f"{r.get('greedy_plain_vs_masked_token_match_rate')} | "
            f"{r.get('tf_logits_max_abs_error')} | {r.get('tf_topk_overlap')} | "
            f"{r.get('greedy_latency_s')} | {r['peak_gpu_memory_mb']} | "
            f"{r['trusted_bytes']} | {r['gpu_bytes']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
