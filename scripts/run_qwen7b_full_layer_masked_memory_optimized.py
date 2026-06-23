"""Qwen2.5-7B full-layer masked execution -- memory-optimized (Stage 8.4).

Untrusted-GPU masked decoder pipeline. **No TEE is imported or used** here and
no TEE execution is claimed; this script only runs the untrusted masked
decoder/attention/MLP/KV/LM-head path with layerwise folded-weight streaming +
chunked folded down-projection so all 28 decoder layers fit in memory.

ModelScope cache only (local files); never Hugging Face remote download.
Correctness is compared against the extracted-weight plaintext reference
(top1_match_rate / max_abs_error / greedy_token_match).
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
    run_memory_optimized_masked,
)


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def _load_prompts(path: str | None, prompt: str | None) -> list[dict]:
    if path:
        rows = []
        for i, line in enumerate(Path(path).read_text(encoding="utf-8")
                                 .splitlines()):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rows.append({"id": str(rec.get("id", f"p{i}")),
                             "prompt": str(rec["prompt"])})
        return rows
    if prompt:
        return [{"id": "literal", "prompt": prompt}]
    return [{"id": "default", "prompt": "Hello, please answer briefly:"}]


def _build_input_ids(args, tokenizer, vocab_size: int) -> torch.Tensor:
    """Tokenize prompts -> [B, seq_len] (truncate/pad). Dry-run uses random."""
    if tokenizer is None:  # dry-run synthetic tokens
        g = torch.Generator().manual_seed(args.seed)
        return torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                             generator=g)
    prompts = _load_prompts(args.prompt_file, args.prompt)[:args.batch_size]
    while len(prompts) < args.batch_size:
        prompts.append(prompts[-1])
    pad_id = (getattr(tokenizer, "pad_token_id", None)
              or getattr(tokenizer, "eos_token_id", None) or 0)
    rows = []
    for rec in prompts:
        ids = list(tokenizer(rec["prompt"])["input_ids"])[:args.seq_len]
        if len(ids) < args.seq_len:
            ids += [int(pad_id)] * (args.seq_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _load_model(args):
    """Load a real checkpoint from a LOCAL path (ModelScope cache); no remote."""
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
    """Tiny random Qwen2 with the requested layer count -- NO Qwen checkpoint."""
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        HFCausalLMSkeletonConfig, make_random_tiny_hf_causal_lm)
    skel = HFCausalLMSkeletonConfig(
        model_family="qwen2", max_layers=args.num_layers, max_vocab_size=256,
        dtype=torch.float32, device=args.device, seed=args.seed)
    model, mc = make_random_tiny_hf_causal_lm(skel)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model.to(args.device)
    return model, mc, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None,
                    help="local checkpoint dir (ModelScope cache); not used "
                         "with --dry-run")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=64)
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
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny synthetic model; validate 28-layer control flow "
                         "without loading any Qwen checkpoint")
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
    input_ids = _build_input_ids(args, tok, vocab)

    cfg = MemoryOptimizedConfig(
        num_layers=args.num_layers, batch_size=args.batch_size,
        seq_len=args.seq_len, max_new_tokens=args.max_new_tokens,
        device=args.device, dtype=args.dtype, folding_dtype=args.folding_dtype,
        folded_weight_device=args.folded_weight_device,
        layerwise_folding=args.layerwise_folding,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=args.seed)

    report = run_memory_optimized_masked(model, model_config, input_ids, cfg)
    report["dry_run"] = bool(args.dry_run)
    report["input_ids_shape"] = list(input_ids.shape)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    _write_md(Path(args.output_md), report)
    _write_csv(Path(args.output_csv), report)

    print(f"status={report['status']} "
          f"executed_layers={report.get('executed_layers')} "
          f"total_layers={report['total_layers']}")
    if report["status"] == "ok":
        print(f"top1_match_rate={report['top1_match_rate']} "
              f"max_abs_error={report['max_abs_error']:.3e} "
              f"greedy_token_match={report['greedy_token_match']}")
        print(f"peak_memory={report.get('peak_memory')}")
    else:
        print(f"OOM at layer {report.get('oom_layer_index')}: "
              f"{report.get('reason')}")
    print(f"Wrote: {out_json}\nWrote: {args.output_md}\nWrote: {args.output_csv}")
    return 0 if report["status"] == "ok" else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Qwen full-layer masked execution (memory-optimized)", ""]
    L.append(f"- model_type: **{r.get('model_type')}** | dry_run: "
             f"**{r.get('dry_run')}** | TEE used: **{r.get('tee_used')}**")
    L.append(f"- executed_layers: **{r.get('executed_layers')}** / "
             f"total_layers: **{r.get('total_layers')}** | status: "
             f"**{r['status']}**")
    cfg = r.get("config", {})
    L.append(f"- batch={cfg.get('batch_size')} seq={cfg.get('seq_len')} "
             f"max_new_tokens={cfg.get('max_new_tokens')} "
             f"folded_weight_device={cfg.get('folded_weight_device')} "
             f"mlp_down_chunk_size={cfg.get('mlp_down_chunk_size')} "
             f"dtype={cfg.get('dtype')}")
    if r["status"] == "ok":
        L.append(f"- top1_match_rate: **{r['top1_match_rate']}** | "
                 f"max_abs_error: **{r['max_abs_error']:.3e}** | "
                 f"greedy_token_match: **{r['greedy_token_match']}**")
        L.append(f"- peak_memory: `{r.get('peak_memory')}`")
    else:
        L.append(f"- OOM layer index: **{r.get('oom_layer_index')}** | reason: "
                 f"`{r.get('reason')}`")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _write_csv(path: Path, r: dict) -> None:
    rows = []
    for m in r.get("per_layer_memory", []):
        before = m.get("before") or {}
        after = m.get("after") or {}
        rows.append({
            "layer": m["layer"], "phase": m["phase"],
            "before_allocated_mb": before.get("allocated_mb"),
            "after_allocated_mb": after.get("allocated_mb"),
            "after_max_allocated_mb": after.get("max_allocated_mb"),
            "after_reserved_mb": after.get("reserved_mb"),
        })
    cols = ["layer", "phase", "before_allocated_mb", "after_allocated_mb",
            "after_max_allocated_mb", "after_reserved_mb"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
