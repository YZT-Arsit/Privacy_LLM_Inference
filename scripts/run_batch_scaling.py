"""Stage 8.2 -- batch-size scaling on real prompts (Group E).

Runs the real-checkpoint probe at several batch sizes (via --max-prompts on the
prompt file) and aggregates compact scalar metrics into ONE JSON. ModelScope
cache only; no tensor dumps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.modelscope_real_checkpoint_probe import (  # noqa: E402
    ModelScopeRealCheckpointProbeConfig,
    run_modelscope_real_checkpoint_probe,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cache-dir", default="/root/modelscope_cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--folding-dtype", default="float32")
    ap.add_argument("--folded-weight-runtime-dtype", default="float32")
    ap.add_argument("--recovery-dtype", default="float32")
    ap.add_argument("--compare-dtype", default="float32")
    ap.add_argument("--prefill-seq-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--max-layers", default="2")
    ap.add_argument("--mask-mode", default="signed_permutation")
    ap.add_argument("--residual-mask-strategy", default="shared")
    ap.add_argument("--prompt-file",
                    default="outputs/real_prompts_stage8_2.jsonl")
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--output",
                    default="outputs/eval_batch_scaling_0_5b_realprompts.json")
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    max_layers = "all" if args.max_layers == "all" else int(args.max_layers)
    rows = []
    status = "ok"
    for bs in batch_sizes:
        cfg = ModelScopeRealCheckpointProbeConfig(
            model_id=args.model_id, cache_dir=args.cache_dir,
            device=args.device, dtype=args.dtype,
            folding_dtype=args.folding_dtype,
            folded_weight_runtime_dtype=args.folded_weight_runtime_dtype,
            recovery_dtype=args.recovery_dtype, compare_dtype=args.compare_dtype,
            prefill_seq_len=args.prefill_seq_len, decode_steps=args.decode_steps,
            max_layers=max_layers, mask_mode=args.mask_mode,
            residual_mask_strategy=args.residual_mask_strategy,
            prompt_file=args.prompt_file, max_prompts=bs,
            run_hf_baseline=False, seed=args.seed)
        r = run_modelscope_real_checkpoint_probe(cfg)
        if r["status"] != "ok":
            status = r["status"]
            rows.append({"batch_size": bs, "status": r["status"],
                         "reason": r.get("reason")})
            break
        lm = r.get("latency_memory", {})
        mr = r.get("masked_runtime", {})
        rows.append({
            "batch_size": bs,
            "input_ids_shape": r["audit"]["input_ids_shape"],
            "attention_mask_explicit": r["audit"]["attention_mask_explicit"],
            "token_match_rate": mr.get("token_match_rate_vs_extracted"),
            "recovered_logits_err": mr.get("recovered_logits_max_abs_error"),
            "latency_ms": lm.get("masked_runtime_latency_ms"),
            "peak_cuda_memory_mb": lm.get("peak_cuda_memory_mb", {})
                .get("masked_runtime"),
            "tokens_per_second": lm.get("tokens_per_second_masked"),
        })

    out = {
        "stage": "8.2_batch_scaling", "status": status,
        "model_id": args.model_id, "max_layers": args.max_layers,
        "prefill_seq_len": args.prefill_seq_len,
        "decode_steps": args.decode_steps, "device": args.device,
        "batch_sizes": batch_sizes, "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote: {path}")
    print(f"status={status} rows={len(rows)}")
    return 0 if status == "ok" or status.startswith("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
