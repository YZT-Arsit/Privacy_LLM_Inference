"""Stage 8.2 -- top-conference evals CLI (Groups C + D).

Loads a real ModelScope checkpoint ONCE and writes compact, scalar-only JSON
for the output-boundary ablation (C) and the leakage/attack metrics (D1/D2/D3).
ModelScope cache only; never Hugging Face remote. No tensor dumps.
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
)
from pllo.experiments.stage8_2_topconf_evals import run_topconf_evals  # noqa: E402


def _write(path: Path, obj: dict) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8")) / 2 ** 20


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
    ap.add_argument("--mask-mode", default="signed_permutation",
                    choices=["signed_permutation", "block_orthogonal",
                             "dense_orthogonal"])
    ap.add_argument("--residual-mask-strategy", default="shared",
                    choices=["shared", "per_layer"])
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--out-prefix", default="outputs/eval")
    ap.add_argument("--model-tag", default="0_5b")
    args = ap.parse_args()

    max_layers = "all" if args.max_layers == "all" else int(args.max_layers)
    cfg = ModelScopeRealCheckpointProbeConfig(
        model_id=args.model_id, cache_dir=args.cache_dir, device=args.device,
        dtype=args.dtype, folding_dtype=args.folding_dtype,
        folded_weight_runtime_dtype=args.folded_weight_runtime_dtype,
        recovery_dtype=args.recovery_dtype, compare_dtype=args.compare_dtype,
        prefill_seq_len=args.prefill_seq_len, decode_steps=args.decode_steps,
        max_layers=max_layers, mask_mode=args.mask_mode,
        residual_mask_strategy=args.residual_mask_strategy,
        block_size=args.block_size, prompt=args.prompt,
        prompt_file=args.prompt_file, max_prompts=args.max_prompts,
        seed=args.seed)

    r = run_topconf_evals(cfg)
    tag = args.model_tag
    pre = args.out_prefix
    meta = {k: r.get(k) for k in (
        "stage", "status", "reason", "model_id", "local_path", "model_type",
        "total_layers", "max_layers_executed", "hidden_size", "vocab_size",
        "input", "mask_mode", "residual_mask_strategy",
        "token_match_rate_vs_extracted", "required_statement")}

    files = {}
    if r["status"] == "ok":
        files = {
            f"{pre}_output_boundary_ablation_{tag}_realprompts.json":
                {**meta, "boundary_ablation": r["boundary_ablation"]},
            f"{pre}_attack_token_recovery_{tag}_realprompts.json":
                {**meta, "attack_token_recovery": r["attack_token_recovery"]},
            f"{pre}_attack_masked_logits_{tag}_realprompts.json":
                {**meta, "attack_masked_logits": r["attack_masked_logits"]},
            f"{pre}_hidden_structure_leakage_{tag}_realprompts.json":
                {**meta, "hidden_structure_leakage":
                 r["hidden_structure_leakage"]},
        }
    else:
        files = {f"{pre}_topconf_evals_{tag}_SKIPPED.json": meta}

    for path, obj in files.items():
        mb = _write(Path(path), obj)
        print(f"Wrote: {path} ({round(mb, 4)} MB)")
    print(f"status={r['status']}")
    return 0 if r["status"] == "ok" or r["status"].startswith("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
