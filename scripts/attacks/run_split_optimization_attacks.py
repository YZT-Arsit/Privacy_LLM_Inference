"""EIA / BRE(forward) / PIA on real Qwen via a differentiable split downstream.

For each method and split layer k, builds the real F = layers[:k] and the
obfuscated smashed target, then runs the optimization attacks. Records
loss_initial / loss_final / loss_reduction_ratio + token recovery. Honest by
construction: plaintext should be inverted; masked methods have no mask and
should resist (measured failure, NOT blocked).

Bounded by design (small samples / steps). 2000-step / multi-seed runs are NOT
launched automatically — report 500 first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

_IMPL = {"eia": "best_effort_modern_adaptation", "bre": "forward_only_best_effort",
         "pia": "best_effort_no_oracle"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--methods", default="plaintext_gpu,obfuscatune_qwen_orthogonal,"
                   "ours_amulet_style_fresh_pad,ours_non_isometric_variant")
    p.add_argument("--attacks", default="eia,bre,pia")
    p.add_argument("--split-layers", default="4")
    p.add_argument("--max-samples", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--num-restarts", type=int, default=1)
    p.add_argument("--cond", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ks = [int(x) for x in args.split_layers.split(",")]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "attacks" / "split_opt"
    if args.dry_run:
        print(f"[dry-run] split-opt model={args.model_name_or_path or 'tiny'} methods={methods} "
              f"ks={ks} attacks={attacks} steps={args.num_steps} -> {out}")
        return

    import torch
    import pllo.attacks as A
    from pllo.attacks.qwen_split_downstream import build_split_attack_inputs
    from pllo.attacks.result_io import failed_result, write_jsonl
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed, dtype=dtype)
        family, path = "qwen", args.model_name_or_path
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=args.seed, dtype=dtype)
        family, path = "qwen_tiny_random", None
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    nlayers = len(model.model.layers)
    ks = [k for k in ks if 1 <= k <= nlayers]
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len), generator=g)
    if args.device == "cuda" and torch.cuda.is_available():
        ids = ids.to("cuda")

    def _run(attack, inp):
        if attack == "eia":
            return A.run_eia_optimization(inp, target_method=method, num_steps=args.num_steps,
                                          num_restarts=args.num_restarts)
        if attack == "bre":
            return A.run_bre_bisr_attack(inp, target_method=method, num_steps=args.num_steps)
        if attack == "pia":
            return A.run_pia_prompt_inversion(inp, target_method=method, num_steps=args.num_steps)
        raise ValueError(attack)

    results = []
    for k in ks:
        for method in methods:
            inp, _sd = build_split_attack_inputs(model, ids, k=k, method=method, seed=args.seed,
                                                 cond=args.cond, device=args.device, dtype=dtype)
            for attack in attacks:
                try:
                    r = _run(attack, inp)
                except Exception as e:  # noqa: BLE001
                    r = failed_result(attack, attack, "optimization", method,
                                      "split_inference", str(e))
                r.threat_model = "split_inference"
                r.model_family, r.model_name_or_path = family, path
                r.implementation_level = _IMPL.get(attack, r.implementation_level)
                r.notes = (f"split_layer_k={k}; {r.notes}")[:400]
                results.append(r)
                print(f"k={k} {method:32s} {attack:4s} {r.status:8s} "
                      f"tok1={r.metrics.get('token_recovery_top1')} "
                      f"mse={r.metrics.get('mse')}")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, out / "all_attacks.jsonl")
    print(f"wrote {out}/all_attacks.jsonl ({len(results)} results)")


if __name__ == "__main__":
    main()
