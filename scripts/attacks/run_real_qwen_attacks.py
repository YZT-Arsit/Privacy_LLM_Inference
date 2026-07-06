"""Real-Qwen full attack matrix across all four defenses (single model load).

Loads a REAL Qwen2 checkpoint ONCE, captures the plaintext residual state, and
applies each defense's actual cloud-visible transform to the same hidden states
(and to a real weight matrix, for the weight-leakage worst case). Then runs the
attack set against plaintext_gpu / stip_qwen / obfuscatune_qwen_orthogonal /
ours_amulet_style and writes AttackResults + a measured security table.

Structural / cryptanalytic / alignment attacks (nn, kpa, multiset, frequency,
arrowmatch, gram) are MEASURED on the real model. Optimization attacks
(eia/bre/pia) need a differentiable white-box victim map (split_inference); none
is provided at this residual cut, so they block honestly.

Example (server, data disk):
    python scripts/attacks/run_real_qwen_attacks.py \
        --model-name-or-path /root/autodl-tmp/.../Qwen2.5-7B-Instruct \
        --num-tokens 32 --device cpu \
        --output-dir /root/autodl-tmp/pllo_attack_run/attacks/qwen7b_real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_METHODS = "plaintext_gpu,stip_qwen,obfuscatune_qwen_orthogonal,ours_amulet_style"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None,
                   help="local Qwen2 checkpoint; omit for a tiny random Qwen (smoke)")
    p.add_argument("--target-methods", default=DEFAULT_METHODS)
    p.add_argument("--attacks", default="nn,kpa,perm,freq,arrow,gram,eia,bre,pia",
                   help="comma list: nn,kpa,perm,freq,arrow,gram,eia,bre,pia")
    p.add_argument("--num-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--include-fresh-pad", action="store_true",
                   help="also evaluate the hardened ours_amulet_style_fresh_pad variant")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    methods = [m.strip() for m in args.target_methods.split(",") if m.strip()]
    if args.include_fresh_pad and "ours_amulet_style_fresh_pad" not in methods:
        methods.append("ours_amulet_style_fresh_pad")
    want = args.attacks.strip()
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "attacks" / "real_qwen"

    if args.dry_run:
        print(f"[dry-run] real-qwen attacks model={args.model_name_or_path or 'tiny_random_qwen'} "
              f"methods={methods} attacks={want} -> {out}")
        return

    import torch
    import pllo.attacks as A
    from pllo.attacks.permutation_multiset_attack import structural_leakage_probe
    from pllo.attacks.real_qwen_representations import build_real_representations
    from pllo.attacks.result_io import failed_result, write_jsonl
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed,
                          dtype=torch.float32)
        family, path = "qwen", args.model_name_or_path
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=args.seed,
                          dtype=torch.float32)
        family, path = "qwen_tiny_random", None
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, model.config.vocab_size, (1, args.num_tokens), generator=g)

    reps = build_real_representations(model, ids, seed=args.seed,
                                      include_fresh_pad=args.include_fresh_pad)

    def plan_for(method, inp):
        plan = []
        def add(key, fn):
            if want == "all" or key in want:
                plan.append((key, fn))
        add("nn", lambda: A.run_nn_embedding_inversion(inp, target_method=method, use_protected=True))
        add("nn", lambda: A.run_nn_embedding_inversion(inp, target_method=method,
                                                       metric="differential", use_protected=True))
        add("kpa", lambda: A.run_kpa_known_plaintext(inp, target_method=method, mode="linear"))
        add("perm", lambda: structural_leakage_probe(inp, target_method=method))
        add("freq", lambda: A.run_frequency_distribution_attack(inp, target_method=method))
        add("arrow", lambda: A.run_arrowmatch_attack(inp, target_method=method,
                                                     model_family=family, model_name_or_path=path))
        add("gram", lambda: A.run_gram_weight_recovery(inp, target_method=method,
                                                       model_family=family, model_name_or_path=path))
        add("eia", lambda: A.run_eia_optimization(inp, target_method=method, num_steps=args.num_steps))
        add("bre", lambda: A.run_bre_bisr_attack(inp, target_method=method, num_steps=args.num_steps))
        add("pia", lambda: A.run_pia_prompt_inversion(inp, target_method=method, num_steps=args.num_steps))
        return plan

    all_results = []
    for method in methods:
        inp = reps.get(method)
        if inp is None:
            print(f"{method:34s} (no representation built — skipped)")
            continue
        for key, fn in plan_for(method, inp):
            try:
                r = fn()
            except Exception as e:  # noqa: BLE001
                r = failed_result(key, key, "structural", method,
                                  "closed_model_no_weight_access", str(e))
            r.model_family, r.model_name_or_path = family, path
            all_results.append(r)
            print(f"{method:34s} {r.attack_id:28s} {r.status:8s} "
                  f"succ={r.metrics.get('attack_success_rate')}")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_results, out / "all_attacks.jsonl")
    print(f"wrote {out}/all_attacks.jsonl ({len(all_results)} results)")


if __name__ == "__main__":
    main()
