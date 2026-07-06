"""Run the full attack matrix across target methods (local smoke by default).

Runs every implemented attack against every target method, writes per-run
AttackResults to outputs/attacks/all/*.jsonl. Blocked/failed records are kept.
Default: toy source, small sizes, CPU — a fast sanity of the whole matrix.

Example:
    python scripts/attacks/run_all_attacks.py
    python scripts/attacks/run_all_attacks.py --dry-run
    python scripts/attacks/run_all_attacks.py --attacks nn,kpa,arrowmatch
"""

from __future__ import annotations

import argparse

from _attack_common import add_common_args, out_dir, rep_for


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--attacks", default="all",
                   help="comma list: nn,eia,bre,kpa,perm,arrow,pia,freq or 'all'")
    args = p.parse_args()
    methods = [m.strip() for m in args.target_methods.split(",") if m.strip()]
    want = args.attacks.strip()

    if args.dry_run:
        print(f"[dry-run] methods={methods} attacks={want} source={args.source} "
              f"num_steps={args.num_steps} max_samples={args.max_samples}")
        return

    import pllo.attacks as A
    from pllo.attacks.permutation_multiset_attack import structural_leakage_probe
    from pllo.attacks.result_io import failed_result, write_jsonl

    def run_for(method):
        inp, _meta = rep_for(method, args)
        plan = []
        def add(key, fam, fn):
            if want == "all" or key in want:
                plan.append((key, fam, fn))
        add("nn", "structural", lambda: A.run_nn_embedding_inversion(inp, target_method=method, use_protected=True))
        add("nn", "structural", lambda: A.run_nn_embedding_inversion(inp, target_method=method, metric="differential", use_protected=True))
        add("eia", "optimization", lambda: A.run_eia_optimization(inp, target_method=method, num_steps=args.num_steps, num_restarts=args.num_restarts))
        add("bre", "optimization", lambda: A.run_bre_bisr_attack(inp, target_method=method, num_steps=args.num_steps))
        add("bre", "optimization", lambda: A.run_bre_backward_gradient_matching(inp, target_method=method))
        add("kpa", "cryptanalysis", lambda: A.run_kpa_known_plaintext(inp, target_method=method, mode="linear"))
        add("perm", "structural", lambda: structural_leakage_probe(inp, target_method=method))
        add("arrow", "alignment", lambda: A.run_arrowmatch_attack(inp, target_method=method))
        add("pia", "optimization", lambda: A.run_pia_prompt_inversion(inp, target_method=method, num_steps=args.num_steps))
        add("freq", "statistical", lambda: A.run_frequency_distribution_attack(inp, target_method=method))
        results = []
        for key, fam, fn in plan:
            try:
                results.append(fn())
            except Exception as e:
                results.append(failed_result(key, key, fam, method,
                                             "closed_model_no_weight_access", str(e)))
        return results

    all_results = []
    for method in methods:
        rs = run_for(method)
        all_results.extend(rs)
        for r in rs:
            print(f"{method:32s} {r.attack_id:28s} {r.status:8s} "
                  f"succ={r.metrics.get('attack_success_rate')}")
    out = out_dir(args, "all") / "all_attacks.jsonl"
    write_jsonl(all_results, out)
    print(f"wrote {out} ({len(all_results)} results)")


if __name__ == "__main__":
    main()
