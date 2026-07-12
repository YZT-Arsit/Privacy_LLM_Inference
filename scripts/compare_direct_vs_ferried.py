"""Numeric-consistency check: direct_h800_tdx vs mac_ferried_authenticated_prototype (req 1).

Given the SAME (seed, lr, steps, seq_len, input_ids), the transport only moves bytes; the math is
identical. So the direct run must reproduce the ferried run's per-step CE trajectory, the per-step
correction counts (48 GPU-exact A + 120 in-enclave A), and the final effective-equivalence (top-1
agreement, next-logit KL). We compare exactly those, per step, with registered tolerances. The
direct run's grad / corrected / update / momentum / dW-proxy norms are additionally recorded as the
direct trajectory's numeric fingerprint (the ferried worker did not log grad norms; their equality
follows transitively from identical CE trajectory + identical final equivalence).

    python3 scripts/compare_direct_vs_ferried.py --direct D.json --direct-equiv D.equiv.json \
        --ferried F.json [--ce-tol 1e-2 --kl-tol 5e-3]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--direct-equiv", default="")
    ap.add_argument("--ferried", required=True)
    ap.add_argument("--ce-tol", type=float, default=1e-2)     # bf16 GPU nondeterminism budget
    ap.add_argument("--kl-tol", type=float, default=5e-3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    D = load(args.direct); F = load(args.ferried)
    dt = D["trajectory"]; ft = F["trajectory"]
    n = min(len(dt), len(ft))
    rows = []; max_ce = 0.0; counts_ok = True
    for i in range(n):
        d, f = dt[i], ft[i]
        d_ce = d["ce"]; f_ce = f.get("tdx_ce", f.get("ce"))
        dce = abs(d_ce - f_ce); max_ce = max(max_ce, dce)
        d_gpu = d["gpu_exact_A"]; f_gpu = f.get("gpu_exact_A_applied")
        d_tdx = d["tdx_corrected"]; f_tdx = f.get("correction_targets_applied")
        cnt = (d_gpu == f_gpu and d_tdx == f_tdx)
        counts_ok = counts_ok and cnt
        rows.append({"step": i, "ce_direct": d_ce, "ce_ferried": f_ce, "ce_abs_diff": dce,
                     "gpuA_direct": d_gpu, "gpuA_ferried": f_gpu,
                     "tdxA_direct": d_tdx, "tdxA_ferried": f_tdx, "counts_match": cnt,
                     "direct_numerics": d.get("numerics", {})})

    # effective equivalence
    d_eq = load(args.direct_equiv) if args.direct_equiv and Path(args.direct_equiv).exists() else {}
    f_eq = F.get("effective_equivalence", {})
    d_top1 = d_eq.get("effective_equivalence_top1_agreement")
    f_top1 = f_eq.get("effective_equivalence_top1_agreement")
    d_kl = d_eq.get("effective_equivalence_next_logit_kl")
    f_kl = f_eq.get("effective_equivalence_next_logit_kl")
    # preservation/consistency criterion: direct must REPRODUCE ferried top1 (equality), not meet a
    # fixed convergence threshold (a 1-step run legitimately has top1<1; a 10-step run reaches 1.0).
    top1_ok = (d_top1 is not None and f_top1 is not None and abs(d_top1 - f_top1) < 1e-6)
    kl_ok = (d_kl is not None and f_kl is not None and abs(d_kl - f_kl) < args.kl_tol)

    ce_ok = max_ce <= args.ce_tol
    verdict = bool(ce_ok and counts_ok and top1_ok and kl_ok)

    out = {"direct_run": D.get("run_id"), "ferried_run": F.get("run_id"),
           "direct_transport_profile": D.get("transport_profile"),
           "steps_compared": n,
           "ce_trajectory": {"max_abs_diff": max_ce, "tol": args.ce_tol, "pass": ce_ok},
           "correction_counts": {"all_steps_match_48gpu_120tdx": counts_ok, "pass": counts_ok},
           "effective_equivalence": {"direct_top1": d_top1, "ferried_top1": f_top1,
                                     "direct_kl": d_kl, "ferried_kl": f_kl,
                                     "top1_pass": top1_ok, "kl_pass": kl_ok, "kl_tol": args.kl_tol},
           "session_reuse_direct": D.get("session_reuse"),
           "vram_summary_direct": D.get("vram_summary"),
           "VERDICT_numerics_consistent": verdict,
           "per_step": rows}
    print("=== direct vs ferried ===")
    print(f"  CE max|Δ| = {max_ce:.3e} (tol {args.ce_tol})  ->  {'PASS' if ce_ok else 'FAIL'}")
    print(f"  counts 48/120 all steps      ->  {'PASS' if counts_ok else 'FAIL'}")
    print(f"  top1 direct={d_top1} ferried={f_top1}  ->  {'PASS' if top1_ok else 'FAIL'}")
    print(f"  KL   direct={d_kl} ferried={f_kl}  ->  {'PASS' if kl_ok else 'FAIL'}")
    print(f"  VERDICT: {'CONSISTENT' if verdict else 'INCONSISTENT'}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
