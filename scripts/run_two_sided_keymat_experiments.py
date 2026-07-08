"""Variant D lambda sweep: exactness (fp64/fp32/bf16) + conditioning + anchor
attacks (Gram/VMA/ArrowMatch) + Surface-A norm leakage + latency.

Answers: is there a lambda range where exactness stays rounding-level, the
anchor-based weight-alignment attacks substantially drop, and latency overhead is
acceptable? Also measures the Surface-A per-token norm correlation (norm_corr) so
the RMSNorm/norm tension is visible (higher lambda => more norm distortion, at the
cost of RMSNorm exactness — see docs/rmsnorm_exact_norm_impossibility.md).

Emits JSON + Markdown. No noise, no RMSNorm correction, exact linear chain only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _time(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--lambdas", default="0.0,0.01,0.03,0.1,0.3,1.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    out = Path(args.output_dir) if args.output_dir else REPO / "results" / "attacks" / "variant_d_two_sided"
    if args.dry_run:
        print(f"[dry-run] variant-D sweep dim={args.dim} lambdas={lambdas} -> {out}")
        return

    import torch
    from pllo.experiments.anchor_attack_audit import run_anchor_audit
    from pllo.experiments.lifted_attack_surface import stable_state_leakage
    from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat, two_sided_linear

    d, m = args.dim, args.tokens
    dtypes = {"fp64": torch.float64, "fp32": torch.float32}
    if hasattr(torch, "bfloat16"):
        dtypes["bf16"] = torch.bfloat16

    rows = []
    for lam in lambdas:
        row = {"lambda": lam}
        # conditioning (fp64 reference)
        _, _, diag = sample_nonorthogonal_keymat(d, lam, seed=args.seed, dtype=torch.float64)
        row.update({"cond_P": diag.cond, "min_singular": diag.min_singular,
                    "max_singular": diag.max_singular,
                    "keymat_inv_max_abs_err": diag.inverse_max_abs_error})
        # exactness of the two-sided linear across dtypes
        for name, dt in dtypes.items():
            p_in, q_in, _ = sample_nonorthogonal_keymat(d, lam, seed=args.seed, dtype=dt)
            p_out, _, _ = sample_nonorthogonal_keymat(d, lam, seed=args.seed + 100, dtype=dt)
            g = torch.Generator().manual_seed(args.seed)
            x = torch.randn(m, d, dtype=dt, generator=g)
            w = torch.randn(d, d, dtype=dt, generator=g)
            b = torch.randn(d, dtype=dt, generator=g)
            r = two_sided_linear(x, w, b, p_in, q_in, p_out)
            err = float((r["y_tilde"].to(torch.float64) - r["y_ref"].to(torch.float64)).abs().max())
            ref = float(r["y_ref"].to(torch.float64).abs().max()) + 1e-30
            row[f"max_abs_err_{name}"] = err
            row[f"rel_err_{name}"] = err / ref
        # Surface-A per-token norm leakage on the stable state X P_in (fp64)
        p_in, _, _ = sample_nonorthogonal_keymat(d, lam, seed=args.seed, dtype=torch.float64)
        g = torch.Generator().manual_seed(args.seed + 7)
        h = torch.randn(m, d, dtype=torch.float64, generator=g)
        leak = stable_state_leakage(h, h @ p_in, m, 1)
        row["surfaceA_norm_corr"] = leak["norm_corr"]
        row["surfaceA_gram_corr"] = leak["gram_corr"]
        # anchor attacks on the folded weight (fp64)
        aud = run_anchor_audit("two_sided_nonorthogonal_exact", d_in=d, d_out=d,
                               lam=lam, seed=args.seed, dtype=torch.float64)
        row.update({
            "anchor_gram_right_rel_err": aud["gram_right_diag_match_rel_err"],
            "anchor_gram_left_rel_err": aud["gram_left_diag_match_rel_err"],
            "anchor_vma_col_cos_top1": aud["vma_col_cos_top1"],
            "anchor_arrow_match_acc": aud["arrow_match_acc"],
            "anchor_mapping_recovered": aud["weight_mapping_fully_recovered"],
        })
        # latency of the extra masking matmuls (X P_in ; fold once amortized)
        p_in32, _, _ = sample_nonorthogonal_keymat(d, lam, seed=args.seed, dtype=torch.float32)
        x32 = torch.randn(m, d, dtype=torch.float32)
        row["stable_state_mask_ms"] = _time(lambda: x32 @ p_in32, args.reps, args.warmup)
        rows.append(row)
        print(f"lam={lam:4.2f} cond={diag.cond:8.2f} "
              f"err[fp64={row['max_abs_err_fp64']:.1e} fp32={row['max_abs_err_fp32']:.1e} "
              f"bf16={row.get('max_abs_err_bf16', float('nan')):.1e}] "
              f"normcorr={row['surfaceA_norm_corr']:.3f} "
              f"gramL={row['anchor_gram_left_rel_err']:.3f} vma={row['anchor_vma_col_cos_top1']:.3f} "
              f"map_rec={row['anchor_mapping_recovered']}")

    # baseline anchor comparison across variants (fixed lambda=0.1)
    baseline = {}
    for v in ("signed_perm", "dense_right_orthogonal", "two_sided_nonorthogonal_exact"):
        a = run_anchor_audit(v, d_in=d, d_out=d, lam=0.1, seed=args.seed, dtype=torch.float64)
        baseline[v] = {k: a[k] for k in (
            "gram_right_diag_match_rel_err", "gram_left_diag_match_rel_err",
            "vma_col_cos_top1", "arrow_match_acc", "weight_mapping_fully_recovered",
            "hidden_state_unmask_max_abs_err")}

    out.mkdir(parents=True, exist_ok=True)
    (out / "variant_d_sweep.json").write_text(json.dumps(
        {"dim": d, "tokens": m, "lambdas": lambdas, "sweep": rows,
         "anchor_baseline_lambda0.1": baseline}, indent=2))
    md = [f"# Variant D — two-sided non-orthogonal exact keymat (dim={d})", "",
          "Exact for the LINEAR chain (no noise, no RMSNorm correction). Non-orthogonal "
          "P does not pass RMSNorm/nonlinearity exactly (scope: linear chain).", "",
          "## Lambda sweep", "",
          "| lambda | cond(P) | max_abs_err fp64 | fp32 | bf16 | surfaceA norm_corr | "
          "anchor gramL | anchor VMA top1 | mapping recovered | mask matmul ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {lam:g} | {cond:.1f} | {e64:.1e} | {e32:.1e} | {e16} | {nc:.3f} | "
                  "{gl:.3f} | {vma:.3f} | {mr} | {lat:.4f} |".format(
                      lam=r["lambda"], cond=r["cond_P"], e64=r["max_abs_err_fp64"],
                      e32=r["max_abs_err_fp32"],
                      e16=(f"{r['max_abs_err_bf16']:.1e}" if "max_abs_err_bf16" in r else "—"),
                      nc=r["surfaceA_norm_corr"], gl=r["anchor_gram_left_rel_err"],
                      vma=r["anchor_vma_col_cos_top1"], mr=r["anchor_mapping_recovered"],
                      lat=r["stable_state_mask_ms"]))
    md += ["", "## Anchor-attack comparison (lambda=0.1, folded weight, public-weight anchor)", "",
           "| variant | gram_right_rel_err | gram_left_rel_err | VMA col top1 | ArrowMatch | mapping recovered | stable-state unmask err |",
           "|---|---|---|---|---|---|---|"]
    for v, a in baseline.items():
        ue = a["hidden_state_unmask_max_abs_err"]
        md.append(f"| {v} | {a['gram_right_diag_match_rel_err']:.3f} | "
                  f"{a['gram_left_diag_match_rel_err']:.3f} | {a['vma_col_cos_top1']:.3f} | "
                  f"{a['arrow_match_acc']:.3f} | {a['weight_mapping_fully_recovered']} | "
                  f"{('%.1e'%ue) if ue is not None else '—'} |")
    md += ["", "**Reading.** `signed_perm` (A_rightmul production surface) is fully recovered "
           "(mapping acc 1.0 → exact stable-state un-mask). ObfuscaTune's right-orthogonal fold "
           "leaves the LEFT row-Gram invariant (gram_left≈0). Variant D scrambles BOTH Grams and "
           "no anchor attack recovers a mapping — while staying exact (fp64/fp32) for the linear "
           "chain. Surface-A norm_corr falls as lambda grows (non-orthogonal distorts per-token "
           "norm) but that same non-orthogonality breaks RMSNorm exactness — the tension formalised "
           "in docs/rmsnorm_exact_norm_impossibility.md. requires_plaintext_weight_anchor=True: none "
           "of these attacks applies under a private-weights deployment."]
    (out / "variant_d_sweep.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/variant_d_sweep.json/.md")


if __name__ == "__main__":
    main()
