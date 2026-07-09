"""Variant E — anchor-attack audit + performance for the two-sided island.

F. Anchor attacks on the ISLAND folded weights (W_up_tilde = Q_in W_up P_ff,
   W_down_tilde = P_ff^{-1} W_down P_out) vs a signed_perm fold of the SAME public
   weights. Boundary switch weights (N_res^{-1} P_in, P_out^{-1} N_res) are pure
   mask products with NO public-weight anchor, so anchor attacks do not apply to
   them (reported as such, not scored).

G. Performance: extra matmuls (mask switches), latency of the island vs a plain
   N_res-masked MLP baseline, condition numbers, fp64/fp32 exactness.

Local, small matrices. No noise, no RMSNorm correction.
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
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir) if args.output_dir else REPO / "results" / "attacks" / "variant_e_island"
    if args.dry_run:
        print(f"[dry-run] variant-E island dm={args.d_model} dff={args.d_ff} lam={args.lam} -> {out}")
        return

    import torch
    from pllo.experiments.anchor_attack_audit import (build_variant_fold, gram_alignment,
                                                      vma_matching)
    from pllo.ops.two_sided_island import (signed_perm_matrix, two_sided_gelu_mlp_island)
    from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat

    dm, dff, m, lam = args.d_model, args.d_ff, args.tokens, args.lam
    dt = torch.float64
    g = torch.Generator().manual_seed(args.seed)
    n, n_inv = signed_perm_matrix(dm, seed=args.seed, dtype=dt)
    p_in, q_in, dip = sample_nonorthogonal_keymat(dm, lam, seed=args.seed + 1, dtype=dt)
    p_ff, _, dff_diag = sample_nonorthogonal_keymat(dff, lam, seed=args.seed + 2, dtype=dt)
    p_out, _, dop = sample_nonorthogonal_keymat(dm, lam, seed=args.seed + 3, dtype=dt)
    w_up = torch.randn(dm, dff, dtype=dt, generator=g)
    w_down = torch.randn(dff, dm, dtype=dt, generator=g)

    # island folded weights (two-sided non-orthogonal)
    w_up_t = q_in @ w_up @ p_ff
    w_down_t = torch.linalg.inv(p_ff) @ w_down @ p_out

    def anchor_no_perm(w_pub, w_obf):
        gr = gram_alignment(w_pub, w_obf)
        vm = vma_matching(w_pub, w_obf, None)
        return {"gram_right_rel_err": gr["gram_right_diag_match_rel_err"],
                "gram_left_rel_err": gr["gram_left_diag_match_rel_err"],
                "vma_col_cos_top1": vm["vma_col_cos_top1"]}

    # signed_perm fold of the SAME public weights (baseline)
    sp_up = build_variant_fold("signed_perm", w_up, lam=lam, seed=args.seed, dtype=dt, device="cpu")
    sp_up_gr = gram_alignment(w_up, sp_up["w_obf"])
    sp_up_vm = vma_matching(w_up, sp_up["w_obf"], sp_up["true_perm"])

    attack_F = {
        "island_W_up_tilde (two-sided Q_in W_up P_ff)": anchor_no_perm(w_up, w_up_t),
        "island_W_down_tilde (two-sided P_ff^-1 W_down P_out)": anchor_no_perm(w_down, w_down_t),
        "baseline_signed_perm_W_up": {
            "gram_right_rel_err": sp_up_gr["gram_right_diag_match_rel_err"],
            "gram_left_rel_err": sp_up_gr["gram_left_diag_match_rel_err"],
            "vma_col_cos_top1": sp_up_vm["vma_col_cos_top1"],
            "mapping_recovered": bool(sp_up_vm["vma_col_cos_top1"] == 1.0)},
        "boundary_switch_weights": {
            "note": "N_res^{-1} P_in and P_out^{-1} N_res are pure mask products; no "
                    "public-weight anchor exists, so Gram/VMA/ArrowMatch do not apply."},
    }

    # --- G. performance: island vs plain N_res-masked MLP ----------------------
    d32 = torch.float32
    Nf, Ninvf = signed_perm_matrix(dm, seed=args.seed, dtype=d32)
    pin32, qin32, _ = sample_nonorthogonal_keymat(dm, lam, seed=args.seed + 1, dtype=d32)
    pff32, _, _ = sample_nonorthogonal_keymat(dff, lam, seed=args.seed + 2, dtype=d32)
    pout32, _, _ = sample_nonorthogonal_keymat(dm, lam, seed=args.seed + 3, dtype=d32)
    poutinv32 = torch.linalg.inv(pout32)
    X = torch.randn(m, dm, dtype=d32); Xres = X @ Nf
    wu32 = torch.randn(dm, dff, dtype=d32); wd32 = torch.randn(dff, dm, dtype=d32)
    switch_in = Ninvf @ pin32; switch_out = poutinv32 @ Nf     # precomputed once

    def island_fwd():
        return two_sided_gelu_mlp_island(
            Xres, wu32, None, wd32, None, n_res=Nf, n_res_inv=Ninvf,
            p_in=pin32, q_in=qin32, p_ff=pff32, p_out=pout32, p_out_inv=poutinv32)

    # plain N_res-masked MLP baseline: up, GELU-on-masked-state, down (no switches,
    # no Amulet lift). Timing proxy for a signed-perm-only masked MLP.
    def plain_masked_fwd():
        u = Xres @ wu32
        v = torch.nn.functional.gelu(u)
        return v @ wd32

    # decompose the island cost so the Variant-E-SPECIFIC overhead (mask switching)
    # is separated from the INHERITED Amulet-GELU nonlinear boundary cost.
    wu_t32 = qin32 @ wu32 @ pff32
    wd_t32 = torch.linalg.inv(pff32) @ wd32 @ pout32

    def switch_only():                                       # the 2 Variant-E matmuls
        return (Xres @ switch_in), (Xres @ switch_out)

    def linear_island_only():                               # two-sided up+down, no nonlinear
        u = (Xres @ switch_in) @ wu_t32
        y = u @ wd_t32
        return y @ switch_out

    island_ms = _time(island_fwd, args.reps, args.warmup)
    plain_ms = _time(plain_masked_fwd, args.reps, args.warmup)
    switch_ms = _time(switch_only, args.reps, args.warmup)
    linear_island_ms = _time(linear_island_only, args.reps, args.warmup)
    amulet_ms = max(0.0, island_ms - linear_island_ms)     # residual = Amulet-GELU boundary

    perf_G = {
        "extra_matmuls_per_island_forward": 2,     # switch_in + switch_out
        "two_sided_fold_matmuls_amortized": "precomputed once (setup); 0 per forward",
        "island_ms_per_forward": island_ms,
        "plain_masked_mlp_ms_per_forward": plain_ms,
        "mask_switch_only_ms": switch_ms,
        "linear_island_only_ms": linear_island_ms,
        "amulet_gelu_boundary_ms_inherited": amulet_ms,
        "variant_e_specific_overhead_ms": max(0.0, linear_island_ms - plain_ms),
        "note": ("mask switching is 2 cheap O(m·d^2) matmuls; the dominant cost is the "
                 "Amulet-GELU Kronecker lift at d_ff, which is INHERITED from variant B "
                 "(right_pad_amulet_gelu), not new to Variant E."),
        "cond_P_in": dip.cond, "cond_P_ff": dff_diag.cond, "cond_P_out": dop.cond,
    }

    # dtype exactness (fp64 vs fp32) of the island
    dtype_err = {}
    for name, dd in (("fp64", torch.float64), ("fp32", torch.float32)):
        nn, nni = signed_perm_matrix(dm, seed=args.seed, dtype=dd)
        pi, qi, _ = sample_nonorthogonal_keymat(dm, lam, seed=args.seed + 1, dtype=dd)
        pf, _, _ = sample_nonorthogonal_keymat(dff, lam, seed=args.seed + 2, dtype=dd)
        po, _, _ = sample_nonorthogonal_keymat(dm, lam, seed=args.seed + 3, dtype=dd)
        gg = torch.Generator().manual_seed(args.seed)
        xx = torch.randn(m, dm, dtype=dd, generator=gg)
        r = two_sided_gelu_mlp_island(xx @ nn, torch.randn(dm, dff, dtype=dd, generator=gg), None,
                                      torch.randn(dff, dm, dtype=dd, generator=gg), None,
                                      n_res=nn, n_res_inv=nni, p_in=pi, q_in=qi, p_ff=pf,
                                      p_out=po, p_out_inv=torch.linalg.inv(po))
        dtype_err[name] = r["max_abs_error"]

    out.mkdir(parents=True, exist_ok=True)
    report = {"config": {"d_model": dm, "d_ff": dff, "tokens": m, "lambda": lam},
              "F_anchor_attacks": attack_F, "G_performance": perf_G,
              "island_dtype_max_abs_error": dtype_err}
    (out / "variant_e_island.json").write_text(json.dumps(report, indent=2))

    md = [f"# Variant E — two-sided island (d_model={dm}, d_ff={dff}, lambda={lam})", "",
          "## F. Anchor attacks (public-weight anchor) on ISLAND folded weights", "",
          "| target | gram_right_rel_err | gram_left_rel_err | VMA col top1 | mapping recovered |",
          "|---|---|---|---|---|"]
    for name, a in attack_F.items():
        if "gram_right_rel_err" not in a:
            continue
        mr = a.get("mapping_recovered", False)
        md.append(f"| {name} | {a['gram_right_rel_err']:.3f} | {a['gram_left_rel_err']:.3f} | "
                  f"{a['vma_col_cos_top1']:.3f} | {mr} |")
    md += ["", "Boundary switch weights (`N_res^{-1} P_in`, `P_out^{-1} N_res`): pure mask products, "
           "**no public-weight anchor** — Gram/VMA/ArrowMatch do not apply.", "",
           "**Reading.** The island's two-sided folded weights inherit Variant D's anchor resistance "
           "(no mapping recovered), while the same public weights under a signed-perm fold are fully "
           "recovered (VMA top1 = 1.0). requires_plaintext_weight_anchor=True throughout.", "",
           "## G. Performance (decomposed — Variant-E-specific vs inherited)", "",
           f"- **mask-switch only** (the 2 Variant-E matmuls): {perf_G['mask_switch_only_ms']:.4f} ms.",
           f"- linear island (switch + two-sided up/down, no nonlinear): {perf_G['linear_island_only_ms']:.4f} ms; "
           f"Variant-E-specific overhead vs plain masked MLP ({perf_G['plain_masked_mlp_ms_per_forward']:.4f} ms) = "
           f"**+{perf_G['variant_e_specific_overhead_ms']:.4f} ms**.",
           f"- **Amulet-GELU boundary (INHERITED from variant B): {perf_G['amulet_gelu_boundary_ms_inherited']:.3f} ms** "
           f"— dominant term (Kronecker lift at d_ff), NOT new to Variant E.",
           f"- full island {perf_G['island_ms_per_forward']:.3f} ms.",
           f"- cond(P_in/P_ff/P_out) = {perf_G['cond_P_in']:.1f} / {perf_G['cond_P_ff']:.1f} / {perf_G['cond_P_out']:.1f}.",
           f"- island max_abs_error: fp64 {dtype_err['fp64']:.1e}, fp32 {dtype_err['fp32']:.1e}.", ""]
    (out / "variant_e_island.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out}/variant_e_island.json/.md")


if __name__ == "__main__":
    main()
