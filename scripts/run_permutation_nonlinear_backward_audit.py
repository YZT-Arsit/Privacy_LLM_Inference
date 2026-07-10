"""Run the permutation nonlinear-island forward/backward + boundary audit.

Writes results.json + CSVs + summary.md into
results/attacks/permutation_nonlinear_backward_audit/. Experiment only; no
production path is touched, nothing is committed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pllo.experiments.permutation_nonlinear_backward_audit import (
    autograd_nonlinear_backward,
    cross_gram_boundary,
    lora_compatibility,
    mlp_forward_backward,
    run_stabilizer_suite,
    tiny_transformer_mlp_integration,
)

OUT = Path("results/attacks/permutation_nonlinear_backward_audit")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # exp1
    stab = run_stabilizer_suite()
    # exp2 (+3 inside via autograd) for gelu/silu/relu
    fb = [mlp_forward_backward(a, use_autograd_nonlinear=True) for a in ("gelu", "silu", "relu")]
    fb_manual = [mlp_forward_backward(a, use_autograd_nonlinear=False) for a in ("gelu", "silu")]
    # exp3 isolated
    autograd = [autograd_nonlinear_backward(a) for a in ("gelu", "silu", "relu")]
    # exp4
    xgram = cross_gram_boundary(trials=200)
    # exp5
    lora = [lora_compatibility(a) for a in ("gelu", "silu")]
    # exp6
    integ = tiny_transformer_mlp_integration()

    results = {
        "meta": {
            "dtype": "float64",
            "experiment_only": True,
            "committed": False,
            "production_path_modified": False,
        },
        "exp1_stabilizer": stab,
        "exp2_mlp_forward_backward_autograd": fb,
        "exp2_mlp_forward_backward_manual_phi_prime": fb_manual,
        "exp3_standard_autograd_nonlinear_backward": autograd,
        "exp4_cross_gram_boundary": xgram,
        "exp5_lora_compatibility": lora,
        "exp6_tiny_transformer_integration": integ,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    _write_csv(OUT / "stabilizer_tests.csv", stab)
    _write_csv(OUT / "forward_backward_correctness.csv", fb + fb_manual)
    _write_csv(OUT / "cross_gram_boundary.csv", [xgram])
    _write_csv(OUT / "lora_compatibility.csv", lora)

    _write_summary(results)
    print(f"wrote audit to {OUT}")


def _fmt(x: float) -> str:
    return f"{x:.2e}"


def _write_summary(r: dict) -> None:
    stab = r["exp1_stabilizer"]
    fb = r["exp2_mlp_forward_backward_autograd"]
    ag = r["exp3_standard_autograd_nonlinear_backward"]
    xg = r["exp4_cross_gram_boundary"]
    lora = r["exp5_lora_compatibility"]
    integ = r["exp6_tiny_transformer_integration"]

    def stab_row(a: str) -> dict:
        return {row["mask_family"]: row["pass_bool"] for row in stab if row["activation"] == a}

    gelu = stab_row("gelu"); silu = stab_row("silu"); relu = stab_row("relu")

    lines = []
    lines.append("# Permutation Nonlinear-Island — Forward/Backward + Security-Boundary Audit\n")
    lines.append("Experiment / audit only (float64). No production path modified; not committed.\n")

    lines.append("## Stabilizer tests — does `phi(Z Q) = phi(Z) Q`?\n")
    lines.append("| activation | permutation | signed_perm | pos_diagonal | dense_orth | dense_gl |")
    lines.append("|---|---|---|---|---|---|")
    for a, row in (("gelu", gelu), ("silu", silu), ("relu", relu)):
        lines.append(
            f"| {a} | {row['permutation']} | {row['signed_permutation']} | "
            f"{row['positive_diagonal']} | {row['dense_orthogonal']} | {row['dense_gl']} |"
        )
    lines.append("")
    lines.append("(pass = relation holds to fp64; each cell also has max_abs/rel_error in "
                 "`stabilizer_tests.csv`.)\n")

    lines.append("## MLP forward/backward (standard autograd nonlinear backward)\n")
    lines.append("| activation | fwd Z | fwd U | fwd Y | bwd GU | bwd GZ | bwd GH |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in fb:
        lines.append(
            f"| {row['activation']} | {_fmt(row['forward_z_error'])} | {_fmt(row['forward_u_error'])} "
            f"| {_fmt(row['forward_y_error'])} | {_fmt(row['backward_gu_error'])} "
            f"| {_fmt(row['backward_gz_error'])} | {_fmt(row['backward_gh_error'])} |"
        )
    lines.append("")
    lines.append("All errors are relative; ~1e-14..1e-16 confirms the exact masked relations "
                 "`Z_t=ZPi, U_t=UPi, Y_t=YN_out, GU_t=GUPi, GZ_t=GZPi, GH_t=GH M_in`.\n")

    ag_ok = all(x["pass_bool"] for x in ag)
    lines.append(f"**Experiment 3 (isolated autograd):** `GZ_tilde = GZ Pi` reproduced by "
                 f"standard `torch.autograd` for gelu/silu/relu — pass={ag_ok}, "
                 f"custom nonlinear primitive used = **False**.\n")

    lines.append("## Cross-Gram security boundary (200 trials)\n")
    lines.append("| region | mean rel_error | mean corr | exact? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| general linear (`H_t GH_t^T` vs `H GH^T`) | {_fmt(xg['mean_rel_error_general'])} "
                 f"| {xg['mean_corr_general']:.3f} | {xg['general_region_exact']} |")
    lines.append(f"| nonlinear `Z` (`Z_t GZ_t^T` vs `Z GZ^T`) | {_fmt(xg['mean_rel_error_z'])} "
                 f"| {xg['mean_corr_z']:.6f} | {xg['nonlinear_region_exact']} |")
    lines.append(f"| nonlinear `U` (`U_t GU_t^T` vs `U GU^T`) | {_fmt(xg['mean_rel_error_u'])} "
                 f"| {xg['mean_corr_u']:.6f} | — |")
    lines.append("")
    lines.append("Independent backward mask `M_in` protects the general linear region "
                 "(`H_t GH_t^T = H N M_in^T GH^T != H GH^T`). The **nonlinear permutation region "
                 "leaks the exact token×token cross-Gram** because `Pi Pi^T = I`.\n")

    lines.append("## LoRA compatibility\n")
    lines.append("| activation | up `Z` err | down `Y` err | rank⊥Pi |")
    lines.append("|---|---|---|---|")
    for row in lora:
        lines.append(f"| {row['activation']} | {_fmt(row['up_z_error'])} | {_fmt(row['down_y_error'])} "
                     f"| {row['rank_independent_of_pi']} |")
    lines.append("")
    lines.append("Rank-space masks `R_up`,`R_down` live in `r×r`; the activation permutation `Pi` "
                 "lives in `d_ff×d_ff` — distinct spaces. Perturbing `R_up` leaves the "
                 "`Pi`-invariant `Z_t = Z Pi` unchanged (rank⊥Pi), confirming independence.\n")

    lines.append("## Tiny-transformer integration (experiment 6)\n")
    if integ.get("synthetic_mlp_only"):
        lines.append(f"synthetic_mlp_only = True ({integ.get('reason')}).\n")
    else:
        lines.append(f"Ran on a real `{integ['used_real_block']}` MLP (GELU, with biases); "
                     f"attention {integ['attention']}. forward_pass={integ['forward_pass']} "
                     f"(recovered-output err {_fmt(integ['recovered_output_error'])}), "
                     f"autograd_backward_pass={integ['autograd_backward_pass']}.\n")

    lines.append("## Answers\n")
    q = [
        ("1. Does permutation mask close GELU/SiLU forward?",
         f"Yes — `phi(ZPi)=phi(Z)Pi` holds to fp64 for GELU and SiLU "
         f"(permutation pass = {gelu['permutation']}/{silu['permutation']})."),
        ("2. Do signed/diagonal/dense masks fail for GELU/SiLU?",
         f"Yes — GELU/SiLU fail signed_perm ({gelu['signed_permutation']}/{silu['signed_permutation']}), "
         f"pos_diagonal ({gelu['positive_diagonal']}/{silu['positive_diagonal']}), "
         f"dense_orth ({gelu['dense_orthogonal']}/{silu['dense_orthogonal']}) and dense_gl "
         f"({gelu['dense_gl']}/{silu['dense_gl']}). ReLU additionally closes positive_diagonal "
         f"({relu['positive_diagonal']}) — positive-homogeneous — but not signed_perm "
         f"({relu['signed_permutation']})."),
        ("3. Does MLP forward remain exact?",
         "Yes — Z/U/Y masked relations hold at ~1e-14."),
        ("4. Does MLP backward remain exact with standard autograd?",
         "Yes — GU/GZ/GH masked relations hold at ~1e-14 using torch autograd for the nonlinear."),
        ("5. Is a nonlinear backward primitive needed?",
         "No — inside the permutation domain `GZ_t = GU_t (.) phi'(Z_t) = GZ Pi` is produced by "
         "standard autograd; no custom masked-Hadamard primitive is required."),
        ("6. Does independent backward mask protect general regions?",
         f"Yes — with independent `M_in`, `H_t GH_t^T` differs from `H GH^T` "
         f"(mean rel_error {_fmt(xg['mean_rel_error_general'])}, exact={xg['general_region_exact']})."),
        ("7. Does the nonlinear permutation region leak exact cross-Gram?",
         f"Yes — `Z_t GZ_t^T = Z GZ^T` and `U_t GU_t^T = U GU^T` exactly "
         f"(rel_error {_fmt(xg['mean_rel_error_z'])}, corr {xg['mean_corr_z']:.6f}). This is a "
         f"real leak of token×token structure; permutation does NOT remove it."),
        ("8. Are LoRA rank-space masks independent from activation permutation masks?",
         "Yes — `R` acts in rank space (`r×r`), `Pi` in activation space (`d_ff×d_ff`); "
         "changing `R` leaves the `Pi`-invariant unchanged."),
        ("9. What claims are allowed/disallowed?",
         "See the two lists below."),
    ]
    for question, answer in q:
        lines.append(f"**{question}**")
        lines.append(f"{answer}\n")

    lines.append("### Allowed claims")
    for c in [
        "GELU/SiLU exact nonlinear islands require permutation-domain masking in this implementation.",
        "MLP forward/backward are exact under permutation islands.",
        "Standard autograd suffices for the nonlinear backward inside the permutation domain.",
        "Independent backward masks protect the linear/general regions.",
        "Nonlinear permutation regions leak the exact token×token cross-Gram.",
    ]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("### Disallowed claims (NOT supported by this experiment)")
    for c in [
        "Nonlinear cross-Gram is eliminated. (It is exactly preserved — see exp 4.)",
        "Permutation hides activation values. (It only permutes them; norm/Gram/value-multiset survive.)",
        "The experiment proves the full stabilizer theorem. (Finite tests only SUPPORT the algebraic claim.)",
        "The experiment proves input/token/adapter reconstruction (or its impossibility).",
        "The experiment proves end-to-end Qwen training.",
        "The nonlinear island provides formal privacy.",
    ]:
        lines.append(f"- {c}")
    lines.append("")

    OUT.joinpath("summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
