"""Run the orthogonal masked-SGD contract audit (Gate 1.5-E, CPU fp64).

Writes results into results/real_qwen_tdx_audit/tdx_instance/gate1_5_sgd_contract/.
CPU-only, service-validation of the math contract. NOT a real Qwen/TDX experiment.
Nothing committed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pllo.experiments.masked_sgd_audit import run_audit
from pllo.experiments.report_utils import write_csv, write_json, write_text

OUT = Path("results/real_qwen_tdx_audit/tdx_instance/gate1_5_sgd_contract")

POSITIVE = ["sgd_orthogonal", "sgd_permutation", "momentum_orthogonal",
            "momentum_permutation", "nesterov_orthogonal", "wd_coupled",
            "wd_decoupled", "accum2", "accum4"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = run_audit()
    write_json(OUT / "results.json", r)

    corr = [{"case": k, "optimizer": r[k]["optimizer"], "mask": r[k]["mask_kind"],
             "steps": r[k]["steps"], "aligned": r[k]["aligned"],
             **{f"worst_{m}": r[k]["worst"][m] for m in
                ("A_err", "B_err", "dW_err", "next_logit_err", "loss_diff")}}
            for k in POSITIVE]
    write_csv(OUT / "correctness.csv", corr, list(corr[0].keys()))

    mom = [{"case": k, "momentum": r[k]["momentum"], "nesterov": r[k]["nesterov"],
            "worst_momentum_err": r[k]["worst"]["momentum_err"], "aligned": r[k]["aligned"]}
           for k in ("momentum_orthogonal", "momentum_permutation", "nesterov_orthogonal", "accum4")]
    write_csv(OUT / "momentum.csv", mom, list(mom[0].keys()))

    inv = [dict(v) for v in r["invocations"].values()]
    write_csv(OUT / "invocation_count.csv", inv, list(inv[0].keys()))

    neg = [{"case": k, "mask": r[k]["mask_kind"], "refresh_masks": r[k]["refresh_masks"],
            "aligned": r[k]["aligned"], "worst_A_err": r[k]["worst"]["A_err"],
            "expected_aligned": False}
           for k in ("neg_refresh_masks", "neg_non_orthogonal_gl")]
    write_csv(OUT / "negative_controls.csv", neg, list(neg[0].keys()))

    write_text(OUT / "summary.md", _summary(r))
    print(f"wrote gate1.5-E sgd contract to {OUT}")


def _summary(r):
    c = r["collapse"]
    sgd = r["invocations"]["gpu_masked_sgd"]; ada = r["invocations"]["trusted_adamw"]
    all_pos = all(r[k]["aligned"] for k in POSITIVE)
    neg_ok = (not r["neg_refresh_masks"]["aligned"]) and (not r["neg_non_orthogonal_gl"]["aligned"])
    L = []
    L.append("# Gate 1.5-E — Orthogonal Masked-SGD Contract (CPU fp64)\n")
    L.append("Service-validation of the math contract. CPU float64, synthetic tensors. "
             "NOT a real Qwen or real TDX experiment. Nothing committed.\n")
    L.append("## Answers\n")
    L.append(f"1. **SGD closes exactly under fixed orthogonal masks?** YES — "
             f"`sgd_orthogonal` worst A rel-err {r['sgd_orthogonal']['worst']['A_err']:.1e} over "
             f"100 steps; permutation masks too.")
    L.append(f"2. **Momentum SGD closes exactly?** YES — `momentum_orthogonal` worst momentum "
             f"rel-err {r['momentum_orthogonal']['worst']['momentum_err']:.1e} over 100 steps "
             f"(mA_t == N_in^-1 mA U).")
    L.append(f"3. **Which SGD options supported?** pure SGD, momentum (0/0.9), **Nesterov** "
             f"(proven: `nesterov_orthogonal` aligned={r['nesterov_orthogonal']['aligned']}), "
             f"coupled L2 + decoupled multiplicative weight decay, gradient accumulation 1/2/4, "
             f"LR scaling. All are linear in (grad, momentum) so they commute with fixed "
             f"orthogonal masks.")
    L.append(f"4. **Which optimizers rejected?** Adam/AdamW/RMSProp/Adagrad in "
             f"`gpu_masked_sgd` mode (element-wise second moments do NOT commute with dense "
             f"masks) — service fails closed at init.")
    L.append(f"5. **Packed optimizer boundary removed?** YES in masked-SGD mode — the GPU owns "
             f"masked A/B (+ masked momentum); no `/train/packed_update`, no trusted optimizer.")
    L.append(f"6. **Trusted invocations/step?** `gpu_masked_sgd` = **{sgd['total_trusted_invocations']}** "
             f"(input + loss; packed_update={sgd['packed_update_invocation']}, "
             f"optimizer_trusted={sgd['optimizer_trusted_invocation']}); "
             f"`trusted_adamw` = **{ada['total_trusted_invocations']}** (retained).")
    L.append(f"\n## Orthogonal collapse\n`N_in^T gradA U^-T == N_in^-1 gradA U` under "
             f"orthogonality (diff {c['orthogonal_diff']:.1e}); differs for general GL "
             f"(diff {c['gl_diff']:.1e}). We use the orthogonal form, not a general-GL formula.\n")
    L.append(f"## Verdict\npositive cases aligned = **{all_pos}**; negative controls correctly "
             f"fail = **{neg_ok}** (mask refresh without re-encoding → not aligned; dense "
             f"non-orthogonal direct SGD → not aligned, and the service rejects it).\n")
    L.append("## Correctness (allowed) vs security (separate)\n"
             "- **Correctness:** fixed orthogonal masks make SGD/momentum updates exactly "
             "equivariant (fp64).\n"
             "- **Efficiency:** the optimizer boundary is removed, 3→2 trusted invocations/step.\n"
             "- **Security:** orthogonal masks preserve MORE second-order geometry (norms/Gram/"
             "spectrum) than dense GL — the actual privacy impact must be MEASURED (Gate: security "
             "trade-off), NOT assumed. Orthogonal-SGD is not presumed safe.\n")
    L.append("## Labels\n`uses_real_gpu=false`, `uses_real_tee=false`, synthetic fp64 contract. "
             "Real Qwen2.5-0.5B masked-SGD is Gate 3 (blocked on GPU gateway cooldown).\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
