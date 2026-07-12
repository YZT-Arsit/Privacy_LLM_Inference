"""Collect real-gate reports (BF16 L10 SGD + L11 momentum) into empirical_matrix_closure/.
Reads profile_validation/{tag}_bf16_s{seed}_{steps}step.json + msg_{tag}_s{seed}/ per-step
backward/apply JSONs (which carry the per-group BF16 numerics). Robust to partial completion.
"""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PV = REPO / "results/aaai_private_base/full_lora_matrix/profile_validation"
CL = REPO / "results/aaai_private_base/empirical_matrix_closure"
SEEDS = [1234, 2025, 7]


def boot_ci(vals, n=2000, seed=1):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return [None, None]
    means = []; s = seed + 777; N = len(vals)
    for _ in range(n):
        acc = 0.0
        for _ in range(N):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            acc += vals[s % N]
        means.append(acc / N)
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def stat(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": statistics.mean(v),
            "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v), "ci95": boot_ci(v)}


def collect(tag, outdir, optimizer):
    outdir.mkdir(parents=True, exist_ok=True)
    rows, per_group_rows = [], []
    for steps in (1, 10):
        for seed in SEEDS:
            rp = PV / f"{tag}_bf16_s{seed}_{steps}step.json"
            if not rp.exists():
                continue
            r = json.loads(rp.read_text())
            traj = r.get("trajectory", [])
            last = traj[-1] if traj else {}
            rows.append({
                "tag": tag, "optimizer": optimizer, "dtype": "bf16", "seed": seed, "steps": steps,
                "gate_pass": r.get("gate_pass"),
                "attestation_verified": r.get("attestation", {}).get("attestation_verified"),
                "final_ce": last.get("tdx_ce"),
                "ce_trajectory": [round(s["tdx_ce"], 5) for s in traj],
                "top1": r.get("effective_equivalence", {}).get("effective_equivalence_top1_agreement"),
                "kl": r.get("effective_equivalence", {}).get("effective_equivalence_next_logit_kl"),
                "gpu_exact_A_per_step": last.get("gpu_exact_A_applied"),
                "tdx_corrected_per_step": last.get("correction_targets_applied"),
                "correction_missing": last.get("correction_missing_targets"),
                "corrected_grad_zeroed_fraction": last.get("corrected_grad_zeroed_fraction"),
                "all_finite": all(s["step_finite"] for s in traj) if traj else None,
                "logical_invocations": last.get("logical_invocations"),
                "untrusted_gamma_materializations":
                    last.get("backward_counters", {}).get("untrusted_gamma_materializations"),
                "untrusted_correction_matrix_materializations":
                    last.get("backward_counters", {}).get("untrusted_correction_matrix_materializations"),
                "silent_fallbacks": last.get("backward_counters", {}).get("silent_fallbacks"),
            })
            # per-group BF16 numerics from msg backward/apply JSONs (last step)
            msg = PV / f"msg_{tag}_s{seed}"
            bw = msg / f"backward_{steps-1}.json"; ap = msg / f"apply_{steps-1}.json"
            raw = json.loads(bw.read_text()).get("raw_A_per_group_all_layers", {}) if bw.exists() else {}
            cor = json.loads(ap.read_text()).get("corrected_A_per_group_all_layers", {}) if ap.exists() else {}
            for grp in ("qkv", "gate_up", "o_down"):
                d = raw.get(grp, {})
                per_group_rows.append({"tag": tag, "seed": seed, "steps": steps, "group": grp,
                    "phase": "raw_grad", "max_abs": d.get("max_abs"),
                    "min_abs_nonzero": d.get("min_abs_nonzero"),
                    "bf16_zeroed_fraction": d.get("bf16_zeroed_fraction"),
                    "bf16_overflow": d.get("bf16_overflow"), "nan_inf": d.get("nan_inf")})
            for grp in ("qkv", "gate_up"):
                d = cor.get(grp, {})
                per_group_rows.append({"tag": tag, "seed": seed, "steps": steps, "group": grp,
                    "phase": "corrected_grad", "max_abs": d.get("max_abs"),
                    "min_abs_nonzero": d.get("min_abs_nonzero"),
                    "bf16_zeroed_fraction": d.get("bf16_zeroed_fraction"),
                    "bf16_overflow": d.get("bf16_overflow"), "nan_inf": d.get("nan_inf")})
    if not rows:
        print(f"[{tag}] no reports yet"); return None
    with open(outdir / "gate_results.csv", "w", newline="") as f:
        cols = [k for k in rows[0] if k != "ce_trajectory"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    if per_group_rows:
        with open(outdir / "per_group_bf16_numerics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_group_rows[0].keys())); w.writeheader(); w.writerows(per_group_rows)
    # 3-seed statistics per steps
    summary = {"tag": tag, "optimizer": optimizer, "seeds": SEEDS}
    for steps in (1, 10):
        srows = [r for r in rows if r["steps"] == steps]
        if not srows:
            continue
        summary[f"{steps}step"] = {
            "n_seeds": len(srows), "all_gate_pass": all(r["gate_pass"] for r in srows),
            "all_attestation_verified": all(r["attestation_verified"] for r in srows),
            "top1": stat([r["top1"] for r in srows]),
            "kl": stat([r["kl"] for r in srows]),
            "final_ce": stat([r["final_ce"] for r in srows]),
            "all_finite": all(r["all_finite"] for r in srows),
            "security_counters_all_zero": all(
                (r["untrusted_gamma_materializations"] == 0 and
                 r["untrusted_correction_matrix_materializations"] == 0 and
                 r["silent_fallbacks"] == 0) for r in srows),
            "gpu_exact_A_per_step": srows[0]["gpu_exact_A_per_step"],
            "tdx_corrected_per_step": srows[0]["tdx_corrected_per_step"]}
    (outdir / "statistics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("step")}, indent=2, default=str)[:800])
    return summary


def main():
    collect("L10BF16", CL / "bf16_gate", "o1c_hybrid_sgd")
    collect("L11MOM", CL / "momentum_real", "o1c_hybrid_momentum")


if __name__ == "__main__":
    main()
