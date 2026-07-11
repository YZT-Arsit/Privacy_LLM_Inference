"""Post-process the fp64 equivalence matrix trajectories into clean aggregates +
statistical analysis. Handles the O1-A numerical INSTABILITY on gamma-fed targets
(NaN/diverged) as a first-class finding rather than letting it poison medians.

Emits full_lora_matrix/{aggregate_metrics.csv, statistical_analysis.json, run_index.csv}.
"""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FM = REPO / "results/aaai_private_base/full_lora_matrix"
GROUPS = {"o_down": ("o_proj", "down_proj"), "qkv": ("q_proj", "k_proj", "v_proj"),
          "gate_up": ("gate_proj", "up_proj")}
OPTS = {"sgd": "equivalence", "momentum": "momentum", "adamw": "adamw"}
KEYS = ["O1A_A_update_relerr", "O1C_A_update_relerr", "O1A_B_update_relerr",
        "O1C_B_update_relerr", "O1A_dW_cos", "O1C_dW_cos",
        "O1A_optstate_relerr", "O1C_optstate_relerr", "loss_absdiff_O1C"]


def finite(vals):
    return [v for v in vals if not (math.isnan(v) or math.isinf(v))]


def p95(sv):
    return sv[min(len(sv) - 1, int(0.95 * len(sv)))] if sv else float("nan")


def boot_ci(vals, n=2000, seed=0):
    """Deterministic bootstrap 95% CI of the mean (no Random module: LCG)."""
    vals = finite(vals)
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    means = []
    s = seed + 12345
    N = len(vals)
    for _ in range(n):
        acc = 0.0
        for _ in range(N):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            acc += vals[s % N]
        means.append(acc / N)
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def main():
    agg_rows, stat = [], {}
    run_index = []
    for opt, sub in OPTS.items():
        path = FM / sub / f"{opt}_full_trajectory.csv"
        rows = list(csv.DictReader(open(path)))
        steps = sorted({int(r["step"]) for r in rows})
        for S in steps:
            srows = [r for r in rows if int(r["step"]) == S]
            for gname, projs in GROUPS.items():
                grows = [r for r in srows if r["proj"] in projs]
                for key in KEYS:
                    raw = [float(r[key]) for r in grows]
                    fin = finite(raw)
                    nan_frac = 1 - len(fin) / max(1, len(raw))
                    sv = sorted(fin)
                    agg_rows.append({
                        "optimizer": opt, "step": S, "group": gname, "metric": key,
                        "n": len(raw), "finite_n": len(fin),
                        "nan_or_inf_fraction": round(nan_frac, 4),
                        "max": max(fin) if fin else float("nan"),
                        "median": statistics.median(fin) if fin else float("nan"),
                        "p95": p95(sv),
                        "worst_layer": (max(grows, key=lambda r: (float(r[key])
                                        if not (math.isnan(float(r[key])) or math.isinf(float(r[key]))) else -1))["layer"]
                                        if fin else "n/a")})
        # per-seed stats for the headline O1-C exactness + O1-A instability at final step
        Smax = max(steps)
        fin_rows = [r for r in rows if int(r["step"]) == Smax]
        stat[opt] = {}
        for gname, projs in GROUPS.items():
            grows = [r for r in fin_rows if r["proj"] in projs]
            o1c = [float(r["O1C_A_update_relerr"]) for r in grows]
            o1a = [float(r["O1A_A_update_relerr"]) for r in grows]
            o1c_f = finite(o1c); o1a_f = finite(o1a)
            stat[opt][gname] = {
                "O1C_A_update_max": max(o1c_f) if o1c_f else None,
                "O1C_A_update_mean": statistics.mean(o1c_f) if o1c_f else None,
                "O1C_A_update_ci95": boot_ci(o1c),
                "O1C_exact": (max(o1c_f) < 1e-10) if o1c_f else False,
                "O1A_A_update_max_finite": max(o1a_f) if o1a_f else None,
                "O1A_nan_or_inf_fraction": round(1 - len(o1a_f) / max(1, len(o1a)), 4),
                "O1A_diverged": (1 - len(o1a_f) / max(1, len(o1a))) > 0,
            }
        run_index.append({"optimizer": opt, "trajectory_csv": str(path.relative_to(REPO)),
                          "rows": len(rows), "steps": ",".join(map(str, steps)),
                          "seeds": "1234,2025,7", "layers": 24, "precision": "fp64"})

    with open(FM / "aggregate_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys())); w.writeheader(); w.writerows(agg_rows)
    (FM / "statistical_analysis.json").write_text(json.dumps({
        "headline": {
            "O1C_exact_all_optimizers_all_groups": all(
                stat[o][g]["O1C_exact"] for o in OPTS for g in GROUPS),
            "O1C_worst_A_update_relerr": max(stat[o][g]["O1C_A_update_max"]
                for o in OPTS for g in GROUPS if stat[o][g]["O1C_A_update_max"] is not None),
            "O1A_exact_only_on_o_down": all(stat[o]["o_down"]["O1A_A_update_max_finite"] is not None
                and stat[o]["o_down"]["O1A_A_update_max_finite"] < 1e-10 for o in OPTS),
            "O1A_gate_up_numerically_unstable_diverges": any(stat[o]["gate_up"]["O1A_diverged"] for o in OPTS),
            "O1A_qkv_drifts_finite": {o: stat[o]["qkv"]["O1A_A_update_max_finite"] for o in OPTS},
        },
        "per_optimizer_per_group": stat}, indent=2))
    with open(FM / "run_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_index[0].keys())); w.writeheader(); w.writerows(run_index)
    print(json.dumps(json.loads((FM / "statistical_analysis.json").read_text())["headline"], indent=2))


if __name__ == "__main__":
    main()
