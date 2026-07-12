"""Figure 3 — training convergence: SST-2 dev accuracy vs optimizer step for the L5
(plaintext-equivalent) and L12 (protected) arms across seeds, from real eval_history.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figcommon import convergence_cell, utility_summary, FIGDIR  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = [1234, 2025, 7]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"L5": "#1f77b4", "L12": "#2e7d32"}
    for prof in ["L5", "L12"]:
        first = True
        for s in SEEDS:
            cell = convergence_cell(f"sst2_conv_{prof}_s{s}")
            if not cell:
                continue
            steps = [c[0] for c in cell]; acc = [c[1] for c in cell]
            ax.plot(steps, acc, marker="o", ms=4, color=colors[prof], alpha=0.75,
                    label=(f"{prof} ({'plaintext-eq fp32' if prof=='L5' else 'protected bf16'})" if first else None))
            first = False
    us = utility_summary()
    if us:
        l0 = us["arms"]["L0_base_floor_acc"]
        ax.axhline(l0, ls="--", color="gray", lw=1)
        ax.text(ax.get_xlim()[1], l0 + 0.001, f"L0 base floor {l0:.3f}", color="gray",
                ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("SST-2 dev accuracy")
    ax.set_title("Figure 3. Training convergence — protected (L12) tracks the plaintext-equivalent "
                 "reference (L5)\n(best-dev checkpoint selection + early stopping; 3 seeds each)", fontsize=9.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    out = FIGDIR / "fig3_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
