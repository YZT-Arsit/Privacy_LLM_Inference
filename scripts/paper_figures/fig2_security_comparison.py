"""Figure 2 — security attack comparison (control vs protected), from the real
security_run_manifest. Each attack uses its own natural success/leakage metric in [0,1];
the caption + per-group labels make the metric explicit. No fabricated numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figcommon import security_manifest, FIGDIR  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    m = security_manifest()
    if not m:
        print("no manifest"); return
    h = m["headline_metrics"]
    # (label, metric_name, control_value, protected_value) each in [0,1]
    rows = [
        ("S1 repr.\ninversion", "recovery cosine", 1.0, max(0.0, h["S1"]["strict_inversion_cosine_middle"])),
        ("S2 LoRA\nrecovery", "ΔW recovery (1-relerr)", 1.0, max(0.0, 1.0 - h["S2"]["ours_plaintext_dW_rel_err"])),
        ("S3 logit\nconf. leak", "confidence corr.", h["S3"]["B1_perm_only"]["confidence_corr"], h["S3"]["B2_monomial"]["confidence_corr"]),
        ("S4 gradient\ninversion", "token recovery", h["S4"]["plaintext_b1_token_acc"], h["S4"]["plaintext_b1_token_acc"]),
        ("S5 member-\nship (AUC)", "ROC-AUC", h["S5"]["auc_B0_plaintext"], h["S5"]["auc_B1_monomial"]),
        ("S6 KV-cache\ninversion", "token top-1", h["S6"]["plaintext_KV_top1"], h["S6"]["masked_transfer_top1"]),
    ]
    labels = [r[0] for r in rows]
    ctrl = [r[2] for r in rows]
    prot = [r[3] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    b1 = ax.bar(x - w / 2, ctrl, w, label="plaintext / control (attacker succeeds)", color="#d1495b")
    b2 = ax.bar(x + w / 2, prot, w, label="protected / ours", color="#2e7d32")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.text(len(rows) - 0.5, 0.52, "random-ish (AUC/50%)", color="gray", fontsize=7, ha="right")
    for xi, r in zip(x, rows):
        ax.text(xi, -0.09, r[1], ha="center", va="top", fontsize=7, style="italic", color="#444")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("attacker success / leakage (per-attack metric, [0,1])")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 2. Attack success: plaintext/control vs protected (private-base, Qwen2.5-0.5B)\n"
                 "S4 bars are EQUAL by design — the orthogonal mask is a transparent basis change for "
                 "gradient inversion (defense = aggregation)", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.bar_label(b1, fmt="%.2f", fontsize=7, padding=1)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=1)
    fig.subplots_adjust(bottom=0.22)
    out = FIGDIR / "fig2_security_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
