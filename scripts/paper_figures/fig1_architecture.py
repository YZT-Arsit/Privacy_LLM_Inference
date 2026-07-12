"""Figure 1 — system architecture (schematic; no experimental numbers).

Untrusted A10 GPU (transformed weights + masked states only) <-> trusted Intel TDX enclave
(labels, masks, FP32 optimizer). Draws the trust boundary and the per-step data plane.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figcommon import FIGDIR  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def box(ax, x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=1.5))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=11, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + 0.04, y + h - 0.16 - i * 0.11, ln, ha="left", va="top", fontsize=8.5)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.3, 1.0, 4.2, 4.0, "Untrusted A10 GPU",
        ["holds ONLY transformed *_tilde:", " • W_tilde (base, folded γ)",
         " • H_tilde = H·Nr (residual)", " • A_tilde, B_tilde (masked LoRA)",
         " • KV_tilde (orthogonal per head)", "runs masked forward/backward",
         "NEVER sees: W, H, masks, labels,", "  loss, per-class logits"], "#e8f0fe")
    box(ax, 5.5, 1.0, 4.2, 4.0, "Trusted Intel TDX enclave",
        ["holds secrets:", " • masks Nr, B, S, P, Pi, U",
         " • private labels / loss", " • FP32 optimizer (m, v)",
         "computes: CE + dlogits,", "  AdamW un-fold/step/re-fold",
         "attested (TDX quote verified)"], "#e6f4ea")
    # trust boundary
    ax.plot([5.0, 5.0], [0.6, 5.4], "r--", lw=1.6)
    ax.text(5.0, 5.5, "trust boundary", color="red", ha="center", fontsize=9)
    # arrows
    a1 = FancyArrowPatch((4.5, 3.6), (5.5, 3.6), arrowstyle="->", mutation_scale=14, lw=1.4, color="#333")
    a2 = FancyArrowPatch((5.5, 2.9), (4.5, 2.9), arrowstyle="->", mutation_scale=14, lw=1.4, color="#333")
    a3 = FancyArrowPatch((4.5, 2.2), (5.5, 2.2), arrowstyle="<->", mutation_scale=14, lw=1.4, color="#333")
    for a in (a1, a2, a3):
        ax.add_patch(a)
    ax.text(5.0, 3.75, "masked logits (bf16)", ha="center", fontsize=7.5)
    ax.text(5.0, 3.05, "authenticated dlogits", ha="center", fontsize=7.5)
    ax.text(5.0, 2.35, "packed trusted grads / AdamW (HMAC)", ha="center", fontsize=7.5)
    ax.text(5.0, 0.35, "per optimizer step: 2 trusted round trips (ce_batch + adamw); "
                       "A10 holds no labels/masks/optimizer state",
            ha="center", fontsize=8, style="italic")
    ax.set_title("Figure 1. Private-base TEE↔GPU architecture (schematic)", fontsize=12)
    out = FIGDIR / "fig1_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
