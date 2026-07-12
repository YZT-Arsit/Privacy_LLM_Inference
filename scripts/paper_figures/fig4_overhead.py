"""Figure 4 — communication / TEE overhead, from the real profiling-gate artifact.
Left: per-optimizer-step component breakdown at batch 16. Right: sec/batch vs physical
batch size (sweep characterization). No fabricated numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figcommon import profiling_gate, FIGDIR  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    pg = profiling_gate()
    if not pg:
        print("no profiling gate"); return
    sw = pg["sweep_characterization_only"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: component breakdown at batch 16
    b16 = sw["16"]
    comps = [("TDX AdamW\n(enclave, fixed)", b16["adamw_tdx_ms"], "#8e44ad"),
             ("CE network\ntransfer", b16["ce_net_ms"], "#c0392b"),
             ("GPU forward", b16["gpu_fwd_ms"], "#2980b9"),
             ("GPU backward", b16["gpu_bwd_ms"], "#27ae60")]
    names = [c[0] for c in comps]; vals = [c[1] for c in comps]; cols = [c[2] for c in comps]
    bars = axL.bar(names, vals, color=cols)
    axL.bar_label(bars, fmt="%.0f ms", fontsize=8)
    axL.set_ylabel("ms per optimizer step (batch 16)")
    axL.set_title("Per-step cost breakdown\n(TDX round trip dominates, inherent)", fontsize=10)
    axL.tick_params(axis="x", labelsize=8)

    # Right: sec/batch vs batch size
    bs = sorted(int(k) for k in sw)
    spb = [sw[str(b)]["sec_per_batch"] for b in bs]
    eps = [sw[str(b)]["examples_per_sec"] for b in bs]
    axR.plot(bs, spb, marker="o", color="#c0392b", label="sec / batch")
    axR.set_xlabel("physical batch size"); axR.set_ylabel("sec / batch", color="#c0392b")
    axR.set_xticks(bs)
    axR.axvline(16, ls="--", color="gray", lw=1)
    axR.text(16, min(spb), " selected\n (=eff. batch)", fontsize=7, color="gray", va="bottom")
    ax2 = axR.twinx()
    ax2.plot(bs, eps, marker="s", color="#2980b9", label="examples / sec")
    ax2.set_ylabel("examples / sec", color="#2980b9")
    axR.set_title("Throughput vs batch size\n(only 16 preserves the frozen effective batch)", fontsize=10)

    fig.suptitle("Figure 4. Communication / TEE overhead — SST-2 L12 protected training (real A10 + Intel TDX)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGDIR / "fig4_overhead.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
