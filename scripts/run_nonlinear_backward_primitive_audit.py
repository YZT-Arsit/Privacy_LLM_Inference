"""Run the Scheme A nonlinear-backward primitive audit and write artifacts.

Writes to results/attacks/nonlinear_backward_primitive_audit/:
    summary.md, results.json, silu_forward_backward.csv,
    swiglu_forward_backward.csv, cross_gram_after_nonlinear.csv, efficiency.csv

Experiment/audit runner; does not touch the production path.

    python scripts/run_nonlinear_backward_primitive_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_backward_primitive_audit import run_all  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def _silu_rows(res: dict) -> list[dict]:
    s = res["silu"]
    rows = []
    v = s["trusted_shortcut"]
    rows.append({"variant": "trusted_shortcut", "forward_rel": v["forward"]["rel"],
                 "forward_max_abs": v["forward"]["max_abs"],
                 "backward_rel": v["backward"]["rel"], "backward_max_abs": v["backward"]["max_abs"],
                 "gpu_side": v["gpu_side"], "exposes_plaintext_in_tee": v["exposes_plaintext_in_tee"],
                 "forward_status": v["forward_status"], "backward_status": v["backward_status"],
                 "runtime_ms": "", "activation_gram_corr": ""})
    v = s["amulet_dense"]
    rows.append({"variant": "amulet_dense", "forward_rel": v["forward"]["rel"],
                 "forward_max_abs": v["forward"]["max_abs"],
                 "backward_rel": v["backward_hadamard_attempt"]["rel"],
                 "backward_max_abs": v["backward_hadamard_attempt"]["max_abs"],
                 "gpu_side": v["gpu_side"], "exposes_plaintext_in_tee": v["exposes_plaintext_in_tee"],
                 "forward_status": v["forward_status"], "backward_status": v["backward_status"],
                 "runtime_ms": v["forward_runtime_ms"], "activation_gram_corr": ""})
    v = s["permutation"]
    rows.append({"variant": "permutation", "forward_rel": v["forward"]["rel"],
                 "forward_max_abs": v["forward"]["max_abs"],
                 "backward_rel": v["backward"]["rel"], "backward_max_abs": v["backward"]["max_abs"],
                 "gpu_side": v["gpu_side"], "exposes_plaintext_in_tee": v["exposes_plaintext_in_tee"],
                 "forward_status": v["forward_status"], "backward_status": v["backward_status"],
                 "runtime_ms": "", "activation_gram_corr": v["activation_gram_corr"]})
    return rows


def _swiglu_rows(res: dict) -> list[dict]:
    s = res["swiglu"]
    rows = []
    v = s["trusted_shortcut"]
    rows.append({"variant": "trusted_shortcut", "forward_rel": v["forward"]["rel"],
                 "backward_dG_rel": v["backward_dG"]["rel"], "backward_dU_rel": v["backward_dU"]["rel"],
                 "gpu_side": v["gpu_side"], "exposes_plaintext_in_tee": v["exposes_plaintext_in_tee"],
                 "forward_status": v["forward_status"], "backward_status": v["backward_status"],
                 "runtime_ms": "", "activation_gram_corr": ""})
    v = s["amulet_dense"]
    rows.append({"variant": "amulet_dense", "forward_rel": v["forward"]["rel"],
                 "backward_dG_rel": v["backward_dG_hadamard_attempt"]["rel"],
                 "backward_dU_rel": v["backward_dU_hadamard_attempt"]["rel"],
                 "gpu_side": v["gpu_side"], "exposes_plaintext_in_tee": v["exposes_plaintext_in_tee"],
                 "forward_status": v["forward_status"], "backward_status": v["backward_status"],
                 "runtime_ms": v["forward_runtime_ms"], "activation_gram_corr": ""})
    v = s["permutation"]
    rows.append({"variant": "permutation", "forward_rel": v["forward"]["rel"],
                 "backward_dG_rel": v["backward_dG"]["rel"], "backward_dU_rel": v["backward_dU"]["rel"],
                 "gpu_side": v["gpu_side"], "exposes_plaintext_in_tee": v["exposes_plaintext_in_tee"],
                 "forward_status": v["forward_status"], "backward_status": v["backward_status"],
                 "runtime_ms": "", "activation_gram_corr": v["activation_gram_corr"]})
    return rows


def _cross_gram_rows(res: dict) -> list[dict]:
    cg = res["cross_gram_after_nonlinear"]
    rows = []
    for name in ("trusted_shortcut_A0_dual", "trusted_shortcut_A1_dense", "permutation_A1"):
        v = cg[name]
        rows.append({"variant": name, "trials": cg["trials"],
                     "corr_mean": v["corr_mean"], "corr_abs_mean": v["corr_abs_mean"],
                     "corr_std": v["corr_std"],
                     "exact_cross_gram_leak": v["exact_cross_gram_leak"],
                     "has_backward": v["has_backward"], "gpu_side": v["gpu_side"],
                     "activation_gram_corr_mean": v.get("activation_gram_corr_mean", ""),
                     "activation_gram_leak": v.get("activation_gram_leak", "")})
    v = cg["amulet_dense"]
    rows.append({"variant": "amulet_dense", "trials": cg["trials"], "corr_mean": "",
                 "corr_abs_mean": "", "corr_std": "",
                 "exact_cross_gram_leak": v["exact_cross_gram_leak"],
                 "has_backward": v["has_backward"], "gpu_side": "",
                 "activation_gram_corr_mean": "", "activation_gram_leak": ""})
    return rows


def _summary_md(res: dict) -> str:
    s, sw = res["silu"], res["swiglu"]
    cg = res["cross_gram_after_nonlinear"]
    tt = res["tiny_transformer_integration"]
    ans = res["answers"]

    def f(x): return f"{x:.2e}"

    a0 = cg["trusted_shortcut_A0_dual"]; a1 = cg["trusted_shortcut_A1_dense"]
    perm = cg["permutation_A1"]
    lines = [
        "# Scheme A — Nonlinear Backward Primitive Audit (SiLU / SwiGLU)",
        "",
        "Experiment/audit module (`src/pllo/experiments/nonlinear_backward_primitive_audit.py`); "
        "no production path modified. Scoped to the minimal SiLU/SwiGLU island that "
        "Qwen/LLaMA use — **not** full Qwen training. Uses the real "
        "`pllo.ops.amulet_right_mask_islands` forward primitive; all math float64.",
        "",
        "## Core finding",
        "",
        "The Amulet/Kronecker right-mask island gives a genuine **GPU-side forward** "
        "primitive for SiLU and SwiGLU under a *dense* mask (`U n → phi(U) n` via a "
        "Kronecker lift), exact to ~1e-16. But the **A1 backward is blocked**: the "
        "nonlinear backward needs elementwise (Hadamard) products of tensors carried "
        "under *independent* masks (`GY·M_out ⊙ SiLU'(X)·N_in`), and the Hadamard "
        "product does not commute with dense masks. Even granting a perfect masked "
        "derivative (obtained via the same lift, err "
        f"`{f(s['amulet_dense']['masked_derivative_err']['rel'])}`), the Hadamard step "
        f"has rel error `{s['amulet_dense']['backward_hadamard_attempt']['rel']:.2f}` — "
        "`blocked_by_hadamard`. The Amulet lift cannot rescue it because the lift "
        "requires both Hadamard operands to share the *same* mask and unit-selection "
        "(that is A0, which re-leaks the exact cross-Gram), not an independent A1 mask.",
        "",
        "## Answers",
        "",
        f"1. **GPU-side SiLU forward implemented?** **Yes** — Amulet dense island, rel "
        f"`{f(s['amulet_dense']['forward']['rel'])}`, runtime "
        f"`{s['amulet_dense']['forward_runtime_ms']:.2f} ms`.",
        f"2. **GPU-side SiLU A1 backward implemented?** **No** — `blocked_by_hadamard` "
        f"(Hadamard attempt rel `{s['amulet_dense']['backward_hadamard_attempt']['rel']:.2f}`).",
        f"3. **GPU-side SwiGLU forward implemented?** **Yes** — Amulet dense island, rel "
        f"`{f(sw['amulet_dense']['forward']['rel'])}`.",
        f"4. **GPU-side SwiGLU A1 backward implemented?** **No** — `blocked_by_hadamard` "
        f"(dU rel `{sw['amulet_dense']['backward_dU_hadamard_attempt']['rel']:.2f}`, "
        f"dG rel `{sw['amulet_dense']['backward_dG_hadamard_attempt']['rel']:.2f}`).",
        f"5. **Does any variant avoid exact cross-Gram?** Trusted-shortcut A1 (TEE, not "
        f"GPU) and permutation-A1 both break the exact cross-Gram (A0 dual = "
        f"{a0['corr_mean']:.3f} → exact leak; A1-dense = {a1['corr_mean']:+.3f}; "
        f"permutation-A1 = {perm['corr_mean']:+.3f}). **But** the only GPU-side "
        f"backward (permutation) leaks the activation Gram "
        f"(corr {perm['activation_gram_corr_mean']:.3f}) — it does not deliver A1 "
        f"dense-mask privacy. So: **no dense GPU-side variant both trains and avoids "
        f"the cross-Gram**.",
        f"6. **Fast enough in small dims?** The forward island runs in a few ms at "
        f"m=4,d=8, but the Kronecker lift blows the working tensors up by the factor "
        f"`k` per side (see efficiency.csv), so it scales poorly to real dims.",
        f"7. **Tiny-transformer nonlinear training without trusted shortcut?** "
        f"**No** — `{tt['status']}` (masked SwiGLU forward works, A1 backward blocked). "
        f"The permutation alternative trains but needs permutation masks throughout "
        f"(incompatible with the dense masks linear/LoRA A1 needs, and leaks the Gram).",
        f"8. **Is full Scheme A still blocked?** **Yes.**",
        "",
        "## SiLU forward/backward",
        "",
        "| variant | fwd rel | bwd rel | gpu-side | TEE plaintext | fwd status | bwd status | act-Gram corr |",
        "|---|---|---|---|---|---|---|---|",
        f"| trusted_shortcut | {f(s['trusted_shortcut']['forward']['rel'])} | "
        f"{f(s['trusted_shortcut']['backward']['rel'])} | False | True | "
        f"trusted_shortcut_only | trusted_shortcut_only | — |",
        f"| amulet_dense | {f(s['amulet_dense']['forward']['rel'])} | "
        f"{s['amulet_dense']['backward_hadamard_attempt']['rel']:.2f} | True | False | "
        f"forward_only | **blocked_by_hadamard** | — |",
        f"| permutation | {f(s['permutation']['forward']['rel'])} | "
        f"{f(s['permutation']['backward']['rel'])} | True | False | backward_passed | "
        f"backward_passed | **{s['permutation']['activation_gram_corr']:.3f} (leaks)** |",
        "",
        "## SwiGLU forward/backward",
        "",
        "| variant | fwd rel | dG rel | dU rel | gpu-side | bwd status | act-Gram corr |",
        "|---|---|---|---|---|---|---|",
        f"| trusted_shortcut | {f(sw['trusted_shortcut']['forward']['rel'])} | "
        f"{f(sw['trusted_shortcut']['backward_dG']['rel'])} | "
        f"{f(sw['trusted_shortcut']['backward_dU']['rel'])} | False | "
        f"trusted_shortcut_only | — |",
        f"| amulet_dense | {f(sw['amulet_dense']['forward']['rel'])} | "
        f"{sw['amulet_dense']['backward_dG_hadamard_attempt']['rel']:.2f} | "
        f"{sw['amulet_dense']['backward_dU_hadamard_attempt']['rel']:.2f} | True | "
        f"**blocked_by_hadamard** | — |",
        f"| permutation | {f(sw['permutation']['forward']['rel'])} | "
        f"{f(sw['permutation']['backward_dG']['rel'])} | "
        f"{f(sw['permutation']['backward_dU']['rel'])} | True | backward_passed | "
        f"**{sw['permutation']['activation_gram_corr']:.3f} (leaks)** |",
        "",
        "## Cross-Gram after nonlinear (200 trials)",
        "",
        "| variant | has bwd | gpu-side | cross-Gram corr | exact leak | act-Gram corr |",
        "|---|---|---|---|---|---|",
        f"| trusted_shortcut A0 dual | True | False | {a0['corr_mean']:+.4f} | "
        f"**{a0['exact_cross_gram_leak']}** | — |",
        f"| trusted_shortcut A1 dense | True | False | {a1['corr_mean']:+.4f} | "
        f"{a1['exact_cross_gram_leak']} | — |",
        f"| permutation A1 | True | True | {perm['corr_mean']:+.4f} | "
        f"{perm['exact_cross_gram_leak']} | **{perm['activation_gram_corr_mean']:.3f}** |",
        f"| amulet_dense | False | — | — | not_applicable | — |",
        "",
        "## Allowed claims",
        "",
        "- \"The Amulet right-mask island is a genuine GPU-side **forward** primitive "
        "for SiLU and SwiGLU under dense masks (exact to fp).\"",
        "- \"The A1 (independent-dense-mask) **backward** for SiLU/SwiGLU is blocked: it "
        "requires Hadamard products under independent dense masks, which do not "
        "commute (`blocked_by_hadamard`), and the Amulet lift cannot supply it "
        "without collapsing to shared-mask A0.\"",
        "- \"A working GPU-side nonlinear backward exists only under permutation masks, "
        "which leak the activation Gram — so it does not achieve A1 dense-mask privacy.\"",
        "- \"trusted_shortcut is exact but recovers plaintext inside the TEE; it is the "
        "correctness upper bound, not a GPU-side primitive.\"",
        "- \"Full Scheme A masked-domain training remains blocked by the nonlinear "
        "backward primitive.\"",
        "",
        "## Disallowed claims",
        "",
        "- ❌ \"GPU-side SiLU/SwiGLU A1 backward is supported.\" (blocked_by_hadamard, measured)",
        "- ❌ \"The Amulet island enables masked-domain nonlinear training.\" (forward only)",
        "- ❌ \"Permutation masks solve Scheme A nonlinear privacy.\" (activation Gram leaks)",
        "- ❌ \"Standard autograd gives A1 nonlinear backward.\" (autograd = A0 dual, which "
        "leaks the exact cross-Gram; not used here)",
        "- ❌ \"Full Scheme A is unblocked / Qwen nonlinear training works.\" (not run; blocked)",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(
        REPO_ROOT / "results" / "attacks" / "nonlinear_backward_primitive_audit"))
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = run_all(seed=args.seed)

    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    _write_csv(out / "silu_forward_backward.csv", _silu_rows(res))
    _write_csv(out / "swiglu_forward_backward.csv", _swiglu_rows(res))
    _write_csv(out / "cross_gram_after_nonlinear.csv", _cross_gram_rows(res))
    _write_csv(out / "efficiency.csv", res["efficiency"])
    (out / "summary.md").write_text(_summary_md(res))

    print(f"wrote artifacts to {out}")
    for p in sorted(out.glob("*")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
