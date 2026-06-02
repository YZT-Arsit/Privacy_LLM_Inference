#!/usr/bin/env python
"""Stage 7.5 — master runner for the paper artifact pipeline.

Sequence:

1. ``run_paper_artifact_consolidation`` — aggregate ``outputs/*.json``
   into ``paper_results/{csv,markdown,latex,json}/``.
2. ``run_measured_runtime_evaluation`` — local wall-time emulation
   benchmarks (NOT real TEE).
3. ``run_paper_claims_audit`` — supported / proxy_supported /
   unsupported classification.
4. Build ``paper_results/figures/*.png`` from the aggregated tables.
5. Write ``paper_results/summary.md``.

No new ops / probes / attackers. Pure aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pllo.experiments.measured_runtime_evaluation import (  # noqa: E402
    MeasuredRuntimeEvaluationConfig,
    run_measured_runtime_evaluation,
)
from pllo.experiments.paper_artifact_consolidation import (  # noqa: E402
    PaperArtifactConsolidationConfig,
    run_paper_artifact_consolidation,
)
from pllo.experiments.paper_claims_audit import (  # noqa: E402
    PaperClaimsAuditConfig,
    run_paper_claims_audit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument(
        "--paper-results-dir", type=Path,
        default=PROJECT_ROOT / "paper_results",
    )
    p.add_argument("--num-warmup", type=int, default=2)
    p.add_argument("--num-repeats", type=int, default=5)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _placeholder_note(
    path: Path, title: str, reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    placeholder = path.with_suffix(".missing.md")
    placeholder.write_text(
        f"# {title}\n\n_Placeholder — figure skipped: {reason}_\n",
        encoding="utf-8",
    )


def _fig_correctness_errors(
    consolidation: dict[str, Any], outpath: Path,
) -> None:
    rows = [
        r for r in consolidation["correctness_summary"]
        if isinstance(r.get("value"), (int, float)) and r.get("value") is not None
    ]
    if not rows:
        _placeholder_note(outpath, "Correctness errors", "no numeric rows")
        return
    labels = [f"{r['stage']}: {r['component'][:24]}" for r in rows]
    values = [float(r["value"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(rows))))
    ax.barh(labels, values)
    ax.set_xscale("log")
    ax.set_xlabel("metric value (log scale)")
    ax.set_title("Correctness summary — per-stage metrics")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


def _fig_security_risk_matrix(
    consolidation: dict[str, Any], outpath: Path,
) -> None:
    rows = consolidation["security_proxy_summary"]
    if not rows:
        _placeholder_note(outpath, "Security risk matrix", "no rows")
        return
    risk_order = ["low", "medium", "needs_more_evaluation", "high"]
    risk_to_score = {r: i for i, r in enumerate(risk_order)}
    labels = [
        f"{r['stage']}::{r['attack_family'][:30]}" for r in rows
    ]
    scores = [risk_to_score.get(r["risk_level"], 2) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(rows))))
    ax.barh(labels, scores)
    ax.set_xticks(list(range(len(risk_order))))
    ax.set_xticklabels(risk_order)
    ax.set_xlabel("risk level")
    ax.set_title("Security proxy risk matrix")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


def _fig_boundary_call_reduction(
    consolidation: dict[str, Any], outpath: Path,
) -> None:
    rows = consolidation["workload_summary"]
    if not rows:
        _placeholder_note(outpath, "Boundary call reduction", "no rows")
        return
    labels = [r["method"][:36] for r in rows]
    calls = [
        float(r["boundary_calls"]) if r["boundary_calls"] is not None else 0.0
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rows))))
    ax.barh(labels, calls)
    ax.set_xlabel("online boundary calls (per forward × forwards)")
    ax.set_title("Boundary call counts per method")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


def _fig_lora_training_errors(
    consolidation: dict[str, Any], outpath: Path,
) -> None:
    rows = consolidation["lora_training_summary"]
    rows = [
        r for r in rows
        if isinstance(r.get("loss_diff"), (int, float))
        and r.get("loss_diff") is not None
    ]
    if not rows:
        _placeholder_note(outpath, "LoRA training errors", "no rows")
        return
    labels = [f"{r['stage']}: {r['training_scope'][:30]}" for r in rows]
    losses = [float(r["loss_diff"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(rows))))
    ax.barh(labels, losses)
    ax.set_xscale("log")
    ax.set_xlabel("max loss diff (log scale)")
    ax.set_title("LoRA training loss diff per stage")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


def _fig_rank_inference_risk(
    consolidation: dict[str, Any], outpath: Path,
) -> None:
    rows = [
        r for r in consolidation["security_proxy_summary"]
        if "rank" in r.get("attack_family", "").lower()
        or "dummy" in r.get("attack_family", "").lower()
    ]
    if not rows:
        _placeholder_note(outpath, "Rank inference risk", "no rows")
        return
    risk_order = ["low", "medium", "needs_more_evaluation", "high"]
    risk_to_score = {r: i for i, r in enumerate(risk_order)}
    labels = [
        f"{r['stage']}::{r['attack_family'][:40]}" for r in rows
    ]
    scores = [risk_to_score.get(r["risk_level"], 2) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(rows))))
    ax.barh(labels, scores)
    ax.set_xticks(list(range(len(risk_order))))
    ax.set_xticklabels(risk_order)
    ax.set_xlabel("risk level")
    ax.set_title("Rank inference / dummy strategy proxy risk")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


def _fig_timing_proxy_before_after(
    outpath: Path, outputs_dir: Path,
) -> None:
    path = outputs_dir / "lora_training_timing_proxy.json"
    if not path.exists():
        _placeholder_note(outpath, "Timing proxy before/after", f"missing {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        _placeholder_note(outpath, "Timing proxy before/after", str(e))
        return
    off = data.get("leakage_tasks_off") or {}
    eq = data.get("leakage_tasks_proxy_equalized") or {}
    tasks = sorted(set(list(off.keys()) + list(eq.keys())))
    if not tasks:
        _placeholder_note(outpath, "Timing proxy before/after", "no tasks")
        return
    off_acc = [off.get(t, {}).get("classification_accuracy", 0.0) for t in tasks]
    eq_acc = [eq.get(t, {}).get("classification_accuracy", 0.0) for t in tasks]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(tasks))
    width = 0.4
    ax.bar([i - width / 2 for i in x], off_acc, width, label="off")
    ax.bar([i + width / 2 for i in x], eq_acc, width, label="proxy_equalized")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("classifier accuracy")
    ax.set_title("Training timing proxy: off vs proxy_equalized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


def _fig_measured_runtime(
    runtime_report: dict[str, Any], outpath: Path,
) -> None:
    rows = [
        r for r in runtime_report["rows"]
        if r.get("mean_ms") is not None
    ]
    if not rows:
        _placeholder_note(outpath, "Measured runtime", "no rows")
        return
    labels = [r["component"] for r in rows]
    means = [float(r["mean_ms"]) for r in rows]
    stds = [float(r["std_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(rows))))
    ax.barh(labels, means, xerr=stds, capsize=4)
    ax.set_xlabel("mean wall-time (ms) — local emulation, NOT real TEE")
    ax.set_title("Measured runtime (local emulation)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------


def _summary_md(
    consolidation: dict[str, Any],
    runtime: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    inv = consolidation["artifact_inventory"]
    missing = consolidation["missing_artifacts"]
    by_slot: dict[str, int] = {}
    for r in inv:
        by_slot[r["slot"]] = by_slot.get(r["slot"], 0) + 1
    present = sum(1 for r in inv if r["status"] == "present")
    lines: list[str] = []
    lines.append("# Stage 7.5 — Paper Artifact Summary\n")
    lines.append(
        "_This is the consolidated paper-side summary of Stage 1 → 7.4."
        " No new ops, no new attacks; pure aggregation of"
        " ``outputs/*.json``. **No real TEE wall-time, no formal /"
        " cryptographic / semantic security claims.**_\n"
    )

    lines.append("## 1. Artifact Inventory\n")
    lines.append(f"- Total artifacts surveyed: **{len(inv)}**")
    lines.append(f"- Present: **{present}**")
    lines.append(f"- Missing: **{len(missing)}**")
    lines.append(
        "- By slot: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_slot.items())
        )
    )
    lines.append("")
    lines.append("See `paper_results/markdown/artifact_inventory.md`.")
    lines.append("")

    lines.append("## 2. Correctness Summary\n")
    lines.append(
        f"- Rows: **{len(consolidation['correctness_summary'])}**"
    )
    lines.append(
        "See `paper_results/markdown/correctness_summary.md` /"
        " `paper_results/latex/correctness_summary.tex`."
    )
    lines.append("")

    lines.append("## 3. Security Proxy Summary\n")
    lines.append(
        f"- Rows: **{len(consolidation['security_proxy_summary'])}**"
    )
    risk_counts: dict[str, int] = {}
    for r in consolidation["security_proxy_summary"]:
        key = r.get("risk_level") or "n/a"
        risk_counts[key] = risk_counts.get(key, 0) + 1
    lines.append(
        "- Risk distribution: " + ", ".join(
            f"{k}={v}" for k, v in sorted(risk_counts.items())
        )
    )
    lines.append(
        "See `paper_results/markdown/security_proxy_summary.md` /"
        " `paper_results/latex/security_proxy_summary.tex`."
    )
    lines.append("")

    lines.append("## 4. Runtime Summary (local emulation only)\n")
    lines.append(
        "**This is local emulation, NOT real TEE wall-time.** No"
        " real sleep, no real runtime gating."
    )
    for r in runtime["rows"]:
        if r.get("mean_ms") is None:
            lines.append(
                f"- `{r['component']}` — _skipped_:"
                f" {r.get('skipped_with_reason')}"
            )
        else:
            lines.append(
                f"- `{r['component']}` ({r['variant']}): mean ="
                f" **{r['mean_ms']:.3f} ms**, median ="
                f" {r['median_ms']:.3f} ms, std ="
                f" {r['std_ms']:.3f} ms, repeats={r['num_repeats']}."
            )
    lines.append(
        "See `paper_results/markdown/measured_runtime.md` /"
        " `paper_results/latex/measured_runtime.tex`."
    )
    lines.append("")

    lines.append("## 5. LoRA Training Summary\n")
    lines.append(
        f"- Rows: **{len(consolidation['lora_training_summary'])}**"
    )
    lines.append(
        "See `paper_results/markdown/lora_training_summary.md`."
    )
    lines.append("")

    lines.append("## 6. Limitations\n")
    lines.append(
        f"- Aggregated limitation rows:"
        f" **{len(consolidation['limitations_summary'])}**"
    )
    lines.append(
        "See `paper_results/markdown/limitations_summary.md`."
    )
    lines.append(
        "Recurring themes: no formal / cryptographic / semantic"
        " security; no real TEE wall-time; padded_rank still visible;"
        " optimizer / loss remain trusted; no PEFT integration; no"
        " hardware side-channel evaluation; no full Qwen / TinyLlama"
        " LoRA fine-tuning."
    )
    lines.append("")

    lines.append("## 7. Claims Audit\n")
    lines.append(
        "- Status counts: " + ", ".join(
            f"{k}={v}" for k, v in
            sorted(audit["counts_by_status"].items())
        )
    )
    lines.append(
        "See `paper_results/markdown/paper_claims_audit.md` /"
        " `paper_results/latex/paper_claims_audit.tex`."
    )
    lines.append("")

    lines.append("## 8. Missing Artifacts\n")
    if not missing:
        lines.append("_All registered artifacts are present._")
    else:
        for r in missing:
            lines.append(
                f"- `{r['artifact_path']}` ({r['status']}) — slot={r['slot']}"
            )
    lines.append("")

    lines.append("## 9. Next Paper-Writing Plan\n")
    lines.append(
        "- Draft the system-model + threat-model sections using"
        " `paper_claims_audit.md` (unsupported claims must NOT appear"
        " as guarantees)."
    )
    lines.append(
        "- Draft the correctness theorem section as empirical"
        " verification over the per-stage `correctness_summary` rows;"
        " do NOT make formal proof claims."
    )
    lines.append(
        "- Draft the security evaluation section from"
        " `security_proxy_summary` + `lora_training_timing_proxy.json`."
        " Always word as 'proxy-evaluated', never 'secure'."
    )
    lines.append(
        "- Draft the experiment section by interleaving the consolidated"
        " CSV / Markdown / LaTeX tables + the `paper_results/figures/`"
        " PDFs / PNGs; keep the 'this is local emulation' disclaimer in"
        " the runtime caption."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_results_dir
    outputs_dir = args.outputs_dir

    consolidation = run_paper_artifact_consolidation(
        PaperArtifactConsolidationConfig(
            outputs_dir=str(outputs_dir),
            paper_results_dir=str(paper_dir),
            strict=args.strict,
        )
    )
    runtime = run_measured_runtime_evaluation(
        MeasuredRuntimeEvaluationConfig(
            output_dir=str(paper_dir),
            num_warmup=args.num_warmup,
            num_repeats=args.num_repeats,
            device=args.device,
            dtype=args.dtype,
            strict=args.strict,
        )
    )
    audit = run_paper_claims_audit(
        PaperClaimsAuditConfig(
            paper_results_dir=str(paper_dir),
            outputs_dir=str(outputs_dir),
        )
    )

    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _fig_correctness_errors(consolidation, figures_dir / "correctness_error_summary.png")
    _fig_security_risk_matrix(consolidation, figures_dir / "security_risk_matrix.png")
    _fig_boundary_call_reduction(consolidation, figures_dir / "boundary_call_reduction.png")
    _fig_lora_training_errors(consolidation, figures_dir / "lora_training_errors.png")
    _fig_rank_inference_risk(consolidation, figures_dir / "rank_inference_risk.png")
    _fig_timing_proxy_before_after(figures_dir / "timing_proxy_before_after.png", outputs_dir)
    _fig_measured_runtime(runtime, figures_dir / "measured_runtime_summary.png")

    summary_md = _summary_md(consolidation, runtime, audit)
    (paper_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    print("Wrote", paper_dir / "summary.md")
    print(
        f"consolidation_rows="
        f"correctness={len(consolidation['correctness_summary'])},"
        f"security={len(consolidation['security_proxy_summary'])},"
        f"workload={len(consolidation['workload_summary'])},"
        f"lora_training={len(consolidation['lora_training_summary'])},"
        f"limitations={len(consolidation['limitations_summary'])}"
    )
    print(
        f"runtime_measured={sum(1 for r in runtime['rows'] if r.get('mean_ms') is not None)}"
        f" runtime_skipped={sum(1 for r in runtime['rows'] if r.get('mean_ms') is None)}"
    )
    print(f"audit_counts={audit['counts_by_status']}")


if __name__ == "__main__":
    main()
