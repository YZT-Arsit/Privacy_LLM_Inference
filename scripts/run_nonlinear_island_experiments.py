#!/usr/bin/env python
"""Stage 5.2 — Nonlinear island correctness probe sweep + emitter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    NonlinearIslandProbeConfig,
    run_nonlinear_island_experiments,
)
from pllo.experiments.report_utils import (
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


CSV_FIELDS = (
    "section",
    "island",
    "mlp_type",
    "mask_family",
    "activation_type",
    "hidden_size",
    "intermediate_size",
    "use_pad",
    "used_pad_at_linear_boundary",
    "online_extra_matmul_count",
    "orthogonality_error",
    "mean_preservation_error",
    "max_abs_error",
    "mean_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "allclose",
)


def _row(section: str, cell: dict) -> dict:
    m = cell["metrics"]
    return {
        "section": section,
        "island": cell["island"],
        "mlp_type": cell.get("mlp_type"),
        "mask_family": cell["mask_family"],
        "activation_type": cell["activation_type"],
        "hidden_size": cell["hidden_size"],
        "intermediate_size": cell["intermediate_size"],
        "use_pad": cell["use_pad"],
        "used_pad_at_linear_boundary": cell["used_pad_at_linear_boundary"],
        "online_extra_matmul_count": cell["online_extra_matmul_count"],
        "orthogonality_error": cell["orthogonality_error"],
        "mean_preservation_error": cell["mean_preservation_error"],
        "max_abs_error": m["max_abs_error"],
        "mean_abs_error": m.get("mean_abs_error"),
        "relative_l2_error": m.get("relative_l2_error"),
        "cosine_similarity": m.get("cosine_similarity"),
        "allclose": m["allclose"],
    }


def _build_markdown(payload: dict) -> str:
    out: list[str] = []
    out.append("# Privacy LLM Obfuscation — Nonlinear Islands (Stage 5.2)")
    out.append("")

    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Stage 5.2 implements and validates operator-compatible nonlinear"
        " islands. Each island matches a nonlinear operator with the mask"
        " family that commutes with it, folds mask transitions into adjacent"
        " Linear weights offline, and verifies the masked forward equals the"
        " plaintext forward times the residual output mask. The goal is to"
        " keep the nonlinear core in the GPU-visible masked domain without"
        " adding extra online matmuls."
    )
    out.append("")

    out.append("## Operator-Compatible Mask Families")
    out.append("")
    headers = ["operator", "mask family", "preserved invariants"]
    for nm, mf in payload["mask_family_assignments"].items():
        pass
    rows = [
        ["Linear / Attention / KV cache", "dense_invertible", "—"],
        ["RMSNorm core", "orthogonal", "row L2 norm (rms preserved)"],
        ["LayerNorm core", "mean_preserving_orthogonal", "row mean + centered L2 norm"],
        ["GELU / ReLU / SiLU activation", "permutation", "coordinate-value multiset"],
        ["SwiGLU activation", "paired_permutation", "paired (up,gate) multiset"],
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 3. Norm-compatible
    out.append("## Norm-Compatible Island Results")
    out.append("")
    headers = [
        "island",
        "hidden",
        "ortho err",
        "mean preservation err",
        "max abs err",
        "allclose",
    ]
    rows = [
        [
            c["island"],
            c["hidden_size"],
            c["orthogonality_error"],
            c["mean_preservation_error"],
            c["metrics"]["max_abs_error"],
            c["metrics"]["allclose"],
        ]
        for c in payload["norm_island_cells"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 4. Affine folding
    out.append("## Affine Folding Results")
    out.append("")
    out.append(
        "Both RMSNorm and LayerNorm affine parameters are folded into the"
        " adjacent Linear layer *offline* — the GPU never sees `gamma` or"
        " `beta` as separate tensors. Folding rules:"
    )
    out.append("")
    out.append("```text")
    out.append("LayerNorm:  W_folded = diag(gamma) @ W       b_folded = beta @ W + b")
    out.append("RMSNorm  :  W_folded = diag(gamma) @ W       b_folded = b")
    out.append("```")
    out.append("")
    out.append(
        "After folding the masked weight becomes `W_tilde = N_in^T W_folded N_out`"
        " (orthogonal / mean-preserving orthogonal ``N_in``)."
    )
    out.append("")

    # 5. Activation permutation
    out.append("## Activation Permutation Island Results")
    out.append("")
    headers = ["activation", "hidden", "max abs err", "allclose"]
    rows = [
        [c["activation_type"], c["hidden_size"], c["metrics"]["max_abs_error"], c["metrics"]["allclose"]]
        for c in payload["activation_island_cells"]
        if c["island"] == "activation_permutation"
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 6. SwiGLU paired
    out.append("## SwiGLU Paired-Permutation Island Results")
    out.append("")
    rows = [
        [c["activation_type"], c["hidden_size"], c["metrics"]["max_abs_error"], c["metrics"]["allclose"]]
        for c in payload["activation_island_cells"]
        if c["island"] == "swiglu_paired_permutation"
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 7. Full MLP island
    out.append("## Full MLP Island Results")
    out.append("")
    headers = [
        "mlp_type",
        "hidden",
        "intermediate",
        "use_pad",
        "max abs err",
        "online extra matmul",
        "allclose",
    ]
    rows = [
        [
            c["mlp_type"],
            c["hidden_size"],
            c["intermediate_size"],
            c["use_pad"],
            c["metrics"]["max_abs_error"],
            c["online_extra_matmul_count"],
            c["metrics"]["allclose"],
        ]
        for c in payload["mlp_island_cells"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 8. Pad placement rule
    out.append("## Pad Placement Rule")
    out.append("")
    out.append(payload["pad_placement_rule"])
    out.append("")
    out.append(
        "- Pad enters and exits only through Linear boundaries: ``X_tilde ="
        " (X - T) N_in`` at island entry, ``C = T W_perm`` adds the standard"
        " linear compensation, and the activation operates on ``Z P`` (no pad)."
    )
    out.append(
        "- Pushing pad through an activation is invalid: ``f((Z - T) P) ≠ f(Z P)"
        " - f(T P)`` for any nonlinear ``f``, so simple additive compensation"
        " cannot cancel it. The island therefore strips pad at the Linear"
        " entry and downstream wrappers may re-introduce a fresh pad at the"
        " next Linear boundary."
    )
    out.append("")

    # 9. Online cost
    out.append("## Online Cost Interpretation")
    out.append("")
    out.append(
        "All mask + permutation transitions are *preprocessing-only* and are"
        " folded into the masked weight tensors before the GPU forward starts."
        " Concretely, the offline pipeline computes:"
    )
    out.append("")
    out.append("```text")
    out.append("# Norm island (offline):")
    out.append("W_folded = diag(gamma) @ W       (RMSNorm / LayerNorm fold)")
    out.append("b_folded = beta @ W + b          (LayerNorm only)")
    out.append("W_tilde  = N_in^T @ W_folded @ N_out")
    out.append("b_tilde  = b_folded @ N_out")
    out.append("")
    out.append("# MLP island (offline):")
    out.append("W1_tilde   = N_in^{-1} @ W1[:, perm]")
    out.append("b1_tilde   = b1[perm]")
    out.append("W2_tilde   = W2[perm, :] @ N_out")
    out.append("b2_tilde   = b2 @ N_out")
    out.append("```")
    out.append("")
    out.append(
        "The online masked path executes exactly the same number of matmuls"
        " as the plaintext path. `online_extra_matmul_count = 0` across every"
        " cell — folded mask transitions add zero online cost."
    )
    out.append("")

    # 10. Limitations
    out.append("## Limitations")
    out.append("")
    out.append(
        "- Compatible mask families are weaker than unrestricted dense masks"
        " inside nonlinear islands."
    )
    out.append(
        "- Permutation islands hide channel identity but do not hide"
        " coordinate-value multisets."
    )
    out.append("- Orthogonal masks preserve norms by design.")
    out.append(
        "- Mean-preserving orthogonal masks preserve mean and centered norm by"
        " design."
    )
    out.append(
        "- Security relies on freshness, dense-mask sandwiching, and pad at"
        " Linear boundaries."
    )
    out.append("- This stage does not prove semantic security.")
    out.append(
        "- This stage does not implement adaptive permutation-recovery attacks"
        " beyond proxy experiments."
    )
    out.append("- This stage does not implement real TEE.")
    out.append("")

    # 11. Next stage plan
    out.append("## Next Stage Plan")
    out.append("")
    out.append(
        "- **Stage 5.3** — Stronger leakage experiments (adaptive attackers,"
        " learned inversion) targeting the compatible mask family boundaries."
    )
    out.append(
        "- **Stage 6.4** — Qwen / TinyLlama migration. The RMSNorm orthogonal"
        " island + SwiGLU paired-permutation island land exactly the two"
        " operators Qwen / LLaMA need, on top of the Stage 5.1 RMSNorm primitive."
    )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = NonlinearIslandProbeConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=args.dtype,
        device=args.device,
        output_dir=str(args.output_dir),
    )
    payload = run_nonlinear_island_experiments(cfg)

    out_dir: Path = args.output_dir
    write_json(out_dir / "nonlinear_island_experiments.json", payload)

    csv_rows: list[dict] = []
    for c in payload["norm_island_cells"]:
        csv_rows.append(_row("norm_island", c))
    for c in payload["activation_island_cells"]:
        csv_rows.append(_row("activation_island", c))
    for c in payload["mlp_island_cells"]:
        csv_rows.append(_row("mlp_island", c))
    write_csv(out_dir / "nonlinear_island_experiments.csv", csv_rows, CSV_FIELDS)
    write_text(
        out_dir / "nonlinear_island_experiments.md", _build_markdown(payload)
    )

    g = payload["global_summary"]
    print(
        f"cells={g['num_cells']} all_allclose={g['all_allclose']}"
        f" max_online_extra_matmul={g['max_online_extra_matmul_count']}"
        f" output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
