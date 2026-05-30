#!/usr/bin/env python
"""Stage 5.1 norm primitive experiments — sweep + JSON/CSV/Markdown emitter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    RMSNormOrthogonalProbeConfig,
    TrustedNormProbeConfig,
    run_rmsnorm_orthogonal_probe,
    run_trusted_norm_probe,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


SWEEP_TRUSTED = {
    "norm_type": ("layernorm", "rmsnorm"),
    "batch_size": (1, 2),
    "seq_len": (4, 8),
    "hidden_size": (64, 128),
    "use_pad": (True, False),
}

SWEEP_ORTHOGONAL_HIDDEN = (64, 128)


CSV_FIELDS = (
    "experiment",
    "norm_type",
    "batch_size",
    "seq_len",
    "hidden_size",
    "use_pad",
    "weight_present",
    "bias_present",
    "max_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "allclose",
    "y_tilde_invariant_max_abs_error",
    "reference_max_abs_error",
    "orthogonality_error",
    "rms_preservation_error",
    "normalized_state_error",
    "scalar_gamma_max",
    "vector_gamma_max",
    "allclose_without_gamma",
    "allclose_with_scalar_gamma",
    "allclose_with_vector_gamma",
)


def _trusted_row(r: dict) -> dict:
    cfg = r["config"]
    m = r["metrics"]
    yti = r["y_tilde_invariant_metrics"]
    ref = r["reference_metrics"]
    return {
        "experiment": "trusted_norm",
        "norm_type": cfg["norm_type"],
        "batch_size": cfg["batch_size"],
        "seq_len": cfg["seq_len"],
        "hidden_size": cfg["hidden_size"],
        "use_pad": cfg["use_pad"],
        "weight_present": r["weight_present"],
        "bias_present": r["bias_present"],
        "max_abs_error": m["max_abs_error"],
        "relative_l2_error": m["relative_l2_error"],
        "cosine_similarity": m["cosine_similarity"],
        "allclose": m["allclose"],
        "y_tilde_invariant_max_abs_error": yti["max_abs_error"],
        "reference_max_abs_error": ref["max_abs_error"],
        "orthogonality_error": None,
        "rms_preservation_error": None,
        "normalized_state_error": None,
        "scalar_gamma_max": None,
        "vector_gamma_max": None,
        "allclose_without_gamma": None,
        "allclose_with_scalar_gamma": None,
        "allclose_with_vector_gamma": None,
    }


def _orthogonal_row(r: dict, hidden: int) -> dict:
    return {
        "experiment": "rmsnorm_orthogonal_probe",
        "norm_type": "rmsnorm",
        "batch_size": r["config"]["batch_size"],
        "seq_len": r["config"]["seq_len"],
        "hidden_size": hidden,
        "use_pad": None,
        "weight_present": None,
        "bias_present": None,
        "max_abs_error": r["normalized_state_error"],
        "relative_l2_error": None,
        "cosine_similarity": None,
        "allclose": r["allclose_without_gamma"],
        "y_tilde_invariant_max_abs_error": None,
        "reference_max_abs_error": None,
        "orthogonality_error": r["orthogonality_error"],
        "rms_preservation_error": r["rms_preservation_error"],
        "normalized_state_error": r["normalized_state_error"],
        "scalar_gamma_max": r["gamma_commutation_error"]["scalar_gamma_max"],
        "vector_gamma_max": r["gamma_commutation_error"]["vector_gamma_max"],
        "allclose_without_gamma": r["allclose_without_gamma"],
        "allclose_with_scalar_gamma": r["allclose_with_scalar_gamma"],
        "allclose_with_vector_gamma": r["allclose_with_vector_gamma"],
    }


def _build_markdown(payload: dict) -> str:
    out: list[str] = []
    out.append("# Privacy LLM Obfuscation — Norm Primitive (Stage 5.1)")
    out.append("")

    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Stage 5.1 validates a unified trusted norm primitive for both"
        " LayerNorm and RMSNorm under the project's right-multiply mask"
        " convention, plus a restricted feasibility probe for RMSNorm under"
        " orthogonal masks. The trusted primitive standardises the existing"
        " trusted-LayerNorm shortcut used in Stages 2–6.2 — it does not yet"
        " execute norm on the GPU side."
    )
    out.append("")

    out.append("## Why general right-mask does not commute with LayerNorm")
    out.append("")
    out.append(
        "General right masks do not commute with LayerNorm. For ``X N`` with a"
        " non-orthogonal invertible ``N``, the column-wise mean and variance"
        " change in ways that depend on the off-diagonal mixing in ``N``, so"
        " ``LayerNorm(X N) ≠ LayerNorm(X) N``. The Stage 5.1 trusted primitive"
        " therefore continues to recover plaintext ``X`` on the trusted side,"
        " run the actual norm in cleartext, and re-mask the output."
    )
    out.append("")

    # ---- Trusted Norm primitive correctness ----
    out.append("## Trusted norm primitive correctness")
    out.append("")
    headers = [
        "norm_type",
        "batch_size",
        "seq_len",
        "hidden_size",
        "use_pad",
        "max output err",
        "y_tilde invariant err",
        "reference err",
        "allclose",
    ]
    rows = []
    for r in payload["trusted_norm"]:
        cfg = r["config"]
        m = r["metrics"]
        rows.append(
            [
                cfg["norm_type"],
                cfg["batch_size"],
                cfg["seq_len"],
                cfg["hidden_size"],
                cfg["use_pad"],
                m["max_abs_error"],
                r["y_tilde_invariant_metrics"]["max_abs_error"],
                r["reference_metrics"]["max_abs_error"],
                m["allclose"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    # ---- Restricted RMSNorm orthogonal probe ----
    out.append("## Restricted RMSNorm orthogonal-mask feasibility")
    out.append("")
    out.append(
        "If ``N`` is an orthogonal matrix, ``rms(X N) = rms(X)`` and"
        " ``normalize(X N) = normalize(X) N``. This is a restricted"
        " feasibility result — orthogonal masks form a strict subset of the"
        " general invertible-right-mask family used by the project, and they"
        " carry a different security profile that Stage 5.3 would have to"
        " evaluate before any GPU-side RMSNorm protocol could land."
    )
    out.append("")
    headers = [
        "hidden_size",
        "orthogonality err",
        "rms preservation err",
        "normalized state err",
        "scalar gamma err",
        "vector gamma err",
        "allclose w/o gamma",
        "allclose scalar gamma",
        "allclose vector gamma",
    ]
    rows = []
    for hidden, r in payload["rmsnorm_orthogonal"].items():
        rows.append(
            [
                hidden,
                r["orthogonality_error"],
                r["rms_preservation_error"],
                r["normalized_state_error"],
                r["gamma_commutation_error"]["scalar_gamma_max"],
                r["gamma_commutation_error"]["vector_gamma_max"],
                r["allclose_without_gamma"],
                r["allclose_with_scalar_gamma"],
                r["allclose_with_vector_gamma"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    # ---- Gamma commutation analysis ----
    out.append("## Gamma commutation analysis")
    out.append("")
    out.append(
        "Vector gamma breaks simple right-mask commutation. RMSNorm scales the"
        " normalised hidden state element-wise by ``gamma ∈ R^H``. For a"
        " scalar ``gamma`` (a single broadcast value) the right multiply by"
        " ``N`` commutes; for a vector ``gamma`` the mapping ``g ⊙ Z``"
        " applied before ``Z N`` mixes channels differently than after, so"
        " ``gamma ⊙ (Z N) ≠ (gamma ⊙ Z) N`` in general. The probe's"
        " ``vector_gamma_max`` column quantifies this gap — the report"
        " calls vector-gamma RMSNorm out as the dominant blocker for any"
        " GPU-side RMSNorm protocol on production checkpoints (LLaMA / T5 /"
        " Qwen all ship per-channel gamma)."
    )
    out.append("")

    # ---- Limitations ----
    out.append("## Limitations")
    out.append("")
    out.append(
        "- TrustedNormPrimitive still runs norm in the trusted side."
    )
    out.append(
        "- It standardizes the current trusted shortcut but does not eliminate"
        " trusted compute."
    )
    out.append(
        "- General right masks do not commute with LayerNorm."
    )
    out.append(
        "- General right masks do not commute with RMSNorm unless"
        " norm-preserving restrictions (orthogonal masks) are imposed."
    )
    out.append(
        "- Vector gamma breaks simple right-mask commutation in RMSNorm."
    )
    out.append("- This stage does not implement GELU / activation obfuscation.")
    out.append("- This stage does not implement real TEE.")
    out.append("- This stage does not claim formal security.")
    out.append("")

    # ---- Next stage plan ----
    out.append("## Next stage plan")
    out.append("")
    out.append(
        "- **Stage 5.2** — Activation primitive feasibility (GELU / SwiGLU /"
        " ReLU). Mirrors this stage: trusted primitive wrapper first,"
        " restricted-mask feasibility probe second."
    )
    out.append(
        "- **Stage 5.3** — Security proxy experiments for the orthogonal-mask"
        " restriction (what does sampling N from O(H) leak vs sampling from"
        " GL(H))."
    )
    out.append(
        "- **Stage 6.4** — Qwen / ModelScope migration once a GPU-side norm"
        " primitive exists for the per-channel-gamma case."
    )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    # ---- 1. Trusted norm primitive sweep ----
    trusted_results: list[dict] = []
    for norm_type in SWEEP_TRUSTED["norm_type"]:
        for batch_size in SWEEP_TRUSTED["batch_size"]:
            for seq_len in SWEEP_TRUSTED["seq_len"]:
                for hidden_size in SWEEP_TRUSTED["hidden_size"]:
                    for use_pad in SWEEP_TRUSTED["use_pad"]:
                        cfg = TrustedNormProbeConfig(
                            norm_type=norm_type,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            hidden_size=hidden_size,
                            use_pad=use_pad,
                            dtype=args.dtype,
                            device=args.device,
                            seed=args.seed,
                        )
                        trusted_results.append(run_trusted_norm_probe(cfg))

    # ---- 2. Restricted RMSNorm orthogonal probe ----
    ortho_results: dict[int, dict] = {}
    for hidden in SWEEP_ORTHOGONAL_HIDDEN:
        cfg = RMSNormOrthogonalProbeConfig(
            hidden_size=hidden,
            num_trials=16,
            dtype=args.dtype,
            device=args.device,
            seed=args.seed,
        )
        ortho_results[hidden] = run_rmsnorm_orthogonal_probe(cfg)

    payload = {
        "trusted_norm": trusted_results,
        "rmsnorm_orthogonal": ortho_results,
    }

    out_dir: Path = args.output_dir
    write_json(out_dir / "norm_experiments.json", payload)

    csv_rows: list[dict] = []
    for r in trusted_results:
        csv_rows.append(_trusted_row(r))
    for hidden, r in ortho_results.items():
        csv_rows.append(_orthogonal_row(r, hidden))
    write_csv(out_dir / "norm_experiments.csv", csv_rows, CSV_FIELDS)
    write_text(out_dir / "norm_experiments.md", _build_markdown(payload))

    all_trusted_allclose = all(r["metrics"]["allclose"] for r in trusted_results)
    all_y_tilde_invariant = all(
        r["y_tilde_invariant_metrics"]["allclose"] for r in trusted_results
    )
    ortho_summary = {
        hidden: {
            "allclose_without_gamma": r["allclose_without_gamma"],
            "allclose_with_scalar_gamma": r["allclose_with_scalar_gamma"],
            "allclose_with_vector_gamma": r["allclose_with_vector_gamma"],
        }
        for hidden, r in ortho_results.items()
    }
    print(
        f"trusted_cells={len(trusted_results)} all_allclose={all_trusted_allclose}"
        f" y_tilde_invariant_allclose={all_y_tilde_invariant}"
        f" rmsnorm_orthogonal={ortho_summary}"
        f" output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
