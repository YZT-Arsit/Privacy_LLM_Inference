"""Runner for the Stage 7.6h norm-mask granularity experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.norm_granularity_low_interaction import (  # noqa: E402
    NormGranularityConfig,
    run_norm_granularity_low_interaction,
    write_reports,
)


def main() -> None:
    cfg = NormGranularityConfig()
    report = run_norm_granularity_low_interaction(cfg=cfg)
    json_path, md_path = write_reports(report, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"status={report['status']} modes={report['modes_evaluated']}")
    for mode in report["modes_evaluated"]:
        m = report["per_mode_results"][mode]
        d = m["diagnostics"]
        print(
            f"  [{mode:8s}] match={m['sequence_exact_match']} "
            f"chunk_size={m['chunk_size']} "
            f"q_per_row={d['norm_q_is_per_row']} "
            f"lm_head_max={d['lm_head_recovery_max_abs_error']:.2e} "
            f"h_hat_max={d['h_hat_layer_entry_invariant_max_abs_error']:.2e}"
        )
    print()
    print("Norm + Gram leakage audit (layer-entry boundary):")
    leak = report["norm_and_gram_leakage_audit"]
    for mode in ("sequence", "chunk", "token"):
        m = leak[mode]
        print(
            f"  [{mode:8s}] row_norm={m['row_norm_error']:.2e} "
            f"full_gram={m['full_gram_error']:.2e} "
            f"offdiag_gram={m['off_diagonal_gram_error']:.2e} "
            f"within_chunk={m['within_chunk_gram_error']:.2e} "
            f"cross_chunk={m['cross_chunk_gram_error']:.2e}"
        )
    print(
        f"  different_prompt_gram_distance="
        f"{leak['different_prompt_gram_distance']:.4f}"
    )


if __name__ == "__main__":
    main()
