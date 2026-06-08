"""Runner for the Stage 7.6f low-interaction operator-compatible correctness
experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.modern_decoder_low_interaction_correctness import (  # noqa: E402
    LowInteractionConfig,
    run_low_interaction_correctness,
    write_reports,
)


def main() -> None:
    cfg = LowInteractionConfig()
    report = run_low_interaction_correctness(cfg=cfg)
    json_path, md_path = write_reports(report, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print()
    print(
        "status={status} main_mode={main_mode} "
        "main_layer_invariant={main_layer_invariant!r} "
        "trusted_fallback_used_in_main_path={trusted_fallback_used_in_main_path} "
        "intermediate_tee_reentry={intermediate_tee_reentry} "
        "online_boundary_round_trips_per_decode_step={online_boundary_round_trips_per_decode_step} "
        "use_pad={use_pad}".format(**report)
    )
    corr = report["correctness"]
    print(
        "greedy_token_match_rate={r} sequence_exact_match={m} "
        "h_hat_invariant_max={i} prefill_max={p} decode_max={d}".format(
            r=corr["greedy_token_match_rate"],
            m=corr["sequence_exact_match"],
            i=corr["h_hat_layer_entry_invariant_max_abs_error"],
            p=corr["prefill_logits_max_abs_error"],
            d=corr["decode_step_logits_max_abs_error_max"],
        )
    )
    leak = report["norm_leakage_audit"]
    print(
        "row_norm_error={a} gram_matrix_error={b} "
        "same_prompt_fresh_Q_gram_linkability={c} "
        "different_prompt_gram_distance={d}".format(
            a=leak["row_norm_error"], b=leak["gram_matrix_error"],
            c=leak["same_prompt_fresh_Q_gram_linkability"],
            d=leak["different_prompt_gram_distance"],
        )
    )


if __name__ == "__main__":
    main()
