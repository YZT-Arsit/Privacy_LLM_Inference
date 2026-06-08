"""Runner for the Stage 7.6g RoPE-safe low-interaction correctness
experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.modern_decoder_rope_safe_low_interaction import (  # noqa: E402
    RopeSafeLowInteractionConfig,
    run_rope_safe_low_interaction_correctness,
    write_reports,
)


def main() -> None:
    cfg = RopeSafeLowInteractionConfig()
    report = run_rope_safe_low_interaction_correctness(cfg=cfg)
    json_path, md_path = write_reports(report, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print()
    print(
        "status={status} rope_mask_mode={rope_mask_mode!r} "
        "rope_transient_plain_qk_visible={rope_transient_plain_qk_visible} "
        "rope_transient_plain_v_visible={rope_transient_plain_v_visible} "
        "qkv_projection_outputs_masked_directly={qkv_projection_outputs_masked_directly} "
        "trusted_fallback_used_in_main_path={trusted_fallback_used_in_main_path} "
        "intermediate_tee_reentry={intermediate_tee_reentry} "
        "online_boundary_round_trips_per_decode_step={online_boundary_round_trips_per_decode_step} "
        "use_pad={use_pad}".format(**report)
    )
    corr = report["correctness"]
    print(
        "greedy_token_match_rate={r} sequence_exact_match={m} "
        "rope_commutation_max={c} qk_score_invariant_max={q} "
        "kv_cache_invariant_max={k}".format(
            r=corr["greedy_token_match_rate"],
            m=corr["sequence_exact_match"],
            c=corr["rope_commutation_max_abs_error"],
            q=corr["qk_score_invariant_max_abs_error"],
            k=corr["kv_cache_invariant_max_abs_error"],
        )
    )
    leak = report["rope_pair_norm_leakage_audit"]
    print(
        "rope_pair_norm_leakage={l} rope_pair_norm_max_abs_error={a} "
        "rope_commutation_audit_err={b}".format(
            l=leak["rope_pair_norm_leakage"],
            a=leak["rope_pair_norm_max_abs_error"],
            b=leak["rope_commutation_max_abs_error_audit"],
        )
    )


if __name__ == "__main__":
    main()
