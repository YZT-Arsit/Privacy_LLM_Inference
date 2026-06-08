"""Runner for the Stage 7.6i attention-privacy-modes experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.attention_privacy_modes import (  # noqa: E402
    AttentionPrivacyModesConfig,
    run_attention_privacy_modes,
    write_reports,
)


def main() -> None:
    cfg = AttentionPrivacyModesConfig()
    report = run_attention_privacy_modes(cfg=cfg)
    json_path, md_path = write_reports(report, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"status={report['status']} modes={report['modes_evaluated']}")
    for mode in report["modes_evaluated"]:
        c = report["comparison"][mode]
        d = report["per_mode_results"][mode]["diagnostics"]
        print(
            f"  [{mode:32s}] "
            f"exact={c['exact']} "
            f"one_rt={c['one_round_trip']} "
            f"hidden={c['attention_hidden']} "
            f"rt/step={c['online_boundary_round_trips_per_decode_step']} "
            f"greedy={c['greedy_token_match_rate']} "
            f"lm_max={d['lm_head_recovery_max_abs_error']:.2e} "
            f"rowconst_err={d['row_constant_blinding_softmax_max_abs_error']:.2e} "
            f"nonconst_err={d['nonconstant_blinding_softmax_max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    main()
