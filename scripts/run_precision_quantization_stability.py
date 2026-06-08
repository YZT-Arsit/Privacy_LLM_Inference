"""Runner for Stage 7.8b precision/quantization stability."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.precision_quantization_stability import (  # noqa: E402
    PrecisionStabilityConfig,
    run_precision_quantization_stability,
    write_reports,
)


def main() -> None:
    rep = run_precision_quantization_stability(cfg=PrecisionStabilityConfig())
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(f"status={rep['status']}")
    print("Orthogonal mask:")
    for r in rep["orthogonal_mask"]["per_precision"]:
        print(
            f"  [{r['precision_mode']:32s}] err={r['logits_max_abs_error_vs_float64_plain']:.2e} "
            f"greedy={r['greedy_token_match_rate']}"
        )
    print("Dense cond sweep (last row per cond):")
    for sweep in rep["condition_sweep"]:
        last = sweep["per_precision"][-1]
        print(
            f"  [cond={sweep['condition_number']:.1f} {last['precision_mode']:32s}] "
            f"err={last['logits_max_abs_error_vs_float64_plain']:.2e}"
        )


if __name__ == "__main__":
    main()
