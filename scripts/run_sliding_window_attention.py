"""Runner for Stage 7.8a sliding window attention."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.sliding_window_attention import (  # noqa: E402
    SlidingWindowConfig,
    run_sliding_window_attention,
    write_reports,
)


def main() -> None:
    rep = run_sliding_window_attention(cfg=SlidingWindowConfig())
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(
        f"status={rep['status']} greedy={rep['greedy_token_match_rate']} "
        f"seq_exact={rep['sequence_exact_match']}"
    )
    for r in rep["per_window_results"]:
        print(
            f"  [w={r['window_size']!s:>4} {r['attention_privacy_mode']:30s}] "
            f"score_inv={r['score_invariant_max_abs_error']:.2e} "
            f"kv_inv={r['kv_window_invariant_max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    main()
