"""Runner for Stage 7.7d multi-session batching simulation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.multi_session_batching import (  # noqa: E402
    MultiSessionBatchingConfig,
    run_multi_session_batching,
    write_reports,
)


def main() -> None:
    cfg = MultiSessionBatchingConfig()
    rep = run_multi_session_batching(cfg=cfg)
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(f"status={rep['status']}")
    for r in rep["per_session_results"]:
        print(
            f"  sid={r['session_id']} greedy={r['greedy_token_match_rate']} "
            f"lm_max={r['lm_head_recovery_max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    main()
