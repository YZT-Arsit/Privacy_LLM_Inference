"""Runner for the Stage 7.7 paper experiment suite aggregator."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.paper_experiment_suite import (  # noqa: E402
    PaperExperimentSuiteConfig,
    run_paper_experiment_suite,
    write_reports,
)


def main() -> None:
    cfg = PaperExperimentSuiteConfig(outputs_dir=REPO_ROOT / "outputs")
    rep = run_paper_experiment_suite(cfg=cfg)
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    s = rep["paper_claims_summary"]
    print(
        f"status={rep['status']} supported={s['supported']} "
        f"proxy={s['proxy_supported']} unsupported={s['unsupported']}"
    )
    for key, info in rep["stages"].items():
        print(f"  [{key:55s}] status={info.get('status', 'n/a')}")


if __name__ == "__main__":
    main()
