"""Runner for Stage 7.8c generation processor coverage."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.generation_processor_coverage import (  # noqa: E402
    GenerationProcessorCoverageConfig,
    run_generation_processor_coverage,
    write_reports,
)


def main() -> None:
    rep = run_generation_processor_coverage(
        cfg=GenerationProcessorCoverageConfig()
    )
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(
        f"status={rep['status']} "
        f"recovery_err={rep['logit_recovery_max_abs_error']:.2e}"
    )
    for k, v in rep["processors"].items():
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
