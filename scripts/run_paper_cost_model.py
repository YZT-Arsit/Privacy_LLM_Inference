"""Runner for Stage 7.7f paper cost model."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.paper_cost_model import (  # noqa: E402
    PaperCostModelConfig,
    run_paper_cost_model,
    write_reports,
)


def main() -> None:
    cfg = PaperCostModelConfig()
    rep = run_paper_cost_model(cfg=cfg)
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(f"status={rep['status']} modes={len(rep['modes_evaluated'])}")
    for m_name in rep["modes_evaluated"]:
        info = rep["real_config_estimates"][m_name]
        print(
            f"  [{m_name:55s}] rt={info['round_trips_per_decode_step']} "
            f"accel_ops={info['accelerator_compute_ops']:.2e}".replace(
                "e+", "e"
            )
        )


if __name__ == "__main__":
    main()
