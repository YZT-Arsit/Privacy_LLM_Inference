"""Runner for Stage 7.8d decoder-only component coverage audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.decoder_component_coverage_audit import (  # noqa: E402
    run_decoder_component_coverage_audit,
    write_reports,
)


def main() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    rep = run_decoder_component_coverage_audit(outputs_dir=outputs_dir)
    j, m = write_reports(rep, outputs_dir=outputs_dir)
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    s = rep["summary"]
    print(
        f"status={rep['status']} covered={s['covered']} "
        f"partial={s['partial']} unsupported={s['unsupported']}"
    )


if __name__ == "__main__":
    main()
