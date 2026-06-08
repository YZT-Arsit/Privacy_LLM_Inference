"""Runner for Stage 7.7g paper claims audit v2."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.paper_claims_audit_v2 import (  # noqa: E402
    run_paper_claims_audit_v2,
    write_reports,
)


def main() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    rep = run_paper_claims_audit_v2(outputs_dir=outputs_dir)
    j, m = write_reports(rep, outputs_dir=outputs_dir)
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    s = rep["summary"]
    print(
        f"status={rep['status']} supported={s['supported']} "
        f"proxy={s['proxy_supported']} cost_only={s['cost_model_only']} "
        f"unsupported={s['unsupported']}"
    )


if __name__ == "__main__":
    main()
