"""Runner for Stage 5.8 -- Lookup Nonlinear Cost Proxy."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.lookup_nonlinear_cost_proxy import (  # noqa: E402
    LookupNonlinearCostProxyConfig,
    run_lookup_nonlinear_cost_proxy,
    write_reports,
)


def main() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    cfg = LookupNonlinearCostProxyConfig(
        batch_size=1,
        seq_len=128,
        intermediate_size=11008,
        bit_widths=(4, 6, 8),
        entry_bytes=2,
        num_layers=1,
        num_tables_policy="per_layer_shared",
        run_microbench=True,
        microbench_intermediate_size=1024,
        microbench_seq_len=128,
        repeats=10,
        seed=0,
    )
    report = run_lookup_nonlinear_cost_proxy(cfg)
    j, c, m = write_reports(report, outputs_dir=str(outputs_dir))
    print(f"Wrote: {j}")
    print(f"Wrote: {c}")
    print(f"Wrote: {m}")
    print(
        f"status={report['status']} "
        f"formal_security_claim={report['formal_security_claim']} "
        f"cryptographic_lookup_implemented="
        f"{report['cryptographic_lookup_implemented']}"
    )


if __name__ == "__main__":
    main()
