"""Runner for Stage 5.7 — Permutation-Invariant Leakage Audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.permutation_invariant_leakage import (  # noqa: E402
    PermutationInvariantLeakageConfig,
    run_permutation_invariant_leakage,
    write_reports,
)


def main() -> None:
    cfg = PermutationInvariantLeakageConfig(
        output_dir=str(REPO_ROOT / "outputs"),
    )
    report = run_permutation_invariant_leakage(cfg)
    json_path, csv_path, md_path = write_reports(
        report, outputs_dir=str(REPO_ROOT / "outputs"),
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(
        f"status={report['status']} "
        f"formal_security_claim={report['formal_security_claim']}"
    )
    for bundle, info in report["per_bundle"].items():
        per_tensor = info["per_tensor"]
        labels = [
            audit.get("statistical_leakage_label")
            for tn, scope_map in per_tensor.items()
            for audit in scope_map.values()
            if "statistical_leakage_label" in audit
        ]
        print(
            f"  [{bundle:36s}] tensors_with_metrics={len(labels)}  "
            f"labels={sorted(set(labels))}"
        )


if __name__ == "__main__":
    main()
