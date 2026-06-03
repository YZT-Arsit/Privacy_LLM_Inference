"""Stage 7.6c runner: emit the reviewer-risk audit + revision plan.

Reads paper_draft/latex/sections + paper_results/markdown + docs/
runtime_boundary.md (read-only). Writes the audit artifacts under
paper_draft/. Does NOT touch outputs/ or paper_results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.reviewer_risk_audit import (  # noqa: E402
    REVIEWER_QAS,
    RISK_ITEMS,
    run_audit,
)


def main() -> None:
    result = run_audit(write=True)
    severity_counts = {}
    priority_counts = {}
    for it in result.items:
        severity_counts[it.severity] = severity_counts.get(it.severity, 0) + 1
        priority_counts[it.priority] = priority_counts.get(it.priority, 0) + 1

    wording_counts = {"safe": 0, "risky": 0, "unsafe": 0}
    for h in result.wording_hits:
        wording_counts[h.classification] = wording_counts.get(h.classification, 0) + 1

    qa_status = {"answered": 0, "partial": 0, "missing": 0}
    for q in result.qas:
        qa_status[q.answer_status] = qa_status.get(q.answer_status, 0) + 1

    print("Wrote:")
    for key, path in sorted(result.output_paths.items()):
        print(f"  {key}: {path}")
    print()
    print(f"risk_items={len(result.items)} "
          f"severity={severity_counts} priority={priority_counts}")
    print(f"reviewer_qas={len(result.qas)} status={qa_status}")
    print(f"wording_hits={len(result.wording_hits)} counts={wording_counts}")
    print(f"outputs_modified=False paper_results_modified=False")


if __name__ == "__main__":
    main()
