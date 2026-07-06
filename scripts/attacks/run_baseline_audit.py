"""Audit ObfuscaTune-Qwen + STIP-Qwen implementation correctness.

Verifies STIP Theorem 1 (transformed-weight forward recovers plaintext),
per-linear transforms, ObfuscaTune orthogonal/linear equivalences, boundary
labels, and flags the paper-spec gaps. Writes outputs/baseline_audit/*.json/.md.

Example:
    python scripts/attacks/run_baseline_audit.py
    python scripts/attacks/run_baseline_audit.py --model-name-or-path /path/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "outputs" / "baseline_audit"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        print(f"[dry-run] baseline audit model={args.model_name_or_path or 'tiny_random_qwen'} -> {OUT}")
        return

    import torch
    from pllo.attacks.baseline_audit import run_baseline_audits
    rep = run_baseline_audits(model_name_or_path=args.model_name_or_path, seed=args.seed,
                              dtype=torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "obfuscatune_audit.json").write_text(json.dumps(rep["obfuscatune"], indent=2, default=str))
    (OUT / "stip_audit.json").write_text(json.dumps(rep["stip"], indent=2, default=str))

    def md(title, r):
        lines = [f"# {title}", "", f"status: **{r['status']}**", ""]
        for k, v in r.get("correctness", {}).items():
            lines.append(f"- correctness.{k}: {v}")
        for k, v in (r.get("qwen_adaptation") or r.get("boundary_labels") or {}).items():
            lines.append(f"- {k}: {v}")
        for g in r.get("paper_spec_gaps_flagged", []):
            lines.append(f"- GAP: {g}")
        return "\n".join(lines) + "\n"

    (OUT / "obfuscatune_audit.md").write_text(md("ObfuscaTune-Qwen audit", rep["obfuscatune"]))
    (OUT / "stip_audit.md").write_text(md("STIP-Qwen audit", rep["stip"]))
    print(f"ObfuscaTune: {rep['obfuscatune']['status']}  STIP: {rep['stip']['status']}")
    print(f"STIP theorem1_err={rep['stip']['correctness']['theorem1_recovery_max_abs_error']:.2e} "
          f"holds={rep['stip']['correctness']['theorem1_holds']}")
    print(f"wrote {OUT}/obfuscatune_audit.json/.md + stip_audit.json/.md")


if __name__ == "__main__":
    main()
