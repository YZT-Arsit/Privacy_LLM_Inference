"""Stage 7.3 -- repository hygiene audit CLI (read-only; deletes nothing)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.repo_hygiene import (  # noqa: E402
    RepoHygieneConfig,
    render_audit_markdown,
    run_repo_hygiene_audit,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--output", default="outputs/repo_hygiene_audit.json")
    ap.add_argument("--warn-mb", type=int, default=10)
    ap.add_argument("--fail-mb", type=int, default=100)
    args = ap.parse_args()

    cfg = RepoHygieneConfig(repo_root=args.repo_root, warn_mb=args.warn_mb,
                            fail_mb=args.fail_mb)
    report = run_repo_hygiene_audit(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(render_audit_markdown(report), encoding="utf-8")

    s = report["summary"]
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"git_available={report['git_available']} "
          f"tracked_generated={s['tracked_generated_count']} "
          f"untracked_generated={s['untracked_generated_count']} "
          f"outputs_size_mb={s['outputs_total_size_mb']}")
    if s["tracked_generated_count"]:
        print("NOTE: tracked generated artifacts found — see "
              "manual_decision_needed (this stage deletes nothing).")


if __name__ == "__main__":
    main()
