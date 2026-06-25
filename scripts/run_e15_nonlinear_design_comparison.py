"""E15 dual-design comparison CLI.

Gathers ALL already-produced experiment report JSONs for each nonlinear design
(``current`` / ``trusted_shortcut``) and emits the five comparison tables plus a
conservative recommendation block. A recommendation is emitted only when BOTH
designs have complete evidence for the axis being compared (see
:mod:`pllo.experiments.nonlinear_design_comparison`).

Reports are grouped by backend preferring each report's own ``nonlinear_backend``
field; if absent, the report is attributed to the flag that supplied it.

Examples::

    python scripts/run_e15_nonlinear_design_comparison.py \\
        --current-json outputs/cur_decode.json \\
        --current-json outputs/cur_build.json \\
        --trusted-shortcut-json outputs/ts_decode.json \\
        --output-json outputs/e15.json --output-md outputs/e15.md

    # generic form
    python scripts/run_e15_nonlinear_design_comparison.py \\
        --backend-json current=outputs/cur_decode.json \\
        --backend-json trusted_shortcut=outputs/ts_decode.json \\
        --output-json outputs/e15.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_design_comparison import (  # noqa: E402
    build_comparison,
    load_json,
    render_md,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    normalize_nonlinear_backend,
    parse_nonlinear_backends,
)


def _add(grouped, backend, report, supplied_by):
    """Attribute a report to a backend, preferring its own nonlinear_backend."""
    if report is None:
        return
    canon = supplied_by
    own = report.get("nonlinear_backend") if isinstance(report, dict) else None
    if own:
        try:
            canon = normalize_nonlinear_backend(own)
        except Exception:                                    # noqa: BLE001
            canon = supplied_by
    grouped.setdefault(canon, []).append(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current-json", action="append", default=[],
                    help="result JSON for backend current (repeatable)")
    ap.add_argument("--trusted-shortcut-json", action="append", default=[],
                    help="result JSON for backend trusted_shortcut (repeatable)")
    ap.add_argument("--backend-json", action="append", default=[],
                    metavar="BACKEND=PATH",
                    help="generic BACKEND=PATH report (repeatable)")
    ap.add_argument("--nonlinear-backends", default="current,trusted_shortcut",
                    help="comma-separated designs to compare")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    backends = parse_nonlinear_backends(args.nonlinear_backends)

    grouped: dict = {}
    for p in args.current_json:
        _add(grouped, "current", load_json(p),
             normalize_nonlinear_backend("current"))
    for p in args.trusted_shortcut_json:
        _add(grouped, "trusted_shortcut", load_json(p),
             normalize_nonlinear_backend("trusted_shortcut"))
    for spec in args.backend_json:
        if "=" not in spec:
            print("ignoring malformed --backend-json %r (want BACKEND=PATH)"
                  % spec, file=sys.stderr)
            continue
        be, _, path = spec.partition("=")
        try:
            canon = normalize_nonlinear_backend(be)
        except Exception as exc:                             # noqa: BLE001
            print("ignoring --backend-json with unknown backend %r: %s"
                  % (be, exc), file=sys.stderr)
            continue
        _add(grouped, canon, load_json(path), canon)

    # ensure every compared backend has an entry (possibly empty)
    for b in backends:
        grouped.setdefault(b, [])

    report = build_comparison(grouped, backends=backends)

    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        om = Path(args.output_md)
        om.parent.mkdir(parents=True, exist_ok=True)
        om.write_text(render_md(report), encoding="utf-8")

    rec = report["recommendation"]
    print("=== E15 nonlinear design comparison ===")
    print("backends=%s" % report["backends"])
    print("reports=%s" % report["num_reports_by_backend"])
    print("recommendation_status=%s" % rec["recommendation_status"])
    print("final_recommendation=%s" % rec["final_recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
