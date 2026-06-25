"""E16 nonlinear-design ablation CLI.

Gathers already-produced per-design report JSONs (mirroring the E15 input style)
and emits the nonlinear ablation table (current vs trusted_shortcut) as JSON +
Markdown + CSV + LaTeX.

Examples::

    python scripts/run_e16_nonlinear_ablation_report.py \\
        --current-json outputs/cur_decode.json \\
        --current-json outputs/cur_build.json \\
        --trusted-shortcut-json outputs/ts_decode.json \\
        --output-json outputs/e16.json --output-md outputs/e16.md \\
        --output-csv outputs/e16.csv --output-tex outputs/e16.tex

    python scripts/run_e16_nonlinear_ablation_report.py \\
        --backend-json current=outputs/cur.json \\
        --backend-json trusted_shortcut=outputs/ts.json \\
        --output-json outputs/e16.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_ablation import (  # noqa: E402
    build_nonlinear_ablation,
    load_json,
    render_csv,
    render_latex,
    render_md,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    normalize_nonlinear_backend,
    parse_nonlinear_backends,
)


def _add(grouped, report, supplied_by):
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
    ap.add_argument("--current-json", action="append", default=[])
    ap.add_argument("--trusted-shortcut-json", action="append", default=[])
    ap.add_argument("--backend-json", action="append", default=[],
                    metavar="BACKEND=PATH")
    ap.add_argument("--nonlinear-backends", default="current,trusted_shortcut")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-tex", default=None)
    args = ap.parse_args()

    backends = parse_nonlinear_backends(args.nonlinear_backends)

    grouped: dict = {}
    for p in args.current_json:
        _add(grouped, load_json(p), normalize_nonlinear_backend("current"))
    for p in args.trusted_shortcut_json:
        _add(grouped, load_json(p),
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
        _add(grouped, load_json(path), canon)

    for be in backends:
        grouped.setdefault(be, [])

    report = build_nonlinear_ablation(grouped, backends=backends)

    def _write(path, text):
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    _write(args.output_json,
           json.dumps(report, indent=2, default=str))
    _write(args.output_md, render_md(report))
    _write(args.output_csv, render_csv(report))
    _write(args.output_tex, render_latex(report))

    print("=== E16 nonlinear ablation ===")
    print("backends=%s" % report["backends"])
    print("rows=%d" % len(report["rows"]))
    print("deltas_summary=%s" % report["deltas_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
