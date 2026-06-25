"""E12 runner: latency / overhead baselines across deployment backends.

Loads whichever per-backend decode reports are provided, builds the latency
comparison table, and writes JSON + Markdown + CSV + LaTeX. Pure parsing -- no
torch / CUDA. Missing inputs are simply absent rows (graceful).

Example::

    python scripts/run_e12_latency_baselines.py \\
        --plaintext-h800-json outputs/plaintext_h800_decode.json \\
        --folded-h800-local-json outputs/qwen7b_folded_full_local_probe.json \\
        --folded-h800-remote-json outputs/qwen7b_folded_remote_decode_probe.json \\
        --tdx-attested-remote-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \\
        --output-json outputs/e12_latency_baselines.json \\
        --output-csv outputs/e12_latency_baselines.csv \\
        --output-md  outputs/e12_latency_baselines.md \\
        --output-tex outputs/e12_latency_baselines.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.latency_baselines import (  # noqa: E402
    BACKENDS,
    build_latency_table,
    load_json,
    render_csv,
    render_latex,
    render_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for b in BACKENDS:
        ap.add_argument("--%s-json" % b.replace("_", "-"), default=None,
                        dest=b)
    ap.add_argument("--output-json", default="outputs/e12_latency_baselines.json")
    ap.add_argument("--output-csv", default="outputs/e12_latency_baselines.csv")
    ap.add_argument("--output-md", default="outputs/e12_latency_baselines.md")
    ap.add_argument("--output-tex", default="outputs/e12_latency_baselines.tex")
    args = ap.parse_args()

    reports_by_backend = {}
    for b in BACKENDS:
        rep = load_json(getattr(args, b))
        if rep is not None:
            reports_by_backend[b] = rep

    table = build_latency_table(reports_by_backend)

    def _write(path, text):
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    _write(args.output_csv, render_csv(table))
    _write(args.output_md, render_md(table))
    _write(args.output_tex, render_latex(table))

    print("=== E12 latency baselines ===")
    print("backends_present: " + (", ".join(table["backends_present"])
                                  or "(none)"))
    for r in table["rows"]:
        print("%s: total_latency_s=%s per_token_s=%s overhead_vs_plaintext=%s"
              % (r["backend"], r["total_latency_s"], r["latency_per_token_s"],
                 r["overhead_vs_plaintext_h800"]))
    print("\nE12 LATENCY BASELINES WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
