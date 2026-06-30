"""Aggregate extended decode-length + context (seq-len) scaling per nonlinear design.

Reads the extended E3 sweeps:

* ``<e3-extended-dir>/e3_decode_len_scaling_<design>.json``  (sweep over
  max_new_tokens)
* ``<e3-context-dir>/e3_context_<design>_seq<N>.json``        (one per seq_len)

and renders a per-design summary as JSON / Markdown / CSV / LaTeX. Per-design
separation is preserved and NO security claim is made for ``trusted_shortcut``
(its registry security_status is ``not_formally_claimed``). Pure parsing -- no
model / GPU.

Example::

    python scripts/render_extended_latency_context.py \\
        --e3-extended-dir outputs/e3_extended --e3-context-dir outputs/e3_context \\
        --designs current,trusted_shortcut --seq-lens 128,256,512,1024 \\
        --output-json outputs/final/extended_latency_context_summary.json \\
        --output-md   outputs/final/extended_latency_context_summary.md \\
        --output-csv  outputs/final/extended_latency_context_summary.csv \\
        --output-tex  outputs/final/extended_latency_context_summary.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_METRIC_KEYS = ("latency_s", "latency_per_token_s", "tokens_per_s",
                "trusted_bytes", "gpu_bytes", "boundary_calls",
                "peak_gpu_memory_mb", "tokens_exact_match", "token_match_rate")


def _norm_design(name):
    try:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        return normalize_nonlinear_backend(name)
    except Exception:                                       # noqa: BLE001
        return str(name)


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def _rows(rep):
    if not isinstance(rep, dict):
        return []
    rows = rep.get("rows")
    return rows if isinstance(rows, list) and rows else [rep]


def _metrics(row):
    return {k: row.get(k) for k in _METRIC_KEYS if isinstance(row, dict)
            and k in row}


def build_summary(decode_len_by_design: Dict[str, Any],
                  context_by_design: Dict[str, Dict[int, Any]],
                  *, designs: List[str]) -> dict:
    """Build the per-design extended summary. Inputs are already-loaded report
    dicts: ``decode_len_by_design[design]`` is one decode-length sweep report;
    ``context_by_design[design][seq_len]`` is one per-seq_len report."""
    per_design = {}
    for d in designs:
        # decode-length scaling rows keyed by max_new_tokens
        dl = decode_len_by_design.get(d)
        dl_rows = []
        for row in _rows(dl) if dl else []:
            if isinstance(row, dict):
                m = _metrics(row)
                m["max_new_tokens"] = row.get("max_new_tokens")
                m["seq_len"] = row.get("seq_len")
                dl_rows.append(m)
        # context scaling: one report per seq_len
        ctx_rows = []
        for seq_len in sorted(context_by_design.get(d, {})):
            rep = context_by_design[d][seq_len]
            rs = _rows(rep)
            row = rs[-1] if rs else {}
            m = _metrics(row)
            m["seq_len"] = seq_len
            ctx_rows.append(m)
        per_design[d] = {
            "decode_length_scaling": dl_rows,
            "context_scaling": ctx_rows,
            "security_claim": (
                "none -- formal security not claimed for this design"
                if _norm_design(d) == "trusted_shortcut"
                else "boundary security as established for the current design"),
        }
    return {
        "stage": "extended_latency_context_summary",
        "designs": designs,
        "per_design": per_design,
        "limitations": [
            "latency/byte numbers are only as real as their input E3 sweeps "
            "(dry-run inputs are not paper-ready)",
            "NO formal security claim is made for trusted_shortcut "
            "(security_status=not_formally_claimed); this summary reports "
            "correctness/performance only",
            "per-design rows are kept separate; do not average across designs",
        ],
    }


def _flat_rows(summary):
    out = []
    for d, blk in summary["per_design"].items():
        for kind, rows in (("decode_length", blk["decode_length_scaling"]),
                           ("context", blk["context_scaling"])):
            for r in rows:
                row = {"design": d, "scaling": kind}
                row.update(r)
                out.append(row)
    return out


def render_md(summary) -> str:
    L = ["# Extended latency / context summary", "",
         "_designs: %s; per-design separation preserved_"
         % ", ".join(summary["designs"]), ""]
    for d, blk in summary["per_design"].items():
        L += ["## design `%s`" % d, "",
              "_security: %s_" % blk["security_claim"], "",
              "### decode-length scaling", "",
              "| max_new_tokens | latency_s | latency/token | trusted_bytes | "
              "gpu_bytes | boundary_calls |", "| --- | --- | --- | --- | --- | --- |"]
        for r in blk["decode_length_scaling"]:
            L.append("| %s | %s | %s | %s | %s | %s |"
                     % (r.get("max_new_tokens"), r.get("latency_s"),
                        r.get("latency_per_token_s"), r.get("trusted_bytes"),
                        r.get("gpu_bytes"), r.get("boundary_calls")))
        L += ["", "### context (seq-len) scaling", "",
              "| seq_len | latency_s | latency/token | trusted_bytes | gpu_bytes "
              "| peak_gpu_memory_mb |", "| --- | --- | --- | --- | --- | --- |"]
        for r in blk["context_scaling"]:
            L.append("| %s | %s | %s | %s | %s | %s |"
                     % (r.get("seq_len"), r.get("latency_s"),
                        r.get("latency_per_token_s"), r.get("trusted_bytes"),
                        r.get("gpu_bytes"), r.get("peak_gpu_memory_mb")))
        L.append("")
    L += ["## Limitations", ""] + ["- %s" % x for x in summary["limitations"]]
    L.append("")
    return "\n".join(L)


def render_csv(summary) -> str:
    import csv
    import io
    rows = _flat_rows(summary)
    cols = ["design", "scaling", "seq_len", "max_new_tokens", "latency_s",
            "latency_per_token_s", "tokens_per_s", "trusted_bytes", "gpu_bytes",
            "boundary_calls", "peak_gpu_memory_mb", "tokens_exact_match",
            "token_match_rate"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def render_latex(summary) -> str:
    L = ["% extended latency/context summary (per design; no security claim for "
         "trusted_shortcut)",
         "\\begin{tabular}{llrrrr}", "\\hline",
         "design & scaling & x & latency\\_s & lat/token & gpu\\_bytes \\\\",
         "\\hline"]
    for r in _flat_rows(summary):
        x = r.get("max_new_tokens") if r.get("scaling") == "decode_length" \
            else r.get("seq_len")
        L.append("%s & %s & %s & %s & %s & %s \\\\"
                 % (r.get("design"), r.get("scaling"), x, r.get("latency_s"),
                    r.get("latency_per_token_s"), r.get("gpu_bytes")))
    L += ["\\hline", "\\end{tabular}"]
    return "\n".join(L)


def discover(e3_extended_dir, e3_context_dir, designs, seq_lens):
    decode_len = {}
    context = {}
    ext = Path(e3_extended_dir)
    ctx = Path(e3_context_dir)
    for d in designs:
        p = ext / ("e3_decode_len_scaling_%s.json" % d)
        rep = _load(p) if p.is_file() else None
        if rep is not None:
            decode_len[d] = rep
        context[d] = {}
        for s in seq_lens:
            cp = ctx / ("e3_context_%s_seq%d.json" % (d, s))
            crep = _load(cp) if cp.is_file() else None
            if crep is not None:
                context[d][s] = crep
    return decode_len, context


def _split(s):
    return [x.strip() for x in str(s).split(",") if x.strip()] if s else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e3-extended-dir", default="outputs/e3_extended")
    ap.add_argument("--e3-context-dir", default="outputs/e3_context")
    ap.add_argument("--designs", default="A_rightmul,amulet_secure_R")
    ap.add_argument("--seq-lens", default="128,256,512,1024")
    ap.add_argument("--output-json",
                    default="outputs/final/extended_latency_context_summary.json")
    ap.add_argument("--output-md",
                    default="outputs/final/extended_latency_context_summary.md")
    ap.add_argument("--output-csv",
                    default="outputs/final/extended_latency_context_summary.csv")
    ap.add_argument("--output-tex",
                    default="outputs/final/extended_latency_context_summary.tex")
    args = ap.parse_args()

    designs = [_norm_design(d) for d in _split(args.designs)]
    seq_lens = [int(x) for x in _split(args.seq_lens)]
    decode_len, context = discover(args.e3_extended_dir, args.e3_context_dir,
                                   designs, seq_lens)
    summary = build_summary(decode_len, context, designs=designs)

    for path, render in ((args.output_json, None), (args.output_md, render_md),
                         (args.output_csv, render_csv),
                         (args.output_tex, render_latex)):
        if not path:
            continue
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if render is None:
            p.write_text(json.dumps(summary, indent=2, default=str),
                         encoding="utf-8")
        else:
            p.write_text(render(summary), encoding="utf-8")

    print("=== extended latency/context summary ===")
    for d in designs:
        blk = summary["per_design"].get(d, {})
        print("  %s: decode_len rows=%d context rows=%d"
              % (d, len(blk.get("decode_length_scaling", [])),
                 len(blk.get("context_scaling", []))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
