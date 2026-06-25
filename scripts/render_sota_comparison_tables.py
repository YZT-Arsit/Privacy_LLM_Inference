"""Render the SOTA privacy-preserving-inference comparison table.

Loads ``baselines/privacy_inference_methods.yaml`` (a flat list of method
records) and emits the comparison as JSON, Markdown, CSV, and a LaTeX
``tabular``. Unknown values (YAML ``null``) render as ``?`` in Markdown/CSV and
``--`` in LaTeX; null means "not-claimed / unknown", NOT false.

No hard PyYAML dependency: we try to import ``yaml`` and fall back to a small
hand parser robust to this specific flat-list-of-scalars schema.

stdlib only (json / csv / pathlib). main() -> int.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_designs import (  # noqa: E402
    add_nonlinear_backend_arg,
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)

STAGE = "sota_comparison"

# Exact field set (order matters for rendered columns).
REQUIRED_FIELDS: List[str] = [
    "method",
    "paper",
    "year",
    "protects_input",
    "protects_logits",
    "protects_kv",
    "protects_lora",
    "requires_gpu_tee",
    "requires_mpc_fhe",
    "tee_holds_full_model",
    "runs_real_7b",
    "real_attestation",
    "reported_latency",
    "source_type",
    "notes",
]

_VALID_SOURCE_TYPES = {"cited", "reproduced", "estimated", "ours"}


# ---------------------------------------------------------------------------
# YAML loading (pyyaml if present, else a small hand parser for this schema)
# ---------------------------------------------------------------------------


def _scalar(token: str) -> Any:
    """Parse a YAML scalar for this flat schema: null/bool/int/quoted str."""
    t = token.strip()
    if t == "" or t in ("null", "~", "Null", "NULL"):
        return None
    if t in ("true", "True", "TRUE"):
        return True
    if t in ("false", "False", "FALSE"):
        return False
    if (len(t) >= 2) and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        return t[1:-1]
    # try int
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _hand_parse(text: str) -> List[Dict[str, Any]]:
    """Parse the specific ``methods:`` list-of-flat-mappings schema.

    Robust to comments (``# ...``), blank lines, quoted strings (including ``#``
    inside quotes), and the ``- key: value`` / ``  key: value`` indentation we
    use. Not a general YAML parser.
    """
    methods: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    in_methods = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_methods:
            if stripped.startswith("methods:"):
                in_methods = True
            continue
        # a new list item begins with "- "
        if stripped.startswith("- "):
            cur = {}
            methods.append(cur)
            stripped = stripped[2:].strip()
            if not stripped:
                continue
        if cur is None:
            # content before any "-" under methods: -- ignore
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key = key.strip()
        cur[key] = _scalar(_strip_inline_comment(value))
    return methods


def _strip_inline_comment(value: str) -> str:
    """Strip a trailing ``# comment`` not inside a quoted string."""
    s = value.strip()
    if not s:
        return s
    quote = None
    out_chars = []
    for ch in s:
        if quote is None and ch == "#":
            break
        if quote is None and ch in ("'", '"'):
            quote = ch
        elif quote is not None and ch == quote:
            quote = None
        out_chars.append(ch)
    return "".join(out_chars).strip()


def load_methods(path: Path) -> List[Dict[str, Any]]:
    """Load the methods list from a YAML file (pyyaml if available)."""
    if not path.exists():
        raise FileNotFoundError("methods yaml not found: %s" % path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict) and "methods" in data:
            rows = data["methods"]
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [dict(r) for r in (rows or [])]
    except ImportError:
        return _hand_parse(text)


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------


def validate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every row has exactly the required field set (fill missing with
    None, keep field order). Validate source_type."""
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        extra = set(row) - set(REQUIRED_FIELDS)
        if extra:
            raise ValueError(
                "row %d (%r) has unexpected fields: %s"
                % (i, row.get("method"), sorted(extra)))
        norm: Dict[str, Any] = {}
        for f in REQUIRED_FIELDS:
            norm[f] = row.get(f, None)
        st = norm.get("source_type")
        if st is not None and st not in _VALID_SOURCE_TYPES:
            raise ValueError(
                "row %d (%r) has invalid source_type %r (expected one of %s)"
                % (i, norm.get("method"), st, sorted(_VALID_SOURCE_TYPES)))
        out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell_text(value: Any, unknown: str) -> str:
    if value is None:
        return unknown
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = list(REQUIRED_FIELDS)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [_cell_text(row.get(f), "?") for f in headers]
        # escape pipes in free text
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(REQUIRED_FIELDS)
    for row in rows:
        writer.writerow([_cell_text(row.get(f), "?") for f in REQUIRED_FIELDS])
    return buf.getvalue()


def _tex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def render_latex(rows: List[Dict[str, Any]]) -> str:
    ncol = len(REQUIRED_FIELDS)
    colspec = "l" * ncol
    lines = ["\\begin{tabular}{%s}" % colspec, "\\hline"]
    header = " & ".join(_tex_escape(f.replace("_", " ")) for f in REQUIRED_FIELDS)
    lines.append(header + " \\\\")
    lines.append("\\hline")
    for row in rows:
        cells = [_tex_escape(_cell_text(row.get(f), "--")) for f in REQUIRED_FIELDS]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(path: Optional[str], content: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--methods-yaml",
        default=str(REPO_ROOT / "baselines" / "privacy_inference_methods.yaml"),
        help="path to the methods YAML (flat list).")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--output-tex", default=None)
    add_nonlinear_backend_arg(p, required=False, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()

    rows = validate_rows(load_methods(Path(args.methods_yaml)))

    report: Dict[str, Any] = {
        "stage": STAGE,
        "methods_yaml": str(args.methods_yaml),
        "fields": list(REQUIRED_FIELDS),
        "row_count": len(rows),
        "rows": rows,
        "unknown_render": {"markdown_csv": "?", "latex": "--",
                           "meaning": "null = not-claimed/unknown (not false)"},
    }

    if getattr(args, "nonlinear_backend", None):
        canon = normalize_nonlinear_backend(args.nonlinear_backend)
        report["nonlinear_backend"] = canon
        report.update(nonlinear_design_report_fields(canon))

    _write(args.output_json, json.dumps(report, indent=2, default=str))
    _write(args.output_md, render_markdown(rows))
    _write(args.output_csv, render_csv(rows))
    _write(args.output_tex, render_latex(rows))

    print("sota_comparison: %d rows from %s" % (len(rows), args.methods_yaml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
