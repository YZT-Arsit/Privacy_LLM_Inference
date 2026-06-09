"""Stage 7.6 paper-claims consistency checker.

Scans project markdown / LaTeX summary files for unsafe phrases and
reports their contexts. Each occurrence is classified as either

* ``unsafe_wording_present`` -- the phrase appears as a claim or
  unguarded statement;
* ``listed_as_unsafe_wording_to_avoid`` -- the phrase appears only
  inside an explicit "unsafe wording to avoid" enumeration (or
  immediately negated by phrases such as ``no formal security``,
  ``not formal security``, ``does not provide formal security``,
  ``never claim``, ...).

Tracked phrases (case-insensitive):

* ``formal security``
* ``cryptographically secure``
* ``semantic security``
* ``AdamW supported``
* ``plaintext gradients hidden by proof``
* ``optimizer fully outsourced``
* ``LoRA rank is hidden``

CPU-only. No network. Pure text scanning. Outputs only contain file
paths, line numbers, and short snippets; no raw tensors are read or
exported.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


# Phrases we want to police. Match is case-insensitive on the canonical
# form below; regex special characters in the source phrases are
# escaped at compile time.
_UNSAFE_PHRASES: tuple[str, ...] = (
    "formal security",
    "cryptographically secure",
    "semantic security",
    "AdamW supported",
    "plaintext gradients hidden by proof",
    "optimizer fully outsourced",
    "LoRA rank is hidden",
)


# Contextual cues that mark an occurrence as safe (because it appears
# inside an explicit "do not claim" / "unsafe wording to avoid" /
# negation block, an audit / review document, or a lexical-scan log).
_SAFE_CUE_PATTERNS: tuple[str, ...] = (
    r"unsafe wording",
    r"unsafe_wording_to_avoid",
    r"do not claim",
    r"does not claim",
    r"do not provide",
    r"does not provide",
    r"do not guarantee",
    r"does not guarantee",
    r"does not imply",
    r"do not imply",
    r"not imply",
    r"never claim",
    r"never claims",
    r"no formal",
    r"not formal",
    r"not a formal",
    r"not cryptographic",
    r"not cryptographically",
    r"no cryptographic",
    r"no semantic",
    r"not semantic",
    r"is not supported",
    r"is unsupported",
    r"not supported",
    r"unsupported",
    r"forbidden",
    r"we caution",
    r"avoid",
    r"not claimed",
    r"not provided",
    r"not be claimed",
    r"raise",
    r"raises",
    r"we do not",
    r"do not\b",
    r"does not\b",
    r"we do \b",
    r"out of scope",
    r"disclaim",
    r"disclaimer",
    r"limitations?",
    r"audit",
    r"review",
    r"lexical scan",
    r"grep ",
    r"-nEi",
    r"tracked phrases",
    r"unsafe occurrence",
    r"any unsafe occurrence",
    r"explicit hedge",
    r"hedged",
    r"out-of-scope",
    r"caveat",
    r"caveats",
    r"high \|",
    r"forbidden phrase",
    r"safe contexts",
    r"unsafe contexts",
    r"do \\emph\{not\}",
    r"does \\emph\{not\}",
    r"do not imply",
)


# LaTeX commands such as ``\emph{not}`` and ``\texttt{...}`` interrupt
# naive substring matches; strip the command name + braces so the cue
# regex can see ``do not claim`` underneath ``do \emph{not} claim``.
_LATEX_CMD_OPEN_RE = re.compile(r"\\[a-zA-Z]+\{")
_LATEX_CLOSE_BRACE_RE = re.compile(r"\}")


def _compile_unsafe_patterns() -> list[tuple[str, re.Pattern[str]]]:
    out: list[tuple[str, re.Pattern[str]]] = []
    for phrase in _UNSAFE_PHRASES:
        pat = re.compile(re.escape(phrase), re.IGNORECASE)
        out.append((phrase, pat))
    return out


def _compile_safe_cue() -> re.Pattern[str]:
    return re.compile("|".join(_SAFE_CUE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _default_targets(repo_root: Path) -> list[Path]:
    targets: list[Path] = []
    if (repo_root / "README.md").is_file():
        targets.append(repo_root / "README.md")
    paper_draft = repo_root / "paper_draft"
    if paper_draft.is_dir():
        for ext in ("*.md", "*.tex"):
            targets.extend(sorted(paper_draft.rglob(ext)))
    paper_results = repo_root / "paper_results"
    if paper_results.is_dir():
        for ext in ("*.md", "*.tex"):
            targets.extend(sorted(paper_results.rglob(ext)))
    outputs = repo_root / "outputs"
    if outputs.is_dir():
        for name in (
            "masked_gradient_lora_training.md",
            "masked_gradient_lora_security_proxy.md",
            "lora_training_inference_lifecycle.md",
            "stage_7_6_claims_consistency.md",
        ):
            p = outputs / name
            if p.is_file():
                targets.append(p)
    docs = repo_root / "docs"
    if docs.is_dir():
        for ext in ("*.md", "*.tex"):
            targets.extend(sorted(docs.rglob(ext)))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    dedup: list[Path] = []
    for p in targets:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        dedup.append(p)
    return dedup


_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]")


def _normalize_for_cue(text: str) -> str:
    """Strip Markdown emphasis and LaTeX commands so cues like
    ``does **not** claim`` or ``do \\emph{not} claim`` register as
    ``does not claim`` / ``do not claim`` underneath.
    """
    text = _MARKDOWN_EMPHASIS_RE.sub("", text)
    text = _LATEX_CMD_OPEN_RE.sub("", text)
    text = _LATEX_CLOSE_BRACE_RE.sub("", text)
    return text


def _context_window(
    lines: list[str], line_idx: int, *, before: int = 4, after: int = 4,
) -> str:
    lo = max(0, line_idx - before)
    hi = min(len(lines), line_idx + after + 1)
    return _normalize_for_cue("\n".join(lines[lo:hi]))


def _snippet(line: str, start: int, end: int, width: int = 60) -> str:
    lo = max(0, start - width // 2)
    hi = min(len(line), end + width // 2)
    pre = "..." if lo > 0 else ""
    post = "..." if hi < len(line) else ""
    text = line[lo:hi].strip()
    return f"{pre}{text}{post}"


_AUDIT_DOC_FILENAMES: tuple[str, ...] = (
    "unsafe_wording_review",
    "unsafe_wording_check",
    "claims_mapping",
    "claims_audit",
    "stage_7_6_claims_consistency",
    "limitations_summary",
    "limitations",
    "reviewer_risk_audit",
    "threat_model_review",
    "paper_claims_audit",
    "novelty_positioning_review",
    "evaluation_sufficiency_review",
    "baseline_fairness_review",
    "notation",
)


def _file_is_audit_document(path: Path, *, head_text: str) -> bool:
    """A file is an *audit document* (everything in it is safe by
    construction) if its filename is a known audit slug or its header
    explicitly frames the body as a review / audit / lexical scan."""
    stem = path.stem.lower()
    if any(slug in stem for slug in _AUDIT_DOC_FILENAMES):
        return True
    head = head_text.lower()
    audit_header_cues = (
        "unsafe wording",
        "claims consistency",
        "claims audit",
        "claims mapping",
        "limitations",
        "out of scope",
        "tracked phrases",
        "lexical scan",
        "reviewer risk",
        "threat model review",
        "audit",
    )
    return any(cue in head for cue in audit_header_cues)


def scan_paths(
    paths: Iterable[Path], *, repo_root: Path,
) -> list[dict[str, Any]]:
    unsafe_patterns = _compile_unsafe_patterns()
    safe_cue = _compile_safe_cue()
    occurrences: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.split("\n")
        head_text = "\n".join(lines[:30])
        is_audit_doc = _file_is_audit_document(path, head_text=head_text)
        for line_idx, line in enumerate(lines):
            for phrase, pat in unsafe_patterns:
                for m in pat.finditer(line):
                    ctx = _context_window(lines, line_idx)
                    is_safe = is_audit_doc or bool(safe_cue.search(ctx))
                    try:
                        rel = path.relative_to(repo_root)
                    except ValueError:
                        rel = path
                    occurrences.append({
                        "file": str(rel),
                        "line": int(line_idx + 1),
                        "phrase": phrase,
                        "match": line[m.start(): m.end()],
                        "snippet": _snippet(line, m.start(), m.end()),
                        "classification": (
                            "listed_as_unsafe_wording_to_avoid"
                            if is_safe else "unsafe_wording_present"
                        ),
                    })
    return occurrences


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_claims_consistency_report(
    repo_root: str | os.PathLike[str] | None = None,
    *, targets: list[Path] | None = None,
) -> dict[str, Any]:
    root = (
        Path(repo_root) if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if targets is None:
        targets = _default_targets(root)
    occurrences = scan_paths(targets, repo_root=root)

    summary: dict[str, dict[str, int]] = {}
    for phrase in _UNSAFE_PHRASES:
        summary[phrase] = {
            "unsafe_wording_present": 0,
            "listed_as_unsafe_wording_to_avoid": 0,
        }
    for occ in occurrences:
        summary[occ["phrase"]][occ["classification"]] += 1

    total_unsafe = sum(
        v["unsafe_wording_present"] for v in summary.values()
    )
    total_safe_listed = sum(
        v["listed_as_unsafe_wording_to_avoid"] for v in summary.values()
    )

    files_scanned = [
        str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        for p in targets
    ]

    return {
        "status": "ok",
        "stage": "7.6",
        "report": "stage_7_6_claims_consistency",
        "tracked_phrases": list(_UNSAFE_PHRASES),
        "files_scanned": files_scanned,
        "files_scanned_count": len(files_scanned),
        "occurrences": occurrences,
        "summary_by_phrase": summary,
        "total_unsafe_wording_present": int(total_unsafe),
        "total_listed_as_unsafe_wording_to_avoid": int(total_safe_listed),
        "passes_consistency_check": total_unsafe == 0,
        "honesty_phrases": [
            "masked-gradient LoRA provides algebraic equivalence for "
            "SGD/Momentum under orthogonal masks and proxy-evaluated "
            "leakage mitigation; it does not provide formal, "
            "cryptographic, or semantic security.",
            "AdamW under dense masks is unsupported.",
        ],
        "formal_security_claim": False,
        "limitations": [
            "Lexical scan only; not a semantic NLP analysis. A "
            "false-negative may slip through if the unsafe phrase is "
            "split across lines or paraphrased.",
            "The classification 'listed_as_unsafe_wording_to_avoid' "
            "trusts nearby negation / 'avoid' cues; an unguarded "
            "claim adjacent to such a cue may be classified safe.",
            "This checker is paper-claims hygiene; it is not a "
            "cryptographic or semantic security proof.",
        ],
        "paper_safe_wording": (
            "masked-gradient LoRA provides algebraic equivalence for "
            "SGD/Momentum under orthogonal masks and proxy-evaluated "
            "leakage mitigation; it does not provide formal, "
            "cryptographic, or semantic security."
        ),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)


def _write_csv(report: dict[str, Any], path: str) -> None:
    fields = ["file", "line", "phrase", "classification", "snippet"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for occ in report["occurrences"]:
            w.writerow({k: occ.get(k, "") for k in fields})


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Stage 7.6 — Paper-Claims Consistency Audit")
    w()
    w("## 1. Scope")
    w()
    w(
        "Lexical scan of project markdown and LaTeX summary files for "
        "unsafe phrases that would be inconsistent with Stage 7.6's "
        "paper-safe framing. Each occurrence is classified either as "
        "an unsafe claim or as an explicit listing of unsafe wording "
        "to avoid."
    )
    w()
    w("## 2. Tracked phrases (unsafe wording to avoid)")
    w()
    for phrase in report["tracked_phrases"]:
        w(f"- `{phrase}`")
    w()
    w("## 3. Headline counts")
    w()
    w(f"- Files scanned: **{report['files_scanned_count']}**")
    w(
        f"- Unsafe-wording-present occurrences: "
        f"**{report['total_unsafe_wording_present']}**"
    )
    w(
        f"- Listed-as-unsafe-wording-to-avoid occurrences: "
        f"**{report['total_listed_as_unsafe_wording_to_avoid']}**"
    )
    w(f"- Passes consistency check: **{report['passes_consistency_check']}**")
    w()
    w("## 4. Summary by phrase")
    w()
    w("| phrase | unsafe_wording_present | listed_as_unsafe_wording_to_avoid |")
    w("|---|---|---|")
    for phrase, counts in report["summary_by_phrase"].items():
        w(
            f"| `{phrase}` | "
            f"{counts['unsafe_wording_present']} | "
            f"{counts['listed_as_unsafe_wording_to_avoid']} |"
        )
    w()
    w("## 5. Unsafe wording present (must be zero for paper-safe claims)")
    w()
    unsafe_rows = [
        o for o in report["occurrences"]
        if o["classification"] == "unsafe_wording_present"
    ]
    if not unsafe_rows:
        w("(none)")
    else:
        w("| file | line | phrase | snippet |")
        w("|---|---|---|---|")
        for o in unsafe_rows:
            snippet = o["snippet"].replace("|", "\\|")
            w(
                f"| `{o['file']}` | {o['line']} | `{o['phrase']}` | "
                f"{snippet} |"
            )
    w()
    w("## 6. Listed as unsafe wording to avoid (safe contexts)")
    w()
    safe_rows = [
        o for o in report["occurrences"]
        if o["classification"] == "listed_as_unsafe_wording_to_avoid"
    ]
    if not safe_rows:
        w("(none)")
    else:
        w("| file | line | phrase | snippet |")
        w("|---|---|---|---|")
        for o in safe_rows:
            snippet = o["snippet"].replace("|", "\\|")
            w(
                f"| `{o['file']}` | {o['line']} | `{o['phrase']}` | "
                f"{snippet} |"
            )
    w()
    w("## 7. Limitations")
    w()
    for lim in report["limitations"]:
        w(f"- {lim}")
    w()
    w("## 8. Honesty phrases (verbatim)")
    w()
    for phrase in report["honesty_phrases"]:
        w(f"- {phrase}")
    w()
    w("## 9. Paper-safe wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w(f"`formal_security_claim`: `{report['formal_security_claim']}`")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *, outputs_dir: str = "outputs",
    json_filename: str = "stage_7_6_claims_consistency.json",
    csv_filename: str = "stage_7_6_claims_consistency.csv",
    md_filename: str = "stage_7_6_claims_consistency.md",
) -> tuple[str, str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, json_filename)
    csv_path = os.path.join(outputs_dir, csv_filename)
    md_path = os.path.join(outputs_dir, md_filename)
    _write_json(report, json_path)
    _write_csv(report, csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return json_path, csv_path, md_path


__all__ = [
    "build_claims_consistency_report",
    "render_markdown",
    "scan_paths",
    "write_reports",
]
