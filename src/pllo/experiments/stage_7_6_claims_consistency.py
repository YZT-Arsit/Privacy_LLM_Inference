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
from collections import Counter
from dataclasses import dataclass
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
# Bounded report configuration
# ---------------------------------------------------------------------------


@dataclass
class ClaimsReportConfig:
    """Bounds for the claims-consistency report writers.

    By default the writers emit *compact* reports only: summary counts,
    top-offender files/terms, and a capped set of examples per category.
    The full per-occurrence list is **never** serialized unless
    ``write_full_occurrences`` is set, and even then it is capped at
    ``max_full_occurrences``. A hard ``max_report_mb`` guard prevents any
    multi-GB file from being written: if a candidate report would exceed
    the limit, a small summary report is written instead and
    ``report_size_guard_triggered`` is set.
    """

    output_path: str | None = None
    write_json: bool = True
    write_csv: bool = True
    write_md: bool = True
    write_full_occurrences: bool = False
    max_examples_per_category: int = 25
    max_examples_per_file: int = 25
    max_top_files: int = 50
    max_top_terms: int = 50
    max_full_occurrences: int = 100_000
    # float so callers/tests can request a sub-MB guard threshold.
    max_report_mb: float = 100


# ---------------------------------------------------------------------------
# Compact report
# ---------------------------------------------------------------------------


def build_compact_report(
    report: dict[str, Any], cfg: ClaimsReportConfig | None = None,
) -> dict[str, Any]:
    """Project a (possibly huge) in-memory ``report`` onto a bounded,
    serialization-safe structure: summary counts, top offenders, capped
    examples, and truncation flags. The full occurrence list is included
    only when ``cfg.write_full_occurrences`` is set (capped)."""
    cfg = cfg or ClaimsReportConfig()
    occurrences: list[dict[str, Any]] = report.get("occurrences", [])
    total = len(occurrences)

    categories: dict[str, int] = {}
    file_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    per_file_example_count: dict[tuple[str, str], int] = {}
    examples_truncated = False

    for occ in occurrences:
        cls = occ["classification"]
        categories[cls] = categories.get(cls, 0) + 1
        file_counter[occ["file"]] += 1
        term_counter[occ["phrase"]] += 1
        bucket = examples.setdefault(cls, [])
        file_key = (cls, occ["file"])
        seen_for_file = per_file_example_count.get(file_key, 0)
        if (
            len(bucket) < cfg.max_examples_per_category
            and seen_for_file < cfg.max_examples_per_file
        ):
            bucket.append({
                "path": occ["file"],
                "line": occ["line"],
                "term": occ["phrase"],
                "context": occ.get("snippet", ""),
            })
            per_file_example_count[file_key] = seen_for_file + 1
        else:
            examples_truncated = True

    unsafe = int(categories.get("unsafe_wording_present", 0))
    allowed = int(categories.get("listed_as_unsafe_wording_to_avoid", 0))

    top_files = [
        {"path": p, "count": int(c)}
        for p, c in file_counter.most_common(cfg.max_top_files)
    ]
    top_terms = [
        {"term": t, "count": int(c)}
        for t, c in term_counter.most_common(cfg.max_top_terms)
    ]

    compact: dict[str, Any] = {
        "stage": "7.6_claims_consistency",
        "report": report.get("report", "stage_7_6_claims_consistency"),
        "status": report.get("status", "ok"),
        "passes_consistency_check": report.get(
            "passes_consistency_check", unsafe == 0),
        "formal_security_claim": report.get("formal_security_claim", False),
        "tracked_phrases": list(report.get("tracked_phrases", [])),
        "summary": {
            "total_files_scanned": report.get(
                "files_scanned_count", len(report.get("files_scanned", []))),
            "total_occurrences": total,
            "unsafe_occurrences": unsafe,
            "allowed_occurrences": allowed,
            "categories": categories,
            "summary_by_phrase": report.get("summary_by_phrase", {}),
        },
        "top_offender_files": top_files,
        "top_offender_terms": top_terms,
        "examples_by_category": examples,
        "truncation": {
            "examples_truncated": examples_truncated,
            "full_occurrences_included": bool(cfg.write_full_occurrences),
            "full_occurrences_truncated": False,
            "max_examples_per_category": cfg.max_examples_per_category,
            "max_examples_per_file": cfg.max_examples_per_file,
            "max_top_files": cfg.max_top_files,
            "max_top_terms": cfg.max_top_terms,
            "max_full_occurrences": cfg.max_full_occurrences,
        },
        "report_size_guard_triggered": False,
        "paper_safe_wording": report.get("paper_safe_wording", ""),
        "honesty_phrases": list(report.get("honesty_phrases", [])),
        "limitations": list(report.get("limitations", [])),
    }

    if cfg.write_full_occurrences:
        capped = occurrences[: cfg.max_full_occurrences]
        compact["full_occurrences"] = capped
        if total > cfg.max_full_occurrences:
            compact["truncation"]["full_occurrences_truncated"] = True

    return compact


def _minimal_guard_report(
    compact: dict[str, Any], *, candidate_bytes: int, max_bytes: int,
) -> dict[str, Any]:
    """A tiny report written when the candidate would exceed the size
    guard: summary counts only, no examples or full occurrences."""
    return {
        "stage": "7.6_claims_consistency",
        "report": compact.get("report", "stage_7_6_claims_consistency"),
        "status": compact.get("status", "ok"),
        "passes_consistency_check": compact.get(
            "passes_consistency_check", False),
        "formal_security_claim": compact.get("formal_security_claim", False),
        "summary": compact.get("summary", {}),
        "top_offender_files": [],
        "top_offender_terms": [],
        "examples_by_category": {},
        "truncation": {
            **compact.get("truncation", {}),
            "examples_truncated": True,
            "full_occurrences_included": False,
        },
        "report_size_guard_triggered": True,
        "report_size_guard": {
            "candidate_bytes": int(candidate_bytes),
            "max_bytes": int(max_bytes),
            "message": (
                "Candidate report exceeded max_report_mb; wrote summary "
                "only. Re-run with a larger --max-report-mb to emit the "
                "full report."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _json_str(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _write_compact_csv(compact: dict[str, Any], path: str) -> None:
    """Aggregate CSV: one row per summary key / top offender / capped
    example -- never one row per occurrence."""
    fields = ["section", "key", "count", "detail"]
    summary = compact.get("summary", {})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in (
            "total_files_scanned", "total_occurrences",
            "unsafe_occurrences", "allowed_occurrences",
        ):
            w.writerow({"section": "summary", "key": key,
                        "count": summary.get(key, 0), "detail": ""})
        for cat, count in summary.get("categories", {}).items():
            w.writerow({"section": "category", "key": cat,
                        "count": count, "detail": ""})
        for tf in compact.get("top_offender_files", []):
            w.writerow({"section": "top_file", "key": tf["path"],
                        "count": tf["count"], "detail": ""})
        for tt in compact.get("top_offender_terms", []):
            w.writerow({"section": "top_term", "key": tt["term"],
                        "count": tt["count"], "detail": ""})
        for cat, exs in compact.get("examples_by_category", {}).items():
            for ex in exs:
                detail = (
                    f"{ex['path']}:{ex['line']} [{ex['term']}] "
                    f"{ex['context']}"
                )
                w.writerow({"section": "example", "key": cat,
                            "count": 1, "detail": detail})
        if compact.get("report_size_guard_triggered"):
            w.writerow({"section": "guard", "key": "report_size_guard_triggered",
                        "count": 1, "detail": "summary-only report written"})


def _write_occurrences_csv(
    report: dict[str, Any], path: str, max_rows: int,
) -> int:
    """Separate, capped, one-row-per-occurrence CSV. Only written when
    full occurrences are explicitly requested."""
    fields = ["file", "line", "phrase", "classification", "snippet"]
    written = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for occ in report.get("occurrences", [])[:max_rows]:
            w.writerow({k: occ.get(k, "") for k in fields})
            written += 1
    return written


def render_markdown(
    report: dict[str, Any], *,
    config: ClaimsReportConfig | None = None,
    compact: dict[str, Any] | None = None,
) -> str:
    """Bounded Markdown report: summary, top offenders, and capped
    examples only. Full per-occurrence detail is omitted by default."""
    cfg = config or ClaimsReportConfig()
    if compact is None:
        compact = build_compact_report(report, cfg)
    summary = compact["summary"]
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
        "to avoid. Full occurrence details are omitted by default to "
        "keep reports bounded; pass `--write-full-occurrences` for the "
        "capped per-occurrence dump."
    )
    w()
    if compact.get("report_size_guard_triggered"):
        w("> **Note:** the size guard triggered; this report contains "
          "summary counts only. Re-run with a larger `--max-report-mb`.")
        w()
    w("## 2. Tracked phrases (unsafe wording to avoid)")
    w()
    for phrase in compact.get("tracked_phrases", []):
        w(f"- `{phrase}`")
    w()
    w("## 3. Headline counts")
    w()
    w(f"- Files scanned: **{summary.get('total_files_scanned', 0)}**")
    w(f"- Total occurrences: **{summary.get('total_occurrences', 0)}**")
    w(
        f"- Unsafe-wording-present occurrences: "
        f"**{summary.get('unsafe_occurrences', 0)}**"
    )
    w(
        f"- Listed-as-unsafe-wording-to-avoid occurrences: "
        f"**{summary.get('allowed_occurrences', 0)}**"
    )
    w(
        f"- Passes consistency check: "
        f"**{compact.get('passes_consistency_check')}**"
    )
    w()
    w("## 4. Summary by phrase")
    w()
    w("| phrase | unsafe_wording_present | listed_as_unsafe_wording_to_avoid |")
    w("|---|---|---|")
    for phrase, counts in summary.get("summary_by_phrase", {}).items():
        w(
            f"| `{phrase}` | "
            f"{counts.get('unsafe_wording_present', 0)} | "
            f"{counts.get('listed_as_unsafe_wording_to_avoid', 0)} |"
        )
    w()
    w("## 5. Top offender files")
    w()
    top_files = compact.get("top_offender_files", [])
    if not top_files:
        w("(none)")
    else:
        w("| file | count |")
        w("|---|---|")
        for tf in top_files:
            w(f"| `{tf['path']}` | {tf['count']} |")
    w()
    w("## 6. Top offender terms")
    w()
    top_terms = compact.get("top_offender_terms", [])
    if not top_terms:
        w("(none)")
    else:
        w("| term | count |")
        w("|---|---|")
        for tt in top_terms:
            w(f"| `{tt['term']}` | {tt['count']} |")
    w()
    w(
        f"## 7. Examples (capped at {cfg.max_examples_per_category} "
        f"per category)"
    )
    w()
    examples = compact.get("examples_by_category", {})
    if not examples:
        w("(none)")
    else:
        if compact.get("truncation", {}).get("examples_truncated"):
            w("_Examples are truncated; counts above are exact._")
            w()
        w("| category | file | line | phrase | snippet |")
        w("|---|---|---|---|---|")
        for cat, exs in examples.items():
            for ex in exs:
                snippet = str(ex["context"]).replace("|", "\\|")
                w(
                    f"| {cat} | `{ex['path']}` | {ex['line']} | "
                    f"`{ex['term']}` | {snippet} |"
                )
    w()
    w("## 8. Limitations")
    w()
    for lim in compact.get("limitations", []):
        w(f"- {lim}")
    w()
    w("## 9. Honesty phrases (verbatim)")
    w()
    for phrase in compact.get("honesty_phrases", []):
        w(f"- {phrase}")
    w()
    w("## 10. Paper-safe wording")
    w()
    w(f"> {compact.get('paper_safe_wording', '')}")
    w()
    w(f"`formal_security_claim`: `{compact.get('formal_security_claim')}`")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *, outputs_dir: str = "outputs",
    json_filename: str = "stage_7_6_claims_consistency.json",
    csv_filename: str = "stage_7_6_claims_consistency.csv",
    md_filename: str = "stage_7_6_claims_consistency.md",
    config: ClaimsReportConfig | None = None,
) -> tuple[str, str, str]:
    """Write bounded claims-consistency reports.

    Default behavior is compact: no full occurrence list, capped
    examples, aggregate CSV. A hard ``max_report_mb`` guard prevents any
    multi-GB file from ever being written.
    """
    cfg = config or ClaimsReportConfig()
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, json_filename)
    csv_path = os.path.join(outputs_dir, csv_filename)
    md_path = os.path.join(outputs_dir, md_filename)

    compact = build_compact_report(report, cfg)

    # Size guard: estimate from the JSON serialization (the largest of
    # the three formats, since it can carry full_occurrences). If it
    # would exceed the cap, fall back to a summary-only report.
    json_text = _json_str(compact)
    max_bytes = int(cfg.max_report_mb * 1024 * 1024)
    candidate_bytes = len(json_text.encode("utf-8"))
    if candidate_bytes > max_bytes:
        compact = _minimal_guard_report(
            compact, candidate_bytes=candidate_bytes, max_bytes=max_bytes)
        json_text = _json_str(compact)

    guarded = bool(compact.get("report_size_guard_triggered"))

    if cfg.write_json:
        with open(json_path, "w", encoding="utf-8") as fh:
            fh.write(json_text)
    if cfg.write_csv:
        _write_compact_csv(compact, csv_path)
    if cfg.write_md:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(report, config=cfg, compact=compact))

    # Full per-occurrence dump only on explicit request, and never when
    # the size guard fired.
    if cfg.write_full_occurrences and not guarded:
        occ_csv_path = os.path.join(
            outputs_dir, csv_filename.replace(".csv", "_occurrences.csv"))
        _write_occurrences_csv(report, occ_csv_path, cfg.max_full_occurrences)

    return json_path, csv_path, md_path


__all__ = [
    "ClaimsReportConfig",
    "build_claims_consistency_report",
    "build_compact_report",
    "render_markdown",
    "scan_paths",
    "write_reports",
]
