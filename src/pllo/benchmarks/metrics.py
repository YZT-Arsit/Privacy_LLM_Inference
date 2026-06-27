"""Pure-Python benchmark metrics (no sklearn / rouge / numpy required).

All metrics are deterministic and operate on plain Python lists/strings so they
can run in the test sandbox with no heavy deps. ``compute_metric`` is the
dispatcher used by the runner.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Sequence

__all__ = ["accuracy", "exact_match", "extract_numeric_answer",
           "numeric_exact_match", "numeric_values_equal", "macro_f1", "rouge_l",
           "rouge_l_corpus", "token_match_rate", "compute_metric"]

# a single number token: optional sign, digits with optional thousands commas,
# optional decimal part. Used both for the GSM8K ``####`` marker and as fallback.
_NUM = r"[+-]?\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
# GSM8K final-answer marker: ``#### 3`` / ``#### 1,234`` / ``#### -2`` / ``#### 3.5``
# (allow whitespace and an optional leading ``$`` between the marker and number).
_MARKER_RE = re.compile(r"####\s*\$?\s*(" + _NUM + r")")


def _norm(x) -> str:
    return str(x).strip().lower()


def accuracy(preds: Sequence, golds: Sequence) -> float:
    """Fraction of positions where ``pred`` equals ``gold`` (case-insensitive)."""
    n = min(len(preds), len(golds))
    if n == 0:
        return 0.0
    hits = sum(1 for i in range(n) if _norm(preds[i]) == _norm(golds[i]))
    return hits / n


def exact_match(preds: Sequence, golds: Sequence) -> float:
    """Alias of :func:`accuracy` on normalized string equality."""
    return accuracy(preds, golds)


def _canon_number(raw: str) -> str:
    """Normalize a matched number token: strip thousands commas / ``$`` and
    canonicalize via :class:`~decimal.Decimal` (``3.0`` -> ``3``, ``3.50`` ->
    ``3.5``). Falls back to the cleaned raw string if it is not a valid number."""
    cleaned = raw.replace(",", "").replace("$", "").strip()
    try:
        d = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if d == d.to_integral_value():
        return str(d.to_integral_value())          # integer -> no decimal point
    return str(d.normalize())                      # strip trailing zeros


def extract_numeric_answer(text) -> Optional[str]:
    """Extract the canonical numeric answer from ``text`` (GSM8K-style).

    Resolution order:
      1. the number attached to the FINAL ``####`` marker (the official GSM8K
         final-answer convention) -- so trailing explanation numbers never
         override the marked answer;
      2. otherwise the LAST number in the text.

    The result is normalized (thousands commas / ``$`` stripped, compared as a
    :class:`~decimal.Decimal`). Returns ``None`` when there is no number.
    """
    if text is None:
        return None
    s = str(text)
    markers = _MARKER_RE.findall(s)
    if markers:
        return _canon_number(markers[-1])          # final #### marker wins
    matches = _NUM_RE.findall(s)
    if not matches:
        return None
    return _canon_number(matches[-1])              # fallback: last number


def numeric_values_equal(a, b) -> bool:
    """True iff two extracted numeric-answer strings are numerically equal
    (Decimal comparison, so ``"3"`` == ``"3.0"``). Falls back to string equality
    for non-numeric tokens."""
    if a is None or b is None:
        return False
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except InvalidOperation:
        return str(a) == str(b)


def numeric_exact_match(preds: Sequence, golds: Sequence) -> float:
    """Accuracy after extracting the numeric answer from each pred/gold."""
    n = min(len(preds), len(golds))
    if n == 0:
        return 0.0
    hits = 0
    for i in range(n):
        p = extract_numeric_answer(preds[i])
        g = extract_numeric_answer(golds[i])
        if numeric_values_equal(p, g):
            hits += 1
    return hits / n


def macro_f1(preds: Sequence, golds: Sequence, labels: Sequence) -> float:
    """Macro-averaged F1 over ``labels`` (per-label precision/recall/F1)."""
    if not labels:
        return 0.0
    n = min(len(preds), len(golds))
    pn = [_norm(p) for p in preds[:n]]
    gn = [_norm(g) for g in golds[:n]]
    f1s: List[float] = []
    for lab in labels:
        l = _norm(lab)
        tp = sum(1 for i in range(n) if pn[i] == l and gn[i] == l)
        fp = sum(1 for i in range(n) if pn[i] == l and gn[i] != l)
        fn = sum(1 for i in range(n) if pn[i] != l and gn[i] == l)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[len(b)]


def rouge_l(pred: str, ref: str) -> float:
    """LCS-based ROUGE-L F1 (token-level, whitespace split, lowercased)."""
    p = _norm(pred).split()
    r = _norm(ref).split()
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l_corpus(preds: Sequence, refs: Sequence) -> float:
    """Mean per-example ROUGE-L F1 over the corpus."""
    n = min(len(preds), len(refs))
    if n == 0:
        return 0.0
    return sum(rouge_l(preds[i], refs[i]) for i in range(n)) / n


def token_match_rate(a: List, b: List) -> float:
    """Position-wise match rate over the first ``min(len(a), len(b))`` tokens."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def compute_metric(metric_name: str, preds: Sequence, golds: Sequence,
                   **kw) -> float:
    """Dispatch to a metric by name.

    Supported: accuracy, exact_match, numeric_exact_match, macro_f1, rouge_l.
    ``macro_f1`` requires a ``labels`` keyword argument.
    """
    if metric_name == "accuracy":
        return accuracy(preds, golds)
    if metric_name == "exact_match":
        return exact_match(preds, golds)
    if metric_name == "numeric_exact_match":
        return numeric_exact_match(preds, golds)
    if metric_name == "macro_f1":
        return macro_f1(preds, golds, kw.get("labels") or [])
    if metric_name == "rouge_l":
        return rouge_l_corpus(preds, golds)
    raise ValueError("unknown metric: %r" % (metric_name,))
