"""AAAI generation preservation metrics: plaintext-GPU vs ours (A_rightmul).

Thin extension of :mod:`pllo.benchmarks.generation_preservation` (reusing its
tested ``levenshtein`` / exact-match primitives) with the extra metrics the AAAI
validator needs: char-level normalized EDIT DISTANCE (not similarity), length
ratio, LCS ROUGE-L, char-trigram chrF, finish-reason match, plus per-example
comparison + aggregation + divergence case studies.

The AAAI claim is that obfuscated greedy decoding reproduces plaintext greedy
decoding deterministically, so preservation (ideally exact) is the headline; any
divergence is surfaced for the appendix.

stdlib only. No torch.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from pllo.benchmarks.generation_preservation import (
    exact_text_match, exact_token_match, levenshtein)

__all__ = [
    "normalized_edit_distance",
    "length_ratio",
    "rouge_l",
    "chrf",
    "finish_reason_match",
    "token_exact_match_rate",
    "compare_responses",
    "aggregate_preservation",
    "case_studies",
]


def normalized_edit_distance(a: str | None, b: str | None) -> float:
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def length_ratio(plaintext: str | None, ours: str | None) -> float | None:
    p = len(plaintext or "")
    return None if p == 0 else len(ours or "") / p


def token_exact_match_rate(toks_a: Sequence[int] | None,
                           toks_b: Sequence[int] | None) -> float | None:
    if not toks_a or not toks_b:
        return None
    n = min(len(toks_a), len(toks_b))
    if n == 0:
        return 0.0
    eq = sum(1 for i in range(n) if toks_a[i] == toks_b[i])
    return eq / max(len(toks_a), len(toks_b))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(plaintext: str | None, ours: str | None) -> float:
    pa, pb = (plaintext or "").split(), (ours or "").split()
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    lcs = _lcs_len(pa, pb)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(pb), lcs / len(pa)
    return 2 * prec * rec / (prec + rec)


def _char_ngrams(s: str, n: int) -> list[str]:
    s = s or ""
    if len(s) >= n:
        return [s[i:i + n] for i in range(len(s) - n + 1)]
    return [s] if s else []


def chrf(plaintext: str | None, ours: str | None, n: int = 3) -> float:
    ref, hyp = _char_ngrams(plaintext or "", n), _char_ngrams(ours or "", n)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(hyp), overlap / len(ref)
    return 2 * prec * rec / (prec + rec)


def finish_reason_match(a: str | None, b: str | None) -> bool | None:
    if a is None and b is None:
        return None
    return a == b


def _field(x, key):
    return None if isinstance(x, str) else (x or {}).get(key)


def _resp(x):
    return x if isinstance(x, str) else (x or {}).get("response", "")


def compare_responses(plaintext: dict[str, Any] | str,
                      ours: dict[str, Any] | str) -> dict[str, Any]:
    """Per-example preservation metrics. Each arg is a response record
    (``{response, token_ids, finish_reason}``) or a raw string."""
    pt, ot = _resp(plaintext), _resp(ours)
    btok, ctok = _field(plaintext, "token_ids"), _field(ours, "token_ids")
    lr = length_ratio(pt, ot)
    return {
        "exact_response_match": exact_text_match(pt, ot),
        "exact_token_match": exact_token_match(btok, ctok),
        "token_exact_match_rate": token_exact_match_rate(btok, ctok),
        "normalized_edit_distance": round(normalized_edit_distance(pt, ot), 6),
        "length_ratio": round(lr, 6) if lr is not None else None,
        "rouge_l": round(rouge_l(pt, ot), 6),
        "chrf": round(chrf(pt, ot), 6),
        "finish_reason_match": finish_reason_match(
            _field(plaintext, "finish_reason"), _field(ours, "finish_reason")),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else None


def aggregate_preservation(per_example: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_example)
    if n == 0:
        return {"num_compared": 0}
    return {
        "num_compared": n,
        "exact_response_match_rate":
            round(sum(1 for e in per_example
                      if e.get("exact_response_match")) / n, 6),
        "exact_token_match_rate":
            _mean([1.0 if e.get("exact_token_match") else 0.0 for e in per_example
                   if e.get("exact_token_match") is not None]),
        "mean_token_exact_match_rate":
            _mean([e.get("token_exact_match_rate") for e in per_example]),
        "mean_normalized_edit_distance":
            _mean([e.get("normalized_edit_distance") for e in per_example]),
        "mean_length_ratio": _mean([e.get("length_ratio") for e in per_example]),
        "mean_rouge_l": _mean([e.get("rouge_l") for e in per_example]),
        "mean_chrf": _mean([e.get("chrf") for e in per_example]),
        "finish_reason_match_rate":
            _mean([1.0 if e.get("finish_reason_match") else 0.0
                   for e in per_example
                   if e.get("finish_reason_match") is not None]),
    }


def case_studies(plaintext_by_id: dict[str, Any], ours_by_id: dict[str, Any], *,
                 max_cases: int = 20) -> list[dict[str, Any]]:
    """Divergent examples (plaintext != ours), worst edit distance first."""
    cases = []
    for rid in sorted(set(plaintext_by_id) & set(ours_by_id)):
        p, o = plaintext_by_id[rid], ours_by_id[rid]
        cmp = compare_responses(p, o)
        if not cmp["exact_response_match"]:
            cases.append({"id": rid, "plaintext_response": _resp(p),
                          "ours_response": _resp(o), **cmp})
    cases.sort(key=lambda c: c["normalized_edit_distance"], reverse=True)
    return cases[:max_cases]
