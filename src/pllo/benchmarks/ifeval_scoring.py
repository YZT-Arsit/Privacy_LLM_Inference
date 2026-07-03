"""Approximate IFEval instruction-following scorer.

A compact, dependency-light reimplementation of the mechanical IFEval
instruction checks (Zhou et al., 2023). It is *approximate*: word/sentence
tokenisation uses regex rather than nltk, and language detection is only
attempted if ``langdetect`` is importable. Every instruction we do not support
is reported as ``supported=False`` and excluded from the accuracy denominators,
and ``coverage_pct`` reports how many instructions were actually scored -- so
the reported numbers are trustworthy for the covered subset and never inflated
by silently passing unknown instruction types.

Strict vs loose: strict scores the response verbatim; loose scores a set of
lenient transforms (strip markdown ``*``/``_``, drop the first/last line, etc.)
and passes if ANY transform satisfies the instruction -- matching the IFEval
loose definition.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
_SENT_RE = re.compile(r"[^.!?]+[.!?]+", re.S)


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _sentences(text: str) -> List[str]:
    s = [m.group().strip() for m in _SENT_RE.finditer(text)]
    if not s and text.strip():
        return [text.strip()]
    return s


def _rel_ok(value: int, relation: str, target: int) -> bool:
    if relation in ("at least", "at_least"):
        return value >= target
    if relation in ("less than", "less_than"):
        return value < target
    if relation in ("at most", "at_most"):
        return value <= target
    if relation in ("more than", "more_than"):
        return value > target
    if relation in ("exactly",):
        return value == target
    return value >= target


# --------------------------------------------------------------------------- #
# individual instruction checkers: (response, kwargs) -> bool
# --------------------------------------------------------------------------- #
def _c_no_comma(r: str, kw: Dict) -> bool:
    return "," not in r and "，" not in r


def _c_number_words(r: str, kw: Dict) -> bool:
    return _rel_ok(len(_words(r)), kw.get("relation", "at least"),
                   int(kw.get("num_words", 0)))


def _c_number_sentences(r: str, kw: Dict) -> bool:
    return _rel_ok(len(_sentences(r)), kw.get("relation", "at least"),
                   int(kw.get("num_sentences", 0)))


def _c_forbidden_words(r: str, kw: Dict) -> bool:
    low = r.lower()
    return not any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low)
                   for w in kw.get("forbidden_words", []))


def _c_keywords_existence(r: str, kw: Dict) -> bool:
    low = r.lower()
    return all(re.search(r"\b" + re.escape(w.lower()) + r"\b", low)
               for w in kw.get("keywords", []))


def _c_keyword_frequency(r: str, kw: Dict) -> bool:
    kwd = kw.get("keyword", "")
    n = len(re.findall(r"\b" + re.escape(kwd.lower()) + r"\b", r.lower()))
    return _rel_ok(n, kw.get("relation", "at least"),
                   int(kw.get("frequency", 0)))


def _c_letter_frequency(r: str, kw: Dict) -> bool:
    letter = (kw.get("letter") or "").lower()
    n = r.lower().count(letter) if letter else 0
    return _rel_ok(n, kw.get("let_relation", kw.get("relation", "at least")),
                   int(kw.get("let_frequency", kw.get("frequency", 0))))


def _c_highlighted_sections(r: str, kw: Dict) -> bool:
    n = len(re.findall(r"\*[^*\n]+\*", r)) + len(re.findall(r"_[^_\n]+_", r))
    return n >= int(kw.get("num_highlights", 0))


def _c_bullet_lists(r: str, kw: Dict) -> bool:
    n = len(re.findall(r"^\s*[\*\-]\s+\S", r, re.M))
    return n == int(kw.get("num_bullets", 0))


def _c_title(r: str, kw: Dict) -> bool:
    return bool(re.search(r"<<[^>\n]+>>", r))


def _c_lowercase(r: str, kw: Dict) -> bool:
    return r == r.lower() and any(c.isalpha() for c in r)


def _c_uppercase(r: str, kw: Dict) -> bool:
    return r == r.upper() and any(c.isalpha() for c in r)


def _c_capital_word_frequency(r: str, kw: Dict) -> bool:
    caps = [w for w in _words(r) if w.isupper() and len(w) > 1]
    return _rel_ok(len(caps),
                   kw.get("capital_relation", kw.get("relation", "at least")),
                   int(kw.get("capital_frequency", kw.get("frequency", 0))))


def _c_placeholders(r: str, kw: Dict) -> bool:
    return len(re.findall(r"\[[^\]\n]+\]", r)) >= int(
        kw.get("num_placeholders", 0))


def _c_paragraphs(r: str, kw: Dict) -> bool:
    paras = [p for p in re.split(r"\n\s*\*\*\*\s*\n|\n\s*\n", r) if p.strip()]
    return len(paras) == int(kw.get("num_paragraphs", 0))


def _c_postscript(r: str, kw: Dict) -> bool:
    marker = (kw.get("postscript_marker") or "P.S.").lower()
    return marker.lower() in r.lower()


def _c_quotation(r: str, kw: Dict) -> bool:
    s = r.strip()
    return len(s) >= 2 and s[0] in '"“' and s[-1] in '"”'


def _c_end_checker(r: str, kw: Dict) -> bool:
    phrase = (kw.get("end_phrase") or "").strip().lower()
    return r.strip().lower().endswith(phrase) if phrase else False


def _c_two_responses(r: str, kw: Dict) -> bool:
    return "******" in r


def _c_json_format(r: str, kw: Dict) -> bool:
    import json as _json
    s = r.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    try:
        _json.loads(s)
        return True
    except Exception:
        return False


def _c_constrained_response(r: str, kw: Dict) -> bool:
    opts = ["my answer is yes.", "my answer is no.", "my answer is maybe."]
    return r.strip().lower() in opts


def _c_multiple_sections(r: str, kw: Dict) -> bool:
    spliter = kw.get("section_spliter", "Section")
    n = len(re.findall(re.escape(spliter) + r"\s*\d", r))
    return n >= int(kw.get("num_sections", 0))


def _c_nth_paragraph_first_word(r: str, kw: Dict) -> bool:
    paras = [p for p in re.split(r"\n\s*\n", r) if p.strip()]
    n = int(kw.get("num_paragraphs", 0))
    nth = int(kw.get("nth_paragraph", 0))
    first = (kw.get("first_word") or "").lower()
    if len(paras) < n or nth < 1 or nth > len(paras):
        return False
    w = _words(paras[nth - 1])
    return bool(w) and w[0].lower() == first


def _c_response_language(r: str, kw: Dict) -> Optional[bool]:
    try:
        from langdetect import detect  # type: ignore
    except Exception:
        return None  # unsupported without the dependency
    want = (kw.get("language") or "").lower()
    try:
        return detect(r).lower().startswith(want[:2])
    except Exception:
        return False


def _c_repeat_prompt(r: str, kw: Dict) -> bool:
    p = (kw.get("prompt_to_repeat") or "").strip()
    return r.strip().startswith(p) if p else False


_CHECKERS: Dict[str, Callable[[str, Dict], Optional[bool]]] = {
    "punctuation:no_comma": _c_no_comma,
    "length_constraints:number_words": _c_number_words,
    "length_constraints:number_sentences": _c_number_sentences,
    "length_constraints:number_paragraphs": _c_paragraphs,
    "length_constraints:nth_paragraph_first_word": _c_nth_paragraph_first_word,
    "keywords:forbidden_words": _c_forbidden_words,
    "keywords:existence": _c_keywords_existence,
    "keywords:frequency": _c_keyword_frequency,
    "keywords:letter_frequency": _c_letter_frequency,
    "detectable_format:number_highlighted_sections": _c_highlighted_sections,
    "detectable_format:number_bullet_lists": _c_bullet_lists,
    "detectable_format:title": _c_title,
    "detectable_format:json_format": _c_json_format,
    "detectable_format:constrained_response": _c_constrained_response,
    "detectable_format:multiple_sections": _c_multiple_sections,
    "detectable_content:number_placeholders": _c_placeholders,
    "detectable_content:postscript": _c_postscript,
    "change_case:english_lowercase": _c_lowercase,
    "change_case:english_capital": _c_uppercase,
    "change_case:capital_word_frequency": _c_capital_word_frequency,
    "startend:quotation": _c_quotation,
    "startend:end_checker": _c_end_checker,
    "combination:two_responses": _c_two_responses,
    "combination:repeat_prompt": _c_repeat_prompt,
    "language:response_language": _c_response_language,
}

SUPPORTED_INSTRUCTION_IDS = tuple(sorted(_CHECKERS))


def _loose_variants(r: str) -> List[str]:
    """IFEval loose transforms: try each and pass if any satisfies."""
    variants = [r]
    variants.append(r.replace("*", ""))
    lines = [ln for ln in r.split("\n")]
    if len(lines) > 1:
        variants.append("\n".join(lines[1:]).strip())      # drop first line
        variants.append("\n".join(lines[:-1]).strip())     # drop last line
    variants.append(r.replace("*", "").replace("_", "").strip())
    # dedupe preserving order
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def score_instruction(instruction_id: str, response: str,
                      kwargs: Dict) -> Dict[str, Any]:
    """Score a single instruction, strict + loose."""
    fn = _CHECKERS.get(instruction_id)
    if fn is None:
        return {"instruction_id": instruction_id, "supported": False,
                "strict": None, "loose": None}
    kwargs = kwargs or {}
    strict = fn(response, kwargs)
    if strict is None:                      # runtime-unsupported (e.g. langdetect)
        return {"instruction_id": instruction_id, "supported": False,
                "strict": None, "loose": None}
    loose = any(bool(fn(v, kwargs)) for v in _loose_variants(response))
    return {"instruction_id": instruction_id, "supported": True,
            "strict": bool(strict), "loose": bool(loose)}


def score_response(meta: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Score all instructions attached to one IFEval prompt."""
    ids = meta.get("instruction_id_list") or []
    kwlist = meta.get("kwargs") or [{}] * len(ids)
    per = [score_instruction(iid, response, kwlist[i] if i < len(kwlist) else {})
           for i, iid in enumerate(ids)]
    supported = [p for p in per if p["supported"]]
    all_strict = bool(supported) and all(p["strict"] for p in supported)
    all_loose = bool(supported) and all(p["loose"] for p in supported)
    return {
        "per_instruction": per,
        "num_instructions": len(ids),
        "num_supported": len(supported),
        "prompt_strict": all_strict if supported else None,
        "prompt_loose": all_loose if supported else None,
    }


def aggregate_ifeval(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate strict/loose prompt- and instruction-level accuracy over the
    *supported* instructions only, plus coverage."""
    total_inst = sum(s["num_instructions"] for s in scored)
    supp_inst = sum(s["num_supported"] for s in scored)
    inst_strict = inst_loose = 0
    for s in scored:
        for p in s["per_instruction"]:
            if p["supported"]:
                inst_strict += int(p["strict"])
                inst_loose += int(p["loose"])
    prompts_scored = [s for s in scored if s["prompt_strict"] is not None]
    n_ps = len(prompts_scored)
    ps_strict = sum(int(s["prompt_strict"]) for s in prompts_scored)
    ps_loose = sum(int(s["prompt_loose"]) for s in prompts_scored)

    def _r(a, b):
        return round(a / b, 4) if b else None

    return {
        "evaluator": "approximate_ifeval_reimpl",
        "num_prompts": len(scored),
        "num_prompts_scored": n_ps,
        "total_instructions": total_inst,
        "supported_instructions": supp_inst,
        "coverage_pct": round(100.0 * supp_inst / total_inst, 1)
        if total_inst else None,
        "strict_prompt_acc": _r(ps_strict, n_ps),
        "loose_prompt_acc": _r(ps_loose, n_ps),
        "strict_inst_acc": _r(inst_strict, supp_inst),
        "loose_inst_acc": _r(inst_loose, supp_inst),
    }
