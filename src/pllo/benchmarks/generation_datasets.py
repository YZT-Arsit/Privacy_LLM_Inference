"""Dataset loaders + answer scoring for the AAAI generation benchmarks.

Three datasets share one ``{id, prompt}`` generation interface so the unified
runner (``scripts/run_generation_benchmark.py``) can drive any of them:

* **ifeval**  -- instruction-following open generation. JSONL ``{id, prompt}``
  (``prompt``/``instruction``/``text`` accepted); no reference scoring here (the
  IFEval instruction-checker is a separate offline step on the responses JSONL).
* **gsm8k**   -- grade-school math. JSONL ``{question, answer}`` (HF gsm8k schema)
  or ``{id, prompt, reference}``. The gold numeric answer is parsed from the
  ``#### <n>`` suffix; the prediction's answer is the last number in the
  generated text; scoring is exact match on the normalised number.
* **mt_bench** -- two-turn chat. JSONL ``{question_id, turns:[t1, t2], category}``
  (MT-Bench schema) or ``{id, turns:[...]}``. Generation is two-turn (turn 2 is
  conditioned on turn 1's response); the output JSONL is judge-ready.

stdlib only (json / re). No torch, no network, no downloads -- paths are passed
in by the caller.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "DATASETS",
    "load_dataset",
    "load_ifeval",
    "load_gsm8k",
    "load_mt_bench",
    "extract_gsm8k_answer",
    "normalize_number",
    "gsm8k_exact_match",
    "gsm8k_gold_answer",
]

DATASETS = ("ifeval", "gsm8k", "mt_bench")

_PROMPT_KEYS = ("prompt", "instruction", "text", "question")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _first_prompt(ex: dict[str, Any]) -> str:
    for k in _PROMPT_KEYS:
        if str(ex.get(k, "")).strip():
            return str(ex[k])
    return ""


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------

def load_ifeval(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        prompt = _first_prompt(ex)
        if not prompt:
            raise ValueError("ifeval example %r missing prompt" % ex.get("id", i))
        out.append({"id": str(ex.get("id", "ifeval-%d" % i)), "prompt": prompt,
                    "dataset": "ifeval"})
    return out


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def normalize_number(s: str | None) -> str | None:
    """Normalise a numeric string for exact match (strip commas/``$``/trailing
    ``.0``). Returns ``None`` when no number is present."""
    if s is None:
        return None
    m = _NUM_RE.search(str(s))
    if not m:
        return None
    num = m.group(0).replace(",", "")
    try:
        f = float(num)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else repr(f)


def extract_gsm8k_answer(text: str | None) -> str | None:
    """Extract the predicted answer from a model generation: the number after
    ``####`` if present, else the LAST number in the text."""
    if not text:
        return None
    if "####" in text:
        tail = text.split("####")[-1]
        n = normalize_number(tail)
        if n is not None:
            return n
    nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def gsm8k_gold_answer(ex: dict[str, Any]) -> str | None:
    """The gold numeric answer of a GSM8K example (``#### <n>`` suffix of the HF
    ``answer`` field, or an explicit ``reference``/``gold``)."""
    for k in ("reference", "gold", "final_answer"):
        if ex.get(k) is not None:
            return normalize_number(ex.get(k))
    ans = ex.get("answer")
    if ans is None:
        return None
    if "####" in str(ans):
        return normalize_number(str(ans).split("####")[-1])
    return normalize_number(str(ans))


def gsm8k_exact_match(prediction_text: str | None, gold: str | None) -> bool:
    """True iff the prediction's extracted answer equals the gold (normalised)."""
    if gold is None:
        return False
    pred = extract_gsm8k_answer(prediction_text)
    return pred is not None and pred == gold


_GSM8K_INSTRUCTION = (
    "Solve the following grade-school math problem. Show your reasoning, then "
    "give the final numeric answer on its own line after '#### '.\n\n")


def load_gsm8k(path: str | Path, *, add_instruction: bool = True
               ) -> list[dict[str, Any]]:
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        q = _first_prompt(ex)
        if not q:
            raise ValueError("gsm8k example %r missing question/prompt"
                             % ex.get("id", i))
        prompt = (_GSM8K_INSTRUCTION + q) if add_instruction else q
        out.append({"id": str(ex.get("id", ex.get("question_id", "gsm8k-%d" % i))),
                    "prompt": prompt, "reference": gsm8k_gold_answer(ex),
                    "dataset": "gsm8k"})
    return out


# ---------------------------------------------------------------------------
# MT-Bench (two-turn)
# ---------------------------------------------------------------------------

def load_mt_bench(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        turns = ex.get("turns")
        if not turns:
            single = _first_prompt(ex)
            if not single:
                raise ValueError("mt_bench example %r missing turns/prompt"
                                 % ex.get("id", i))
            turns = [single]
        turns = [str(t) for t in turns]
        out.append({
            "id": str(ex.get("question_id", ex.get("id", "mtbench-%d" % i))),
            "prompt": turns[0],          # turn-1 prompt (single-turn interface)
            "turns": turns,
            "category": ex.get("category"),
            "dataset": "mt_bench"})
    return out


def load_dataset(dataset: str, path: str | Path, *, max_examples: int = 0,
                 **kw) -> list[dict[str, Any]]:
    """Load + normalise one of :data:`DATASETS` into a list of example dicts."""
    if dataset == "ifeval":
        rows = load_ifeval(path)
    elif dataset == "gsm8k":
        rows = load_gsm8k(path, **kw)
    elif dataset == "mt_bench":
        rows = load_mt_bench(path)
    else:
        raise ValueError("unknown dataset %r (expected %s)"
                         % (dataset, ", ".join(DATASETS)))
    if max_examples and max_examples > 0:
        rows = rows[:max_examples]
    return rows
