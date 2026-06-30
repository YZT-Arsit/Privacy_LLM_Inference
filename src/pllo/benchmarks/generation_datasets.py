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

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

__all__ = [
    "DATASETS",
    "OPTIONAL_DATASETS",
    "load_dataset",
    "load_ifeval",
    "load_gsm8k",
    "load_mt_bench",
    "load_humaneval",
    "load_mbpp",
    "load_longbench_1024_lite",
    "load_sensitive_prompt",
    "extract_gsm8k_answer",
    "normalize_number",
    "gsm8k_exact_match",
    "gsm8k_gold_answer",
    "GSM8K_FEWSHOT_COT",
    "sha256_file",
    "build_dataset_card",
    "estimate_max_prompt_tokens",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_max_prompt_tokens(rows: list[dict[str, Any]]) -> int:
    """Cheap whitespace-token estimate of the longest prompt (no tokenizer)."""
    best = 0
    for r in rows:
        text = r.get("prompt") or " ".join(r.get("turns", []) or [])
        best = max(best, len(str(text).split()))
    return best


def build_dataset_card(*, dataset_name: str, split: str,
                       rows: list[dict[str, Any]], source_path: str | Path,
                       output_path: str | Path,
                       now: float | None = None) -> dict[str, Any]:
    """Build a reproducibility card for a normalised dataset JSONL."""
    return {
        "dataset_name": dataset_name,
        "split": split,
        "num_examples": len(rows),
        "input_sha256": sha256_file(source_path) if Path(source_path).exists()
        else None,
        "output_sha256": sha256_file(output_path) if Path(output_path).exists()
        else None,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime(now if now is not None
                                                else time.time())),
        "max_prompt_tokens_estimate": estimate_max_prompt_tokens(rows),
    }

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

_IFEVAL_META_KEYS = ("key", "instruction_id_list", "kwargs", "instruction_id",
                     "instruction_kwargs")


def load_ifeval(path: str | Path) -> list[dict[str, Any]]:
    """Normalise an official IFEval prompt file to ``{id, dataset, prompt, meta}``.

    The instruction metadata (``instruction_id_list`` / ``kwargs`` / ``key``) is
    preserved under ``meta`` so the official IFEval instruction-following checker
    can score the responses offline later."""
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        prompt = _first_prompt(ex)
        if not prompt:
            raise ValueError("ifeval example %r missing prompt" % ex.get("id", i))
        meta = {k: ex[k] for k in _IFEVAL_META_KEYS if k in ex}
        ex_id = ex.get("id", ex.get("key", "ifeval-%d" % i))
        out.append({"id": str(ex_id), "dataset": "ifeval", "prompt": prompt,
                    "meta": meta})
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

# A short 2-shot chain-of-thought preamble (kept compact to bound output length).
GSM8K_FEWSHOT_COT = (
    "Solve each grade-school math problem step by step, then give the final "
    "numeric answer on its own line after '#### '.\n\n"
    "Question: Natalia sold 48 clips in April and half as many in May. How many "
    "clips did she sell altogether?\n"
    "Answer: In April she sold 48. In May she sold 48 / 2 = 24. Altogether "
    "48 + 24 = 72.\n#### 72\n\n"
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "How many bolts in total?\n"
    "Answer: Blue is 2 bolts. White is 2 / 2 = 1 bolt. Total 2 + 1 = 3.\n"
    "#### 3\n\n")


def load_gsm8k(path: str | Path, *, prompt_style: str = "zero_shot",
               add_instruction: bool = True) -> list[dict[str, Any]]:
    """Normalise GSM8K to ``{id, dataset, prompt, answer, final_answer, meta}``.

    ``prompt_style``: ``zero_shot`` (instruction + question; default, short
    output) or ``few_shot_cot`` (a compact 2-shot CoT preamble). ``add_instruction``
    keeps back-compat (False == raw question)."""
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        q = _first_prompt(ex)
        if not q:
            raise ValueError("gsm8k example %r missing question/prompt"
                             % ex.get("id", i))
        if prompt_style == "few_shot_cot":
            prompt = GSM8K_FEWSHOT_COT + "Question: " + q + "\nAnswer:"
        elif add_instruction:
            prompt = _GSM8K_INSTRUCTION + q
        else:
            prompt = q
        gold = gsm8k_gold_answer(ex)
        out.append({
            "id": str(ex.get("id", ex.get("question_id", "gsm8k-%d" % i))),
            "dataset": "gsm8k", "prompt": prompt,
            "answer": ex.get("answer"), "final_answer": gold,
            "reference": gold,           # back-compat alias
            "meta": {"prompt_style": prompt_style, "question": q}})
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
            "dataset": "mt_bench",
            "prompt": turns[0],          # turn-1 prompt (single-turn interface)
            "turns": turns,
            "category": ex.get("category"),
            "meta": {"num_turns": len(turns),
                     "reference": ex.get("reference")}})
    return out


# Optional code-generation datasets. Interface only -- NOT part of the AAAI main
# experiment (kept so they can be wired later without a structural change).
OPTIONAL_DATASETS = ("humaneval", "mbpp")


def load_humaneval(path: str | Path) -> list[dict[str, Any]]:
    """OPTIONAL: HumanEval ``{task_id, prompt, entry_point, test, ...}`` ->
    normalised rows (code fields surfaced at the top level). The
    function-execution scorer is a separate offline CPU step (not run on H800)."""
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        prompt = _first_prompt(ex)
        if not prompt:
            raise ValueError("humaneval example %r missing prompt"
                             % ex.get("task_id", i))
        out.append({"id": str(ex.get("task_id", ex.get("id", "he-%d" % i))),
                    "dataset": "humaneval", "prompt": prompt,
                    "entry_point": ex.get("entry_point"),
                    "test": ex.get("test"),
                    "canonical_solution": ex.get("canonical_solution"),
                    "meta": {k: ex[k] for k in ("docstring", "task_id")
                             if k in ex}})
    return out


def load_longbench_1024_lite(path: str | Path) -> list[dict[str, Any]]:
    """OPTIONAL: a 1024-token-capped LongBench-style stress set. **NOT** an
    official LongBench score -- inputs are truncated to <=1024 tokens and used for
    latency / scaling / long-prompt-security stress only."""
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        prompt = _first_prompt(ex)
        if not prompt:
            raise ValueError("longbench_1024_lite example %r missing prompt"
                             % ex.get("id", i))
        out.append({
            "id": str(ex.get("id", ex.get("_id", "lb-%d" % i))),
            "dataset": "longbench_1024_lite", "prompt": prompt,
            "answer": ex.get("answer") or ex.get("answers"),
            "task": ex.get("task") or ex.get("dataset_task"),
            "original_length_estimate": ex.get("original_length_estimate"),
            "truncated": bool(ex.get("truncated", False)),
            "meta": {"not_official_longbench_score": True}})
    return out


def load_sensitive_prompt(path: str | Path) -> list[dict[str, Any]]:
    """A synthetic sensitive-prompt stress set. ``sensitive_spans`` are the
    substrings a transcript scan must NEVER find on a GPU-visible channel."""
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        prompt = _first_prompt(ex)
        if not prompt:
            raise ValueError("sensitive_prompt example %r missing prompt"
                             % ex.get("id", i))
        out.append({
            "id": str(ex.get("id", "sp-%d" % i)),
            "dataset": "sensitive_prompt_1024", "prompt": prompt,
            "sensitive_spans": list(ex.get("sensitive_spans", []) or []),
            "length_bucket": ex.get("length_bucket"),
            "task": ex.get("task", "summarize"),
            "meta": ex.get("meta", {})})
    return out


def load_mbpp(path: str | Path) -> list[dict[str, Any]]:
    """OPTIONAL: MBPP ``{task_id, text/prompt, test_list, ...}`` -> normalised."""
    out = []
    for i, ex in enumerate(_read_jsonl(path)):
        prompt = _first_prompt(ex) or str(ex.get("text", ""))
        if not prompt:
            raise ValueError("mbpp example %r missing text/prompt"
                             % ex.get("task_id", i))
        out.append({"id": str(ex.get("task_id", ex.get("id", "mbpp-%d" % i))),
                    "dataset": "mbpp", "prompt": prompt,
                    "meta": {k: ex[k] for k in ("test_list", "code") if k in ex}})
    return out


def load_dataset(dataset: str, path: str | Path, *, max_examples: int = 0,
                 **kw) -> list[dict[str, Any]]:
    """Load + normalise one of :data:`DATASETS` (or the OPTIONAL code datasets)
    into a list of example dicts."""
    if dataset == "ifeval":
        rows = load_ifeval(path)
    elif dataset == "gsm8k":
        rows = load_gsm8k(path, **kw)
    elif dataset == "mt_bench":
        rows = load_mt_bench(path)
    elif dataset == "humaneval":
        rows = load_humaneval(path)
    elif dataset == "mbpp":
        rows = load_mbpp(path)
    elif dataset == "longbench_1024_lite":
        rows = load_longbench_1024_lite(path)
    elif dataset == "sensitive_prompt_1024":
        rows = load_sensitive_prompt(path)
    else:
        raise ValueError("unknown dataset %r (expected %s)"
                         % (dataset, ", ".join(DATASETS + OPTIONAL_DATASETS)))
    if max_examples and max_examples > 0:
        rows = rows[:max_examples]
    return rows
