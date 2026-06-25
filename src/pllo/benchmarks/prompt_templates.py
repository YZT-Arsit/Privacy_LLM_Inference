"""Deterministic prompt builders for normalized benchmark examples.

Every builder is a pure function of the example: same input -> byte-identical
prompt. ``build_prompt`` dispatches on ``task_type``. Keep these simple; the
point is reproducibility, not prompt engineering.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pllo.benchmarks.task_schemas import TASK_TYPES

__all__ = ["LETTERS", "build_prompt", "build_multiple_choice_prompt",
           "build_generation_exact_prompt", "build_yes_no_prompt",
           "build_classification_prompt", "build_summarization_prompt"]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _strip_letter_prefix(choice: str) -> str:
    """Drop a leading ``A. ``/``B) `` style prefix if the choice already has one."""
    c = choice.strip()
    if len(c) >= 2 and c[0] in LETTERS and c[1] in (".", ")", ":"):
        return c[2:].strip()
    return c


def build_multiple_choice_prompt(example: Dict[str, Any]) -> str:
    lines: List[str] = [str(example["question"]).strip()]
    for i, choice in enumerate(example["choices"]):
        letter = LETTERS[i] if i < len(LETTERS) else str(i)
        lines.append("%s. %s" % (letter, _strip_letter_prefix(str(choice))))
    lines.append("Answer:")
    return "\n".join(lines)


def build_generation_exact_prompt(example: Dict[str, Any]) -> str:
    return "%s\nAnswer:" % str(example["question"]).strip()


def build_yes_no_prompt(example: Dict[str, Any]) -> str:
    passage = example.get("passage")
    parts: List[str] = []
    if passage:
        parts.append(str(passage).strip())
    parts.append(str(example["question"]).strip())
    parts.append("Answer (yes/no):")
    return "\n".join(parts)


def build_classification_prompt(example: Dict[str, Any]) -> str:
    labels = ", ".join(str(x) for x in example["label_space"])
    return "%s\nLabel (one of: %s):" % (str(example["text"]).strip(), labels)


def build_summarization_prompt(example: Dict[str, Any]) -> str:
    return "%s\nSummary:" % str(example["document"]).strip()


_DISPATCH = {
    "multiple_choice": build_multiple_choice_prompt,
    "generation_exact": build_generation_exact_prompt,
    "yes_no": build_yes_no_prompt,
    "classification": build_classification_prompt,
    "summarization": build_summarization_prompt,
}


def build_prompt(example: Dict[str, Any]) -> str:
    """Build a deterministic prompt for ``example`` dispatched on task_type."""
    tt = example.get("task_type")
    if tt not in TASK_TYPES or tt not in _DISPATCH:
        raise ValueError("cannot build prompt for task_type: %r" % (tt,))
    return _DISPATCH[tt](example)
