"""Normalized example schema for public benchmark tasks.

A single dict shape is used across every dataset so prompt builders, metrics and
the runner are dataset-agnostic. ``validate_example`` returns a list of human
readable problems (empty list == valid); ``assert_valid`` raises ``ValueError``.

Canonical normalized example (per task_type)::

    multiple_choice : {id, dataset, task_type="multiple_choice", metric,
                       question, choices: [str, ...], answer}
    generation_exact: {id, dataset, task_type="generation_exact", metric,
                       question, answer}                  # + optional numeric_answer
    yes_no          : {id, dataset, task_type="yes_no", metric,
                       question, answer, label_space: [str, ...]}
    classification  : {id, dataset, task_type="classification", metric,
                       text, label, label_space: [str, ...]}
    summarization   : {id, dataset, task_type="summarization", metric,
                       document, summary}
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["TASK_TYPES", "REQUIRED_KEYS", "TASK_REQUIRED_KEYS",
           "validate_example", "assert_valid"]

TASK_TYPES = (
    "multiple_choice",
    "generation_exact",
    "yes_no",
    "classification",
    "summarization",
)

# Keys every normalized example must carry.
REQUIRED_KEYS = ("id", "dataset", "task_type", "metric")

# Extra required keys per task_type.
TASK_REQUIRED_KEYS = {
    "multiple_choice": ("question", "choices", "answer"),
    "generation_exact": ("question", "answer"),
    "yes_no": ("question", "answer", "label_space"),
    "classification": ("text", "label", "label_space"),
    "summarization": ("document", "summary"),
}


def validate_example(ex: Dict[str, Any]) -> List[str]:
    """Return a list of problems with ``ex`` (empty list means valid)."""
    problems: List[str] = []
    if not isinstance(ex, dict):
        return ["example is not a dict"]
    for k in REQUIRED_KEYS:
        if k not in ex or ex[k] in (None, ""):
            problems.append("missing required key: %s" % k)
    tt = ex.get("task_type")
    if tt not in TASK_TYPES:
        problems.append("invalid task_type: %r (allowed: %s)"
                        % (tt, ", ".join(TASK_TYPES)))
        return problems
    for k in TASK_REQUIRED_KEYS[tt]:
        if k not in ex or ex[k] in (None, ""):
            problems.append("missing %s key: %s" % (tt, k))
    # Shape checks for list-valued fields.
    if tt == "multiple_choice":
        choices = ex.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            problems.append("choices must be a list of >=2 items")
    if tt in ("yes_no", "classification"):
        ls = ex.get("label_space")
        if not isinstance(ls, list) or len(ls) < 2:
            problems.append("label_space must be a list of >=2 items")
    if tt == "classification":
        label = ex.get("label")
        ls = ex.get("label_space")
        if isinstance(ls, list) and label is not None and label not in ls:
            problems.append("label %r not in label_space" % (label,))
    return problems


def assert_valid(ex: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``ex`` is not a valid normalized example."""
    problems = validate_example(ex)
    if problems:
        raise ValueError("invalid example (id=%s): %s"
                         % (ex.get("id") if isinstance(ex, dict) else "?",
                            "; ".join(problems)))
