"""Local converters: read a user-provided file -> normalized example dicts.

No downloads. The user supplies a local CSV/TSV/JSONL file already extracted
from the public dataset (license compliance is the user's responsibility); each
converter yields dicts matching ``pllo.benchmarks.task_schemas``. Sampling is
deterministic (seeded shuffle) and reproducible; dataset cards record both input
and output file sha256 for provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from pllo.benchmarks.metrics import extract_numeric_answer

__all__ = ["convert_mmlu_csv", "convert_ceval_csv", "convert_cmmlu_csv",
           "convert_gsm8k_jsonl", "convert_boolq_jsonl", "convert_agnews_csv",
           "convert_sst2", "convert_summarization_jsonl", "CONVERTERS",
           "deterministic_sample", "sha256_file", "build_dataset_card",
           "iter_jsonl"]

LETTERS = "ABCD"


def _ex_id(dataset: str, split: str, i: int) -> str:
    return "%s-%s-%d" % (dataset, split, i)


def iter_jsonl(path) -> Iterator[Dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Multiple-choice (MMLU / C-Eval / CMMLU): CSV question,A,B,C,D,answer
# ---------------------------------------------------------------------------


def _convert_mc_csv(path, *, dataset: str, split: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            choices = ["%s. %s" % (L, str(row.get(L, "")).strip())
                       for L in LETTERS]
            answer = str(row.get("answer", "")).strip().upper()
            yield {
                "id": _ex_id(dataset, split, i),
                "dataset": dataset,
                "task_type": "multiple_choice",
                "metric": "accuracy",
                "question": str(row.get("question", "")).strip(),
                "choices": choices,
                "answer": answer,
            }


def convert_mmlu_csv(path, *, split: str = "test") -> Iterator[Dict[str, Any]]:
    return _convert_mc_csv(path, dataset="mmlu", split=split)


def convert_ceval_csv(path, *, split: str = "val") -> Iterator[Dict[str, Any]]:
    return _convert_mc_csv(path, dataset="ceval", split=split)


def convert_cmmlu_csv(path, *, split: str = "test") -> Iterator[Dict[str, Any]]:
    return _convert_mc_csv(path, dataset="cmmlu", split=split)


# ---------------------------------------------------------------------------
# GSM8K: JSONL {question, answer}
# ---------------------------------------------------------------------------


def convert_gsm8k_jsonl(path, *, split: str = "test") -> Iterator[Dict[str, Any]]:
    for i, row in enumerate(iter_jsonl(path)):
        answer = str(row.get("answer", "")).strip()
        yield {
            "id": _ex_id("gsm8k", split, i),
            "dataset": "gsm8k",
            "task_type": "generation_exact",
            "metric": "numeric_exact_match",
            "question": str(row.get("question", "")).strip(),
            "answer": answer,
            "numeric_answer": extract_numeric_answer(answer),
        }


# ---------------------------------------------------------------------------
# BoolQ: JSONL {passage, question, answer(bool/yes/no)}
# ---------------------------------------------------------------------------


def _to_yes_no(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return "yes"
    if s in ("false", "no", "0"):
        return "no"
    return s


def convert_boolq_jsonl(path, *, split: str = "validation") -> Iterator[Dict[str, Any]]:
    for i, row in enumerate(iter_jsonl(path)):
        yield {
            "id": _ex_id("boolq", split, i),
            "dataset": "boolq",
            "task_type": "yes_no",
            "metric": "accuracy",
            "passage": str(row.get("passage", "")).strip(),
            "question": str(row.get("question", "")).strip(),
            "answer": _to_yes_no(row.get("answer")),
            "label_space": ["yes", "no"],
        }


# ---------------------------------------------------------------------------
# AG News: CSV {label, title, description}; label 1-4 or name
# ---------------------------------------------------------------------------

_AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def _agnews_label(v: Any) -> str:
    s = str(v).strip()
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(_AGNEWS_LABELS):
            return _AGNEWS_LABELS[idx]
    for lab in _AGNEWS_LABELS:
        if s.lower() == lab.lower():
            return lab
    return s


def convert_agnews_csv(path, *, split: str = "test") -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", "")).strip()
            text = (title + ". " + desc).strip() if title else desc
            yield {
                "id": _ex_id("agnews", split, i),
                "dataset": "agnews",
                "task_type": "classification",
                "metric": "macro_f1",
                "secondary_metric": "accuracy",
                "text": text,
                "label": _agnews_label(row.get("label")),
                "label_space": list(_AGNEWS_LABELS),
            }


# ---------------------------------------------------------------------------
# SST-2: TSV or CSV {sentence, label(0/1)}
# ---------------------------------------------------------------------------

_SST2_LABELS = ["negative", "positive"]


def _sst2_label(v: Any) -> str:
    s = str(v).strip()
    if s in ("0", "1"):
        return _SST2_LABELS[int(s)]
    if s.lower() in _SST2_LABELS:
        return s.lower()
    return s


def convert_sst2(path, *, split: str = "validation") -> Iterator[Dict[str, Any]]:
    p = Path(path)
    delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for i, row in enumerate(reader):
            yield {
                "id": _ex_id("sst2", split, i),
                "dataset": "sst2",
                "task_type": "classification",
                "metric": "accuracy",
                "text": str(row.get("sentence", "")).strip(),
                "label": _sst2_label(row.get("label")),
                "label_space": list(_SST2_LABELS),
            }


# ---------------------------------------------------------------------------
# Summarization (CNN/DailyMail, XSum): JSONL {document, summary}
# ---------------------------------------------------------------------------


def convert_summarization_jsonl(path, *, dataset: str = "cnndm",
                                split: str = "test") -> Iterator[Dict[str, Any]]:
    for i, row in enumerate(iter_jsonl(path)):
        yield {
            "id": _ex_id(dataset, split, i),
            "dataset": dataset,
            "task_type": "summarization",
            "metric": "rouge_l",
            "document": str(row.get("document", "")).strip(),
            "summary": str(row.get("summary", "")).strip(),
        }


def _convert_cnndm(path, *, split: str = "test"):
    return convert_summarization_jsonl(path, dataset="cnndm", split=split)


def _convert_xsum(path, *, split: str = "test"):
    return convert_summarization_jsonl(path, dataset="xsum", split=split)


# Registry keyed by dataset name.
CONVERTERS = {
    "mmlu": convert_mmlu_csv,
    "ceval": convert_ceval_csv,
    "cmmlu": convert_cmmlu_csv,
    "gsm8k": convert_gsm8k_jsonl,
    "boolq": convert_boolq_jsonl,
    "agnews": convert_agnews_csv,
    "sst2": convert_sst2,
    "cnndm": _convert_cnndm,
    "xsum": _convert_xsum,
}


# ---------------------------------------------------------------------------
# Sampling / provenance
# ---------------------------------------------------------------------------


def deterministic_sample(examples: Iterable[Dict[str, Any]],
                         max_examples: Optional[int],
                         seed: int = 0) -> List[Dict[str, Any]]:
    """Seeded shuffle then take the first ``max_examples`` (stable / reproducible).

    ``max_examples`` of ``None`` or <= 0 keeps every example (still shuffled).
    """
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)
    if max_examples is not None and max_examples > 0:
        items = items[:max_examples]
    return items


def sha256_file(path) -> str:
    """Return the hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_card(*, source_name: str, split: str, sample_count: int,
                       task_type: str, metric: str,
                       sampling_seed: int,
                       input_file_sha256: Optional[str] = None,
                       output_file_sha256: Optional[str] = None,
                       license_note: str = "License compliance is the user's "
                       "responsibility; this file was prepared from a "
                       "user-provided local source (no downloads).",
                       **extra) -> Dict[str, Any]:
    """Build a provenance dataset card dict."""
    card = {
        "source_name": source_name,
        "split": split,
        "sample_count": sample_count,
        "task_type": task_type,
        "metric": metric,
        "license_note": license_note,
        "sampling_seed": sampling_seed,
        "input_file_sha256": input_file_sha256,
        "output_file_sha256": output_file_sha256,
    }
    card.update(extra)
    return card
