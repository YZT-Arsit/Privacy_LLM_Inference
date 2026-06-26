"""Held-out statistics for base vs LoRA generations on Dolly-15k.

Dolly responses are open-ended, so this does NOT compute a hard accuracy. It emits
reproducible, paper-appendix-friendly statistics over the base and LoRA response
JSONLs, plus optional lexical overlap with the reference response from the data
JSONL (a simple unigram-F1 and an LCS-based ROUGE-L-like score; no heavy deps).

numpy + standard library only. All metric helpers are import-safe + unit-tested.

Example::

    python scripts/evaluate_dolly_lora_outputs.py \\
        --data-jsonl /root/autodl-tmp/datasets/dolly/dolly_test.jsonl \\
        --base-response-jsonl outputs/lora_dolly/base_plaintext_responses.jsonl \\
        --lora-response-jsonl outputs/lora_dolly/lora_plaintext_responses.jsonl \\
        --output-json outputs/lora_dolly/dolly_eval.json \\
        --output-md   outputs/lora_dolly/dolly_eval.md \\
        --output-jsonl outputs/lora_dolly/dolly_eval_per_example.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9']+")
_HUMAN = re.compile(r"\b(human|user)\s*:", re.IGNORECASE)
_ASSISTANT = re.compile(r"\b(assistant|ai)\s*:", re.IGNORECASE)
_NOTE = re.compile(r"\b(note|disclaimer)\s*:", re.IGNORECASE)


def _tok(s):
    return _WORD.findall((s or "").lower())


def unigram_f1(pred, ref):
    """Bag-of-words unigram F1 between prediction and reference."""
    from collections import Counter
    pt, rt = _tok(pred), _tok(ref)
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    overlap = sum((Counter(pt) & Counter(rt)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pt), overlap / len(rt)
    return 2 * prec * rec / (prec + rec)


def lcs_len(a, b):
    """Length of the longest common subsequence (token level)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_like(pred, ref):
    """LCS-based ROUGE-L-like F1 (simple, dependency-free)."""
    pt, rt = _tok(pred), _tok(ref)
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    lcs = lcs_len(pt, rt)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(pt), lcs / len(rt)
    return 2 * prec * rec / (prec + rec)


def response_features(text):
    t = text if isinstance(text, str) else ""
    return {
        "chars": len(t), "words": len(t.split()),
        "empty": bool(not t.strip()),
        "contains_human_marker": bool(_HUMAN.search(t)),
        "contains_assistant_marker": bool(_ASSISTANT.search(t)),
        "contains_note": bool(_NOTE.search(t)),
    }


def _load_jsonl(path):
    out = {}
    order = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            key = str(rec.get("id", rec.get("key", "ex-%d" % i)))
            out[key] = rec
            order.append(key)
    return out, order


def _side_stats(records, refs):
    """Aggregate stats for one side (base or lora) keyed by id; refs id->response."""
    n = len(records)
    chars = words = 0
    empty = human = assist = note = 0
    fr_counts = {}
    seen_texts = {}
    dup = 0
    f1s, rouges = [], []
    for key, rec in records.items():
        text = rec.get("response", "")
        f = response_features(text)
        chars += f["chars"]
        words += f["words"]
        empty += int(f["empty"])
        human += int(f["contains_human_marker"])
        assist += int(f["contains_assistant_marker"])
        note += int(f["contains_note"])
        fr = rec.get("finish_reason")
        if fr is not None:
            fr_counts[fr] = fr_counts.get(fr, 0) + 1
        norm = (text or "").strip()
        seen_texts[norm] = seen_texts.get(norm, 0) + 1
        ref = refs.get(key)
        if isinstance(ref, str) and ref.strip():
            f1s.append(unigram_f1(text, ref))
            rouges.append(rouge_l_like(text, ref))
    for c in seen_texts.values():
        if c > 1:
            dup += c - 1
    return {
        "num_examples": n,
        "avg_chars": round(chars / n, 2) if n else 0.0,
        "avg_words": round(words / n, 2) if n else 0.0,
        "empty_response_count": empty,
        "contains_human_marker_count": human,
        "contains_assistant_marker_count": assist,
        "contains_note_count": note,
        "finish_reason_counts": dict(sorted(fr_counts.items())),
        "exact_duplicate_rate": round(dup / n, 4) if n else 0.0,
        "unigram_f1_avg": (round(sum(f1s) / len(f1s), 4) if f1s else None),
        "rouge_l_like_lcs_avg": (round(sum(rouges) / len(rouges), 4)
                                 if rouges else None),
        "_total_words": words,
    }


def evaluate(data_records, base_records, lora_records):
    """Compute the full evaluation dict from three id->record maps."""
    refs = {k: v.get("response") for k, v in data_records.items()}
    cats = {}
    for v in data_records.values():
        c = v.get("category")
        if c is not None:
            cats[c] = cats.get(c, 0) + 1
    base = _side_stats(base_records, refs)
    lora = _side_stats(lora_records, refs)
    n = max(base["num_examples"], lora["num_examples"])
    delta = (round((lora["_total_words"] - base["_total_words"]) / n, 2)
             if n else 0.0)
    base.pop("_total_words", None)
    lora.pop("_total_words", None)
    return {
        "stage": "dolly_lora_eval", "dataset": "databricks-dolly-15k",
        "num_examples": n,
        "category_counts": dict(sorted(cats.items())),
        "base": base, "lora": lora,
        "response_length_delta_lora_minus_base": delta,
    }


def _per_example(data_records, base_records, lora_records):
    rows = []
    for key in lora_records:
        ref = (data_records.get(key) or {}).get("response")
        b = (base_records.get(key) or {}).get("response", "")
        l = (lora_records.get(key) or {}).get("response", "")
        rows.append({
            "id": key,
            "category": (data_records.get(key) or {}).get("category"),
            "base_words": len(_tok(b)), "lora_words": len(_tok(l)),
            "base_unigram_f1": (round(unigram_f1(b, ref), 4) if ref else None),
            "lora_unigram_f1": (round(unigram_f1(l, ref), 4) if ref else None),
            "lora_rouge_l_like": (round(rouge_l_like(l, ref), 4) if ref else None),
        })
    return rows


def _markdown(rep):
    b, l = rep["base"], rep["lora"]
    keys = ["num_examples", "avg_chars", "avg_words", "empty_response_count",
            "contains_human_marker_count", "contains_assistant_marker_count",
            "contains_note_count", "exact_duplicate_rate", "unigram_f1_avg",
            "rouge_l_like_lcs_avg"]
    L = ["# Dolly LoRA held-out statistics", "",
         "dataset=%s  num_examples=%s" % (rep["dataset"], rep["num_examples"]),
         "response_length_delta (lora - base, words/ex) = %s"
         % rep["response_length_delta_lora_minus_base"], "",
         "| metric | base | lora |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (k, b.get(k), l.get(k)) for k in keys]
    L += ["", "| finish_reason | base | lora |", "|---|---|---|"]
    frs = sorted(set(b["finish_reason_counts"]) | set(l["finish_reason_counts"]))
    L += ["| %s | %s | %s |" % (fr, b["finish_reason_counts"].get(fr, 0),
                                l["finish_reason_counts"].get(fr, 0)) for fr in frs]
    L += ["", "## category counts", ""]
    L += ["- %s: %s" % (k, v) for k, v in rep["category_counts"].items()]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--base-response-jsonl", required=True)
    ap.add_argument("--lora-response-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-jsonl", default=None)
    args = ap.parse_args()

    data, _ = _load_jsonl(args.data_jsonl)
    base, _ = _load_jsonl(args.base_response_jsonl)
    lora, _ = _load_jsonl(args.lora_response_jsonl)
    rep = evaluate(data, base, lora)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rep, indent=2, default=str),
                                      encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(_markdown(rep), encoding="utf-8")
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as fh:
            for row in _per_example(data, base, lora):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Dolly LoRA eval ===")
    print("num_examples=%s" % rep["num_examples"])
    print("avg_words base=%s lora=%s (delta=%s)"
          % (rep["base"]["avg_words"], rep["lora"]["avg_words"],
             rep["response_length_delta_lora_minus_base"]))
    print("lora unigram_f1=%s rouge_l_like=%s"
          % (rep["lora"]["unigram_f1_avg"], rep["lora"]["rouge_l_like_lcs_avg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
