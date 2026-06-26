"""Offline IFEval scoring of ALREADY-GENERATED responses (no model re-run).

The repo's `run_ifeval_generation.py` produces a responses JSONL but does NOT
score it; `analyze_ifeval_strict_gap.py` only *consumes* evaluator output. This
script closes that gap: it takes the IFEval prompts (official input format) plus a
responses JSONL and computes the official IFEval instruction-following metrics
(strict + loose, prompt-level + instruction-level) WITHOUT touching the model or
the generation path.

It REUSES the official IFEval logic: `evaluation_lib.test_instruction_following_
strict` / `..._loose` from Google's `instruction_following_eval` package (the same
checkers the IFEval paper/leaderboard use). We do not re-implement the instruction
classes -- we only join prompts<->responses and aggregate the official verdicts.

The official package must be importable (vendored, pip-installed, or on
PYTHONPATH). The script probes several locations; if none is found it prints exact
install instructions and exits non-zero (it never silently falls back to a
home-grown scorer, which would not be "official IFEval").

Inputs
------
--input-jsonl     official IFEval prompts: one JSON object per line with
                  ``key`` (or ``id``), ``prompt``, ``instruction_id_list``,
                  ``kwargs`` (list aligned with instruction_id_list).
--response-jsonl  generated responses (run_ifeval_generation.py output): objects
                  with ``prompt`` (+ optional ``id``/``key``) and ``response``.

Outputs
-------
--output-json     summary metrics (strict/loose prompt+instruction accuracy,
                  counts, per-category breakdown).
--output-jsonl    per-example COMBINED records (strict + loose). Sibling files
                  ``<stem>_strict.jsonl`` and ``<stem>_loose.jsonl`` are also
                  written in the exact shape `analyze_ifeval_strict_gap.py`
                  consumes (``follow_instruction_list`` / ``follow_all_instructions``).
--output-md       human-readable summary table.

Example (TDX guest)::

    python scripts/evaluate_existing_ifeval_responses.py \\
      --input-jsonl    /root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ifeval_prompts.jsonl \\
      --response-jsonl outputs/tdx_ifeval/tdx_folded_ifeval541_combined_responses.jsonl \\
      --output-json    outputs/tdx_ifeval/tdx_folded_ifeval541_eval.json \\
      --output-jsonl   outputs/tdx_ifeval/tdx_folded_ifeval541_eval_records.jsonl \\
      --output-md      outputs/tdx_ifeval/tdx_folded_ifeval541_eval.md
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import sys
from pathlib import Path

# Official IFEval InputExample field layout (evaluation_lib accesses .prompt,
# .instruction_id_list, .kwargs, .key). We build our own namedtuple with the same
# fields so we do not depend on importing the official InputExample symbol.
InputExample = collections.namedtuple(
    "InputExample", ["key", "instruction_id_list", "prompt", "kwargs"])


# ---------------------------------------------------------------------------
# official-package discovery (no home-grown fallback)
# ---------------------------------------------------------------------------

_IMPORT_CANDIDATES = (
    "instruction_following_eval",                       # pip / vendored root
    "pllo.third_party.instruction_following_eval",      # vendored under src/pllo
    None,                                               # top-level on PYTHONPATH
)

_INSTALL_HELP = (
    "Official IFEval evaluator (evaluation_lib + instructions_registry) not "
    "found. Install it on this host, e.g.:\n"
    "  pip install langdetect nltk immutabledict absl-py\n"
    "  python -c \"import nltk; nltk.download('punkt'); "
    "nltk.download('punkt_tab')\"\n"
    "  # vendor Google's IFEval code so it is importable, e.g.:\n"
    "  git clone --depth 1 https://github.com/google-research/google-research \\\n"
    "      /tmp/gr && cp -r /tmp/gr/instruction_following_eval \\\n"
    "      $PWD/src/pllo/third_party/instruction_following_eval\n"
    "  touch $PWD/src/pllo/third_party/__init__.py\n"
    "Then re-run (PYTHONPATH=$PWD/src must be set)."
)


def import_ifeval_eval_lib():
    """Return the official `evaluation_lib` module, or None if unavailable."""
    for base in _IMPORT_CANDIDATES:
        mod_name = ("%s.evaluation_lib" % base) if base else "evaluation_lib"
        try:
            return importlib.import_module(mod_name)
        except Exception:                                    # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# loading / joining (pure, unit-tested)
# ---------------------------------------------------------------------------


def load_prompts(path, *, max_examples=0):
    """Load official IFEval prompt records into InputExample list.

    Returns (inputs, skipped) where skipped counts records missing the
    instruction metadata required for scoring."""
    inputs, skipped = [], 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                # noqa: BLE001
                skipped += 1
                continue
            prompt = str(rec.get("prompt", ""))
            iid = rec.get("instruction_id_list")
            if not prompt or not isinstance(iid, list) or not iid:
                skipped += 1
                continue
            kwargs = rec.get("kwargs")
            if not isinstance(kwargs, list) or len(kwargs) != len(iid):
                # official checkers need one kwargs dict per instruction; fill
                # blanks so non-parameterized instructions still score.
                kwargs = [{} for _ in iid]
            else:
                kwargs = [dict(k) if isinstance(k, dict) else {} for k in kwargs]
            key = rec.get("key", rec.get("id", i))
            inputs.append(InputExample(key=key, instruction_id_list=list(iid),
                                       prompt=prompt, kwargs=kwargs))
    if max_examples and max_examples > 0:
        inputs = inputs[:max_examples]
    return inputs, skipped


def load_responses(path):
    """Load a responses JSONL -> (by_prompt, by_key) text maps."""
    by_prompt, by_key = {}, {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            resp = rec.get("response")
            if resp is None:
                continue
            resp = str(resp)
            if rec.get("prompt"):
                by_prompt[str(rec["prompt"])] = resp
            k = rec.get("key", rec.get("id"))
            if k is not None:
                by_key[str(k)] = resp
    return by_prompt, by_key


def build_prompt_to_response(inputs, by_prompt, by_key):
    """Map each input's prompt text -> response (official scorers key on the
    prompt string). Join by prompt text first, then by key/id. Returns
    (prompt_to_response, matched_keys, missing_keys)."""
    p2r, matched, missing = {}, [], []
    for inp in inputs:
        resp = by_prompt.get(inp.prompt)
        if resp is None:
            resp = by_key.get(str(inp.key))
        if resp is None:
            missing.append(inp.key)
            continue
        p2r[inp.prompt] = resp
        matched.append(inp.key)
    return p2r, matched, missing


# ---------------------------------------------------------------------------
# evaluation + aggregation (pure given injectable scorers)
# ---------------------------------------------------------------------------


def evaluate(inputs, prompt_to_response, strict_fn, loose_fn):
    """Run the (official) strict + loose checkers over inputs that have a
    response. Returns (strict_outputs, loose_outputs) aligned lists."""
    strict_out, loose_out = [], []
    for inp in inputs:
        if inp.prompt not in prompt_to_response:
            continue
        strict_out.append(strict_fn(inp, prompt_to_response))
        loose_out.append(loose_fn(inp, prompt_to_response))
    return strict_out, loose_out


def _category(instruction_id):
    """IFEval instruction category = the id prefix before the first ':'."""
    s = str(instruction_id)
    return s.split(":", 1)[0] if ":" in s else s


def aggregate_scores(strict_outputs, loose_outputs):
    """Official IFEval metrics: prompt-level = fraction of prompts where ALL
    instructions are followed; instruction-level = fraction of all instructions
    followed. Computed for strict and loose."""
    def _prompt_acc(outs):
        if not outs:
            return None
        return round(sum(1 for o in outs if o.follow_all_instructions)
                     / len(outs), 6)

    def _inst_counts(outs):
        total = followed = 0
        for o in outs:
            fl = list(o.follow_instruction_list)
            total += len(fl)
            followed += sum(1 for x in fl if x)
        return followed, total

    def _inst_acc(outs):
        followed, total = _inst_counts(outs)
        return (round(followed / total, 6) if total else None)

    s_followed, s_total = _inst_counts(strict_outputs)
    # per-category instruction accuracy (strict + loose)
    cat = collections.defaultdict(lambda: {"strict_followed": 0,
                                           "strict_total": 0,
                                           "loose_followed": 0,
                                           "loose_total": 0})
    for outs, sk, tk in ((strict_outputs, "strict_followed", "strict_total"),
                         (loose_outputs, "loose_followed", "loose_total")):
        for o in outs:
            for iid, ok in zip(o.instruction_id_list, o.follow_instruction_list):
                c = cat[_category(iid)]
                c[tk] += 1
                if ok:
                    c[sk] += 1
    per_cat = {}
    for name in sorted(cat):
        c = cat[name]
        per_cat[name] = {
            "strict_instruction_accuracy":
                (round(c["strict_followed"] / c["strict_total"], 6)
                 if c["strict_total"] else None),
            "loose_instruction_accuracy":
                (round(c["loose_followed"] / c["loose_total"], 6)
                 if c["loose_total"] else None),
            "num_instructions": c["strict_total"],
        }
    return {
        "num_prompts": len(strict_outputs),
        "num_instructions": s_total,
        "strict_prompt_accuracy": _prompt_acc(strict_outputs),
        "loose_prompt_accuracy": _prompt_acc(loose_outputs),
        "strict_instruction_accuracy": _inst_acc(strict_outputs),
        "loose_instruction_accuracy": _inst_acc(loose_outputs),
        "per_instruction_category": per_cat,
    }


def _mode_record(o):
    """One analyze_ifeval_strict_gap-compatible record from an OutputExample."""
    return {
        "prompt": o.prompt,
        "response": o.response,
        "instruction_id_list": list(o.instruction_id_list),
        "follow_instruction_list": [bool(x) for x in o.follow_instruction_list],
        "follow_all_instructions": bool(o.follow_all_instructions),
    }


def build_records(inputs, strict_outputs, loose_outputs):
    """Build the combined + per-mode (strict/loose) per-example record lists.

    inputs is the subset (with responses) that was scored, in the same order as
    the outputs."""
    scored = [inp for inp in inputs if True]  # caller passes scored subset
    combined, strict_recs, loose_recs = [], [], []
    for inp, so, lo in zip(scored, strict_outputs, loose_outputs):
        key = inp.key
        sr = _mode_record(so)
        lr = _mode_record(lo)
        sr["key"] = key
        sr["id"] = key
        lr["key"] = key
        lr["id"] = key
        strict_recs.append(sr)
        loose_recs.append(lr)
        combined.append({
            "key": key, "id": key, "prompt": inp.prompt,
            "response": so.response,
            "instruction_id_list": list(so.instruction_id_list),
            "strict_follow_instruction_list": sr["follow_instruction_list"],
            "strict_follow_all_instructions": sr["follow_all_instructions"],
            "loose_follow_instruction_list": lr["follow_instruction_list"],
            "loose_follow_all_instructions": lr["follow_all_instructions"],
        })
    return combined, strict_recs, loose_recs


# ---------------------------------------------------------------------------
# output rendering
# ---------------------------------------------------------------------------


def _markdown(summary, meta):
    L = ["# IFEval offline evaluation (existing responses)", "",
         "| metric | value |", "|---|---|",
         "| response_jsonl | %s |" % meta.get("response_jsonl"),
         "| num_prompts | %s |" % summary["num_prompts"],
         "| num_instructions | %s |" % summary["num_instructions"],
         "| missing_responses | %s |" % meta.get("missing_responses"),
         "| prompts_skipped_no_metadata | %s |" % meta.get("prompts_skipped"),
         "| **strict_prompt_accuracy** | %s |" % summary["strict_prompt_accuracy"],
         "| **loose_prompt_accuracy** | %s |" % summary["loose_prompt_accuracy"],
         "| strict_instruction_accuracy | %s |"
         % summary["strict_instruction_accuracy"],
         "| loose_instruction_accuracy | %s |"
         % summary["loose_instruction_accuracy"], ""]
    pc = summary.get("per_instruction_category") or {}
    if pc:
        L += ["## per instruction category", "",
              "| category | strict | loose | n |", "|---|---|---|---|"]
        for name, c in pc.items():
            L.append("| %s | %s | %s | %s |"
                     % (name, c["strict_instruction_accuracy"],
                        c["loose_instruction_accuracy"], c["num_instructions"]))
        L.append("")
    return "\n".join(L) + "\n"


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--response-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args(argv)

    evlib = import_ifeval_eval_lib()
    if evlib is None:
        print("ERROR: " + _INSTALL_HELP, file=sys.stderr)
        return 3
    strict_fn = evlib.test_instruction_following_strict
    loose_fn = evlib.test_instruction_following_loose

    inputs, skipped = load_prompts(args.input_jsonl,
                                   max_examples=args.max_examples)
    if not inputs:
        print("ERROR: no scoreable IFEval prompts (need instruction_id_list + "
              "kwargs) in %s" % args.input_jsonl, file=sys.stderr)
        return 3
    by_prompt, by_key = load_responses(args.response_jsonl)
    p2r, matched, missing = build_prompt_to_response(inputs, by_prompt, by_key)
    scored_inputs = [inp for inp in inputs if inp.prompt in p2r]

    strict_out, loose_out = evaluate(inputs, p2r, strict_fn, loose_fn)
    summary = aggregate_scores(strict_out, loose_out)
    combined, strict_recs, loose_recs = build_records(
        scored_inputs, strict_out, loose_out)

    meta = {
        "stage": "ifeval_offline_evaluation",
        "input_jsonl": args.input_jsonl,
        "response_jsonl": args.response_jsonl,
        "prompts_loaded": len(inputs),
        "prompts_skipped": skipped,
        "responses_matched": len(matched),
        "missing_responses": len(missing),
        "missing_response_keys": missing[:50],
        "evaluator": "official_instruction_following_eval",
        "model_rerun": False,
    }
    report = {**meta, **summary}

    jp = Path(args.output_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # combined per-example records + sibling strict/loose files for the analyzer
    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_jsonl, combined)
    strict_path = out_jsonl.with_name(out_jsonl.stem + "_strict.jsonl")
    loose_path = out_jsonl.with_name(out_jsonl.stem + "_loose.jsonl")
    _write_jsonl(strict_path, strict_recs)
    _write_jsonl(loose_path, loose_recs)
    mp = Path(args.output_md)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(_markdown(summary, meta), encoding="utf-8")

    print("=== IFEval offline evaluation ===")
    print("prompts=%d scored=%d missing=%d skipped=%d"
          % (len(inputs), len(scored_inputs), len(missing), skipped))
    print("strict_prompt_acc=%s loose_prompt_acc=%s "
          "strict_inst_acc=%s loose_inst_acc=%s"
          % (summary["strict_prompt_accuracy"], summary["loose_prompt_accuracy"],
             summary["strict_instruction_accuracy"],
             summary["loose_instruction_accuracy"]))
    print("records -> %s (+ %s, %s)"
          % (out_jsonl, strict_path.name, loose_path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
