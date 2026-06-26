"""Diagnose the IFEval strict-vs-loose gap between plaintext and folded runs.

Loose IFEval can match the plaintext baseline while strict still trails. Strict
scoring is unforgiving about output FORMAT (leading/trailing whitespace, wrapping
quotes, casing, bullet/paragraph structure, prompt echo, ...), so a folded run that
answers correctly can still fail strict on formatting. This tool compares the eval
JSONL of two runs (plaintext vs folded) for strict and (optionally) loose, and
surfaces exactly which examples + which instruction categories diverge, plus the
response-format features that drive strict failures.

Inputs are the IFEval evaluator output JSONL (one record per example with
``prompt``, ``response``, ``instruction_id_list``, ``follow_instruction_list``,
``follow_all_instructions``). It analyses ONLY final text responses + evaluator
results -- never logits / hidden / masks / secrets.

Outputs a markdown summary + a JSON summary:
* plaintext-pass-but-folded-fail / folded-pass-but-plaintext-fail example ids;
* per-example instruction_id_list + both follow_instruction_lists;
* failed-instruction category counts (by instruction-id prefix);
* per-response format features (length, whitespace, special-char start/end,
  paragraphs, bullets, quote-wrapping, prompt echo, commas, casing).

Example::

    python scripts/analyze_ifeval_strict_gap.py \\
        --plaintext-strict outputs/plaintext_chat_20/eval_results_strict.jsonl \\
        --folded-strict    outputs/folded_aligned_eosstop_20/eval_results_strict.jsonl \\
        --plaintext-loose  outputs/plaintext_chat_20/eval_results_loose.jsonl \\
        --folded-loose     outputs/folded_aligned_eosstop_20/eval_results_loose.jsonl \\
        --output-md   outputs/debug/ifeval_strict_gap.md \\
        --output-json outputs/debug/ifeval_strict_gap.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
_QUOTES = ('"', "'", "“", "”", "‘", "’", "`")


def _load(path):
    """Load an IFEval eval JSONL -> dict keyed by example id (key/id/prompt)."""
    out = {}
    order = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key", rec.get("id", rec.get("prompt", "ex-%d" % i)))
            key = str(key)
            fil = rec.get("follow_instruction_list")
            rec["_follow"] = list(fil) if isinstance(fil, list) else None
            fa = rec.get("follow_all_instructions")
            if fa is None and rec["_follow"] is not None:
                fa = all(rec["_follow"])
            rec["_pass"] = bool(fa) if fa is not None else None
            rec["_iids"] = list(rec.get("instruction_id_list") or [])
            out[key] = rec
            order.append(key)
    return out, order


def _features(response, prompt):
    """Format features of a single response (strict-eval-relevant), text only."""
    r = response if isinstance(response, str) else ""
    stripped = r.strip()
    words = r.split()
    lines = r.splitlines()
    paragraphs = [p for p in re.split(r"\n\s*\n", r) if p.strip()]
    bullets = [ln for ln in lines if _BULLET.match(ln)]
    upper = sum(1 for c in r if c.isalpha() and c.isupper())
    lower = sum(1 for c in r if c.isalpha() and c.islower())
    first = stripped[:1]
    last = stripped[-1:]
    quote_wrapped = bool(stripped and stripped[0] in _QUOTES
                         and stripped[-1] in _QUOTES)
    echoes = False
    if isinstance(prompt, str) and prompt.strip():
        p = prompt.strip()
        echoes = bool(r.strip().startswith(p[:40]) or (len(p) > 20 and p in r))
    return {
        "char_len": len(r),
        "word_count": len(words),
        "line_count": len(lines),
        "has_leading_whitespace": bool(r[:1].isspace()),
        "has_trailing_whitespace": bool(r[-1:].isspace()),
        "startswith_char": first,
        "startswith_special": bool(first and not first.isalnum()
                                   and not first.isspace()),
        "endswith_char": last,
        "endswith_special": bool(last and not last.isalnum()
                                 and not last.isspace()),
        "paragraph_count": len(paragraphs),
        "bullet_count": len(bullets),
        "quote_wrapped": quote_wrapped,
        "echoes_prompt": echoes,
        "comma_count": r.count(","),
        "uppercase_letters": upper,
        "lowercase_letters": lower,
        "is_all_lowercase": bool(upper == 0 and lower > 0),
        "has_uppercase": bool(upper > 0),
    }


def _category(iid):
    return str(iid).split(":", 1)[0]


def _compare(plain, folded, plain_order):
    """Per-example comparison over the union of ids (plaintext order first)."""
    ids = list(plain_order) + [k for k in folded if k not in plain]
    p_pass_f_fail, f_pass_p_fail = [], []
    failed_cat = {}              # category -> count (folded failed, plaintext ok)
    per_example = []
    for key in ids:
        pe = plain.get(key)
        fe = folded.get(key)
        if pe is None or fe is None:
            continue
        pp, fp = pe["_pass"], fe["_pass"]
        if pp and fp is False:
            p_pass_f_fail.append(key)
        if fp and pp is False:
            f_pass_p_fail.append(key)
        # per-instruction failed categories (folded failed where plaintext passed)
        iids = pe["_iids"] or fe["_iids"]
        pf, ff = pe["_follow"], fe["_follow"]
        if iids and pf and ff and len(pf) == len(iids) and len(ff) == len(iids):
            for j, iid in enumerate(iids):
                if pf[j] and not ff[j]:
                    cat = _category(iid)
                    failed_cat[cat] = failed_cat.get(cat, 0) + 1
        per_example.append({
            "id": key,
            "prompt": pe.get("prompt"),
            "instruction_id_list": iids,
            "plaintext_pass": pp, "folded_pass": fp,
            "plaintext_follow_instruction_list": pf,
            "folded_follow_instruction_list": ff,
            "plaintext_features": _features(pe.get("response"), pe.get("prompt")),
            "folded_features": _features(fe.get("response"), fe.get("prompt")),
        })
    return {
        "plaintext_pass_folded_fail": p_pass_f_fail,
        "folded_pass_plaintext_fail": f_pass_p_fail,
        "failed_instruction_category_counts": dict(
            sorted(failed_cat.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_example": per_example,
        "num_examples": len(per_example),
    }


def _rate(d):
    vals = [e for e in d.values() if e.get("_pass") is not None]
    if not vals:
        return None
    return round(sum(1 for e in vals if e["_pass"]) / len(vals), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plaintext-strict", required=True)
    ap.add_argument("--folded-strict", required=True)
    ap.add_argument("--plaintext-loose", default=None)
    ap.add_argument("--folded-loose", default=None)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    ps, ps_order = _load(args.plaintext_strict)
    fs, _ = _load(args.folded_strict)
    strict = _compare(ps, fs, ps_order)
    strict["plaintext_pass_rate"] = _rate(ps)
    strict["folded_pass_rate"] = _rate(fs)

    loose = None
    if args.plaintext_loose and args.folded_loose:
        pl, pl_order = _load(args.plaintext_loose)
        fl, _ = _load(args.folded_loose)
        loose = _compare(pl, fl, pl_order)
        loose["plaintext_pass_rate"] = _rate(pl)
        loose["folded_pass_rate"] = _rate(fl)

    report = {
        "stage": "ifeval_strict_gap_analysis",
        "inputs": {"plaintext_strict": args.plaintext_strict,
                   "folded_strict": args.folded_strict,
                   "plaintext_loose": args.plaintext_loose,
                   "folded_loose": args.folded_loose},
        "strict": strict, "loose": loose,
        "analysis_scope": "final_text_responses_and_evaluator_results_only",
        "no_secret_fields": True,
    }

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(_markdown(report), encoding="utf-8")

    print("=== IFEval strict-gap analysis ===")
    print("strict pass: plaintext=%s folded=%s" % (strict["plaintext_pass_rate"],
                                                   strict["folded_pass_rate"]))
    if loose:
        print("loose  pass: plaintext=%s folded=%s" % (loose["plaintext_pass_rate"],
                                                       loose["folded_pass_rate"]))
    print("strict plaintext-pass-but-folded-fail: %s"
          % strict["plaintext_pass_folded_fail"])
    print("strict failed instruction categories: %s"
          % strict["failed_instruction_category_counts"])
    return 0


def _fmt_diff(e):
    """One-line per-example format-feature diff (plaintext vs folded)."""
    pf, ff = e["plaintext_features"], e["folded_features"]
    keys = ["char_len", "word_count", "has_leading_whitespace",
            "has_trailing_whitespace", "startswith_special", "endswith_special",
            "paragraph_count", "bullet_count", "quote_wrapped", "echoes_prompt",
            "comma_count", "has_uppercase", "is_all_lowercase"]
    diffs = ["%s: %s|%s" % (k, pf.get(k), ff.get(k)) for k in keys
             if pf.get(k) != ff.get(k)]
    return ", ".join(diffs) if diffs else "(no format-feature differences)"


def _markdown(r):
    L = ["# IFEval strict-gap analysis", "",
         "Scope: final text responses + evaluator results only (no secrets).", ""]
    for name in ("strict", "loose"):
        sec = r.get(name)
        if not sec:
            continue
        L += ["## %s" % name.upper(), "",
              "- pass rate: plaintext=%s folded=%s"
              % (sec.get("plaintext_pass_rate"), sec.get("folded_pass_rate")),
              "- plaintext-pass-but-folded-fail: %s"
              % (sec["plaintext_pass_folded_fail"] or "[]"),
              "- folded-pass-but-plaintext-fail: %s"
              % (sec["folded_pass_plaintext_fail"] or "[]"),
              "- failed instruction category counts: %s"
              % (sec["failed_instruction_category_counts"] or "{}"), ""]
        focus = set(sec["plaintext_pass_folded_fail"])
        if focus:
            L += ["### %s: plaintext-pass / folded-fail examples" % name, ""]
            for e in sec["per_example"]:
                if e["id"] not in focus:
                    continue
                L += ["- **%s**  instructions=%s" % (e["id"], e["instruction_id_list"]),
                      "  - plaintext_follow=%s folded_follow=%s"
                      % (e["plaintext_follow_instruction_list"],
                         e["folded_follow_instruction_list"]),
                      "  - format diff (plaintext|folded): %s" % _fmt_diff(e)]
            L += [""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
