"""Fetch + localise the optional AAAI datasets (run on TDX / a networked host).

NEVER run on the H800 (no internet). Reads a LOCAL raw file (already downloaded)
or, only if ``--source-url`` is given, downloads it on this networked host, then
normalises to a unified JSONL + writes a dataset card with sha256s. The converted
JSONL + cards are then ``rsync``'d to the H800.

Datasets:
* ``humaneval`` -- ``{id, dataset, prompt, entry_point, test, canonical_solution, meta}``
* ``mbpp`` (with ``--include-mbpp``) -- ``{id, dataset, prompt, test_list, code, meta}``
* ``longbench_1024_lite`` -- 1024-token-capped stress set; the card marks
  ``not_official_longbench_score=true`` and ``used_for="latency/security/long-prompt-stress"``.

Examples::

    python scripts/fetch_aaai_optional_datasets.py --dataset humaneval \\
      --raw-input <RAW_HUMANEVAL_JSONL> --output-dir <CONVERTED_DIR>
    python scripts/fetch_aaai_optional_datasets.py --dataset longbench_1024_lite \\
      --raw-input <RAW_LONGBENCH_JSONL> --max-seq-len 1024 --output-dir <DIR>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_datasets import (  # noqa: E402
    build_dataset_card, load_humaneval, load_longbench_1024_lite, load_mbpp,
    sha256_file)


def _maybe_download(source_url, dest):
    """Download on THIS networked host only (never the H800). Returns the path."""
    import urllib.request
    print("[fetch] downloading %s -> %s" % (source_url, dest))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(source_url, dest)        # noqa: S310 (TDX host)
    return dest


def _truncate_longbench(rows, max_seq_len):
    """Cap each prompt to <= max_seq_len whitespace tokens; mark truncation."""
    out = []
    for r in rows:
        words = str(r.get("prompt", "")).split()
        orig = len(words)
        truncated = orig > max_seq_len
        if truncated:
            r = dict(r, prompt=" ".join(words[:max_seq_len]))
        r["original_length_estimate"] = orig
        r["truncated"] = truncated
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["humaneval", "mbpp", "longbench_1024_lite"])
    ap.add_argument("--raw-input", default=None,
                    help="local raw file (preferred; no network)")
    ap.add_argument("--source-url", default=None,
                    help="optional download URL (THIS networked host only)")
    ap.add_argument("--raw-cache", default=None,
                    help="where to save a downloaded raw file")
    ap.add_argument("--include-mbpp", action="store_true", default=False)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--license", default=None)
    args = ap.parse_args()

    if args.dataset == "mbpp" and not args.include_mbpp:
        print("NOTE: MBPP is optional; pass --include-mbpp to convert it.",
              file=sys.stderr)
        return 0

    raw = args.raw_input
    if raw is None and args.source_url:
        raw = _maybe_download(args.source_url,
                              args.raw_cache or (Path(args.output_dir)
                                                 / ("raw_%s" % args.dataset)))
    if not raw:
        print("ERROR: provide --raw-input (local) or --source-url (networked host)",
              file=sys.stderr)
        return 3

    if args.dataset == "humaneval":
        rows = load_humaneval(raw)
    elif args.dataset == "mbpp":
        rows = load_mbpp(raw)
    else:
        rows = _truncate_longbench(load_longbench_1024_lite(raw), args.max_seq_len)
    if args.max_examples and args.max_examples > 0:
        rows = rows[:args.max_examples]

    out_dir = Path(args.output_dir)
    cards = out_dir / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / ("%s.jsonl" % args.dataset)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = build_dataset_card(dataset_name=args.dataset, split="test", rows=rows,
                              source_path=raw, output_path=out_jsonl)
    card["source"] = args.source_url or str(raw)
    card["source_sha256"] = sha256_file(raw) if Path(raw).exists() else None
    card["license"] = args.license
    if args.dataset == "longbench_1024_lite":
        card["not_official_longbench_score"] = True
        card["used_for"] = "latency/security/long-prompt-stress"
        card["max_seq_len"] = args.max_seq_len
    Path(cards / ("%s_card.json" % args.dataset)).write_text(
        json.dumps(card, indent=2), encoding="utf-8")
    print("[fetch] %s: %d examples -> %s" % (args.dataset, len(rows), out_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
