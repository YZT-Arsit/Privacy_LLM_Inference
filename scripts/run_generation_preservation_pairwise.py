"""Pairwise open-ended generation preservation (folded current vs plaintext).

Consumes two per-backend generation reports from
``scripts/run_generation_preservation_benchmark.py`` -- the plaintext baseline and
the protected folded (current) candidate -- and emits a pairwise preservation
report (JSON/CSV/MD) matching examples by id:

* exact_text_match / exact_token_match (when token ids are available),
* normalized_edit_similarity, output_length_delta (chars + tokens),
* latency + audit carried from the candidate report.

CURRENT design only (a trusted_shortcut candidate is refused). No LLM judge, no
subjective quality scoring -- every metric is an objective string/token compare.

Example::

    python scripts/run_generation_preservation_pairwise.py \\
        --baseline-json out/gen_plaintext.json \\
        --candidate-json out/gen_folded.json \\
        --output-json out/gen_pairwise.json --output-md out/gen_pairwise.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_preservation import (  # noqa: E402
    load_json,
    pairwise_generation_preservation,
    render_pairwise_csv,
    render_pairwise_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-json", required=True,
                    help="plaintext_local generation report")
    ap.add_argument("--candidate-json", required=True,
                    help="folded_remote (current) generation report")
    ap.add_argument("--min-exact-token-match", type=float, default=0.95)
    ap.add_argument("--min-edit-similarity", type=float, default=0.95)
    ap.add_argument("--min-exact-text-match", type=float, default=0.90)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    baseline = load_json(args.baseline_json)
    candidate = load_json(args.candidate_json)
    if baseline is None:
        print("ERROR: cannot read baseline %s" % args.baseline_json,
              file=sys.stderr)
        return 3
    if candidate is None:
        print("ERROR: cannot read candidate %s" % args.candidate_json,
              file=sys.stderr)
        return 3

    try:
        report = pairwise_generation_preservation(
            baseline, candidate,
            min_exact_token_match_rate=args.min_exact_token_match,
            min_edit_similarity=args.min_edit_similarity,
            min_exact_text_match_rate=args.min_exact_text_match)
    except ValueError as exc:                                # current-only guard
        print("ERROR: %s" % exc, file=sys.stderr)
        return 3

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_csv:
        pc = Path(args.output_csv)
        pc.parent.mkdir(parents=True, exist_ok=True)
        pc.write_text(render_pairwise_csv(report), encoding="utf-8")
    if args.output_md:
        pm = Path(args.output_md)
        pm.parent.mkdir(parents=True, exist_ok=True)
        pm.write_text(render_pairwise_md(report), encoding="utf-8")

    a = report["aggregate"]
    print("=== generation preservation pairwise (%s vs %s) ==="
          % (report["candidate_backend"], report["baseline_backend"]))
    print("num_compared=%s token_ids_available=%s"
          % (a["num_compared"], report["token_ids_available"]))
    print("exact_text_match_rate=%s exact_token_match_rate=%s "
          "mean_edit_sim=%s" % (a["exact_text_match_rate"],
                                a["exact_token_match_rate"],
                                a["mean_normalized_edit_similarity"]))
    print("generation_preserved=%s paper_ready=%s"
          % (report["generation_preserved"], report["paper_ready"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
