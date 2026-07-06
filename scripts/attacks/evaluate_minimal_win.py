"""Minimal-win evaluator: does any ours candidate strictly beat ObfuscaTune?

Reads a measured security table (security_attack_table_measured.json) and, per
ours candidate column, compares it to obfuscatune_qwen_orthogonal row-by-row.
Leakage metrics are "higher = more leakage" (attack success / recovery), so a
STRICT WIN for ours means ours_cell + threshold <= obfuscatune_cell.

Rules (per the round spec):
  * default threat model rows vs weight-leakage worst-case rows are separated;
    worst-case (ArrowMatch/Gram) NEVER counts toward the default verdict.
  * if either side is missing/blocked/non-numeric on a row, that row is not a win
    for anyone (a blocked attack is not scored as a defense win).
  * threshold default 0.1 on [0,1] leakage/recovery metrics.

Verdict:
  * defense_claim_supported            — some candidate has >=1 default strict win
                                         and no default strict loss.
  * defense_claim_supported_with_tradeoffs — some candidate has >=1 default win but
                                         also >=1 default loss.
  * needs_more_measurements            — no numeric default comparisons at all.
  * pivot_to_security_characterization_recommended — comparisons exist but no
                                         candidate achieves a default strict win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORST_CASE = "weight_leakage_worst_case"


def _num(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _compare(rows, obf_col, ours_col, threshold):
    default_wins, default_losses, worst_wins, worst_losses = [], [], [], []
    skipped = []
    for r in rows:
        probe = r.get("attack_probe")
        o, c = _num(r.get(obf_col)), _num(r.get(ours_col))
        bucket_w, bucket_l = ((worst_wins, worst_losses) if r.get("threat_model") == WORST_CASE
                              else (default_wins, default_losses))
        if o is None or c is None:
            skipped.append({"probe": probe, "obfuscatune": r.get(obf_col), "ours": r.get(ours_col),
                            "threat_model": r.get("threat_model")})
            continue
        if o - c >= threshold:
            bucket_w.append({"probe": probe, "obfuscatune": o, "ours": c, "delta": o - c,
                             "threat_model": r.get("threat_model")})
        elif c - o >= threshold:
            bucket_l.append({"probe": probe, "obfuscatune": o, "ours": c, "delta": c - o,
                             "threat_model": r.get("threat_model")})
    return {"default_wins": default_wins, "default_losses": default_losses,
            "worst_case_wins": worst_wins, "worst_case_losses": worst_losses,
            "skipped_rows": skipped}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True, help="security_attack_table_measured.json")
    p.add_argument("--obfuscatune-col", default="obfuscatune_qwen_orthogonal")
    p.add_argument("--ours-cols", default="ours_amulet_style_signed_perm,"
                   "ours_amulet_style_fresh_pad,ours_non_isometric_variant")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    table = json.loads(Path(args.table).read_text())
    rows = table["rows"]
    ours_cols = [c.strip() for c in args.ours_cols.split(",") if c.strip()]
    present = set(table.get("methods", []))
    per_candidate = {}
    for col in ours_cols:
        if col not in present:
            per_candidate[col] = {"status": "column_absent"}
            continue
        per_candidate[col] = _compare(rows, args.obfuscatune_col, col, args.threshold)

    # verdict
    any_numeric = any(isinstance(v, dict) and (v.get("default_wins") or v.get("default_losses"))
                      for v in per_candidate.values())
    clean_win = [c for c, v in per_candidate.items()
                 if isinstance(v, dict) and v.get("default_wins") and not v.get("default_losses")]
    tradeoff_win = [c for c, v in per_candidate.items()
                    if isinstance(v, dict) and v.get("default_wins") and v.get("default_losses")]
    if clean_win:
        verdict = "defense_claim_supported"
    elif tradeoff_win:
        verdict = "defense_claim_supported_with_tradeoffs"
    elif not any_numeric:
        verdict = "needs_more_measurements"
    else:
        verdict = "pivot_to_security_characterization_recommended"

    report = {
        "verdict": verdict,
        "threshold": args.threshold,
        "obfuscatune_column": args.obfuscatune_col,
        "candidates_with_clean_default_win": clean_win,
        "candidates_with_tradeoff_default_win": tradeoff_win,
        "per_candidate": per_candidate,
        "recommended_paper_positioning": (
            "Claim a concrete, threat-model-scoped advantage: ours fresh/non-isometric "
            "variants strictly resist the known-plaintext (static-mask) attack that breaks "
            "ObfuscaTune's fixed R." if clean_win or tradeoff_win else
            "No strict default-threat-model advantage found; reposition as a systematic "
            "security characterization of linear/orthogonal/permutation/norm-preserving "
            "obfuscation (what each leaks and under which threat model), not a blanket "
            "'ours beats ObfuscaTune' claim."),
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "minimal_win_report.json").write_text(json.dumps(report, indent=2, default=str))

    md = ["# Minimal-win report", "", f"**Verdict: {verdict}**", "",
          f"threshold={args.threshold}; baseline column = `{args.obfuscatune_col}`", ""]
    for col, v in per_candidate.items():
        md.append(f"## {col}")
        if v.get("status") == "column_absent":
            md += ["", "_column absent from table_", ""]; continue
        def _fmt(items):
            return "; ".join(f"{i['probe']}(obf={i['obfuscatune']:.3f} vs ours={i['ours']:.3f}, "
                             f"Δ={i['delta']:.3f})" for i in items) or "none"
        md += ["",
               f"- default strict wins: {_fmt(v['default_wins'])}",
               f"- default strict losses: {_fmt(v['default_losses'])}",
               f"- worst-case wins (NOT default): {_fmt(v['worst_case_wins'])}",
               f"- worst-case losses (NOT default): {_fmt(v['worst_case_losses'])}",
               f"- skipped (missing/blocked either side): "
               f"{', '.join(s['probe'] for s in v['skipped_rows']) or 'none'}", ""]
    md += ["## Recommended paper positioning", "", report["recommended_paper_positioning"], ""]
    (out / "minimal_win_report.md").write_text("\n".join(md) + "\n")
    print(f"VERDICT: {verdict}")
    print(f"clean default wins: {clean_win}; tradeoff: {tradeoff_win}")
    print(f"wrote {out}/minimal_win_report.json/.md")


if __name__ == "__main__":
    main()
