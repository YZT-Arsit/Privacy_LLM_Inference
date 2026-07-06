"""Norm-leakage audit on real Qwen: why is per-token norm=1.0 and what breaks it.

Writes outputs/baseline_audit/norm_leakage_audit.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--num-tokens", type=int, default=64)
    p.add_argument("--non-isometric-cond", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "baseline_audit"
    if args.dry_run:
        print(f"[dry-run] norm audit model={args.model_name_or_path or 'tiny_random_qwen'} -> {out}")
        return

    import torch
    from pllo.attacks.norm_leakage_audit import audit_norm_leakage
    from pllo.attacks.real_qwen_representations import build_real_representations
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed, dtype=torch.float32)
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=args.seed, dtype=torch.float32)
    ids = torch.randint(0, model.config.vocab_size, (1, args.num_tokens),
                        generator=torch.Generator().manual_seed(args.seed))
    reps = build_real_representations(model, ids, seed=args.seed,
                                      non_isometric_cond=args.non_isometric_cond)
    rep = audit_norm_leakage(reps)
    rep["model"] = args.model_name_or_path or "tiny_random_qwen"

    out.mkdir(parents=True, exist_ok=True)
    (out / "norm_leakage_audit.json").write_text(json.dumps(rep, indent=2, default=str))
    cols = ["method", "transform_type", "is_norm_preserving_theoretically",
            "whether_fresh_per_token", "whether_kronecker_enabled",
            "norm_preserved_exactly_max_rel_err", "measured_norm_leakage_pearson",
            "breaks_norm_leakage"]
    md = ["# Norm-leakage audit (real Qwen)", "", f"model: {rep['model']}", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for m, r in rep["per_method"].items():
        md.append("| " + " | ".join(
            (f"{r[c]:.3e}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    md += ["", "## Recommendation", "", rep["recommendation"]]
    if rep["norm_leakage_breakers"]:
        md += ["", f"**Breaks norm leakage:** {', '.join(rep['norm_leakage_breakers'])}"]
    else:
        md += ["", "**Breaks norm leakage:** none of the measured variants."]
    (out / "norm_leakage_audit.md").write_text("\n".join(md) + "\n")
    for m, r in rep["per_method"].items():
        print(f"{m:34s} preserve_err={r['norm_preserved_exactly_max_rel_err']:.2e} "
              f"corr={r['measured_norm_leakage_pearson']:.3f} breaks={r['breaks_norm_leakage']}")
    print(f"wrote {out}/norm_leakage_audit.json/.md")


if __name__ == "__main__":
    main()
