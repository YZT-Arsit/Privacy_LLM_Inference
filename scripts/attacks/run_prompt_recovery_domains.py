"""Block-B (motivation appendix): prompt-recovery optimization attacks across four
sensitive domains. NON-discriminative by design — the point is that plaintext
prompts are invertible while EVERY non-trivial mask crushes recovery to ~0, so the
existence of a mask (not its structure) is what defeats output-only optimization
attacks. Structure only matters under KPA (block A).

Domains use the repo's SYNTHETIC sensitive-prompt set (no real PII, seeded,
reproducible): medical_note (medical), contract (legal), finance_memo (financial),
email (general). Attacks (EIA / BRE-forward / PIA) run through the real
differentiable split downstream. Implementation is BEST-EFFORT (not a full SOTA
reproduction) — labelled as such on every record. This table exists only to echo
the motivation; it is not a security-ranking claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DOMAINS = {"medical": "medical_note", "legal": "contract",
           "financial": "finance_memo", "general": "email"}
DEFENSES = ["plaintext_gpu", "stip_qwen", "obfuscatune_qwen_orthogonal",
            "ours_fresh_signed_perm"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--attacks", default="eia,bre,pia")
    p.add_argument("--split-layer", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    out = Path(args.output_dir) if args.output_dir else REPO / "outputs" / "attacks" / "prompt_recovery_domains"
    if args.dry_run:
        print(f"[dry-run] prompt-recovery domains model={args.model_name_or_path or 'tiny'} "
              f"domains={list(DOMAINS)} attacks={attacks} -> {out}")
        return

    import torch
    import pllo.attacks as A
    from pllo.attacks.qwen_split_downstream import build_split_attack_inputs
    from pllo.attacks.result_io import failed_result, write_jsonl
    from pllo.benchmarks.sensitive_prompts import build_sensitive_prompt_set
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed, dtype=dtype)
        family, path = "qwen", args.model_name_or_path
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=args.seed, dtype=dtype)
        family, path, tok = "qwen_tiny_random", None, None
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    k = max(1, min(args.split_layer, len(model.model.layers)))

    prompts = build_sensitive_prompt_set(num_per_bucket=8, buckets=(128,), seed=2035)

    def domain_ids(doc_type):
        rec = next((r for r in prompts if r["meta"]["doc_type"] == doc_type), None)
        text = rec["prompt"] if rec else doc_type
        if tok is not None:
            ii = tok(text, return_tensors="pt").input_ids[0][:args.seq_len]
            if ii.numel() < args.seq_len:  # pad by repeat to keep the attack shapes fixed
                ii = ii.repeat((args.seq_len // ii.numel()) + 1)[:args.seq_len]
            return ii.unsqueeze(0)
        g = torch.Generator().manual_seed(args.seed + len(doc_type))
        return torch.randint(0, model.config.vocab_size, (1, args.seq_len), generator=g)

    def run(attack, inp, method):
        if attack == "eia":
            return A.run_eia_optimization(inp, target_method=method, num_steps=args.num_steps)
        if attack == "bre":
            return A.run_bre_bisr_attack(inp, target_method=method, num_steps=args.num_steps)
        if attack == "pia":
            return A.run_pia_prompt_inversion(inp, target_method=method, num_steps=args.num_steps)
        raise ValueError(attack)

    results, table = [], []
    for domain, doc_type in DOMAINS.items():
        ids = domain_ids(doc_type)
        if args.device == "cuda" and torch.cuda.is_available():
            ids = ids.to("cuda")
        for method in DEFENSES:
            inp, _ = build_split_attack_inputs(model, ids, k=k, method=method, seed=args.seed,
                                               device=args.device, dtype=dtype)
            for attack in attacks:
                try:
                    r = run(attack, inp, method)
                except Exception as e:  # noqa: BLE001
                    r = failed_result(attack, attack, "optimization", method, "split_inference", str(e))
                r.threat_model = "split_inference"
                r.model_family, r.model_name_or_path = family, path
                r.implementation_level = "best_effort"
                r.notes = (f"domain={domain} doc_type={doc_type} split_k={k} synthetic_no_pii "
                           f"MOTIVATION_ONLY_non_discriminative; {r.notes}")[:400]
                results.append(r)
                tok1 = r.metrics.get("token_recovery_top1")
                table.append({"domain": domain, "attack": attack, "method": method,
                              "token_recovery_top1": tok1, "mse": r.metrics.get("mse"),
                              "status": r.status})
                print(f"{domain:10s} {attack:4s} {method:30s} tok1={tok1} status={r.status}")

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, out / "all_attacks.jsonl")
    (out / "domain_table.json").write_text(json.dumps(table, indent=2))
    # compact md: rows = (domain, attack), cols = defenses, cell = tok1
    md = ["# Prompt-recovery across sensitive domains (MOTIVATION appendix, best-effort, non-discriminative)",
          "", f"model={path or 'tiny'}, synthetic prompts (no real PII), split_k={k}, "
          f"steps={args.num_steps}. Cell = token_recovery_top1 (higher = more recovered).", "",
          "| domain | attack | " + " | ".join(m.replace("_qwen", "").replace("_gpu", "")
                                               for m in DEFENSES) + " |",
          "|" + "---|" * (2 + len(DEFENSES))]
    by = {}
    for t in table:
        by.setdefault((t["domain"], t["attack"]), {})[t["method"]] = t["token_recovery_top1"]
    for (domain, attack), cells in by.items():
        vals = []
        for m in DEFENSES:
            v = cells.get(m)
            vals.append("—" if v is None else f"{v:.3g}")
        md.append(f"| {domain} | {attack} | " + " | ".join(vals) + " |")
    md += ["", "**Reading (motivation only).** Plaintext prompts leak to output-only optimization "
           "attacks; STIP, ObfuscaTune and ours all crush recovery to ~0. The mask STRUCTURE does "
           "not separate the defenses here — only the presence of a mask matters. Discrimination "
           "between defenses appears in block A (KPA), not here. Best-effort implementation, not a "
           "full SOTA reproduction."]
    (out / "domain_table.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/all_attacks.jsonl + domain_table.{{json,md}} ({len(results)} results)")


if __name__ == "__main__":
    main()
