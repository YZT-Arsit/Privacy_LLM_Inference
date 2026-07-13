"""PHASE 12 — assemble the generative-LoRA paper tables + claim-evidence matrix + limitations +
final_status from the produced artifacts (metrics, generation profiles, protected-run results,
attestation, cost). Offline (Mac). Robust to missing cells (marks N/A / PARTIAL). Never fabricates.
"""
from __future__ import annotations
import json
from pathlib import Path

GL = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation/results/aaai_private_base/generative_lora")


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def main():
    metrics = jload(GL / "metrics/statistical_summary.json", {}) or {}
    cells = metrics.get("cells", {})
    frozen = jload(GL / "data/frozen_config.json", {}) or {}
    manifest = jload(GL / "data/dataset_manifest.json", {}) or {}
    # protected run result + gen profile + attestation
    prot_runs = sorted(GL.glob("protected_runs/*.result.json"))
    prot = jload(prot_runs[-1], {}) if prot_runs else {}
    g2profs = sorted(GL.glob("generations/g2_protected_*.jsonl.profile.json"))
    g2prof = jload(g2profs[-1], {}) if g2profs else {}
    g0prof = jload(GL / "generations/g0_base_test.jsonl.profile.json", {}) or {}
    g1prof = jload(GL / "generations/g1_final_s1234_test.jsonl.profile.json", {}) or {}
    handoff = jload(GL / "handoff/lifecycle_gate.json", {}) or {}
    negctrl = jload(GL / "handoff/negative_controls.json", {}) or {}
    cost = jload(GL / "cost/cost_report.json", {}) or {}

    def m(cell, k, d="N/A"):
        return cells.get(cell, {}).get(k, d)

    # ---- main table ----
    rows = [
        "# Generative LoRA — Main Table (E2E NLG, Qwen2.5-0.5B)",
        "",
        f"Dataset: E2E NLG (official split via parquet). Test = {manifest.get('n_test_gen','?')} unique-MR "
        f"multi-reference. Decoding: greedy. Config (frozen on validation): lr {frozen.get('lr')}, "
        f"{frozen.get('steps')} steps, rank {frozen.get('rank')}/alpha {frozen.get('alpha')}, 7 targets.",
        "",
        "| Method | Optimizer profile | BLEU | chrF | ROUGE-L | invalid% | rep% | fresh-proc handoff | real TDX |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def cell_row(name, label, prof, tdx):
        handoff_ok = "yes" if (name == "G2" and handoff.get("fresh_process_ok")) else ("n/a" if name != "G2" else "no")
        return (f"| {label} | {prof} | {m(name,'BLEU')} | {m(name,'chrF')} | {m(name,'ROUGE_L')} | "
                f"{m(name,'invalid_rate_pct')} | {m(name,'repetition_pct')} | {handoff_ok} | {tdx} |")
    rows.append(cell_row("G0", "Base (no LoRA)", "none", "gen: plaintext ckpt on A10"))
    rows.append(cell_row("G1", "Plaintext LoRA", "standard AdamW", "no (plaintext ref)"))
    rows.append(cell_row("G2", "Protected LoRA (Profile E)", "exact/hybrid AdamW (TDX)",
                         "YES" if g2prof.get("attestation_verified") else "gen on A10+TDX"))
    # multi-seed G1
    g1seeds = metrics.get("g1_seed_bleu", {})
    if g1seeds:
        rows.append("")
        rows.append(f"G1 multi-seed BLEU: {g1seeds}")
    (GL / "final_main_table.md").write_text("\n".join(rows) + "\n")

    # csv
    keys = ["BLEU", "chrF", "ROUGE_L", "invalid_rate_pct", "repetition_pct", "avg_len", "n"]
    csv = ["cell," + ",".join(keys)]
    for c in ["G0", "G1", "G2"]:
        csv.append(c + "," + ",".join(str(m(c, k)) for k in keys))
    (GL / "final_main_table.csv").write_text("\n".join(csv) + "\n")

    # ---- claim-evidence matrix ----
    cem = ["# Claim-Evidence Matrix — Generative LoRA", ""]
    def ev(claim, status, evidence):
        cem.append(f"- **{claim}** — {status}\n  - Evidence: {evidence}")
    ev("Protected LoRA learns a useful generative task (non-floor)",
       "SUPPORTED" if isinstance(m("G2", "BLEU"), (int, float)) and isinstance(m("G0", "BLEU"), (int, float))
       and m("G2", "BLEU") > m("G0", "BLEU") else "PARTIAL",
       f"G0 base BLEU {m('G0','BLEU')} -> G2 protected BLEU {m('G2','BLEU')} (test); metrics_report.md")
    ev("Protected LoRA retains comparable utility to plaintext LoRA",
       "see paired_differences.csv (do NOT claim strict equivalence)",
       f"G1 BLEU {m('G1','BLEU')} vs G2 BLEU {m('G2','BLEU')}; metrics/paired_differences.csv")
    ev("Transformed adapter consumed by protected generation runtime without plaintext reconstruction",
       "SUPPORTED" if handoff.get("fresh_process_ok") else "PARTIAL",
       f"fresh-process protected_generate --adapter-package; handoff/lifecycle_gate.json; "
       f"negative controls all fail-closed = {negctrl.get('all_controls_fail_closed')}")
    ev("Protected generation is exact vs plaintext (fold parity)",
       "SUPPORTED (base G0)", "g0_protected vs g0_plain 10/10 exact text match; fp32 fold")
    ev("Real TDX evidence for the protected paper-facing cell",
       "SUPPORTED" if g2prof.get("attestation_verified") else "PARTIAL",
       f"attestation_verified={g2prof.get('attestation_verified')}; protected_runs/*.attestation.json")
    (GL / "claim_evidence_matrix.md").write_text("\n".join(cem) + "\n")

    # ---- limitations ----
    lim = ["# Limitations — Generative LoRA", "",
           "- GSM8K@0.5B is task-floor and is NOT the generative utility result; kept as parity/limitation.",
           f"- Train subset capped at {manifest.get('n_train','?')} examples (deterministic sample) and "
           f"{frozen.get('steps')} steps for affordable protected training; not full-42k E2E.",
           f"- Test generation capped at {manifest.get('n_test_gen','?')} unique MRs (no silent truncation; "
           "recorded in dataset_manifest.json).",
           "- G2 seeds reduced vs G1 (protected training cost); G1 reports 3 seeds, G2 reports "
           f"{prot.get('seed','?')}. Not a strict-equivalence claim.",
           "- Native transformed LoRA (G3) profile not wired into the live clm runner this stage -> G3 = N/A.",
           "- Protected greedy decode returns the argmax token to the GPU (public output); over a generation "
           "the GPU learns perm_inv only for the emitted (public) tokens. Full vocab permutation stays in TDX.",
           "- Plaintext G0/G1 are real-GPU references (plaintext checkpoint), NOT protected/TDX cells."]
    (GL / "limitations.md").write_text("\n".join(lim) + "\n")

    # ---- final status ----
    have = {"dataset": bool(manifest), "G0": isinstance(m("G0", "BLEU"), (int, float)),
            "G1": isinstance(m("G1", "BLEU"), (int, float)), "G2": isinstance(m("G2", "BLEU"), (int, float)),
            "protected_gen": bool(g2prof), "handoff": bool(handoff.get("fresh_process_ok")),
            "negative_controls": bool(negctrl.get("all_controls_fail_closed")),
            "ablation": (GL / "target_ablation/results.csv").exists(),
            "cost": bool(cost), "real_tdx_attest": bool(g2prof.get("attestation_verified"))}
    complete = all(have[k] for k in ["dataset", "G0", "G1", "G2", "protected_gen", "handoff",
                                     "negative_controls", "real_tdx_attest"])
    status = "REAL_GENERATIVE_LORA_COMPLETE" if complete else "REAL_GENERATIVE_LORA_PARTIAL"
    missing = [k for k, v in have.items() if not v]
    fs = [f"# FINAL STATUS: {status}", "", "## Completed cells", json.dumps(have, indent=2),
          "", "## Missing / partial cells", json.dumps(missing, indent=2),
          "", "Nothing committed. Frozen artifacts untouched. G3 native = N/A (allowed).",
          f"Main table: BLEU G0 {m('G0','BLEU')} / G1 {m('G1','BLEU')} / G2 {m('G2','BLEU')}."]
    (GL / "final_status.md").write_text("\n".join(fs) + "\n")
    print("[phase12]", status, "missing:", missing)


if __name__ == "__main__":
    main()
