# Claim–Evidence Matrix (v2)

| Claim | Status | Evidence | Guardrail |
|---|---|---|---|
| Protected LoRA achieves competitive corpus-level generative quality | SUPPORTED WITH METRIC QUALIFICATION | Mean corpus BLEU: G1 64.076, G2 64.559; mean Δ +0.483 | Corpus chrF is lower for G2 on average; report the full metric table |
| Mean corpus BLEU is close to plaintext across three seeds | SUPPORTED DESCRIPTIVELY | Three-seed mean Δ +0.483 | “Close” is descriptive; no equivalence margin was preregistered |
| One seed improves and two show no significant corpus-BLEU difference | SUPPORTED | Seed 1234 CI excludes zero; seed 7 and 2025 CIs include zero | Within-seed paired bootstrap inference only |
| Protected runs exhibit greater across-seed variability | SUPPORTED DESCRIPTIVELY | Corpus-BLEU SD: G2 1.362 vs G1 0.206 | Only three seeds |
| Strict equivalence | NOT ESTABLISHED | No equivalence margin or TOST | Do not infer equivalence from failure to reject zero difference |
| BF16-matched interpretation | PENDING | No matched terminal artifact | Await GPU owner |
| Final T1–T4 cost-quality result | PENDING | Only T1 terminal | T2/T3 running; T4 pending at snapshot |
| Deployable transformed-adapter closure | PENDING | Preparation/negative controls only | Requires deployment artifact |
| Corrected MIA result | PENDING | CPU evaluator ready; shadow collections absent | Old source-confounded AUC=1 excluded |
| ObfuscaTune system comparison | PENDING | 52 CPU fidelity tests only | No matched system measurement |
