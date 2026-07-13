# Final status

`FROM_SCRATCH_BLACKBOX_EVALUATION_PARTIAL`

Hard-stop reason: the frozen victim is `Qwen/Qwen2.5-0.5B (HF)`, not a backbone trained completely from scratch. Phase 0 therefore fails and prohibits attack execution.

## Completed

- Repository/branch/HEAD audit.
- Checkpoint, tokenizer, architecture and transformed-package provenance audit.
- Code/log search for public checkpoint loading and missing from-scratch evidence.
- Source hash inventory.
- Explicit partial tables and claim/limitation records.
- Verification that no GPU attack job was launched and no frozen artifact was overwritten.

## Missing cells

- Q0 remainder: B0/B1/B2/B3 runtime view enforcement; immutable attack-corpus schema/manifests; replay metadata checks; preregistered `delta_MS` and `delta_MIA` margins.
- Q1: one-time B0/B1/B2/B3 victim transcript collection and frozen reusable attack corpora.
- Phase 2: package replay R0, R1, R2, R3 and R4.
- Phase 3: query, logit, intermediate, attention, gradient/update, transformed-initialization and architecture-aware model stealing across all budgets and attacker initializations.
- Phase 4: A0 unrelated-public, A1 independent-from-scratch and A2 same-lineage-positive-control alignment attacks.
- Phase 5: direct, transfer, invariant, attention, gradient/dlogits, adapter-trajectory and confidence/loss MIA with multiple seeds and confidence intervals.
- Phase 6: C0/C1/C2/C3 cross-phase joint attacks and mask-refresh controls.
- Phase 7: attention-score leakage and B2-with/without-attention ablation.
- Phase 8: preregistered B2-B0 equivalence/non-inferiority tests.
- Phase 9: required B0/B3/ours/ObfuscaTune/permutation/no-mask/static-vs-refreshed baselines.
- Phase 10 numerical outputs: per-attack results, paired differences and confidence intervals.

No black-box or black-box-like security claim is made.
