# Private-base experiment matrix (FROZEN v1.0)

Order and gating for the private-base main line. **Checkpoint rule:** stop after
items 1–6 and report before launching the first new H800 run. W1/S1 infrastructure
is implemented before (not after) any long stability run.

## Immediate execution order
| # | id | task | compute | gate |
|---|---|---|---|---|
| 1 | — | design_spec + threat_model freeze | local | — |
| 2 | — | code-path provenance audit | local | FAIL if any private-base run loads a plaintext checkpoint in the GPU worker |
| 3 | — | trusted packager + plaintext-absence scan + `private_base_required` guard | local/offline | scan must show zero plaintext tensors in the GPU package |
| 4 | — | LoRA-aware differentiable folded forward (7 projections, 3 mask spaces) | local | no plaintext HF forward in protected path |
| 5 | C1 | CPU full-block/full-model correctness + counters | CPU | all counters 0; forward/grad recover to fp tol |
| 6 | O1 | optimizer equivalence matrix (A/B/C) + preconditioner leakage audit | local | O1-B not paper-safe until leakage evaluated |
| — | **REPORT + STOP** | first checkpoint report | — | before first H800 run |
| 7 | — | real 0.5B unified **one-step** private-base run | H800+TDX | one-step passes |
| 8 | W1 | residual-chain private-weight alignment attack (no_anchor/output_anchor/oracle_anchor) | local | **go/no-go**: if no_anchor or output_anchor recovers a functional model, private-base claim FAILS |
| 9 | S1 | orth vs non-orth same-profile matrix (cond 1.5–32) | local(+H800) | joint correctness+security+perf per cell |
| 10 | — | real 0.5B unified **ten-step** | H800+TDX | no error growth |
| 11 | T1 | fresh real TDX integration | H800+TDX | new quote/nonce/key; report_data binds private-base mode |
| 12 | A1/A2 | private-weight + activation/data attack matrices | local | honest partial-leakage reporting |
| 13 | G1 | unified generation + LoRA share one package | H800 | same_package_hash / same_base_weight_view / same_qkv_forward |
| 14 | — | 7B only after the scale gate | H800+TDX | see scale gate |

## O1 — optimizer equivalence (item 6)
- **O1-A** orthogonal direct masked SGD (2 inv/step) — correctness control.
- **O1-B** non-orthogonal preconditioned masked SGD (2 inv/step):
  `A_tilde=N_in^{-1} A U` ⇒ update `A_tilde -= lr (N_in^T N_in)^{-1} grad_A_tilde (U^T U)`;
  `B_tilde=U^{-1} B N_out` ⇒ `B_tilde -= lr (U^T U)^{-1} grad_B_tilde (N_out^T N_out)`.
  Paper-safe ONLY if the preconditioner does not reveal `N^T N` / undo the Gram defense.
- **O1-C** trusted non-orthogonal optimizer, packed grads to TDX (3 inv/step) — reference.
Outputs: `o1_optimizer/{formulas,correctness.csv,trajectory.csv,preconditioner_visibility,leakage_implications,invocation_count,summary}`.

## C1 — real operator correctness (item 5→7)
Per-layer: recovered forward error, mask-invariant error, gradA/gradB error+cosine,
residual-domain fingerprint, finite, backend, plaintext-materialization count.
Counters required 0: `plaintext_base_weight_materializations`,
`plaintext_embedding_materializations`, `plaintext_hidden_materializations`,
`nonlinear_trusted_calls`, `trusted_nonlinear_ops_count`, `silent_fallbacks`.
Run sequence: fp32 1-step → bf16 1-step → bf16 10-step. **No 50-step yet.**

## W1 — residual-chain alignment attack (item 8, go/no-go, before 7B)
Attacker owns the static `W_tilde_l = N_l^{-1} W_l N_{l+1}` chain. Settings:
no_anchor / output_anchor / oracle_anchor. Methods: chained factorization/alignment,
alt-min, Procrustes/CCA, spectral/Gram propagation, residual-loop + Q/K + norm +
head/embed-tying constraints. Static folded-package attack mandatory; fresh online
activation masks do NOT count as a defense against the offline package attacker.

## S1 — orth vs non-orth (item 9)
Same implementation, S1-orth vs S1-nonorth, condition sweep {1.5,2,4,8,16,32}.
Report correctness + security + performance jointly per cell. Never mix
"correctness from orthogonal, security from non-orthogonal" as one profile.

## A1/A2 — attack matrices (item 12)
A1 private-weight: orthogonal self-Gram, non-orth generalized Gram, multi-layer
alignment, multi-view, known-plaintext, chosen-input, embed/head tying,
compensation-term, LoRA functional extraction, rank/subspace recovery, black-box
cloning. Public-base only as a labeled control. A2 activation/data: hidden-state
inversion, token reconstruction, value-multiset at islands, ICA/BSS,
attention-score inversion, KV inversion, gradient inversion, membership,
linkability, monomial-scale/perm recovery. Report partial leakage honestly.

## Scale gate (before 7B)
All true: package has no plaintext base weights; real 0.5B unified 1-step + 10-step
pass; W1 done; S1 orth/nonorth done; core A1 done; real TDX session done; generation
+ LoRA share one package; no silent fallback. Then 7B: one generation, one unified
LoRA step, one 10-step if affordable, core W1/S1 cells, perf comparison.
