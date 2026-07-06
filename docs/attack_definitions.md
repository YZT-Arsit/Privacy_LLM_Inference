# Attack definitions

Eight attack families, their inputs/outputs/metrics/threat model, and exactly
what we DO and DO NOT claim. Grounding: `docs/attack_paper_audit.md`.

| attack_id | family | impl level | threat model |
|---|---|---|---|
| nn_embedding_inversion | structural | full | public_model / worst_case |
| eia_optimization | optimization | best_effort | public_model |
| bre_bisr_forward | optimization | best_effort | split_inference |
| bre_bisr_backward | optimization | **blocked** | split_inference |
| kpa_known_plaintext | cryptanalysis | full | known_plaintext |
| multiset_permutation_leakage | structural | full | closed / public |
| arrowmatch_weight_alignment | alignment | full | weight_leakage_worst_case |
| pia_prompt_inversion | optimization | best_effort | split_inference |
| frequency_distribution | statistical | full | closed_model |

## NN embedding inversion (`nn_embedding_inversion`)
NN of observed embedding-space vectors vs an embedding table (cosine/L2/dot), or
the **EDNN differential** variant (`metric="differential"`, matches
`v − lshift(v)` — invariant to glide-reflection). In: observed/protected
vectors + table + ids. Out: `token_recovery_top{1,5,10,100}`,
`mean_token_accuracy`, `sequence_exact_match`. Blocked with no table.
**full.** EDNN only breaks difference-preserving obfuscations.

## EIA optimization (`eia_optimization`)
Relaxed soft-token / continuous-embedding optimization so a differentiable
`downstream` reproduces an observed representation (MSE/cosine/combined, Adam,
restarts, temperature, entropy reg, hard projection). Out: `mse`,
`cosine_similarity`, token accuracy. **best_effort** — simplified modern
adaptation of the Song&Raghunathan family, not a verbatim reproduction.

## BRE / BiSR (`bre_bisr_forward`, `bre_bisr_backward`)
Forward (Eq.7-8): continuous embedding optimization with **cosine** loss vs
observed smashed data + NN projection to tokens — **best_effort**, NOT full
BiSR. Backward (Eq.9): gradient matching on a dummy vocab-distribution label —
**blocked**: an inference-only harness has no split-FT activation gradients
`grad(x̃_trk)`; needs a fine-tuning loop (see `server_run_plan.md`).

## KPA (`kpa_known_plaintext`)
`mode="linear"`: solve `R_hat` from `(X, XR)` (lstsq/ridge), held-out
deobfuscation error, sample-complexity sweep `{H/4,H/2,H,2H,4H}`, token recovery.
`mode="permutation"`: recover a feature permutation from `(X, Xπ)` by column
matching (Hungarian). Fresh per-sample padding ⇒ failure = **intended
protection** (labelled). Blocked with no pairs. **full.**

## Permutation / multiset (`multiset_permutation_leakage`)
`structural_leakage_probe`: coordinate-multiset (sorted-L1 Hungarian) + per-token
norm + pairwise-distance preservation + permutation recovery. Out:
`multiset_leakage_score`, `norm_profile_match_accuracy`,
`distance_profile_match_accuracy`, `permutation_recovery_accuracy`.
`run_autoregressive_decode`: the paper's **Alg.1/Alg.3** greedy autoregressive
decode — forward each vocab candidate, match its last-row layer-`l` state to the
observed states (sorted-L1 for permuted; positional for unpermuted) within `ε`.
**full.** Requires the model + table (public model / captured after ≥1 attention).

## ArrowMatch (`arrowmatch_weight_alignment`)
S1 (Eq.1): `σ(i)=argmin_j cos_dist(w_obf^i, w_pre^j)` per weight matrix (greedy
or Hungarian). S2: length ratio `‖w_pre‖/‖w_obf‖`. Out:
`permutation_recovery_accuracy`, `alignment_accuracy`. **full.** Requires
`model_weights={public,obfuscated}` → **weight_leakage_worst_case**, NOT the
default threat model. Also `load_external_arrowmatch` for external results.

## PIA (`pia_prompt_inversion`)
Phase 1 (Eq.13): constrained continuous optimization
`min ‖F(v̂)−A‖² + λ Σ min_t ‖v̂_i−E(t)‖²` + per-dim clip. Phase 2 (Eq.16):
adaptive discretization — greedy autoregressive activation calibration over a
top-K embedding candidate set. Out: token accuracy, `rouge_l_f1`,
`edit_distance`, `prompt_recovery_success`. **best_effort** — the oracle-LLM
semantic-speculation set `S_s` is OMITTED (labelled).

## Frequency / distribution (`frequency_distribution`)
Norm/frequency rank correlation + norm-multiset recovery between plaintext and
protected states. Out: `frequency_rank_correlation`, `multiset_leakage_score`.
**full** (supplementary). Permutation/orthogonal preserve per-token norm →
leak; fresh-pad/matrix-mixing break it.
