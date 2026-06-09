# Security Claims Table

This is the paper-side consolidation of every theorem and claim from
`docs/PAPER_THEORY_OUTLINE.md`, graded using the five-tier vocabulary
defined in `docs/PAPER_EVALUATION_MAP.md`:

- `algebraically_proven` — direct algebraic identity verified at machine precision
- `experimentally_validated` — measured behaviour with conservative thresholds
- `proxy_evaluated` — named-attacker proxy only
- `cost_proxy_only` — table-size / bandwidth / microbenchmark cost
- `unsupported` — explicit declination (raise, `False`, `unsupported`)

**No row in this table claims formal, cryptographic, or semantic security.**
`formal_security_claim` is `False` in every artifact that records it.


## 1. Linear and boundary

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T1 | Right-action mask correctness of linear layers: `(X N)(N^T W) + b = X W + b` | `algebraically_proven` | `outputs/ours_runtime_api_validation.json` |
| T2 | Boundary pad cancellation: `[X | Z] [W; 0_{p × d_out}] = X W` | `algebraically_proven` | `outputs/ours_runtime_api_validation.json` |


## 2. Attention

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T3 | QK score invariance under per-head right-action mask: `(Q N_h)(K N_h)^T = Q K^T` | `algebraically_proven` | `outputs/attention_experiments.json` |
| T4 | KV-cache append invariance under per-call mask binding | `algebraically_proven` | `outputs/ours_runtime_api_validation.json` (prefill / decode_step / generate); `outputs/modern_decoder_model_wrapper_smoke.json` |
| T6 | Post-RoPE per-head masking preserves T3 | `algebraically_proven` | `outputs/modern_decoder_model_wrapper_smoke.json` |
| T7 | Grouped-query attention with per-K/V mask tying | `algebraically_proven` | `outputs/modern_decoder_model_wrapper_smoke.json` |


## 3. RMSNorm and SwiGLU island

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T5 | RMSNorm commutes with orthogonal residual mask under same-coordinate-system normaliser | `algebraically_proven` | `outputs/modern_decoder_model_wrapper_smoke.json` |
| T8 | SwiGLU compatible-island correctness per call | `algebraically_proven` | `outputs/modern_decoder_model_wrapper_smoke.json`; `outputs/ours_runtime_api_validation.json` |


## 4. Permutation-invariant leakage

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T8a | Compatible island preserves permutation-invariant statistics (known leakage) | `experimentally_validated` | `outputs/permutation_invariant_leakage.json` |
| T9 | Permutation-invariant statistics theorem (norms / sorted multiset / quantiles unchanged by column permutation) | `algebraically_proven` + `experimentally_validated` | `outputs/permutation_invariant_leakage.json` |
| T9-attacker | Named-attacker risk under `fresh_perm_only` and `fresh_perm_plus_sandwich_plus_pad` bundles (linear inverter / MLP inverter / signature-matching permutation recovery / linkability) | `proxy_evaluated` | `outputs/real_activation_attacks.json`; `outputs/real_token_activation_attacks.json`; `outputs/stronger_attackers.json` |
| T9-fixed-perm-ablation | Fixed-perm sanity baseline shows attacker DOES recover (used to validate the freshness contract) | `experimentally_validated` | `outputs/real_activation_attacks.json` (`fixed_permutation_debug` bundle) |


## 5. Lookup cost proxy

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T10 | Two-input lookup table size scales as `2^(2b)`; online lookup bandwidth is `B·S·D·entry_bytes`; per-channel tables `2^(2b)·D·entry_bytes` are reported as `impractical_proxy_only` | `cost_proxy_only` | `outputs/lookup_nonlinear_cost_proxy.json` |
| T10-secure-lookup-implementation | A secure lookup primitive (garbled circuit / MPC / FHE / Tabula / FLUTE) is implemented in this artifact | `unsupported` | `outputs/lookup_nonlinear_cost_proxy.json` records `cryptographic_lookup_implemented = False` |


## 6. Masked-gradient LoRA

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T11 | Rank-space-mixed masked LoRA forward correctness: `X_tilde A_tilde B_tilde = X A B N_y` | `algebraically_proven` | `outputs/masked_gradient_lora_training.json` |
| T12 | Gradient relations `grad_A_tilde = N_x^T grad_A M`, `grad_B_tilde = M^T grad_B N_y` and masked-loss equality | `algebraically_proven` | `outputs/masked_gradient_lora_training.json` |
| T13 | Masked-SGD / masked-momentum-SGD equivalence to plaintext SGD / momentum SGD under trusted-side recovery | `algebraically_proven` | `outputs/masked_gradient_lora_training.json` |
| T13-leakage | GPU-visible parameters / gradients under (i) true-rank inference, (ii) real-vs-dummy subspace separation, (iii) cross-step linkability (fresh vs fixed masks) | `proxy_evaluated` | `outputs/masked_gradient_lora_security_proxy.json` |
| T14 | Cancellation-padded rank: `A_pad B_pad = A_real B_real` at initialisation | `algebraically_proven` | `outputs/masked_gradient_lora_training.json` (`initial_dummy_contribution_norm`) |


## 7. AdamW dense-mask limitation

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| T15-limitation | Dense masked AdamW exactness is unsupported because coordinate-wise second moments are not invariant under dense orthogonal mixing | `algebraically_proven` (limitation) | `docs/PAPER_THEORY_OUTLINE.md` §9 (counterexample) |
| T15-gate | `masked_adamw_step_unsupported(...)` raises `DenseMaskedAdamWUnsupported` rather than silently approximating | `experimentally_validated` | `outputs/masked_gradient_lora_training.json` (`adamw_dense_mask_unsupported.status = "explicitly_raised_as_designed"`) |
| T15-claim | Dense masked AdamW exactness is claimed by this artifact | `unsupported` | `outputs/masked_gradient_lora_training.json` records `masked_gradient_lora_adamw_dense_mask_supported = False` |


## 8. Paper-safe wording

| ID | Claim | Grade | Artifact |
|---|---|---|---|
| C-PaperSafe | No file in `paper_draft/`, `paper_results/`, `outputs/`, `README.md`, or `docs/` contains an unguarded unsafe phrase (`formal security`, `cryptographically secure`, `semantic security`, `AdamW supported`, `plaintext gradients hidden by proof`, `optimizer fully outsourced`, `LoRA rank is hidden`) | `experimentally_validated` (lexical scan) | `outputs/stage_7_6_claims_consistency.json` (`total_unsafe_wording_present = 0`, `passes_consistency_check = True`); `outputs/paper_claims_audit_v2.json` |
| C-FormalSecurity | Formal, cryptographic, or semantic security is claimed by this artifact | `unsupported` | All `formal_security_claim = False` records |


## 9. Headline non-claims (verbatim)

- *This is a lookup cost proxy, not a secure lookup implementation.*
- *No garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic lookup protocol is implemented.*
- *Lookup-style nonlinear protection may improve value hiding, but this stage evaluates only table-size and memory-access costs.*
- *The current compatible island is faster and lower-memory but preserves permutation-invariant activation statistics.*
- *Dense masked AdamW is not claimed because coordinate-wise second moments are not invariant under dense orthogonal mixing.*
- *Masked SGD is algebraically equivalent under orthogonal masks.*
- *The GPU never receives plaintext LoRA adapters or plaintext LoRA gradients in this experiment.*
- *No real TEE or GPU wall-time is measured.*
- *No formal, cryptographic, or semantic security is claimed.*


## 10. Reference

- Theorems and proof sketches: `docs/PAPER_THEORY_OUTLINE.md`
- Artifact → claim mapping and grading vocabulary: `docs/PAPER_EVALUATION_MAP.md`
- Lexical consistency check: `outputs/stage_7_6_claims_consistency.{json,csv,md}` and `outputs/paper_claims_audit_v2.{json,md}`
- LoRA training-to-inference visibility lifecycle: `outputs/lora_training_inference_lifecycle.{json,csv,md}`
