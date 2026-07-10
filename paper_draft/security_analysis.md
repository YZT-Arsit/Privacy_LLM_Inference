# 7. Security Analysis

Every result in this section is **proxy-evaluated under a specific, named attacker**. No formal, cryptographic, semantic, or differential-privacy claim is made. The wording "is bounded under tested proxy attackers" replaces "is secure" everywhere in this section. The mapping between proxy attackers and claim status is summarized in **Table 5 (security proxy summary)** and in `paper_draft/claims_mapping.md`.

The risk-level taxonomy used below is exactly the one produced by the artifact pipeline:

- `low` — under the named attacker the recovered signal is close to random chance in our tests;
- `medium` — there is a residual signal but it is not strong enough to recover the target;
- `needs_more_evaluation` — the attacker does not succeed in our tests but a stronger attacker may; we do not claim the property is hidden;
- `high` — the attacker succeeds in our tests; this is a known leakage channel, reported faithfully.

## 7.1 Transcript obfuscation analysis

Under the full mitigation bundle (fresh per-call permutation + dense Linear sandwich + boundary pad), the GPU-visible transcript per call is `(X_tilde, W_tilde, A_tilde, B_tilde, K_tilde, V_tilde, Y_tilde)`. We instantiate the following adaptive proxy attackers on this transcript and report their outcomes:

- **Ridge inverter on `(X_tilde, Y_tilde)` pairs.** Attempts a linear least-squares recovery of `X` from masked outputs. *Result: bounded close to random chance in our evaluated setting.*
- **Small MLP inverter.** Adds a non-linear inverse head. *Result: bounded close to random chance in our evaluated setting.*
- **Signature / Sinkhorn permutation recovery.** Attempts to recover the per-call island permutation from the masked transcript. *Result: bounded close to random chance under fresh per-call masks; this is `proxy-evaluated`, not formally proven.*
- **Linkability AUC.** Attempts to link two masked traces to the same plain input. *Result: bounded close to AUC 0.5 (random) in our tested configurations.*

Status: **proxy_supported.** Artifact evidence in `outputs/adaptive_island_attacks.json`, `outputs/real_activation_attacks.json`, `outputs/real_token_activation_attacks.json`, `outputs/stronger_attackers.json`.

## 7.2 Activation recovery and linkability proxy

For the modern-decoder block and model wrappers, we evaluate the same family of attackers against block-level masked activations and model-level real-token traces. Under the full mitigation bundle, the worst-case attacker accuracy in our test set stays near random chance; the artifact tags this as `needs_more_evaluation` because we cannot exclude a stronger attacker. The Stage 7.5 consolidation flags this with risk level `needs_more_evaluation` rather than `low`.

The paper-safe wording, taken verbatim from the claims audit: *"Under our adaptive proxy attackers (ridge, small MLP, signature / Sinkhorn permutation recovery, linkability), the full mitigation bundle keeps the worst-case attacker close to random chance in our tested configurations."*

## 7.3 KV cache and generation trace proxy

The KV cache invariant (Section 5.5, Theorem 6) preserves attention scores; the masked cache itself is randomized per call by `N_K, N_V`. We do not currently evaluate a *generation-trace* recovery attacker (i.e., one that takes the full masked decode transcript and attempts to recover the prompt-to-output mapping in plain space), so this row is explicitly reported as `needs_more_evaluation` rather than `low` in `outputs/real_token_activation_attacks.json`. The output tokens *themselves* are visible to the GPU by the threat model (Section 3); this is an allowed leakage channel.

## 7.4 LoRA adapter extraction proxy

For LoRA forward, the adapter extraction attacker observes `(X_tilde, A_tilde, B_tilde, Y_tilde)` over multiple calls and attempts to recover `(A, B)` (or any rotated version thereof). Under fresh masks per call plus rank padding, the attacker's recovery is bounded; the membership-style linkability AUC is reduced by `Δ AUC = +0.463` versus the `fixed_masks_fixed_u` baseline in `outputs/lora_security_proxy.json`. Status: **proxy_supported, `needs_more_evaluation` risk level.**

The paper-safe wording: *"Under our LoRA adapter / gradient leakage proxy attackers, fresh masks + pad bring the linkability AUC close to 0.5 (random chance) in our tests."*

## 7.5 Gradient leakage proxy

For LoRA backward, the gradient extraction attacker observes `(G_tilde, dA_tilde, dB_tilde)` over multiple calls. Under fresh masks + pad, the gradient-side membership linkability AUC is reduced by `Δ AUC = +0.478` versus the `fixed_masks_fixed_u` baseline in `outputs/lora_gradient_security_proxy.json`. Status: **proxy_supported, `needs_more_evaluation` risk level.**

We explicitly do *not* claim "gradient leakage is impossible"; the contract is that *under the named proxy attackers in our tested configurations*, the AUC is close to random chance. Rank still leaks from shape under Stage 7.0 / 7.1 (Stage 7.2 addresses shape; see 7.6).

## 7.5b Nonlinear-island structural leakage (cross-Gram and value multiset)

The operator-compatible nonlinear islands (paired channel permutations for GELU/ReLU/SwiGLU) are exact but not value-hiding: inside an island the GPU observes the pre-activation up to a per-call channel permutation, so it can compute the token×token activation–gradient Gram `Z_tilde G_Z_tilde^T = Z G_Z^T` and the per-token value multiset (sorted rows of `Z_tilde`) exactly, and during training the per-token gradient norm `||G_Z_tilde|| = ||G_Z||`. We evaluate whether this recovers user data with a five-attacker panel (token reconstruction, private label/target, membership, LoRA adapter/`ΔW` extraction, future-token) — see `results/attacks/nonlinear_cross_gram_attack_audit/`.

**Main setting (private base weights).** Because the attacker does not hold plaintext `W`, it cannot map an observed value multiset to a vocabulary item, solve the per-call channel permutation, or fit a surrogate adapter. In our tests **token reconstruction, target/label recovery, future-token prediction, and adapter/`ΔW` extraction all fail** (token top-1 and future-token top-1 fall to the frequency baseline; adapter `ΔW` cosine ≈ 0). The one residual channel is **membership inference at AUC ≈ 0.61** (medium), riding the model-agnostic gradient-norm signal, which does not require knowing `W`. The pure Gram matrix alone — without the co-leaked value multiset — carries no vocabulary identity (token top-1 at baseline). We classify the nonlinear cross-Gram / value-multiset as **auxiliary structural leakage**: exact and non-zero, *not* eliminated, but under the evaluated attackers it did not recover user tokens, targets, or a functional adapter. Risk: `needs_more_evaluation` for reconstruction channels, `medium` for membership.

**Public-base ablation (negative setting).** When the base weights are instead public — the assumption of much prior obfuscation work, included here only as an ablation — the same leak is **high**: value-multiset matching against `sort(emb·W)` gives exact token reconstruction (top-1 ≈ 1.0), enabling target recovery (AUC ≈ 0.82), future-token recovery, and (via the recovered global channel permutation) full `ΔW` adapter extraction (cosine ≈ 1.0). This ablation is what motivates the private-base threat model: the danger is the co-leaked pre-activation value multiset *in combination with plaintext weights*, not the Gram matrix in isolation.

## 7.6 Rank leakage and dummy hardening proxy

Rank padding (Section 5.9) hides the *true* rank `r` from tensor shape; `r_pad` itself remains visible to the GPU. We evaluate four spectral-inference proxy detectors on the masked transcript:

- **Spectral cliff.** Uses the gap between the `r`-th and `(r+1)`-th singular value.
- **Energy.** Uses cumulative singular-value energy.
- **Elbow.** Uses the curvature of the singular-value spectrum.
- **Ensemble.** Combines the three.

Across `true_ranks ∈ {2, 4, 8}` with `paired_cancellation_dummy`, the spectral-inference risk is reported as `needs_more_evaluation` (worst-case across detectors), and the gradient-inference risk is `high`. Stage 7.4 evaluates stronger dummy distributions; the worst-case spectral-rank-inference risk is `high`, the worst-case gradient-rank-inference risk is `high`, the dummy-strategy classifier accuracy is `0.476` (chance `0.143`) yielding risk `medium`, and the cross-layer linkage across strategies is `low`.

We faithfully report all four risk levels, including the high ones. Specifically: **we do not claim that rank padding hides the LoRA rank cryptographically, and we do not claim `padded_rank` is hidden.** See Limitations.

## 7.7 Timing and metadata leakage proxy

We evaluate a cost-model timing classifier on training-step latency in `outputs/lora_training_timing_proxy.json`. With constant-time mode off, the classifier achieves above-random accuracy. With `proxy_equalized` mode on (per-step trusted compute is padded to the upper-bucket latency), the worst-case classifier accuracy drops to `0.5124` (near chance for a two-class task). Risk level: `low`. This is **cost-model proxy only**; we do not claim real wall-time hardness and we do not evaluate hardware side-channels.

The paper-safe wording: *"Under our cost-model timing proxy, equalizing per-step latency to the upper bucket reduces classifier accuracy to near random chance."*

## 7.8 Cross-layer linkage proxy

Cross-layer adapter linkage is `high` under shared-mask / shared-`U` baselines (by construction — this baseline does *not* refresh masks), `proxy_supported, low risk` under fresh masks per module + paired cancellation across the tested multi-layer configuration in `outputs/multilayer_lora_security_proxy.json`. The Stage 7.4 stronger-dummy ensemble preserves the cross-layer linkage at `low`.

## 7.9 Claims boundary

The full security analysis above is summarized in **Figure 6 (security risk matrix)** and consolidated in `paper_results/markdown/security_proxy_summary.md`. To be explicit:

- We **do not** claim the system is secure.
- We **do not** claim semantic, cryptographic, or formal indistinguishability.
- We **do not** claim resistance to a compromised TEE.
- We **do not** claim resistance to hardware side-channels.
- We **do not** claim the padded LoRA rank is hidden.
- We **do not** claim real TEE wall-time.
- We **do** claim, under the named proxy attackers on the tested configurations, that the worst-case proxy classifier is reduced to (close to) random chance for the items labeled `low` in the artifact, and that residual leakage in the items labeled `needs_more_evaluation`, `medium`, and `high` is reported faithfully and not re-classified as `low`.

Section 8 quantifies these claims against research questions; Section 9 enumerates every limitation that follows from the boundary above.
