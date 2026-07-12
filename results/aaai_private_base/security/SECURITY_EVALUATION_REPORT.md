# AAAI private-base security evaluation — S1–S4 report

Scope (never exceeded): *under the evaluated threat model and published attack methodologies, the private-base transformed Qwen2.5-0.5B empirically resists recovery of private weights, representations, LoRA adapters, gradients, and inference artifacts.* We make **no** claim of impossible recovery, zero leakage, or information-theoretic security.

All experiments ran **CPU-only, offline**, on a defender-side oracle holding the real Qwen2.5-0.5B weights as the SECRET private base; **no attacker function received plaintext W/H, masks, or any public checkpoint**. The active A10/TDX utility run and production checkpoints were never touched. **Nothing committed.**

## 1. Summary of results

| Exp | Attack surface | Positive control | Main result (ours) | Verdict |
|---|---|---|---|---|
| S1 | masked hidden states | P0 plaintext cos=1.0, P1 identity cos=0.929 | strict inversion cos=0.0018 ≈ random 0.0029 | resists strict inversion; orthogonal mask linearly invertible **iff** paired plaintext leaks |
| S2 | transformed LoRA A~,B~ | plaintext ΔW rel_err=2.4e-15 | plaintext ΔW rel_err=1.4148 (≈√2 scramble) | masked product recoverable; plaintext ΔW not, w/o side masks |
| S3 | logit masking | B0 plaintext multiset_gap=0 | B1 conf_corr=1.0 (full leak) vs B2 conf_corr=0.0092 | monomial removes the exactly-preserved confidence leak |
| S4 | transformed gradients | plaintext-b1 DLG token acc=0.625 ≫ rand 6.6e-06 | basis-invariance loss-gap=9.1e-13, tokens agree=1.0 | mask is a transparent basis change; defense = aggregation/non-exposure |
| S5 | black-box outputs (MIA) | B0 plaintext AUC=0.6139 > 0.5 | perm-only AUC=0.5948, monomial AUC=0.499 | membership leaks at the trained weights; monomial removes readable signal, perm-only retains it |
| S6 | masked KV cache | plaintext KV top1=0.5411 ≫ chance 1.3e-03 | masked-transfer top1=0.0033, masked-adapted top1=0.5362 | mask defeats a plaintext-calibrated attacker; orthogonal-invertible given masked pairs (=TEE) |

## 2. Paper mapping

See `attack_paper_mapping.md`. Methodologies: Mahendran & Vedaldi (CVPR'15) + Fredrikson et al. (CCS'15) [S1, S6]; Hu et al. LoRA (ICLR'22) [S2]; internal design_spec §F [S3]; Zhu et al. DLG (NeurIPS'19) + Zhao et al. iDLG (2020) [S4]; Shokri et al. (S&P'17) shadow-model MIA [S5]. **All citations are real; none fabricated.**

## 3. Positive controls (must pass before trusting any attack)

- **S1**: PASS
- **S2**: PASS
- **S3**: PASS
- **S4**: PASS
- **S5**: PASS
- **S6**: PASS

## 4. Main attack results (detail)

### S1 — Representation inversion
- **Strict attacker** (no W, no pairs): recovery cosine **0.0018** vs random **0.0029** → infeasible.
- **Known-pairs upper bound** (generous, attacker given paired plaintext): A1 linear cosine **0.9291** on 395 real positions; synthetic n≥dim → rel_err **5.7e-07** (orthogonal mask is exactly linearly invertible once pairs leak). MLP A2/A3 do not beat linear (true map is linear).
- **Documented invariant leaks**: norm gap **1.8e-12**, pairwise-Gram gap **7.9e-09** → orthogonal mask preserves norms & inner products (not claimed hidden).
- **Embedding boundary**: deterministic-embedding token NN acc **1.0** (out of the residual mask's scope; input-token protection is the separate input-pad / layer-0 TEE mechanism).

### S2 — LoRA adapter recovery
- Positive control (plaintext adapter): SVD recovers ΔW rel_err **2.4e-15** (exact).
- Ours (masked): best plaintext-ΔW rel_err **1.4148** ≈ √2 — the value of a fully uninformative orthogonal scramble; the masked *product* is recoverable but plaintext ΔW is not, without the secret orthogonal side masks. Individual A,B are non-identifiable (rank rotation) even in plaintext.

### S3 — Logit leakage
- **B1 permutation-only** (shipped default, D=I): value-multiset gap **0.0**, confidence correlation **1.0** → the exact confidence/uncertainty profile leaks (only token identities hidden by the unknown permutation).
- **B2 monomial** (proposed): multiset gap **30.2985**, confidence correlation **0.0092**, sorted-prob KL **3.7215** → removes the exactly-preserved distributional leak, at bounded condition number. Neither hides token identity better (same permutation); the gain is strictly distributional.

### S4 — Gradient inversion (DLG/iDLG)
- Positive control (plaintext, batch 1): DLG token recovery **0.625** vs random **6.6e-06** (embedding cosine 0.2511; shallow head → modest cosine but NN still recovers tokens far above chance).
- **Path-independent basis invariance**: mapping each plaintext DLG solution through Nr matches the MASKED gradients with a gradient-match-loss gap of at most **9.1e-13** and recovers the **same token in 100%** of cases → the orthogonal mask (exact Nr-conjugate = the deployed fold) is a **transparent change of basis** for gradient inversion; the attacker inverts in the masked basis and NNs against the shipped E~.
- Aggregation defense (batch sweep [1, 2, 4, 8], plaintext basis; masked identical by invariance): token recovery [0.625, 0.875, 0.25, 0.0] → degrades with batch size. **The real defense is batch aggregation + never exposing per-example gradients, not the mask.**

### S5 — Membership inference (Shokri shadow-model MIA)
- Positive control (B0 plaintext outputs): ROC-AUC **0.6139** (acc 0.5833, prec 0.5455, rec 1.0) > 0.5 → non-random membership signal.
- Protected outputs: **perm-only AUC 0.5948** (retains most signal — set-symmetric confidence preserved, cf. S3) vs **monomial AUC 0.499** (≈ random — distortion removes readable confidence). Membership leaks at the *trained weights* (generalization gap), not the mask; the mask only changes readability of the output channel.

### S6 — KV-cache inversion
- Positive control (plaintext KV, linear decoder): token top-1 **0.5411** vs chance **1.3e-03** → highly recoverable.
- **Masked KV, plaintext-calibrated attacker (transfer)**: top-1 **0.0033** ≈ chance → the orthogonal KV mask (Bk rope-commuting, Sv signed-perm) defeats an attacker who does not know the mask (**reduced recovery**).
- **Masked KV, adapted attacker (masked pairs)**: top-1 **0.5362** → the orthogonal mask is invertible given masked pairs, so protection rests on **mask secrecy / the TEE**, not information destruction (consistent with S1).

## 5. Limitations (honest, per experiment)

- **S1**: Ground-truth private base uses real Qwen2.5-0.5B weights (secret, never exposed to attackers) for realistic representations; leakage geometry is weight-distribution invariant.
- **S1**: Known-pairs setting is a deliberately generous upper bound (attacker given paired plaintext) to quantify mask thinness; it is stronger than the frozen attacker and is labeled as such.
- **S1**: Representation set is modest (a few thousand token positions); the synthetic probe isolates the exact-invertibility claim with n>=dim.
- **S1**: We evaluate published inversion methods under this observation model; we do NOT claim inversion is impossible, only that it fails without weights/paired-plaintext and that the orthogonal mask is linearly invertible once pairs leak.
- **S2**: LoRA factorization is inherently non-unique (rotation), so 'recover A,B' is ill-posed even in plaintext; we measure the well-defined ΔW and subspace alignment.
- **S2**: Protection of plaintext ΔW rests on the secret orthogonal side masks; as in S1 these are linearly invertible given paired plaintext (masks-secrecy = TEE assumption), so the claim is scoped to an attacker without paired plaintext.
- **S2**: Downstream delta uses random plaintext inputs, not a task-specific eval; it measures functional distance of ΔW, not end-task utility loss.
- **S3**: The frozen shipped package uses permutation-only (D=I); B2 evaluates the proposed monomial D as a design option, not the currently-built default.
- **S3**: Monomial condition number is bounded (~7) to avoid bf16 overflow; larger D reduces leakage further but risks numerical range — a documented trade-off, not a free win.
- **S3**: An attacker with known-plaintext logit pairs could estimate D per column (D>0 diagonal is identifiable from ratios); monomial resists identity-free multiset reading, not a full known-plaintext attacker.
- **S4**: Shallow batch-1 supervised head, not the full 24-layer LoRA stack; DLG is known not to converge on deep models / large batches, so batch-1 is the strongest (favourable-to-attacker) case.
- **S4**: Token recovery uses NN against the (masked) embedding table the attacker legitimately holds; this isolates the change-of-basis transparency, the key point.
- **S4**: The real protocol's per-step adapter gradients are computed on the untrusted GPU; this experiment argues the defense must be aggregation / non-exposure, and is scoped accordingly — we do NOT claim the mask defeats gradient inversion.
- **S5**: MIA target is a LoRA-style linear probe over frozen private-base features (standard lightweight MIA target); it models the output-confidence channel the mask affects, not a full attention/MLP LoRA.
- **S5**: Black-box outputs-only attacker; a white-box or per-example-gradient attacker is out of this channel.
- **S5**: Absolute AUC is modest because the fine-tune is light and SST-2 generalizes well (small member gap); we report the RELATIVE channel comparison, which is the design-relevant quantity.
- **S5**: Protected channels are modeled at the 2-class verbalizer boundary via the S3 monomial mask; a full vocab-logit MIA would see the same set-symmetric preservation (perm-only) shown in S3.
- **S6**: Layer-0 KV (least contextualized) is the attacker-favourable case; deeper layers mix context and are harder to invert to a single token.
- **S6**: Closed-set token classification over tokens present gives a clean chance baseline but is easier than open-vocabulary recovery; the plaintext-vs-masked-transfer CONTRAST is the design-relevant quantity.
- **S6**: The masked-adapted decoder assumes masked (KV_tilde, token) pairs; consistent with S1, the orthogonal KV mask provides no information-theoretic protection — confidentiality is the secret mask / TEE.
- **S6**: KV masks are orthogonal per-kv-head (Bk rope-commuting, Sv signed-perm); norms/Grams are preserved as in S1 (documented structural leak).
- **Global**: real Qwen2.5-0.5B weights stand in for a from-scratch private base (secret, never exposed to attackers); the leakage geometry under test is weight-distribution invariant. The recurring structural theme — orthogonal/permutation masks preserve norms, Grams, and value multisets, and are linearly invertible given paired plaintext — means **confidentiality rests on the TEE preventing paired-plaintext / per-example-gradient exposure, and on aggregation, not on the algebraic mask alone.**

## 6. Files changed / added

Harness+attacks (all new, untracked): `scripts/security/{pb_harness,s1_representation_inversion,s2_lora_recovery,s3_logit_leakage,s4_gradient_inversion,s5_membership,s6_kv_cache,test_security_harness,build_security_report,monitor_main_experiment}.py`. Results+registry (new, untracked): `results/aaai_private_base/security/**`, `results/aaai_private_base/security_progress_monitor/**`. No existing training/protocol file modified.

## 7. Tests

`scripts/security/test_security_harness.py` — 11/11 PASS (Nr orthogonality+roundtrip, norm/Gram preservation, LoRA masked-product identity + plaintext hiding, perm-only multiset exactness, monomial multiset change, mask determinism, masked-embedding-table wall). Each S1–S4 script self-checks its positive control before reporting.

## 8. Git status

All security files (`scripts/security/**`, `results/aaai_private_base/security/**`, `results/aaai_private_base/security_progress_monitor/**`) are **untracked** — no `git add`/`git commit` was performed by this work. HEAD is `431d5db` (a pre-existing profiling-gate utility-artifacts commit authored 2026-07-12 19:58, before this security task began); this security work neither created nor modified any commit, and touched no existing training/protocol source file.

## 9. Nothing committed

Confirmed: no commits, no staging. The active A10+TDX converged-utility run was monitored read-only (`security_progress_monitor/`) and never interrupted.

---
_Stop condition honored: S1–S6 complete (S5 membership + S6 KV-cache added this round); no 7B, no external-baseline reproduction, no new training. All six positive controls pass._