# Adversary probe — findings (in progress)

Reference defense being compared: **CONJFORMER** (Yukhimchuk et al., arXiv 2606.16461,
Jun 2026) — orthogonal-equivariant transformer, and its attack suite (NN inversion,
norm-based, two-sided ALS-Procrustes, Gram alignment, attention fingerprint).
Also: **Lin et al. EDNN** (EMNLP 2024) — element-wise differential NN + the
"fixed-point nonexistence" security requirement for embedding obfuscation.

## Setup correction (important)

Our A_rightmul folded masks are **orthogonal and structured** (signed permutation for
residual/input, pairwise rotation for attention), *required* by the compatible-mask
conditions — NOT general-invertible. So the earlier hope "non-orthogonal N_in defeats
Procrustes/norm" does **not** hold as stated. The real picture:

- We do **not** fine-tune → folded weights use the **exact public W** (`W̃ = N_in⁻¹ W N_out`,
  `W = W_public`). CONJFORMER instead relies on fine-tuning to make `W̃ ≠ W` so alignment
  degrades. So our exact-W setting is a priori *easier* for weight alignment — must test.
- We add a **linear-boundary additive pad**: GPU input operand `X̃ = (X − T) N_in`.

## Synthetic probe results (scripts/attacks/synthetic_alignment_probe.py)

d=64, vocab=2000, 5 families sharing the input mask. Numbers = token-recovery % (top1/top10).

| attack | no pad | pad 0.1 | pad 0.5 |
|---|---|---|---|
| A0 raw NN (no defense, ceiling) | 100 / 100 | — | — |
| A1 norm-based | **100** / — | 1.95 / — | 0.39 / — |
| A2 shared-mask alignment (no pad-knowledge) | ~0 / 1–8 | ~0 / 1–8 | ~0 / 1–6 |

random baseline ≈ 0.05% top1, 0.5% top10.

### Solid, defensible conclusions
1. **RETRACTED (2026-07-02): the additive pad does NOT defeat the norm attack.**
   Rigorous data-flow proof (folded_worker.py / embedding_artifact.py): the GPU's actual
   layer-0 input is `h_tilde = X @ N_0` where `N_0` is the ORTHOGONAL residual mask and the
   input is PAD-FREE (the linear-boundary pad is applied only inside `_linear` when forming the
   q/k/v operand — it never enters the residual stream / RMSNorm input). Orthogonal `N_0`
   preserves row norms, so `‖h_tilde_row‖ = ‖X_row‖` EXACTLY → the true per-token embedding
   norm leaks to the GPU regardless of the pad. Synthetic confirms: norm attack on the real
   `h_tilde` = 100% top-1 at every pad scale (0.0/0.1/0.5); the pad only perturbs the *padded
   Linear operand* `(X−T)N_in`, which is NOT the surface that carries the embedding norm.
   → The norm attack is a REAL, currently-unmitigated vulnerability, and WORSE for us than for
   CONJFORMER: they flatten embedding-norm structure by fine-tuning the embedding table
   (their App. D.4); we do not fine-tune, so that mitigation is unavailable. (Synthetic uses
   distinctive per-token norms so it reads 100%; on real Qwen top-1 will be lower because many
   tokens share norms, but the exact structural leak `‖h_tilde‖=‖X‖` holds.)
   Open mitigation options: residual-stream additive pad (needs compensation through every
   nonlinear island — defeats the boundary-local design), pre-mask embedding normalization
   (changes semantics), non-orthogonal residual mask (breaks A_rightmul correctness), or honest
   disclosure as a limitation.
2. **CONJFORMER's two-sided weight-alignment attack is structurally unidentifiable against us.**
   Because each Linear family uses an **independent** output mask `N_out^i` (vs CONJFORMER's
   single **global** right factor `U`), the per-family objective `‖W̃_i − A W_i B_i‖` is **zero
   for every orthogonal A** (A·W_i and W̃_i always share singular values ⇒ some orthogonal B_i
   fits exactly). The attack that works for CONJFORMER (aggregate the shared U across blocks via
   Gram) has no shared right factor to aggregate here.

### Open question (NOT yet resolved — do not claim "safe")
3. The only structure left is the **shared input mask** `N_in` (common to q/k/v/gate/up), attackable
   via the **left Gram** `W̃_i W̃_iᵀ = N_in⁻¹ (W_i W_iᵀ) N_in⁻ᵀ` (right mask cancels, `W_i` known).
   Our Riemannian solver drives the Gram residual to ~0 but the recovered A does **not** invert the
   mask (token recovery near-random). This is ambiguous between (a) genuine non-identifiability
   (joint orthogonal stabilizer of `{W_iW_iᵀ}` larger than ±I / discards phase) and (b) our solver
   hitting a spurious residual-0 minimum. Must be settled by a clean identifiability argument +
   validation on the **real Qwen-7B weights** before any security claim.

## A3 — Attention-logit fingerprint attack: FULL BREAK (most severe finding)

scripts/attacks/attention_fingerprint_probe.py. Result: **100% top-1 token recovery** with as
few as **12** (layer,head) self-attention logits; `mask_invariance_maxerr = 0.0` and
`pad_invariance_maxerr = 0.0` (exact).

Why it works and why it is structural:
- A_rightmul certifies `attention_qk_scores_preserved` (needed for correctness) ⇒ the GPU softmax
  runs over the TRUE attention logits `Q̂K̂ᵀ = QKᵀ`. The single-position self-attention logit per
  (layer,head) is a deterministic, mask-INVARIANT fingerprint `f(token) ∈ R^{B·H}`.
- The attacker recomputes `f` for the whole vocab from PUBLIC weights (mask-invariance makes the
  observed and table `f` match), then NN-matches. It can obtain the observed `f` by running
  single-token offline inference on the masked embeddings (the model runs on its GPU).
- **This bypasses mask secrecy entirely** — no mask/pad recovery is needed; a mask-invariant
  quantity directly identifies the token.
- The linear-boundary pad is compensated (C_pad) before attention ⇒ does not perturb the logit.
- We do NOT fine-tune ⇒ the fingerprint table is EXACT. CONJFORMER's only mitigation (App. D.5:
  fine-tuning shifts `f` off the table) is unavailable to us ⇒ strictly worse than CONJFORMER.

Real Qwen2.5-7B has ≈28×28 = 784 such features → massively over-determined; expect a full break.
This is the core tension of running nonlinearities on masked state with structure-preserving
masks: preserving attention scores = exposing a token fingerprint.

**REAL-WEIGHT CONFIRMED (Qwen2.5-7B, scripts/attacks/real_attention_fingerprint.py):**
784-dim fingerprint, 20000 candidate tokens, with a realistic fp32-table/bf16-query precision gap:
**top-1 = 98.5%, top-10 = 98.6%, top-100 = 99.0%**. (The <100% is bf16 precision flipping a few
near-collisions; result at results/attacks/real_attention_fingerprint_qwen7b.json.) The
attention-fingerprint attack is a confirmed full break of token privacy on the real model.

## Not yet implemented
- **Attention-fingerprint attack** (CONJFORMER §3.5): the one leak we likely SHARE — if
  `Q̂K̂ᵀ = QKᵀ` (pairwise-rotation correctness), the GPU sees true attention logits. We don't
  fine-tune, so the "fine-tuning shifts fingerprints" mitigation is unavailable to us. This is a
  priority and a likely genuine exposure.
- **EDNN** (Lin et al.): our orthogonal `N_in` is not the glide-reflection `I − 2/d·E`, so the
  neighbor-difference invariant does not hold; also worth a formal "fixed-point nonexistence" check.
- **Real-weight validation** on the deployed Qwen-7B folded package (adversary = GPU operator,
  runs on H800 which has both public Qwen weights and the folded package).
