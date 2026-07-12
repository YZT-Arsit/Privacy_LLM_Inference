# Mask-component ablation (A0–A5) — private-base design

Source: `ablation_results.json` (CPU, real Qwen2.5-0.5B harness; security columns reuse the
already-run S1–S6). **This table shows each component's contribution to correctness /
attack-resistance / cost. It is NOT a formal security proof.**

| Ablation | Correctness (exact fold?) | Security effect (matching attack) | Efficiency | Verdict |
|---|---|---|---|---|
| **A0 full method** | ✅ exact (recovery rel_err ~1e-16, KL 0) | S1 strict cos **0.002**, S3 conf-corr **0.009**, S5 AUC **0.499**, S6 transfer top1 **0.003** | 2 trusted calls/step, 1.14 MB/ex | all surfaces at protected (near-random) level |
| **A1 remove feature mask** (Nr=I) | ✅ exact (identity) | **breaks representation/KV confidentiality**: S1 strict cos → **1.0**, S6 top1 → **0.541** (plaintext levels) | unchanged (mask is baked into package) | feature mask = the **S1/S6 representation-confidentiality** component |
| **A2 remove nonlinear-perm masking** | ❌ **BROKEN**: SiLU commutes with a permutation (err **0**) but not with a signed/general mask (err **4.16**; SwiGLU rel_err large) | n/a (not deployable); would expose plaintext pre-activations | would need nonlinear islands in the TEE | permutation-compatible nonlinear handling is a **CORRECTNESS requirement** |
| **A3 remove rank mask** (U=I) | ✅ exact (U cancels in ÃB̃) | S2 plaintext ΔW rel_err stays **1.41** (side masks, not U, protect ΔW); U only hides the A/B factor split | unchanged | **marginal** — factor-split hiding only |
| **A4 remove monomial logit mask** (D=I) | ✅ exact (perm-only still exact) | **restores confidence leak**: S3 conf-corr **1.0** (vs 0.009), S5 MIA AUC **0.595** (vs 0.499) | O(V), no extra RPC | monomial mask = the **logit/confidence + membership-readability** component (S3/S5) |
| **A5 remove mask refresh** (static) | ✅ exact | **static-mask KPA**: cross-step known-plaintext accumulates → mask recovered (rel_err **1e-7**); refreshed stays under-determined (per-step rel_err **0.82**) | refresh = re-fold/re-bind (amortized) | mask refresh = **cross-step known-plaintext-accumulation** resistance |

## Reading of the ablation (honest)
- **Correctness is owned by A2** alone: only the sign-free permutation on the nonlinear
  (SwiGLU) region lets the masked forward reconstruct the plaintext op exactly. The feature/
  rank/logit/refresh masks are all exact-invertible, so removing them does not change
  correctness — only the attack surface.
- **Security contributions are separable and component-specific**: feature mask ⇒ S1/S6
  (representation & KV), monomial ⇒ S3/S5 (logit confidence & membership readability),
  refresh ⇒ cross-step KPA. Removing each restores exactly the corresponding leak.
- **The rank mask (A3) is marginal** for ΔW confidentiality — S2 already showed plaintext ΔW
  is protected by the orthogonal *side* masks, not the rank mask; U only hides the factor split.
- Consistent with the recurring S1–S6 theme: the masks are exact orthogonal/permutation
  operators; their confidentiality is real only under the TEE assumptions (no paired
  plaintext, refreshed masks, aggregated gradients) — **not claimed as formal security**.
