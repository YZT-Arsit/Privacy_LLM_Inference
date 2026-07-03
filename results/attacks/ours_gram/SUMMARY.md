# GO/NO-GO: residual-mask Gram inversion attack on OUR A_rightmul folded scheme

Self-audit security experiment (H800, real Qwen2.5-7B-Instruct, 2026-07-03).
Script: `scripts/attacks/residual_mask_gram_attack.py`. Result:
`results/attacks/ours_gram/qwen7b_gram_attack.json`.

## Verdict: **NOT SECURE** — uniquely solvable, not underdetermined. Change the design.

An honest-but-curious server that holds (i) the public Qwen weights, (ii) the
offloaded folded weights `W'`, and (iii) the masked residual states `X·N_res` that
cross to the GPU, **recovers the secret residual mask exactly and inverts 98.5% of
user tokens.**

| metric | value |
|---|---|
| permutation recovery, per projection (all 140, min/median/max) | **1.000 / 1.000 / 1.000** |
| sign recovery (global-flip-invariant), per projection | **1.000** |
| aggregated N_res recovery | **exact** |
| **token recovery top-1 / top-10 / top-100** | **98.5 / 98.5 / 98.5 %** |
| random-mask baseline top-1 | 0.0 % (chance 0.017 %) |

Every single one of the 140 attacked projections (q/k/v/gate/up × 28 layers)
recovers the entire global mask on its own. The residual 1.5% of token-recovery is
embedding-table degeneracy (identical rows fail NN at every k — top-1==top-100), not
attack error: the mask is recovered *exactly*.

## Why it breaks (the exploitable invariant)

The A_rightmul default uses a **single global signed-permutation residual mask
`N_res`** (`residual_mask_strategy="shared"`) and **folds exact public weights** (we
do NOT fine-tune). For every left-shared projection `i ∈ {q,k,v,gate,up}` the fold is

    W'_i = N_res⁻¹ · (g_i ⊙ W_i) · M_i        (M_i = orthogonal output mask)

The **left self-Gram cancels the output mask** (M_i M_iᵀ = I):

    W'_i W'_iᵀ = N_res⁻¹ (A_i A_iᵀ) N_res⁻ᵀ = P · (A_i A_iᵀ) · Pᵀ,
        A_i = g_i⊙W_i  (PUBLIC),   P = N_res⁻¹ = N_resᵀ  (signed permutation)

So `diag(W'_i W'_iᵀ)` is a **permutation of `diag(A_i A_iᵀ)`**, which is public.
Because we fold *exact* public weights, `G'_ii` equals `G_{π(i)π(i)}` to fp64/fp32
precision, so the permutation is recovered by an exact 1-NN diagonal match (works even
where the coarse "distinct-diagonal" fraction looks low: 100% recovery held on the
worst projection, diag-distinct 0.013). Signs come from the off-diagonals. With one
global mask, the 140 projections are 140 exact equations for one unknown → massively
over-determined → exact.

**What does NOT help (verified):**
- The **pairwise-rotation attention masks** and **SwiGLU permutation** — they are the
  *output* masks `M_i`, which cancel in the left Gram. No protection.
- The **linear-boundary pad** `(X−T)N_in` — additive, on the forward path; it does not
  change `W' = N_res⁻¹ W N_out`, so the weight-Gram mask recovery is untouched.
- CONJFORMER's *cross*-Gram `(W'_K)ᵀW'_Q` does NOT apply to us (our masks are on the
  right); the **left self-Gram** is the correct, and here decisive, surface.

## Root cause vs CONJFORMER — the tension with our own advantage

CONJFORMER resists this exact attack **only because it fine-tunes**: its offloaded
weights are the fine-tuned `W̃ ≠ W`, so the attacker's public-`W` Gram is only
approximate and decays as fine-tuning proceeds (their Fig. 3). **Our zero-training
advantage — folding the exact public instruct model — is precisely what makes the
mask exactly recoverable from the weights.** A structured orthogonal mask on exact
public weights is an *exact* conjugation the adversary can invert.

## Design implications (go/no-go = must change)

Recovery is exact and per-projection, so this is not a "harden the constant" fix.
Candidate directions (each needs its own security test):
1. **Break the Gram identifiability**: dimension-expanding redundant padding on the
   *weights* (not just the input), so `W'` is no longer a clean orthogonal conjugate
   of a known public matrix and `diag(W'W'ᵀ)` is no longer a permutation of a public
   diagonal.
2. **Don't fold exact public weights** (introduce a private transform / light
   fine-tune) — but that surrenders the zero-training / serve-stock-model advantage
   and inherits CONJFORMER's costs.
3. **Non-orthogonal or non-monomial residual masks** so the output mask does not
   cancel cleanly and the diagonal no longer leaks the permutation — must be checked
   against the ALS/Procrustes continuous attack too.

Until one of these is validated, **do not claim the folded residual stream hides the
input under this threat model** (attacker with public weights + offloaded folded
weights + masked residual states).

## Caveats / faithfulness
- Attack uses only public `W` + folded `W'`; the ground-truth `N_res` is used *only*
  to score recovery. Output masks (real pairwise-rotation / SwiGLU-perm) and RMSNorm
  gains are folded in, and the self-Gram empirically cancels them (residual ~1e-14).
- Storage precision provides NO protection (confirmed, `qwen7b_gram_attack_bf16.json`):
  simulating a **bf16-stored** folded package drops a *single* projection's permutation
  recovery to ~65%, but the 140-projection **majority vote still recovers N_res exactly
  (perm_agg_acc = 1.0)** → token recovery 98.5%, identical to fp64/fp32. The redundancy
  across q/k/v/gate/up × 28 layers defeats per-projection bf16 noise.
