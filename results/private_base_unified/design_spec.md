# Private-base unified design spec (FROZEN v1.0)

Status: **frozen**. This is the single source of truth for the private-base main
line. The public GitHub README's public-weight scope is NOT the current threat
model. Frozen 2026-07-11. Do not silently deviate; a change bumps the version and
re-binds the runtime hash.

The prior `archive_rank_masked_sgd_50step` result (real Qwen2.5-0.5B, real H800,
real TDX private loss, rank-space masked SGD, 50-step stable) is **PASS but with
plaintext base weights + plaintext activations**. It is frozen and MUST NOT be
extended or used to support any private-base claim.

## Protected assets (must stay masked / trusted)
base weights, embeddings, user inputs, hidden states, KV cache, LoRA adapters,
LoRA gradients, labels/loss, optimizer state (where applicable), logits.

## Untrusted GPU/host must NEVER receive (plaintext)
checkpoint tensors, embeddings, hidden states, LoRA parameters/gradients, labels,
logits.

## A. Residual / feature state
    H_tilde_l = H_l · N_l
`N_l` is secret and well-conditioned. The GPU only ever holds `H_tilde`.

## B. Linear weights (GPU receives only the transformed weight)
    W_tilde_l = N_in^{-1} · W_l · N_out
For a row-vector activation convention `y = x W^T`, the folded operator is applied
so that `x_tilde @ W_tilde == (x W) @ N_out` with `x_tilde = x N_in` (see
`llama_qwen_single_block.fold_hf_single_block_weights` for the existing orthogonal
instance; private-base generalises `N` beyond signed permutations — see §H).

## C. QKV (RoPE first in logical coords, then feature masks)
    Q_rope = RoPE(Q) ;  K_rope = RoPE(K)
    Q_tilde = Q_rope · R
    K_tilde = K_rope · R^{-T}
    V_tilde = V · S
=>  Q_tilde · K_tilde^T = Q_rope · R · R^{-1} · K_rope^T = Q_rope · K_rope^T
Score-preservation holds for ANY invertible `R` (orthogonal OR non-orthogonal).
Generation and LoRA training use the SAME forward representation.

**Frozen open issue (OI-1, ordering):** "RoPE in logical coords then mask" means
the projection must present `Q_rope`/`K_rope` before `R` is applied. Realising this
with `plaintext_hidden_materializations = 0` requires either (a) the Q/K projection
output basis to commute with RoPE so `R` can be folded pre-RoPE (pair structure),
or (b) computing RoPE in the masked basis with correspondingly transformed tables.
The unified forward MUST take route (a)/(b) and MUST NOT materialise plaintext
`Q_rope`/`K_rope`. This is audited in the exposure inventory and gated in C1.

## D. KV cache (masked + append-compatible)
    K_cache_tilde = K_cache_rope · R^{-T}
    V_cache_tilde = V_cache · S
Appending a new masked token's `K_tilde`/`V_tilde` is exact because `R`, `S` are
per-(head,feature) constants, not per-position.

## E. Nonlinear execution
    execution_profile = paper_safe
    nonlinear_backend  = A_rightmul
    nonlinear_trusted_calls = 0
    plaintext_hidden_materializations = 0
(gated by `pllo.experiments.execution_profile`.)

## F. Logits (O(V) bounded-condition monomial mask; NO dense vocab mask)
    logits_tilde = logits · D · Pi
`D` nonzero, bounded condition number, no extreme bf16 scales (see
`pllo.masks.monomial_logit_mask`). Permutation-only (`D = I`) is the baseline.

## G. Attention leakage (explicitly recorded, NOT claimed away)
The paired-feature construction (§C) exposes the EXACT token-token attention score
matrix `Q_rope K_rope^T` (and softmax `probs`) on the untrusted GPU, unless a
stronger primitive is added. This is the known attention-fingerprint leak.
**Do NOT claim:** attention-score confidentiality, zero leakage, or formal
security. Mitigation currently limited to relocating layer-0 attention into the
TEE (partial; layers 1+ still expose scores).

## H. Base-weight security (candidate, NOT proven)
Non-orthogonal feature masks (`N`, `R`, `S`) are a CANDIDATE defence against direct
orthogonal-Gram alignment (the self-Gram attack recovers an *orthogonal* mixer
exactly). They are NOT a proven security guarantee. Their real strength is decided
by W1 (residual-chain alignment attack) and A1 (private-weight attack matrix). Do
not mark a non-orthogonal profile paper-safe / secure before W1/S1/A1.

## Separated mask spaces (never one symbol for all domains)
- **Feature masks** `N_in, N_out, R, S` — hidden states, base weights, Q/K/V, KV cache.
- **LoRA rank masks** `U` — rank space only.
- **Nonlinear permutations** `Pi_nl` — nonlinear-compatible islands only.
- **Vocabulary mask** `D_vocab, Pi_vocab` — logits boundary only.
The manifest and `config_digest` record these domains separately.

## Effective masked linear with LoRA (all 7 targets)
    W_eff_tilde = W_base_tilde + A_tilde · B_tilde
targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. Base term
frozen (detached); LoRA term differentiable in the masked adapters. Correct mask
convention per projection (in-basis / out-basis) documented in the folded-forward
module.

## Optimizer profiles (see o1_optimizer/)
- **O1-A** orthogonal feature+rank masks, direct GPU masked SGD, 2 trusted inv/step
  (correctness control, NOT the strongest private-weight profile).
- **O1-B** non-orthogonal feature masks, preconditioned masked SGD (2 inv/step) —
  paper-safe ONLY if the preconditioner does not leak the secret Gram (audited).
- **O1-C** non-orthogonal feature masks, packed LoRA grads to TDX, trusted optimizer,
  3 trusted inv/step (deployable correctness/security reference).

## What must never be claimed from this spec alone
- Private-base security merely because the package is transformed.
- Non-orthogonal-mask security before W1/S1/A1 complete.
- Attention-score confidentiality / zero leakage / formal security.
