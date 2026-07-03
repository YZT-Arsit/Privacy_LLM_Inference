# STEP 2: can the attacker peel the Linear-boundary pad and expose hidden states?

Follow-up to the Step-1 mask recovery (`results/attacks/ours_gram/`). Once the masks
are recovered, does the fresh per-module Linear-boundary pad `T_l` still hide the
intermediate hidden states `H_l`? Real Qwen2.5-7B-Instruct, layer 14, real
activations. Script: `scripts/attacks/pad_recovery_attack.py`. Data:
`qwen7b_pad_recovery_scale0.1.json` (production scale 0.1).

## Verdict: **pad provides NO protection of hidden states** — joint attack exposes all H_l (cosine 1.0)

Threat model: attacker has public `W_l`, the recovered masks, the GPU-visible
compensation `C_pad,l = T_l W_l N_out,l`, and the padded operand `H̃_l = (H_l−T_l)N_in,l`.
Core equation after cancelling the recovered `N_out`: `C'_l = T_l W_l`.

| module | class | d_in→d_out | left-null | single-layer T-relerr | **joint T-relerr** | H_cos single | **H_cos joint** |
|---|---|---|---|---|---|---|---|
| q_proj | square | 3584→3584 | 0 | 0.04 | 0.04 | 1.00 | **1.00** |
| o_proj | square | 3584→3584 | 0 | 0.06 | 0.06 | 1.00 | **1.00** |
| gate_proj | up-proj | 3584→18944 | 0 | 0.00 | 0.00 | 1.00 | **1.00** |
| up_proj | up-proj | 3584→18944 | 0 | 0.00 | 0.00 | 1.00 | **1.00** |
| k_proj | down-proj | 3584→512 | 3072 | **0.93** | 0.15 | 0.99 | **1.00** |
| v_proj | down-proj | 3584→512 | 3072 | **0.92** | 0.15 | 0.99 | **1.00** |
| down_proj | down-proj | 18944→3584 | 15360 | **0.90** | 0.00 | 0.85 | **1.00** |

**Decisive metric = H_cos_joint (hidden-state recovery cosine). It is 1.00 for every
module.** The intermediate hidden states are fully exposed once the masks are known.

## Per-shape-class analysis (answers the go/no-go)

**Single-layer pseudoinverse `T̂ = C' W⁺`** recovers `T` projected onto `W`'s row
space; the left-null-space component (dim `d_in − rank W`) is lost:
- **Square (q, o)** and **up-projection (gate, up)**: full row rank → null=0 → `T`
  recovered → **pad fails single-layer** (T-relerr ≈ 0, H_cos = 1.0).
- **Output-reduced (k, v: d→d_kv; down: 4d→d)**: large left null (3072, 3072, 15360)
  → single-layer `T` recovery fails (T-relerr 0.90–0.93) → **pad survives single-layer**.

**Cross-module joint attack breaks the output-reduced layers too** (the decisive test):
- **k, v** read the *same* attn-norm input as **q**. q is square → solve `T_q` → recover
  the shared input `Z1 = H̃_q N_res⁻¹ + T_q` → `T_k = Z1 − H̃_k N_res⁻¹`, `T_v` likewise.
  → H_cos_joint = 1.00.
- **down**'s input is the SwiGLU intermediate `I = SiLU(gate)⊙up`, reconstructable from
  the (un-maskable) gate/up outputs → `T_down = I − H̃_down N_swiglu⁻¹` → **joint T-relerr
  = 0.00**, H_cos_joint = 1.00.
- (The 0.15 residual T-relerr on k/v is fp32 noise propagated from q's pseudoinverse on
  a *tiny* pad vector — it does not affect H recovery, which is cosine 1.00.)

## The pad is weak even before the attack (scale)

The production pad scale (0.1) barely perturbs activations: `pad_to_activation_ratio`
is 0.14–0.61, and the random-`T` baseline already recovers `H` at cosine 0.96–0.98.
So the pad hides very little to begin with, and the joint attack removes what remains
(cosine 1.00). [A large-scale run confirming the attack peels even a *strong* pad:
`qwen7b_pad_recovery_scale10.json` — see below.]

## Conclusion

Combined with Step 1 (mask recovered exactly → 98.5% token recovery), **the
Linear-boundary pad adds no security** in this threat model: single-layer it survives
only on the output-reduced k/v/down layers, and the cross-module joint attack
(shared-input for q/k/v, upstream SwiGLU reconstruction for down) recovers every pad
and exposes every intermediate hidden state to cosine 1.00. The pad is boundary-local
and additively compensated, so it never changes `W' = N⁻¹ W N_out` and cannot help the
mask-recovery step either. **Do not rely on the fresh pad as a hidden-state defense.**
