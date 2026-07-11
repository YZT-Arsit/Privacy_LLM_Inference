# Exact implemented parameterization (per LoRA target)

Row-vector convention (code: `y = x @ W.T`, `x` is `(T,in)`). Residual/hidden mask `Nr`
is an orthogonal signed permutation. `ga`=input_layernorm.weight, `gm`=post_attention_
layernorm.weight (both PRIVATE RMSNorm gains). `B`=per-head RoPE-commuting rotation,
`S`=per-head V signed-perm, `P`=SwiGLU intermediate permutation. All of B/S/P/Nr are
orthogonal; `diag(ga)`, `diag(gm)` are NOT.

For each target: `x_tilde = x_plain @ T_in`, `y_tilde = y_plain @ T_out`, base fold
`W_base_tilde = T_out^T W_plain T_in^{-T}`, LoRA fold `A_tilde = A_plain T_in^{-T}`,
`B_tilde = T_out^T B_plain` (worker trains `A_tilde,B_tilde` with an orthogonal rank mask
`U` on the r-dim, which does not affect T_in/T_out).

| target | x_plain | x_tilde | T_in | T_out | T_in orth? | T_out orth? |
|---|---|---|---|---|---|---|
| q_proj | rmsnorm(H)⊙ga | rmsnorm(H)@Nr | diag(1/ga)@Nr | blockdiag(B, n_h) | **NO** (diag 1/ga) | yes |
| k_proj | rmsnorm(H)⊙ga | rmsnorm(H)@Nr | diag(1/ga)@Nr | blockdiag(B, n_kv) | **NO** | yes |
| v_proj | rmsnorm(H)⊙ga | rmsnorm(H)@Nr | diag(1/ga)@Nr | blockdiag(S, n_kv) | **NO** | yes |
| o_proj | attn_out | attn_out@blockdiag(S,n_h) | blockdiag(S, n_h) | Nr | yes | yes |
| gate_proj | rmsnorm(Hm)⊙gm | rmsnorm(Hm)@Nr | diag(1/gm)@Nr | P | **NO** (diag 1/gm) | yes |
| up_proj | rmsnorm(Hm)⊙gm | rmsnorm(Hm)@Nr | diag(1/gm)@Nr | P | **NO** | yes |
| down_proj | silu(gate)*up | (silu*up)@P | P | Nr | yes | yes |

**Exact source of non-orthogonality:** the RMSNorm gain `gamma` is FOLDED into the base
weight (so the worker's normed input `x_tilde` carries no gamma), hence bridging
`x_tilde -> x_plain` requires `T_in = diag(1/gamma) @ Nr`. `T_in^T T_in = Nr^T
diag(1/gamma^2) Nr != I`. Only o_proj and down_proj read a non-normed input (attention
output / SwiGLU intermediate), so their `T_in` is a pure orthogonal/permutation mask.
