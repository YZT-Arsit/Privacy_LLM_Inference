# Corrected update (O1-B on GPU / O1-C in TEE)

For every MIXED target (q,k,v,gate,up):
  gA_tilde_corrected = gA_tilde @ (T_in^T T_in)^{-1} = gA_tilde @ (Nr^T diag(gamma^2) Nr)
  A_tilde' = A_tilde - lr * gA_tilde_corrected
  momentum: v_A = mom*v_A + gA_tilde_corrected ; A_tilde' = A_tilde - lr*v_A
  B update unchanged (already exact); weight decay (if enabled) applies in the same metric.

Numerically (fp64, real gamma), the corrected update reproduces folded plaintext SGD to
machine precision for ALL targets (O1B A-update err ~1e-16, momentum ~1e-16), while naive
O1-A drifts (see per_target_results.csv).

The correction needs: (T_in^T T_in)^{-1} = Nr^T diag(gamma^2) Nr  -- i.e. Nr AND gamma^2.
It does NOT need T_out Grams (T_out orthogonal) nor rank-mask Grams (U orthogonal).

Two correction matrices per layer (input_layernorm for q/k/v; post_attention_layernorm
for gate/up), each 896x896, shared across the projections fed by that norm.
