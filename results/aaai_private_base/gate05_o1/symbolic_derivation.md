# Symbolic derivation: masked SGD vs folded plaintext SGD

Loss identical (T_out orthogonal => MSE/CE basis-invariant). With `A_tilde = A_plain
T_in^{-T}`, `B_tilde = T_out^T B_plain`:

Gradients (chain rule, row-vector):
  dL/dA_plain = dL/dA_tilde @ T_in^{-1}     (verified numerically: gradA_relation_err ~ 1e-16)
  dL/dB_plain = T_out @ dL/dB_tilde

Plaintext SGD then fold-to-masked:
  A_plain' = A_plain - lr dL/dA_plain
  fold: A_tilde_target = A_plain' T_in^{-T} = A_tilde - lr (dL/dA_tilde)(T_in^T T_in)^{-1}
  B_tilde_target = T_out^T B_plain' = B_tilde - lr (T_out^T T_out)(dL/dB_tilde)

Naive masked SGD (O1-A): A_tilde' = A_tilde - lr dL/dA_tilde,  B_tilde' = B_tilde - lr dL/dB_tilde.

Therefore:
  **A update exact  <=>  T_in^T T_in = I**  (holds only for o_proj, down_proj)
  **B update exact  <=>  T_out^T T_out = I** (holds for ALL targets)

So B is always exact; A is exact only for the two non-gamma-fed targets. The required A
correction is a right-multiply by  (T_in^T T_in)^{-1} = Nr^T diag(gamma^2) Nr.
Same relation holds for the momentum buffer (the buffer must live in the corrected
metric), verified numerically.
