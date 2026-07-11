# Leakage analysis of the O1 correction

The exact-equivalence correction tensor is `M = Nr^T diag(gamma^2) Nr` per RMSNorm/layer.

- **Eigenvalues(M) = gamma^2** (Nr orthogonal). So publishing/using M on the untrusted
  worker reveals the MULTISET of squared RMSNorm gains gamma^2 -- a PRIVATE base-weight.
  The eigenvector basis is Nr-rotated, so the assignment of gamma^2 to features is hidden
  without Nr, but the gamma spectrum itself leaks. gamma ranges are wide (input_layernorm
  cond up to 204; post_attention up to ~8e5), so the spectrum is informative.
- This is strictly MORE than the no-reference private-weight baseline (which never
  exposes gamma). **O1-B (GPU correction) therefore weakens the private-weight claim and
  is NOT paper_safe.**
- **O1-C** applies M inside TDX (which legitimately holds gamma and Nr); the untrusted
  worker sends packed masked gradients and receives corrected transformed updates -- M
  never touches the GPU. Cost: +1 trusted crossing/step; packed gradient ~3.4 MB for the
  5 gamma-fed targets x 24 layers (small vs the 25 MB logit ferry).
- The correction canNOT be folded into a static GPU operator without materializing M on
  the worker (it multiplies the per-step gradient, and any on-device realization exposes
  the gamma^2 spectrum).
- o_proj and down_proj need NO correction (orthogonal T_in) -> GPU-exact, no leak.
