# Limitations — full LoRA validation matrix

## Precision & where exactness is decided
- **Parameter-equivalence is an exact-arithmetic property and is decided at fp64** (per the
  plan). The comprehensive matrix (3 optimizers × 3 seeds × 24 layers × steps{1,10,50}) runs
  fp64 on the real Qwen2.5-0.5B checkpoint. The real H800 run is fp32 for the clean-equivalence
  gate; it validates the systems integration (in-enclave correction, counters, transport) and
  model-level equivalence (top1 1.0), NOT exactness.
- **BF16 note (CORRECTED).** An earlier draft claimed the tiny gate/up correction components
  (min_abs 5.9e-17) "vanish in BF16." That was **wrong**: BF16 shares FP32's 8-bit exponent
  (range ~1e-38), so those components are representable (with reduced mantissa) and do **not**
  underflow. The **real BF16 L10 gate was subsequently run on real H800+TDX** (see
  `empirical_matrix_closure/bf16_gate/`): stable, finite, `corrected_grad_zeroed_fraction ≈ 0`,
  correction executes in-enclave; equivalence degrades to top1 ~0.976 / KL ~8e-3 vs the fp64
  plaintext reference — this is BF16 *mantissa* precision, not instability. Exactness remains an
  fp64 property; BF16 is the deployment-numerics validation.

## O1-A instability (a finding, not just drift)
- On gate/up, naive O1-A is not merely approximate — it **numerically diverges** (NaN in 66/144
  fp64 cases by step 50 at lr 1e-3) because the uncorrected update overshoots along small-γ
  directions. This is stronger than "functionally aligned"; O1-A is unusable on the γ-fed MLP
  targets. O1-C is exact there.

## Transport
- `transport_profile = mac_ferried_authenticated_prototype`. Per-step wall time (~184 s) is
  dominated by cross-cloud SSH ferrying (25.5 MB logits each way + 3.48 MB correction each way),
  NOT by TDX compute (CE+dlogits + correction ≈ 0.05–0.1 s in-enclave). **Ferried end-to-end
  latency is NOT a deployment-performance claim.** `direct_h800_tdx_transport` is registered,
  not yet built.

## Real-hardware coverage actually run vs preregistered
- **Run:** L10 real H800+TDX 1-step (top1 1.0) and 10-step (monotonic CE 0.454→~0.29), seed 1234,
  fp32, full counters + in-enclave correction. Rank-mask ablation (fp64). Full fp64 equivalence
  matrix (3 opt × 3 seeds × 24 layers).
- **Preregistered / budget-limited (honestly not claimed as done):** real 50-step 3-seed L10;
  real momentum (L11) and trusted-AdamW (L12) end-to-end on hardware (the exactness is fp64-proven
  and L12's TDX-basis AdamW is implemented + fp64-validated, but a full real-hardware L11/L12 ferry
  run was not executed); BF16-forward gate; utility to convergence on full official splits.

## Utility
- L10 utility EQUALS L1 by construction (exactness proof). The protected deployment provenance +
  held-out CE are real and package-native; full GSM8K exact-match / SST-2 accuracy on the official
  splits (generation harness) is preregistered budget, not a convergence claim here.

## Scope / threat model (unchanged)
- No cryptographic-security or zero-leakage claim. `L9` protects rank-only, NOT private base
  weights — not compared as if it did. External privacy baselines remain inference-only
  (LoRA_training_support = N/A); no unverified literature baselines implemented in this matrix.
