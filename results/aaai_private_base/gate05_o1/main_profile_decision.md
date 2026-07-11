# Gate 0.5 main-profile decision

**Primary finding:** the RMSNorm-gamma fold makes the LoRA INPUT transform
`T_in = diag(1/gamma) @ Nr` non-orthogonal for the 5 gamma-fed targets (q,k,v,gate,up),
so ordinary masked SGD (O1-A) is a PRECONDITIONED update there, NOT plaintext-equivalent.
o_proj and down_proj are fully orthogonal (exact under O1-A).

Measured (fp64, real gamma, one primary seed):
- O1-A A-update error @10 steps: q 0.105, k 0.334, v 0.150, gate/up ~7e-4, o/down ~1e-17.
- O1-A momentum A-error @5: q 0.18, k 0.53, v 0.26.
- O1-B/O1-C corrected: ~1e-16 (exact) for all targets, incl. momentum.
- O1-A effective-dW cosine vs plaintext @10: 0.76 (k) - 1.0 (o/down).

**Profile assessment**
- O1-A (current GPU masked SGD): paper_safe, but NOT exact parameter-equivalent for the 5
  gamma-fed targets. Retain as an EFFICIENCY ABLATION (effective-weight/functionally
  aligned), explicitly NOT described as exact parameter-equivalent.
- O1-B (corrected GPU): exact, but materializes `Nr^T diag(gamma^2) Nr` (gamma^2 multiset)
  on the untrusted worker -> weakens the private-weight claim -> **REJECTED as paper_safe**.
- O1-C (trusted-equivalent optimizer): exact + no GPU gamma leak; +1 packed
  gradient-correction crossing/step (~3.4 MB) for the 5 gamma-fed targets. **SELECTED.**
- O1-D (orthogonality-restored forward): **INFEASIBLE** without either changing the
  plaintext LoRA function (placing LoRA on the pre-gamma normed state changes semantics)
  or exposing gamma (RMSNorm equivariance forces an ORTHOGONAL residual mask, and diag(gamma)
  cannot be made orthogonal nor applied in the masked domain without leak).

## DECISION
    MAIN_PROFILE = trusted_plaintext_equivalent_optimizer   (O1-C, hybrid)
      - o_proj, down_proj : GPU-exact masked SGD (O1-A), no correction, no crossing
      - q,k,v,gate,up     : trusted gradient correction in TDX (O1-C)
    EFFICIENCY_ABLATION = O1-A uniform GPU (functionally aligned, NOT exact-parameter)

Not selected by latency. Corrected-GPU (O1-B) rejected on leakage; restored-orthogonal
(O1-D) infeasible under the frozen model semantics.

## Preregistered for the L0-L8 matrix (not blocking this decision)
- Real-TDX O1-C end-to-end run (gradient-correction crossing) + bf16 full-H800 diagnostic.
- 3 fixed seeds (primary seed used here; seeds 2025, 7 preregistered).
- Momentum + AdamW under O1-C (AdamW already trusted-side per prior audit).
