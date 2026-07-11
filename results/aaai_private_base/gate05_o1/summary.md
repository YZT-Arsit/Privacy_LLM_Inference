# Gate 0.5 -- O1 gamma-fold & optimizer equivalence -- SUMMARY

**Non-orthogonality source:** RMSNorm gamma folded into the base weight => LoRA input
transform `T_in = diag(1/gamma) @ Nr`, `T_in^T T_in = Nr^T diag(1/gamma^2) Nr != I`.
Cond(T_in): q/k/v ~204, gate/up ~8e5. o_proj/down_proj orthogonal (cond 1).

**Classification:** q,k,v,gate,up = MIXED (A non-orthogonal, B orthogonal); o_proj,
down_proj = O (both exact). No target is fully NO.

**Is O1-A parameter-equivalent?** NO globally. B-factor exact for all; A-factor exact
only for o_proj/down_proj. A-drift up to 33% (SGD) / 53% (momentum) on q/k/v.

**Only functionally aligned:** O1-A effective-dW cosine vs plaintext 0.76-1.0.

**Corrected (O1-B/C):** machine-exact (1e-16). Correction = Nr^T diag(gamma^2) Nr, which
leaks the gamma^2 multiset if applied on the GPU (O1-B, rejected) but is safe in TDX
(O1-C, +1 packed ~3.4MB crossing/step).

**O1-D restored-orthogonal:** INFEASIBLE (RMSNorm forces orthogonal residual mask; gamma
cannot be orthogonalized or applied in the masked domain without leak/semantic change).

**DECISION: MAIN_PROFILE = trusted_plaintext_equivalent_optimizer (O1-C hybrid)** --
GPU-exact for o_proj/down_proj, trusted gradient correction for the 5 gamma-fed targets.
O1-A retained as efficiency ablation (must NOT be called exact-parameter-equivalent).

Transport: mac_ferried_authenticated_prototype (not final latency). bf16 full-H800 +
real-TDX O1-C + 3-seed preregistered for the matrix. Nothing committed.
