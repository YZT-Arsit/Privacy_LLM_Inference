# Gate 1.5-E — Orthogonal Masked-SGD Contract (CPU fp64)

Service-validation of the math contract. CPU float64, synthetic tensors. NOT a real Qwen or real TDX experiment. Nothing committed.

## Answers

1. **SGD closes exactly under fixed orthogonal masks?** YES — `sgd_orthogonal` worst A rel-err 2.0e-11 over 100 steps; permutation masks too.
2. **Momentum SGD closes exactly?** YES — `momentum_orthogonal` worst momentum rel-err 2.1e-12 over 100 steps (mA_t == N_in^-1 mA U).
3. **Which SGD options supported?** pure SGD, momentum (0/0.9), **Nesterov** (proven: `nesterov_orthogonal` aligned=True), coupled L2 + decoupled multiplicative weight decay, gradient accumulation 1/2/4, LR scaling. All are linear in (grad, momentum) so they commute with fixed orthogonal masks.
4. **Which optimizers rejected?** Adam/AdamW/RMSProp/Adagrad in `gpu_masked_sgd` mode (element-wise second moments do NOT commute with dense masks) — service fails closed at init.
5. **Packed optimizer boundary removed?** YES in masked-SGD mode — the GPU owns masked A/B (+ masked momentum); no `/train/packed_update`, no trusted optimizer.
6. **Trusted invocations/step?** `gpu_masked_sgd` = **2** (input + loss; packed_update=0, optimizer_trusted=0); `trusted_adamw` = **3** (retained).

## Orthogonal collapse
`N_in^T gradA U^-T == N_in^-1 gradA U` under orthogonality (diff 0.0e+00); differs for general GL (diff 3.4e+01). We use the orthogonal form, not a general-GL formula.

## Verdict
positive cases aligned = **True**; negative controls correctly fail = **True** (mask refresh without re-encoding → not aligned; dense non-orthogonal direct SGD → not aligned, and the service rejects it).

## Correctness (allowed) vs security (separate)
- **Correctness:** fixed orthogonal masks make SGD/momentum updates exactly equivariant (fp64).
- **Efficiency:** the optimizer boundary is removed, 3→2 trusted invocations/step.
- **Security:** orthogonal masks preserve MORE second-order geometry (norms/Gram/spectrum) than dense GL — the actual privacy impact must be MEASURED (Gate: security trade-off), NOT assumed. Orthogonal-SGD is not presumed safe.

## Labels
`uses_real_gpu=false`, `uses_real_tee=false`, synthetic fp64 contract. Real Qwen2.5-0.5B masked-SGD is Gate 3 (blocked on GPU gateway cooldown).
