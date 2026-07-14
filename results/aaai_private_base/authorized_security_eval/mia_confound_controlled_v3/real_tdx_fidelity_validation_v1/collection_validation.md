# Collection validation

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample fidelity-validation subset

Status: **INVALID_COLLECTION_MISMATCH**

- Primary collection: 250/250 rows, 611/611 columns, all finite, no duplicates, attestation verified.
- Corrective recapture: 250/250 rows under a new nonce and fresh verified attestation.
- V0 exact-text agreement: 250/250 in the primary capture and 250/250 in the recapture.
- Primary median absolute feature difference: 1.907e-06; 99th percentile: 2.355e-04.
- Exactly one feature has disjoint ranges: `hidden.final.last_l2`.

The frozen simulated collector reads PEFT/HF `hidden_states[-1]`, which contains the terminal RMSNorm gamma. In the transformed package, that gamma is folded into `lm_head`; the untrusted GPU materializes only the pre-norm residual and the gamma-free RMSNorm core. The first capture measured the core, and the preserved recapture measured the pre-norm residual. Neither is semantically equivalent to the simulated feature. Materializing gamma on the GPU would violate the protected protocol, so the mismatch cannot be repaired by another valid collection.
