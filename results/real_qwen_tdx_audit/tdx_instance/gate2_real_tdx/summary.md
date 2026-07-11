# Gate 2 — real TDX service validation (evidence)

**gate2_pass = True**  (sgd=True, adamw=True, negatives=True)

Flags: uses_real_tee=True, uses_real_gpu=False,
uses_real_qwen=False, service_validation_only=True.
guest_hostname=iZ2zebdihkqll10ragqp0cZ.

The exact trusted training-service code runs INSIDE the real Intel TDX guest. Each
profile: fresh nonce -> /train/challenge (guest binds it + the guest ECDH public key
into report_data and produces a REAL TD Quote) -> DCAP QVL appraisal (verifier +
relying_party -v: PCK cert chain + ES384 policy signature) -> external verifier
re-derives report_data over every bound field and enforces DEBUG=false -> attested
X25519 ECDH -> AEAD-wrapped real endpoints over the SSH tunnel.

This is SERVICE VALIDATION ONLY with SYNTHETIC protocol tensors — NOT Qwen, NOT GPU.

### gpu_masked_sgd
- attestation_verified: **True** | quote_chain(QVL+ES384 sig): True | overall_appraisal_result: 1 | report_data_match: True
- measurement_policy_passed: True | DEBUG: False | quote_version: 5 | appraisal_source: relying_party_v_a
- mr_td: `8a56b29ae530d23ee602628eff3b7fdf5e159d4a55d62b24...`
- runtime_hash: `ec24ab8ef0801e4e5ed33a7711bd879c21562502b603d058...`
- trusted invocations/step: **2** (expected 2, ok=True) | trusted_optimizer_calls: 0
- logits_loss: loss_abs_err=0.00e+00, dlogits_abs_err=1.42e-07
- packed_update rejected in SGD mode: True

### trusted_adamw
- attestation_verified: **True** | quote_chain(QVL+ES384 sig): True | overall_appraisal_result: 1 | report_data_match: True
- measurement_policy_passed: True | DEBUG: False | quote_version: 5 | appraisal_source: relying_party_v_a
- mr_td: `8a56b29ae530d23ee602628eff3b7fdf5e159d4a55d62b24...`
- runtime_hash: `5a841ebe9fa1dae862d71a2cc27fb12b84cd582a60876ed3...`
- trusted invocations/step: **3** (expected 3, ok=True) | trusted_optimizer_calls: 1
- logits_loss: loss_abs_err=0.00e+00, dlogits_abs_err=1.42e-07
- packed_update: adapter_abs_err=2.68e-06 (real AdamW verified)


## Negative / fail-closed matrix (real wire)
- [PASS] wrong_runtime_hash_rejected: AttestationFailClosed
- [PASS] challenge_optimizer_mode_mismatch_rejected: /train/challenge: HTTP 400: {"error": "FailClosed: challenge optimizer_mode mism
- [PASS] aead_replay_rejected: /train/logits_loss: HTTP 400: {"error": "decode rejected: KexError: non-monotoni
- [PASS] aead_tamper_rejected: /train/init: HTTP 400: {"error": "decode rejected: KexError: AEAD authentication
- [PASS] oversized_rejected: declared > cap (server rejects on Content-Length)
- [PASS] sgd_packed_update_rejected: /train/packed_update: HTTP 400: {"error": "FailClosed: packed_update not allowed
