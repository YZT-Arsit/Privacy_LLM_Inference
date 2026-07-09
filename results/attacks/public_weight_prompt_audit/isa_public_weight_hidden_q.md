# V3_isa_public_weight_hidden_q

- **threat_model**: public weights; attacker observes H_obs = H Q; Q secret in TEE
- **attacker_observation**: masked hidden states; can synthesise (H, H Q) pairs iff Q static
- **attacker_knowledge**: plaintext weights; k known-plaintext pairs under the same Q
- **attack_objective**: recover Q, then recover input tokens
- **metric**: Q_recovery_error, token_recovery_top1, required_known_rows_to_break
- **result**: no-known-plaintext top1 = 0.00e+00 (~0); STATIC Q: token recovery breaks at ~768 known rows (approximate Q already argmaxes correctly), and Q is recovered EXACTLY at known_rows >= d (1024 rows, Q err ~1e-14); fresh Q per session resists (token top1 = 0.00e+00).
- **claim_survives**: True
- **note**: Survives ONLY without known plaintext AND/OR with fresh Q per session. Static Q + >= d known plaintext rows is fully broken; this is reported, not hidden.

## Known-plaintext attack on STATIC Q (hidden dim d=768)

- no-known-plaintext token top1: 0.00e+00
- known rows to break TOKENS (approx Q): 768
- known rows to recover Q EXACTLY (>= d): 1024
- fresh Q per session token top1: 0.00e+00 (resists=True)

| known_rows | Q_recovery_err | token_top1 | broken |
|---|---|---|---|
| 1 | 1.67e-01 | 0.000 | False |
| 16 | 1.73e-01 | 0.000 | False |
| 64 | 1.66e-01 | 0.000 | False |
| 256 | 1.49e-01 | 0.727 | False |
| 768 | 6.15e-02 | 1.000 | False |
| 1024 | 1.46e-14 | 1.000 | True |
