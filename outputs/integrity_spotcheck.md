# Integrity Spot-Check Prototype

_Stage 7.7e: probabilistic detector for a malicious accelerator returning corrupted masked tensors. Prototype only, not full verifiable computation._

## Configuration

| Field | Value |
|---|---|
| batch_size | 4 |
| seq_len | 32 |
| hidden_size | 64 |
| vocab_size | 256 |
| head_dim | 16 |
| num_kv_heads | 4 |
| num_steps | 32 |
| checked_fractions | [0.0, 0.05, 0.1, 0.25, 0.5] |
| num_trials_per_setting | 50 |
| corruption_magnitude | 0.1 |

## Per-Mode Detection Curves

### `no_check`

| checked_fraction | empirical_detection_rate | expected_detection | false_positive_rate | extra_trusted_ops |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 0.05 | 0.0 | 0.0 | 0.0 | 0 |
| 0.1 | 0.0 | 0.0 | 0.0 | 0 |
| 0.25 | 0.0 | 0.0 | 0.0 | 0 |
| 0.5 | 0.0 | 0.0 | 0.0 | 0 |

### `spot_check_linear_projection`

| checked_fraction | empirical_detection_rate | expected_detection | false_positive_rate | extra_trusted_ops |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 0.05 | 0.04 | 0.05 | 0.0 | 26214 |
| 0.1 | 0.12 | 0.1 | 0.0 | 52428 |
| 0.25 | 0.22 | 0.25 | 0.0 | 131072 |
| 0.5 | 0.3 | 0.5 | 0.0 | 262144 |

### `spot_check_lm_head_slice`

| checked_fraction | empirical_detection_rate | expected_detection | false_positive_rate | extra_trusted_ops |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 0.05 | 0.06 | 0.05 | 0.0 | 104857 |
| 0.1 | 0.12 | 0.1 | 0.0 | 209715 |
| 0.25 | 0.18 | 0.25 | 0.0 | 524288 |
| 0.5 | 0.62 | 0.5 | 0.0 | 1048576 |

### `spot_check_kv_cache_append`

| checked_fraction | empirical_detection_rate | expected_detection | false_positive_rate | extra_trusted_ops |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 0.05 | 0.06 | 0.05 | 0.0 | 204 |
| 0.1 | 0.06 | 0.1 | 0.0 | 409 |
| 0.25 | 0.24 | 0.25 | 0.0 | 1024 |
| 0.5 | 0.48 | 0.5 | 0.0 | 2048 |

## Sanity

| Field | Value |
|---|---|
| no_check_corruption_undetected | True |
| linear_projection_detection_increases_with_checked_fraction | True |
| no_false_alarm_under_clean_curves | True |

## Policy Flags

| Field | Value |
|---|---|
| active_adversary_integrity_supported | probabilistic spot-check only |
| full_verifiable_computation | False |
| malicious_accelerator_privacy_not_addressed | True |

## Limitations

- CPU local emulation only; no real attestation, no real TEE / GPU.
- Detection rate is parameterised by checked_fraction; with sample size 50 the empirical rate is noisy and is checked monotonically rather than to exact theoretical value.
- This is NOT a verifiable computation primitive, NOT a ZK proof, NOT an authenticated dataflow.
- The adversary is assumed to commit to a fixed corruption location per call; adaptive corruption that observes which items the TEE chose to spot-check would lower the effective detection rate.
- Malicious accelerator can still mount denial-of-service or selectively corrupt UN-checked tokens.
- Privacy under a malicious accelerator (rather than integrity) is NOT addressed by this mode.
- Not formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> We prototype a probabilistic spot-check defence against an active adversary that returns corrupted masked tensors. The TEE recomputes a random fraction of the boundary linear and detects any mismatch. This is a lightweight prototype, not full verifiable computation; detection is per-call probabilistic, false-positive-free under correct execution, and explicitly does not address privacy under a malicious accelerator.

## Unsafe Wording to Avoid

- Active malicious accelerator fully handled.
- Verifiable computation provided.
- Authenticated dataflow.
- Cryptographic integrity proof.

