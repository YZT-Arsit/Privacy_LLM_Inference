# Stage 7.6 — Masked-Gradient LoRA Security Proxy

Proxy audit of leakage from GPU-visible masked parameters and masked gradients. The GPU never receives plaintext LoRA adapters or plaintext LoRA gradients in this experiment. No formal, cryptographic, or semantic security is claimed. This is a CPU-only algebraic and proxy-leakage experiment, not a real TEE/GPU training benchmark.

## Fresh-masks-per-step trials

| trial | rank_proxy_match_true_rank | rank_proxy_label | abs_mean_cos_A | abs_mean_cos_B | linkability_label |
|---|---|---|---|---|---|
| 0 | False | low_proxy_risk | 0.0169 | 0.0977 | low_proxy_risk |
| 1 | False | low_proxy_risk | 0.088 | 0.0532 | low_proxy_risk |
| 2 | False | low_proxy_risk | 0.039 | 0.0443 | low_proxy_risk |

## Fixed-masks baseline trials

| trial | rank_proxy_match_true_rank | rank_proxy_label | abs_mean_cos_A | abs_mean_cos_B | linkability_label |
|---|---|---|---|---|---|
| 0 | False | low_proxy_risk | 0.0333 | 0.2547 | low_proxy_risk |
| 1 | False | low_proxy_risk | 0.1349 | 0.3035 | medium_proxy_risk |
| 2 | False | low_proxy_risk | 0.1142 | 0.2565 | low_proxy_risk |

## Limitations

- Proxy attacks only -- NOT a formal security proof.
- True-rank inference depends on the dummy strategy and on the mixer's invariance properties; the cancellation-padded paired-strategy used here is not guaranteed to hide the rank against all spectral attacks.
- Fixed-mask baseline is included for reference; the default Stage 7.6 policy is fresh masks per step.
- No real TEE / GPU runtime; no hardware side-channel evaluation.
- No formal cryptographic / semantic / differential-privacy security is claimed.
- Raw tensors, masks, adapters, and gradients are NEVER exported.

`formal_security_claim`: `False`

