# Trusted-boundary footprint — Profile E vs Profile N

| Item held inside TDX | Profile E (EXACT_REFERENCE_L12) | Profile N (NATIVE_TRANSFORMED_ADAMW) |
|---|---|---|
| Private labels / targets | yes | yes |
| CE + dlogits boundary | yes | yes |
| Session bindings + monotonic ledger | yes | yes |
| FP32 plaintext LoRA master | **yes** | **no** |
| AdamW m (first moment) | **yes** | **no** (on GPU, transformed coords) |
| AdamW v (second moment) | **yes** | **no** (on GPU, transformed coords) |
| Optional keys / refresh metadata | yes | yes |
| **Trusted optimizer bytes (local gate)** | **49152** | **0** |

Communication (per optimizer step): Profile E returns re-folded masked factors AND retains master+
m+v un-fold/step/re-fold in the enclave; Profile N runs AdamW on the GPU in transformed coordinates
and the enclave only serves CE/dlogits — fewer trusted round trips for the optimizer step itself.
(Exact on-hardware byte/latency counts = deferred Phase 3.4 resource comparison.)

Honest caveats: this reduced footprint is NOT claimed as "minimal TDX path"; Profile N places
additional transformed m/v on the GPU (see `security_delta_report.md`); no malicious-GPU integrity
is claimed for either profile.
