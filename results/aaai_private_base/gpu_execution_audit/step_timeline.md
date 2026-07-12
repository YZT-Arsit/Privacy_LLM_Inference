# S5 direct-transport per-step timeline (real hardware)

- run: `AUDIT_S5-1s-1783820131-eb5b1b`  transport: `direct_h800_tdx`  steps: 1
- mean step wall: **155.84 s**

## Mean fractions of the step wall

| phase | fraction |
|---|---|
| GPU compute (fwd+bwd+apply) | 0.52% |
| CPU processing (serialize/deserialize) | 0.03% |
| Network wait (SSH wire, dominated by throttled TDX→H800 download) | 99.30% |
| TDX enclave (CE + correction) | 0.10% |
| Idle/unattributed | 0.05% |

## Interpretation

The step wall is dominated by **network wait** — the throttled TDX→H800 ingress moving the dlogits (and corrected grads) back to the GPU host. GPU compute is a tiny fraction. This is a transport/network limit, not a CPU-compute or device-placement bug: the forward/backward run on CUDA (S1–S4).