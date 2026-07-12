# Alibaba A10 + TDX migration — summary

**Status: `ALICLOUD_PRIVATE_PATH_VALIDATED_EXPERIMENTS_RESUMED`**. Nothing committed.

The validated private-base O1-C LoRA direct-transport stack was migrated to Alibaba A10 (172.30.25.154)
+ TDX (172.30.25.153) over the **private VPC data plane**; the network bottleneck that deferred long
runs on AutoDL is **gone**, and the one/ten/fifty-step gates pass on real A10+TDX.

## What ran (verified)
| Gate | Result |
|---|---|
| Private-net preflight | RTT 0.143 ms, MTU 8500, 0% loss, direct eth0, no NAT |
| Throughput | ~170 MB/s down / ~182 MB/s up (both ≥100), ~1.15 Gbit/s — **~1700× AutoDL** |
| Package on A10 | root hash **bfd578b8 == pinned** (exact bytes; rebuild rejected on hash mismatch) |
| One-step L10 BF16 | top1 **1.0** vs plaintext, counters 0, attestation SUCCESS/debug-false, wall 0.9 s |
| L10 SGD 10-step | CE 0.4453→0.2921, top1 1.0, 0 missing, ~0.57 s/step |
| L11 momentum 10-step | CE 0.4453→0.2461, buffer transported, ~0.59 s/step |
| L10 50-step (sustained) | CE 0.4453→0.1080, all finite, ~0.54 s/step |

Cross-hardware: A10 (sm_86) vs H800 (sm_90) CE differs ~0.008 (bf16 platform), **within tolerance**;
equivalence top1=1.0 confirms algorithmic alignment (not required bitwise-identical, per spec).

## Deferred (enabled by this validation, not exhaustively run this session)
- **L12 AdamW** hardware gate — requires the trusted **enclave AdamW-state protocol** (m,v held in TDX
  for A of q/k/v/gate/up + B of q/k); not implemented (fp64-only to date).
- Full **50-step 3-seed × multi-family (L1–L5)** matrix — pipeline validated; needs seed plumbing + the
  other L-profiles wired for A10.
- **SST-2 / GSM8K converged utility** (L0/L5/L12, 3 seeds) — real multi-hour training; L12 arm needs the
  AdamW-state protocol.

These are no longer network-blocked; they are compute/protocol-scope items on a validated fast path.

## Cleanup
Secure TDX cleanup verified: 0 temporary keys (removed key rejected), 0 session-key/gamma/labels files,
152 temp payloads removed, 0 experiment processes/listeners; management key + `/dev/tdx_guest` + base env
preserved; attestation evidence pulled durably to `attestation_evidence/`. A10 transient session key +
transport key deleted; package/checkpoint/code retained. A10 ECS **safe to stop/release — user decides**
(not auto-released).
