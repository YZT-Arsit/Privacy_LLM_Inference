# System overhead comparison (private-base) — existing measured artifacts only

Sources: inference latency `results/baselines/obfuscatune_latency_h800.json` (real NVIDIA H800 PCIe, bfloat16, dims {'D': 3584, 'FFN': 18944, 'layers': 28}); training profile `PHASE_profiling_gate.json` (real A10 + Intel TDX). SHA256 in `manifest.json`. **No new experiments were run for this table.**

## A. Algorithmic inference overhead (compute + TEE boundary crossings)

This is the *algorithmic* cost (TEE compute + boundary crossings), independent of the prototype network transport. STIP/ObfuscaTune/Amulet are inference-only baselines (`baseline_manifest.json`).

| Setting | prefill 512 (ms) | slowdown | prefill 1024 (ms) | slowdown | TEE boundary crossings |
|---|---|---|---|---|---|
| plaintext (no protection) | 32.1 | 1.000× | 75.5 | 1.000× | 0 |
| **ours** (GPU-TEE) | 32.2 | 1.002× | 75.5 | 1.000× | 2 |
| ours (CPU-TEE) | 77.2 | 2.403× | 229.5 | 3.040× | 2 |
| ObfuscaTune (GPU-TEE) | 34.4 | 1.072× | 79.6 | 1.054× | 224 |
| ObfuscaTune (CPU-TEE) | 2637.3 | 82.089× | 8626.3 | 114.276× | 224 |

**Key algorithmic result:** ours uses **2** TEE boundary crossings (single trusted session, zero nonlinear crossings) → **1.00–1.07× slowdown on a GPU-TEE**, versus ObfuscaTune's **224** crossings. On a CPU-only TEE the crossing count dominates (ours 2.4–3.0× vs ObfuscaTune 82–114×). STIP/Amulet: inference-only, no LoRA training.

## B. Protected LoRA-training overhead (real A10 + Intel TDX, batch 16)

| Component | value | kind |
|---|---|---|
| sec / batch (16 ex) | 0.459 | measured |
| TDX AdamW compute / opt step | 113.6 ms | **algorithmic** (CPU-only enclave, fixed) |
| CE network transfer / opt step | 44.5 ms | **prototype transport** |
| GPU forward / backward | 46.3 / 82.5 ms | GPU workload |
| trusted calls / opt step | 2 | algorithmic |
| comm / example (each way) | 1.14 MB | prototype transport |

## C. Prototype transport vs algorithmic overhead (must not be conflated)

- **Algorithmic overhead** (the paper's claim): 2 TEE crossings, GPU-TEE inference slowdown **≤1.07×**; fixed per-opt-step enclave AdamW (~114 ms on a CPU-only TDX guest).
- **Prototype transport** (implementation, not algorithmic): the per-token decode overhead measured in the SSH-tunnelled A10↔TDX prototype (~9× per token; ~71 ms tunnel RTT + fp32-logit payload) is a **deployment artifact of the prototype network path**, not the algorithm; a co-located enclave / GPU-TEE removes it (cf. the GPU-TEE row above at ≤1.07×). Cross-machine data-plane throughput on the private VPC ≈170 MB/s (`alicloud_a10_migration/**`).

_Honest separation: we report the algorithmic overhead as the contribution, and label the prototype network latency as an implementation artifact of the current two-machine testbed._