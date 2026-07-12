# LoRA baseline decision

Decision rule (frozen): implement an external baseline only if (1) open-source impl exists,
(2) compatible with our threat model (private base weights on an untrusted host + TEE), and
(3) reasonable cost. Otherwise state **why it is N/A** — the paper must explain this, not
silently omit it.

## Decisions
| Baseline | Decision | Reason |
|---|---|---|
| **Plaintext LoRA** (Hu et al. ICLR'22) | **INCLUDE (mandatory)** | exact correctness + utility reference; already run as the **L5** arm (SST-2 L12−L5 = +0.0027). It is the right "no-protection" upper bound for the utility gap. |
| **QLoRA** (Dettmers et al. NeurIPS'23) | **N/A (not a privacy baseline)** | targets quantized-memory efficiency, no privacy threat model, does not protect base weights. A utility/privacy comparison is a category error. Could be cited as an orthogonal efficiency technique only. |
| **DP-SGD / DP-LoRA** (Abadi CCS'16; DP fine-tuning line) | **N/A (different threat model)** | protects *training-data* privacy via gradient noise, at a utility cost, against a *trusted* trainer. It does **not** hide base weights from an untrusted execution host and uses no TEE. Comparing task accuracy would conflate two different privacy axes and mislead. |
| **Federated LoRA (FFA-LoRA, ICLR'24)** | **N/A (different threat model)** | protects cross-client aggregation privacy; every client still holds plaintext weights. No untrusted-host base-weight protection. |
| **Offsite-Tuning** (Xiao et al. '23) | **N/A (lossy, different mechanism)** | shares a compressed *emulator* (lossy), not an exact-fold unified package, and provides no TEE-backed confidentiality; not our setting. |
| **STIP / ObfuscaTune / Amulet** (repo baseline suite) | **INCLUDE for generation + overhead only** | inference-only (`training_support=N/A`); used in the **Phase 5 system-comparison** (latency/comm/TEE workload), **not** as LoRA-training utility baselines. Real artifacts already exist (`results/baselines/`). |

## What the paper compares against (final)
- **Utility (LoRA training)**: ours (protected L12) vs **plaintext LoRA (L5)** and the **base
  model (L0)**. This is the only threat-model-faithful LoRA-training comparison.
- **Ablations** (Phase 4) stand in for "internal baselines" — each removed component shows
  the contribution of the design over weaker variants.
- **Generation + system overhead**: ours vs plaintext vs **STIP / ObfuscaTune** (+ Amulet),
  reusing existing measured artifacts (Phase 5).

## Why we do NOT implement a new external LoRA privacy baseline
No published privacy-preserving-LoRA method shares our threat model (base-weight
confidentiality against an untrusted execution host with a TEE + exact-fold unified package).
The candidates that *do* train LoRA (DP-LoRA, federated LoRA) defend a **different asset**
(data / cross-party privacy), so a head-to-head accuracy comparison would be misleading and
is deliberately omitted, with this justification stated in the paper. This is consistent with
the execution constraint "if an experiment is unnecessary, explain why instead of running it."
