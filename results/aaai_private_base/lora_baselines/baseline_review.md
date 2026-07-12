# LoRA privacy-baseline review (literature-grounded)

Purpose: decide which external baselines are **comparable** to our setting — *private base
weights on an untrusted execution host, TEE (Intel TDX) trust boundary, private input/labels,
private LoRA adapter + gradients, one unified transformed package for generation and LoRA
training*. We do **not** implement a baseline unless (1) an open-source implementation
exists, (2) it is compatible with this threat model, and (3) implementation cost is
reasonable. Citations below are to real, published works; where we are not certain of a
venue we say "as implemented in this repo's baseline suite" instead of asserting one.

| # | Candidate | Paper (real) | Year | Task | Threat model | Protects base weights? | Supports LoRA **training**? | Comparable to us? |
|---|---|---|---|---|---|---|---|---|
| A | **Plaintext LoRA** | Hu et al., *LoRA* | ICLR 2022 | fine-tune + gen | none (no protection) | ✗ | ✅ | ✅ **mandatory reference** (our L5 arm) |
| B | **QLoRA** | Dettmers et al., *QLoRA* | NeurIPS 2023 | efficient fine-tune | none (quantization/efficiency) | ✗ | ✅ | ⚠️ orthogonal goal (memory, not privacy) — N/A as a *privacy* baseline |
| C1 | **DP-SGD / DP fine-tuning** | Abadi et al., *Deep Learning with Differential Privacy* | CCS 2016 | private training | data-privacy via gradient noise | ✗ (no host/weight protection, no TEE) | ✅ | ✗ different threat model (protects *training data* against a trusted trainer, not base weights on an untrusted host) |
| C2 | **DP-LoRA** | DP applied to LoRA (e.g. Yu et al. DP fine-tuning line) | 2022–2023 | private LoRA | membership/record DP | ✗ | ✅ | ✗ different threat model + utility–privacy trade-off axis |
| C3 | **Federated LoRA (e.g. FFA-LoRA)** | Sun et al., *Improving LoRA in Privacy-preserving Federated Learning* | ICLR 2024 | federated fine-tune | multi-party aggregation privacy | ✗ (each client has plaintext weights) | ✅ | ✗ different threat model (cross-client), no untrusted-host base-weight protection |
| D1 | **Offsite-Tuning** | Xiao et al., *Offsite-Tuning* | 2023 | transfer w/o full model | model-owner ↔ data-owner split | partial (shares a compressed emulator) | ✅ (adapter) | ⚠️ related but lossy emulator; no exact-fold / TEE; not our unified-package setting |
| E | **STIP** | secure transformer inference (repo baseline suite) | — | **inference only** | untrusted-server inference | ✓ (inference) | ✗ (`training_support=N/A`, see `baseline_manifest.json`) | ✅ **generation/overhead only** (Phase 5), not LoRA training |
| F | **ObfuscaTune** | *ObfuscaTune* (offsite obfuscated fine-tune/inference) | 2024 | inf. (+ obf. tune) | untrusted-server | partial (obfuscation) | ✗ in our repo integration (`training_support=N/A`) | ✅ **generation/overhead only** (Phase 5) |
| G | **Amulet** | repo baseline suite (nonlinear-island obfuscation) | — | inference | untrusted-server | partial | ✗ (`training_support=N/A`) | ✅ **generation/overhead only** (Phase 5) |

## Key observations
1. **No published method shares our exact LoRA-training threat model** (private base weights
   kept off an untrusted GPU behind a TEE, exact-fold unified package, private adapter +
   gradients). The closest privacy-preserving LoRA works (DP-LoRA, federated LoRA) target a
   *different* asset (training-data privacy / cross-party privacy), not base-weight
   confidentiality against an untrusted execution host.
2. **Inference-only obfuscation baselines** (STIP / ObfuscaTune / Amulet) already exist in
   this repo but have `training_support = N/A` (`baseline_manifest.json`); they are valid for
   the **generation and system-overhead** comparison (Phase 5) but cannot serve as
   LoRA-training utility baselines.
3. **QLoRA** optimizes a different axis (quantized-memory efficiency) with no privacy threat
   model; comparing "privacy utility gap" against it would be a category error.
