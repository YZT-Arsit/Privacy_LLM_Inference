# Phase 2 — Qwen2.5-7B LoRA scale validation (feasibility + scoped evidence)

Goal (per brief): *prove scalability, not benchmark SOTA*. Execution constraint #7: if an
experiment cannot be run, **explain why** instead of fabricating it. Constraint #2: reuse the
existing 7B generation experiment; do not redo generation correctness.

## Feasibility determination (honest)
A full 7B protected-LoRA fine-tune + GSM8K eval is **not runnable in this environment**:
- **No Qwen2.5-7B checkpoint is present** locally (`privacy_llm_data/checkpoints/` holds only
  0.5B) or on the reachable A10 (`/root` holds only `qwen25_05b`, 954 MB). The H800 host is
  offline. A 7B run needs the 7B checkpoint provisioned, a trusted-packager build of the 7B
  private package (offline, hours), and secure transfer.
- Building/transferring the 7B package and training on it is a multi-hour hardware pipeline
  outside this session's reach; we do **not** fabricate numbers for it.

## What IS already established at 7B scale (reuse, frozen)
Scalability of the *transform* — the hard part — is already demonstrated at 7B and is reused,
not re-run:
- **7B generation / fold correctness**: `results/baselines/conjformer/qwen7b_generate_fp32*.json`
  (real Qwen2.5-7B fp32 generation), i.e. the exact-fold masked forward operates at 7B
  (D=3584, 28 layers, FFN=18944).
- **7B security surface**: real-7B attacks exist — `results/attacks/ours_gram/qwen7b_gram_attack*.json`,
  `results/attacks/real_attention_fingerprint_qwen7b.json`, and the fresh-signed-perm matrix
  (`final_security_efficiency/measured_inputs/qwen7b_*`). The S1–S6 leakage geometry (orthogonal/
  permutation masks) is dimension-agnostic and was measured at 7B in these artifacts.
- **7B inference overhead**: `results/baselines/obfuscatune_latency_h800.json` is measured at
  7B dims (D=3584, 28 layers) — ours = 2 TEE crossings, ≤1.07× GPU-TEE slowdown (Phase 5).

## Why 0.5B + frozen-7B jointly cover the scalability claim
The protected-LoRA design is **scale-invariant by construction**: the fold, the per-layer
feature/nonlinear/rank masks, and the trusted A/B split are defined per operator and do not
depend on width/depth. The two axes of "protected LoRA works at 7B" are:
1. **Optimizer exactness under masking** — validated exhaustively at 0.5B (L0–L12: SGD /
   momentum / AdamW exact, top-1 = 1.0, trust-domain counters 0; `full_lora_matrix/**`,
   `gate0_d4/**`). This is model-size-independent (it is a per-factor algebraic identity).
2. **Exact fold + masked forward at 7B** — validated by the frozen 7B generation experiment.
Their composition is exactly "protected LoRA at 7B"; no new algebra appears at 7B.

## Cost projection for a 7B protected-LoRA step (projection, NOT measured — labeled as such)
Scaling the measured 0.5B profile (batch 16): the fixed per-step TDX AdamW (un-fold/AdamW/
re-fold on a CPU-only enclave) scales ~linearly with trusted-factor dimensions (hidden 896→3584
≈ 4×, layers 24→28 ≈ 1.17×) ⇒ ~**0.5–0.8 s/opt-step TDX compute** (vs 0.114 s at 0.5B); GPU
forward/backward scales with the ~14× parameter count and would require a ≥30 GB-class GPU for
batch-16 seq-512 training (A10's 23 GB is borderline; H800/A100 preferred). Trusted round trips
stay **2 per opt step** (scale-invariant) — the key efficiency property. *These are projections
from the 0.5B measurement, explicitly not measured at 7B.*

## Requirements to actually run it (deferred, concrete)
1. Provision Qwen2.5-7B checkpoint on a GPU host (A10 tight; H800/A100 preferred).
2. Trusted-packager build of the 7B private package (fp64-validated fold) + transfer.
3. Protected LoRA one-step gate → short fine-tune (1000 GSM8K train) → GSM8K subset (100 test).
4. Compare plaintext vs protected: loss trajectory, EM, latency, VRAM, trusted-invocation count,
   communication.

## Verdict
7B protected-LoRA *training* is **deferred (environment-limited, not design-limited)**. The
scalability of the transform is supported by the frozen 7B generation/fold + security artifacts
plus the size-independent 0.5B optimizer-exactness validation. Full 7B LoRA fine-tune numbers
are listed as an **unsupported/deferred** item in the gap report.
