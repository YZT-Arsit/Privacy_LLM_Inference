# Minimum paper-valid first run

The first paper-facing run, only after oracle and real-runtime gates pass, is:

- label: `faithful ObfuscaTune-style Qwen LoRA adaptation (transformed-coordinate optimizer)`;
- Qwen2.5-0.5B and the frozen E2E NLG data/prompt/tokenizer;
- q/k/v/o/gate/up/down targets, rank 8, alpha 16, dropout 0;
- seed 1234, 750 steps, 500 direct generations;
- external transformed-coordinate FP32 AdamW state with LR 2e-4, betas
  (.9,.999), epsilon 1e-8, weight decay .01, constant LR and no clipping;
- real A10 plus Intel TDX adaptation with baseline-specific runtime ID;
- utility plus measured messages, bytes, trusted/GPU state, compute/transport
  time, step latency, peak GPU memory and generation speed.

This scope deliberately defers seeds 7/2025, rekeying, a large fail-closed
matrix, canonical/plaintext adapter export, and full three-seed statistics.

## Required caveats

- AdamW trajectory is not plaintext-coordinate equivalent.
- Qwen and Intel TDX are adaptations beyond the paper's GPT-2/two-GPU setup.
- Plaintext Q/K/V and KV cache are exposed under this baseline.
- The paper does not define adapter export/handoff; the first run may reuse the
  same transformed domain but cannot claim a general protected adapter package.
- No comparison should silently treat this threat model as identical to G2.

Current readiness: `OBFUSCATUNE_IMPLEMENTATION_PARTIAL`; real execution remains
prohibited until the minimal oracle and later Qwen/runtime gates pass.
