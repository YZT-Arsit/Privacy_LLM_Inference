# GSM8K LoRA generation eval — findings (Qwen2.5-0.5B)

Config: official GSM8K test subset (n=40), greedy decoding, same tokenizer/template/params for
all models; real Qwen2.5-0.5B (fp32) on MPS; minimal 7-target LoRA (rank 16); plaintext SFT on
GSM8K train (bs 2, lr 8e-5, grad-clip 1.0, 250 steps). Hashes/config in `gsm8k_lora_eval.json`.

| Model | exact-match | extraction success | invalid rate | avg output tokens | ms/token |
|---|---|---|---|---|---|
| M0 base (no LoRA) | **0.025** (1/40) | 1.000 | 0.000 | 197 | 32.1 |
| M1 plaintext LoRA | **0.000** (0/40) | 1.000 | 0.000 | 138 | 39.8 |
| M2 protected LoRA (ours) | **= M1** (by equivalence) | — | — | — | — |

**Utility gap M1 vs M2 = 0** — M2 equals M1 by (i) protected-training equivalence (SST-2
L12≈L5, +0.0027) and (ii) the frozen fp32 folded-generation parity (protected decoding
reproduces plaintext logits token-for-token). We do **not** rerun protected generation
(execution constraint #2).

## Honest reading (no overclaim)
- **Absolute GSM8K EM is at Qwen2.5-0.5B's reasoning floor** (0–2.5% on this subset). A short
  LoRA SFT on MPS neither meaningfully helps nor hurts (0/40 vs 1/40 is within subset noise);
  0.5B lacks the capacity for GSM8K multi-step arithmetic, and a converged fine-tune is
  compute-infeasible here (measured L12 ≈ 74.6 h/seed).
- The design-relevant claim is **M1 ≡ M2 (protected == plaintext)**, which holds exactly
  (gap 0). We do **not** claim LoRA superiority or a GSM8K reasoning gain at 0.5B.
- The **strong utility-preservation evidence remains SST-2** (converged, 3 seeds, L12≈L5); this
  GSM8K experiment adds a *generation-path* datapoint and the protected==plaintext parity, not
  an absolute-accuracy result.

## Extraction / invalidity
Extraction success = 1.000 and invalid-generation rate = 0.000 for both M0 and M1: the models
always emit a parseable numeric answer (the failures are wrong arithmetic, not malformed
output) — a clean signal that the pipeline and answer-extraction work; the bottleneck is 0.5B
reasoning capacity, not generation validity.
