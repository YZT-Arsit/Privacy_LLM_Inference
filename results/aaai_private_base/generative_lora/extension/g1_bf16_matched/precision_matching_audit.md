# G1_BF16_MATCHED precision-matching audit

This control matches the **actual executed G2 L12 implementation**, not fields that are only declared in the older frozen-config prose. It is plaintext and uses no transformed package, TDX, masks, trusted correction, or protected serialization.

| Field | G1 FP32 | G1 BF16 Matched | G2 Protected L12 (actual) |
|---|---|---|---|
| Base checkpoint | plaintext Qwen2.5-0.5B | identical plaintext checkpoint, SHA-256 bound | equivalent transformed package derived from same checkpoint |
| Model/runtime dtype | FP32 | BF16 | BF16 |
| Autocast | disabled | disabled | not used |
| LoRA runtime dtype | FP32 | BF16 copy regenerated from master | BF16 copy regenerated from master |
| Authoritative LoRA dtype | FP32 runtime parameter | FP32 master | FP32 master split between GPU and TDX |
| Runtime gradient dtype | FP32 | BF16, then cast to FP32 | BF16 dlogits/backward, accumulated into FP32 master gradients |
| AdamW m/v dtype | FP32 | FP32 | FP32 |
| Optimizer | PyTorch AdamW | explicit G2-formula AdamW | explicit GPU/TDX AdamW |
| Fused | PyTorch default/unfused in frozen run | false | false |
| Betas / epsilon | 0.9 / 0.999 / 1e-8 | identical | identical |
| Weight decay | 0.0 | 0.01, matching executed G2 | 0.01 in `a10_batch_runner.py` |
| LR schedule | 3% warmup + linear decay | constant 2e-4, matching executed G2 | constant 2e-4 |
| Loss scaling | none | none | none |
| Gradient clipping | 1.0 | none, matching executed G2 | no clipping call in active runner |
| Loss definition | HF shifted CLM mean over supervised tokens | identical shifted CLM mean | TDX mean CE over the same supervised next-token rows |
| Initialization | PEFT default | exact plaintext A0/B0 underlying G2 rank-masked initialization | rank-masked U@A0 and B0@U^T |
| Target modules | all seven | all seven | all seven |
| Rank / alpha / dropout | 8 / 16 / 0 | 8 / 16 / 0 | 8 / 16 / 0 |
| Trainable parameters | 4,399,104 | 4,399,104 | 4,399,104 authoritative factor elements |
| Batch / accumulation | 16 / 1 | 16 / 1 | 16 / 1 |
| Data order | frozen seed schedule | identical hash-bound schedule | identical hash-bound schedule |
| Training steps | 750 | 750 | 750 |
| Generation dtype | FP32 | FP32 | FP32 in completed protected generation profiles |
| Decoding | greedy, 96 max-new | identical | identical |

## Explicit mismatches and interpretation

- G1_BF16_MATCHED uses the Hugging Face plaintext batched forward, while G2 uses the transformed per-example runtime. This is the intended protection-path difference.
- CE and labels are local in the plaintext control but reside in TDX for G2. The supervised positions and mean reduction are matched.
- The historical frozen configuration says weight decay 0.0, warmup/linear decay, and clipping 1.0. The completed G2 runner actually used weight decay 0.01, constant learning rate, and no clipping. This control matches the executed G2 implementation and records the discrepancy rather than silently claiming the declarations matched.
- This control isolates precision and much of the execution-path effect, but it is not strict numerical equivalence because HF batched kernels and the transformed runtime use different operator paths.
