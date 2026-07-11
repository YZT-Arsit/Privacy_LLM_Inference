# Gate 3 — limitations (checkpoint)

1. **Base weights plaintext on GPU.** This run protects the LoRA adapters + optimizer
   (masked-domain SGD) and, in the next milestone, the loss/labels (TDX). The frozen
   base weights execute in plaintext HF Qwen on the GPU. The final private-base threat
   model is NOT claimed here; base folding is the separately-validated inference layer.
2. **Local CE, not TDX CE, in the current exactness run.** `protected_localce` is a
   masking-correctness de-risk only. The real-TDX private-CE boundary is pending.
3. **Logit mask for real vocab is a permutation (planned).** vocab=151936 makes a dense
   head mask infeasible (~92 GB); the real-TDX path will use an O(vocab) permutation head
   mask (hides token↔logit identity + private labels held in TDX), which is weaker than a
   dense mask and leaks the logit-value multiset. To be stated in the final claims.
4. **Training input visible to GPU.** In this LoRA-loss profile the GPU tokenises/embeds
   the training text, so the input is not hidden from the GPU; only the loss/label
   boundary is trusted. Full input protection is the separate staged-obfuscation path.
5. **bf16 numerics.** Residual ΔW/logit errors (<2%) are bf16 rounding, not masking error
   (confirmed by loss_abs_diff=0 and gradA_norm_rel=0). fp32 diagnostics available.
6. Single/10/50-step, momentum, and the negative matrix on the REAL-TDX loop are pending.
