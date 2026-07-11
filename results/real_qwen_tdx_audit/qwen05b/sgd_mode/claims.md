# Gate 3 — claims status (checkpoint, NOT a PASS)

Run context: real H800 (autodl-container-35eb4cbcf2-9d84e25e), real Qwen2.5-0.5B
(494,032,768 params, bf16, verified — see model_manifest.json), profile gpu_masked_sgd.

## VALIDATED (real, on real Qwen2.5-0.5B / real H800)
- **Real model + real GPU forward/backward.** 494M-param pretrained Qwen2.5-0.5B loaded
  in bf16 on the H800; real forward + real `loss.backward()` through 96 LoRA-wrapped
  attention modules (q/k/v/o_proj across all 24 layers).
- **Genuine masked-domain SGD (not plaintext SGD relabelled).** The GPU holds ONLY the
  masked adapters A_t=U·A, B_t=B·Uᵀ (fixed orthogonal rank-space mask U per layer) and
  the optimizer updates those masked leaves directly. Never materialises plaintext A/B.
- **Exactness vs plaintext SGD** (protected local-CE vs plaintext reference, one step):
  loss_abs_diff = **0.0**; max gradA Frobenius-norm rel error = **0.0** (orthogonal masks
  preserve ‖·‖_F exactly); max ΔW rel error = 0.0077 (bf16); next-logits rel error =
  0.0177 (bf16); top1 agreement = 0.996; all finite.
- **No trusted optimizer / no packed_update:** packed_update_calls = 0, trusted_optimizer
  _calls = 0 in the protected path.
- Real 64-sample zh/en dataset (32+32), train/val split, fixed order (gate3_data.json).

## NOT YET DONE (so Gate 3 is NOT PASS)
- **Real TDX private-CE loss boundary NOT yet in the loop.** The exactness above used a
  LOCAL cross-entropy on the GPU as a masking-correctness de-risk (clearly labelled mode
  `protected_localce`). The real-TDX closed loop (masked logits -> TDX CE -> masked
  dlogits, with a freshly re-generated + externally verified quote and a new attested
  AEAD session) is the next milestone and has NOT been run.
- No single/10/50-step REAL-TDX run; no momentum run; no negative-matrix on the real run.

## NOT CLAIMED (scope / threat model)
- **Base weights run in plaintext on the GPU here.** This milestone validates the LoRA
  masked-training loop correctness, NOT the final private-base-weight threat model. Folding
  the base (the separately-validated inference layer) is required for that and is not
  applied in this run. Per the Gate-3 rules, the final threat model is therefore NOT claimed.
- 7B is out of scope this stage.
