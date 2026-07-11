# Gate 3 — claims status

Run context: real H800 (autodl-container-35eb4cbcf2-9d84e25e), real Qwen2.5-0.5B
(494,032,768 params, bf16, verified — model_manifest.json), real Intel TDX guest
(39.96.4.252). Profile gpu_masked_sgd. Direct H800<->TDX data plane; Mac = control only.

## VALIDATED (real hardware, real model)
- **Real Qwen2.5-0.5B forward/backward on a real H800.** 494M-param pretrained model in
  bf16; real forward + real `loss.backward()` through 96 LoRA-wrapped attention modules
  (q/k/v/o_proj, all 24 layers).
- **Real fresh TDX attestation (no Gate-2 reuse).** A NEW TD Quote was generated for
  run_id=gate3_qwen05b_sgd, model_id=Qwen2.5-0.5B, optimizer_mode=gpu_masked_sgd,
  gradient_convention=nout_dual, logits_mask_family=vocab_permutation, guest ECDH pubkey,
  and a fresh single-use verifier nonce. The Mac (external verifier) cryptographically
  verified the signed DCAP appraisal (overall_appraisal_result=1, relying_party -v ES384
  signature), measurement policy (DEBUG=false), runtime-hash binding, and report_data
  over every bound field: `attestation_verified=True, reused_gate2=False`.
- **Real TDX private cross-entropy over a direct, attested, AEAD session.** vocab=151936
  O(vocab) PERMUTATION logit mask (no dense head): the H800 sends bf16 masked logits
  Z_tilde=Z[:,pi] directly to TDX; TDX recovers, computes fp32 CE with the private labels
  it holds, returns masked dlogits G_tilde=G[:,pi]; the H800 un-permutes. No dense head
  mask, permutation bijection+inverse verified, ignore-index/causal-shift handled. The
  permutation-CE parity vs plaintext was verified exactly offline (loss diff 0, dlogits 0).
- **GPU-only masked-domain SGD; TDX holds NO adapter/optimizer state.** The GPU updates
  the masked adapters A_t/B_t in place (rank-orthogonal mask); it never materialises
  plaintext A/B; **packed_update=0, trusted_optimizer=0, layer_wise=0**; the loss service
  received an EMPTY LoRA manifest.
- **Single-step PASS (all 12 criteria)** vs an independent plaintext SGD reference (same
  data/seed/init, bf16): loss_abs_diff=2.2e-5; next-logits top1_agreement=1.0;
  min ΔW cosine=0.9994; min gradB cosine=0.9993; no NaN/Inf; no per-layer explosion
  (max 3.6%); **2 trusted invocations** (input/init boundary=1, loss boundary=1).
- Direct connectivity proven: H800->TDX RTT ~23.6 ms; single H800<->TDX SSH tunnel
  (TDX-initiated reverse forward); Mac never relayed tensors (network_topology.md).

## CLAIM DISCIPLINE (corrected wording)
- gradA Frobenius-norm equality proves ORTHOGONAL NORM PRESERVATION, not complete
  gradient equality. (At step 0, gradA==0 because LoRA B=0 init, so its cosine is
  degenerate; gradB carries the signal and matches at cosine 0.9993.)
- The ~3.6% ΔW/gradB and ~1.8% next-logits residuals are **consistent with bf16 /
  order-of-operations effects** — NOT yet proven purely bf16 (fp32/reorder control not
  run). The near-zero loss diff (2.2e-5) and top1=1.0 indicate they are numerical, not a
  masking error, but the stronger claim awaits the fp32 control.
- Current implementation validates **rank-masked LoRA SGD** with a real TDX private-loss
  boundary. It does NOT validate the final private-BASE-weight threat model.

## NOT CLAIMED (scope / threat model)
- **Base weights run in plaintext on the GPU** in this profile. The final private-base
  threat model is therefore NOT claimed; base folding (the separately-validated inference
  layer) is required for that and is not applied here. Explicit limitation.
- The permutation logit mask hides token<->logit identity + keeps labels in TDX, but
  leaks the logit-value multiset; with a plaintext base (GPU holds true Z + pi) it is
  protocol-structural, not adding confidentiality beyond the AEAD. It becomes meaningful
  only with a folded LM head.
- Training input (text) is visible to the GPU (it tokenises/embeds); only the loss/label
  boundary is trusted here.
- 10-step / 50-step / momentum status: see summary.md (single-step is the gating result).
- 7B is out of scope this stage.
