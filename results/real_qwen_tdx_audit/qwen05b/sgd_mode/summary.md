# Gate 3 — real Qwen2.5-0.5B masked LoRA SGD with real TDX private loss

**Single-step: PASS.** 10-step / 50-step: see status below. Flags: uses_real_gpu=true,
uses_real_tee=true, uses_real_qwen=true. Base weights plaintext on GPU (final private-base
threat model NOT claimed — see claims.md / limitations.md).

## Architecture (frozen)
External verifier (Mac) — control only: fresh nonce, DCAP appraisal + policy + runtime
hash + report_data verification, authorization. Data plane: **direct H800 <-> TDX** single
SSH tunnel (TDX-initiated reverse forward; no TDX key on the untrusted GPU; Mac never
relays tensors). Attestation-bound X25519/HKDF/ChaCha20-Poly1305 AEAD, safe codec.

## Answers to the required questions
1. **Real Qwen2.5-0.5B?** Yes — 494,032,768 params, bf16, verified (model_manifest.json).
2. **Real H800 forward/backward?** Yes — real fwd + `loss.backward()` on the H800, 96
   LoRA modules (q/k/v/o_proj, 24 layers).
3. **Real TDX loss?** Yes — private CE computed inside the real TDX guest over the direct
   attested AEAD session (vocab-permutation mask, fp32 CE).
4. **Quote regenerated + verified?** Yes — fresh quote, fresh single-use nonce, no Gate-2
   reuse; Mac verified: attestation_verified=True, DCAP appraisal=SUCCESS, ES384 sig
   verified, DEBUG=false, report_data bound over all fields.
5. **GPU direct masked-domain SGD?** Yes — GPU updates masked A_t/B_t in place; no unmask.
6. **packed_update never called?** Yes — packed_update=0 (loss service got an empty LoRA
   manifest; it holds no adapter state).
7. **TDX holds no adapter/optimizer?** Yes — TDX holds only the vocab permutation + private
   labels; trusted_optimizer=0, layer_wise=0.
8. **2 invocations/step?** Yes — input/init boundary=1 (authorized init: labels + pi),
   loss boundary=1; total=2.
9. **Single vs plaintext SGD diff?** loss_abs_diff=2.2e-5, next-logits top1=1.0,
   min ΔW cosine=0.9994, min gradB cosine=0.9993, all finite. PASS.
10. **10-step aligned?** Yes — protected loss 2.0582->1.7166 tracks plaintext 2.0582->1.7226;
    max per-step loss abs diff 0.0178, final ΔW min cosine 0.9984, next-logits top1=1.0, all
    protected steps finite (ten_step_results.json / ten_step_trajectory.csv).
11. **50-step stable?** <PENDING_50STEP>
12. **bf16 error accumulation?** <PENDING — single-step residual ~3.6% ΔW is consistent
    with bf16/order-of-ops (fp32 control not yet run); multi-step growth reported below.>
13. **Largest-error module?** see single_step_per_layer.csv (uniform ~bf16 across layers).
14. **Performance bottleneck?** the shared-gateway SSH tunnel: ~0.5 MB/s => ~80 s per
    38.6 MB (bf16) logit RPC. Not a system-perf claim; 0.5B = feasibility/correctness.
15. **Proceed to 7B?** Not this stage — hold for user decision (per instruction).

## Single-step metrics (real TDX loop)
- loss (protected/plaintext): 2.0582 / 2.0582; abs diff 2.2e-5
- next-logits: rel 1.8% (bf16), top1 agreement 1.0
- ΔW: max rel 3.6% (bf16), min cosine 0.9994
- gradB: max rel 3.6% (bf16), min cosine 0.9993; gradA=0 at step 0 (B=0 init; degenerate)
- invocations/step: total 2 (input/init 1 + loss 1); packed_update 0; trusted_optimizer 0
- wire: 38.6 MB bf16 each way; RPC latency: init ~1.5 s, logits_loss ~80 s
- attestation: verified=True, reused_gate2=False, DEBUG=false, appraisal=SUCCESS, ES384 sig

## Multi-step + fail-closed status
- 10-step (real TDX): PASS. loss diff <=0.0178, ΔW cos 0.9984, top1 1.0, all finite
  (ten_step_results.json).
- Fail-closed matrix (live service): 5/5 PASS (negative_tests.csv) — wrong run_id, wrong
  optimizer_mode, wrong config_digest, compute-without-session, and wrong runtime-hash
  (external Mac verifier refuses on report_data/runtime mismatch) all rejected.
- 50-step (real TDX): <PENDING_50STEP>

## Gate-3 PASS status
Single-step meets all 12 PASS criteria (single_step_results.json); 10-step trajectory
aligned; fail-closed 5/5. Per the Gate-3 rules, overall Gate 3 is marked PASS only when the
50-step run completes without divergence/NaN; until then it is **single-step + 10-step PASS,
fail-closed PASS / 50-step in progress**.
