# Plaintext protected-asset exposure inventory (private-base threat model)

Repo: `/Users/Hoshino/Desktop/privacy_llm_obfuscation` @ `fix/folded-remote-repeat-debug-and-guards`.
Scope: every place a plaintext PROTECTED asset (base weight, embedding, hidden state, LoRA param/grad, label, logit) is materialized on the UNTRUSTED side today. Ranked by severity. "In GPU process" = runs inside the untrusted CUDA worker process.

Protected assets (must never reach the untrusted GPU in plaintext): base weights, embeddings, hidden states, KV cache, LoRA adapters+gradients, labels/loss, optimizer state, logits.

---

## VERDICT: FAIL

At least one private-base run loads a full plaintext checkpoint inside the untrusted GPU worker and executes the entire plaintext base forward there. Details S1-S4 below.

---

## S1 — CRITICAL — gate3 masked-LoRA TRAINING loads a plaintext checkpoint on the untrusted GPU and runs the entire plaintext base forward there

- Site: `scripts/gate3_worker.py:147` `load_model()` -> `AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)` then `model.to("cuda").eval()` (`:150-151`).
- Forward: `scripts/gate3_worker.py:218` `out = model(input_ids=input_ids, attention_mask=attn)` and `:219` `logits = out.logits.float()`.
- In GPU process: YES. This is THE untrusted H800 training worker (docstring `:1-33`).
- Plaintext assets materialized on the untrusted GPU:
  - base weights (all q/k/v/o/gate/up/down + norms + embed + lm_head) — from_pretrained onto CUDA;
  - embeddings — computed inside `model(input_ids)` (`:218`);
  - every hidden state / residual / RMSNorm / attention / SwiGLU output — plaintext;
  - full plaintext logits — `out.logits.float()` (`:219`) BEFORE any vocab mask;
  - input_ids and the label source (`:207`, `:160-161`) are on CUDA (labels themselves held in TDX for CE, but the token ids are co-resident).
- The docstring ADMITS this: "Base weights run in plaintext here (LoRA-training correctness milestone). The final private-base threat model (folded base) is NOT claimed in this mode." (`gate3_worker.py:31-33`).
- Only the LoRA adapters (A_t,B_t) and the logit permutation are masked. Masking plaintext logits AFTER they already exist on the GPU (`:233-240`) does not un-expose them.
- Same exposure in the multi-step TDX driver: `scripts/gate3_tdx_step.py:75` `load_model(...)`, `:87` and `:102` `model(...)`.

## S2 — CRITICAL — the default in-process generation worker (Qwen7BGpuBackend / MaskedQwenSession) loads a plaintext checkpoint on the untrusted GPU

- Site: `src/pllo/protocol/gpu_worker.py:205` `AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dt, device_map=self.device, ...)` inside `_ensure_session()` of `Qwen7BGpuBackend` (`:150`, `tee_used=False`, `:166`).
- The session then folds weights ON THE FLY from the resident plaintext model: `src/pllo/hf_wrappers/qwen_masked_session.py:163` `_folded_layer()` -> `extract_hf_single_block_weights(self.base.layers[ell], ...)` (`:165`) and `:82` `_extract_boundary(model, ...)` reads the embedding + lm_head.
- In GPU process: YES. Although `MaskedQwenSession` labels its extract/fold steps "trusted setup", they execute in the SAME untrusted worker process that serves `worker_prefill`/`worker_decode`. The plaintext base model object `self.base.layers` is co-resident on the untrusted GPU for the session lifetime.
- Plaintext assets on the untrusted GPU: base weights, embedding table, lm_head, final norm — all held by `self._model`.
- This path is what `Qwen7BGpuBackend` uses whenever a `model_path` (not a folded package) is given.

## S3 — MEDIUM — trusted-setup folded-package builder loads a plaintext checkpoint on a CUDA device

- Site: `scripts/build_qwen7b_folded_package.py:75` `AutoModelForCausalLM.from_pretrained(...)`, default `--device cuda` (`:107`), then `MaskedQwenSession(...)` (`:251`) and `export_folded_layer_tensors` (`:304`).
- In UNTRUSTED GPU process: NO by intent — this is the TDX/trusted-setup phase (docstring `:3`, `created_by=tdx_trusted_setup|trusted_setup`). It legitimately reads plaintext weights to PRODUCE the folded package.
- Severity is MEDIUM (not FAIL) only IF this actually runs inside the attested trusted domain. If a benchmark ever runs this builder on the same untrusted H800 that later serves decode, the plaintext base is exposed. There is no code-level guard binding this builder to TDX; it defaults to `cuda`.

## S4 — BY DESIGN (recorded, not a bug) — the untrusted GPU sees the exact attention SCORE / softmax matrix

- Site: `src/pllo/hf_wrappers/llama_qwen_single_block.py:310` `scores = qr @ kr_rep.transpose(-2,-1) * scale`; softmax `:318-319`.
- The compatible Q/K masks satisfy `Q_tilde K_tilde^T == Q_rope K_rope^T` exactly, so the score matrix and the post-softmax attention weights on the untrusted GPU are IDENTICAL to plaintext. This is the documented A_rightmul attention-fingerprint leak (MEMORY: attention-fingerprint-attack-stage, 98.5% token recovery on Qwen7B).
- Not a hidden-state materialization: the residual hidden stays masked; only the attention probability geometry is revealed. It is a knowingly-accepted property of the frozen convention, mitigated (out of scope here) by layer-0 TEE relocation. Recorded, not counted as a S1/S2-class violation.

---

## What is NOT exposed (the compliant folded-inference path — for contrast)

- `src/pllo/deployment/folded_worker.py` (`apply_folded_prefill:280`, `apply_folded_decode:319`, `apply_folded_head:358`) and the `Qwen7BFoldedPackageGpuBackend` (`gpu_worker.py:318`) load ONLY pre-folded `*_tilde` shards (`load_folded_layer:91`, `load_shard`), never a plaintext checkpoint and never a mask.
- The package writer screens every tensor name against `FORBIDDEN_PACKAGE_SUBSTRINGS` (`folded_package.py:42`) — "mask","perm","n_res","label","grad","optim","scale","delta_w",... — and refuses to write them (`save_shard:93`). The manifest forces `contains_mask_secrets/plaintext_inputs/raw_lora/optimizer_state == False` (`folded_package_manifest.py:73-76`, validated `:213`).
- Mask secrets (embed table, N_0, vocab mask) live only in the TDX-only `embedding_artifact` (`embedding_artifact.py:17-20`: "NEVER sent to the GPU worker").
- The CPU/fp64 training CONTRACT `unified_masked_training.py` proves a masked-base training forward + masked-LoRA backward with `plaintext_hidden_materializations == 0` (`:378-381`) and `nonlinear_trusted_calls == 0` (`:441`) — but it is explicitly NOT a real-model or GPU result (`:469-471`), and it materializes the base weights `w.Wq...` in the same process (synthetic contract, not a deployment split).

## Severity summary

| Sev | Finding | Site | In untrusted GPU proc | Assets |
|---|---|---|---|---|
| CRITICAL | plaintext base+embed+hidden+logits in training | gate3_worker.py:150/:218/:219 | YES | base W, embed, hidden, logits, input_ids |
| CRITICAL | plaintext base loaded by in-proc gen worker | gpu_worker.py:205 + qwen_masked_session.py:165 | YES | base W, embed, lm_head |
| MEDIUM | trusted builder loads plaintext on cuda | build_qwen7b_folded_package.py:75/:107 | NO (if truly in TDX) | base W (to fold) |
| BY-DESIGN | exact attention score/probs visible | llama_qwen_single_block.py:310 | YES | attention geometry (residual stays masked) |
