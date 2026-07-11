# Gate 3.5 — waiting-period report (training path → paper-safe masked execution)

Produced while the Gate-3 50-step rank-masked-SGD run holds the H800 (undisturbed).
Everything below is **local/CPU integration work**: code, unit tests, static audit,
and synthetic small-dimension correctness. **No result here is a real Qwen or GPU
result.** Gate 3 is NOT claimed full-PASS; private-base training is NOT claimed done.

## 1. Current 50-step run health (read-only)
- Process alive on H800 (`gate3_tdx_step.py --phase run`, PID 5048), state `Ssl`
  (sleeping on the TDX loss RPC — expected; GPU compute is ~0.05 s/step, ~79 s/step
  is the throttled tunnel).
- Model resident (~4.0 GB), GPU otherwise idle between RPCs. No NaN/Inf observed in
  the streamed steps so far; loss finite. No interference performed (no kill/restart/
  attach/config-edit). Armed monitor will fire on completion or any failure signature.
- Freeze/archive of the run happens only after it completes (its own run_id/config/
  model-hash/final-loss/ΔW/top1/invocation-count go into the Gate-3 bundle; base +
  activations remain plaintext, stated in limitations).

## 2. execution_profile — default footgun removed (spec B)
`src/pllo/experiments/execution_profile.py`. Three explicit, fail-closed profiles:

| profile | backend | masked inter-block | trusted nl calls | plaintext-hidden | paper_facing | claim |
|---|---|---|---|---|---|---|
| **paper_safe** (default) | A_rightmul / amulet_secure_R; `current`/`trusted_shortcut` **forbidden** | required | cap 0 | cap 0 | true | allowed iff all gates pass |
| legacy_debug | `current` allowed | not required | unbounded (recorded) | recorded | false | **never** |
| experimental | opt-in | not required | recorded | recorded | false | never by default |

- A **missing** profile resolves to `paper_safe`; a **present-but-unknown** profile is
  fail-closed (never silently treated as default).
- A requested-but-forbidden backend is fail-closed, **never silently replaced** by the
  default. Zero-cap counters that are *missing* from a run are themselves violations
  (a "0" that was never reported cannot be certified).
- `profile_manifest_fields()` emits profile + backend + inter-block mode + gate
  pass/fail + `security_claim_allowed` into the manifest; `execution_profile_metadata_hash()`
  binds the profile into config_digest / runtime hash / report_data.
- Tests: `tests/test_execution_profile.py` — **16/16**, covering the full spec-B matrix
  (paper_safe+current → reject; +trusted_shortcut → reject; +plaintext-materialization
  → reject; legacy_debug runnable but claim=false; missing profile → paper_safe).

## 3. Training-path alignment matrix (spec C)
`matrix.json`, `matrix.md`, `missing_integration.md` (this directory).
- **Gate-3 training forward = 2/9 masked probes.** It runs the full **plaintext HF
  forward** (`gate3_worker.py:218`); the only obfuscation is a rank-space LoRA mask that
  **cancels in the forward** (`B_t@A_t==B@A`). It calls **none** of A_rightmul /
  masked-RMSNorm / post-RoPE masking / GQA-masked-cache / SwiGLU-island; inter-block
  hidden is plaintext; backward is masked in **rank space only**.
- **Paper A_rightmul inference path** implements all 10 masked operators, but over
  **detached constant** folded weights → not differentiable w.r.t. any trainable param.
- **Top gap (load-bearing): LoRA-aware folded weight.** `fold_hf_single_block_weights`
  folds only frozen base weights; `_maybe_merge_lora` merges only a *frozen* package.
  No trainable-LoRA fold exists in the repo.

## 4. Unified masked-training CPU contract (spec D+E) — CLOSES EXACTLY
`src/pllo/experiments/unified_masked_training.py` (+ shared kernels
`src/pllo/ops/masked_training_kernels.py`). One tiny synthetic Qwen-shaped decoder
block runs the full pipeline: masked input → masked RMSNorm-core → Q/K/V (LoRA-folded)
→ post-RoPE **pair-permutation** masking (commutes with RoPE) → GQA attention → masked
residual → RMSNorm-core → shared-perm SwiGLU island → masked residual → masked LM head
+ O(V) monomial mask → trusted-loss mock boundary → masked dlogits → **full autograd
backward** → rank-masked LoRA grads → masked SGD. All 12 contract checks pass at fp64
machine epsilon:

- forward recovered output: **1.3e-15**; next-step logits: **8.9e-16**
- per-layer mask invariant (masked residual == plaintext@N): **3e-16**; residual
  genuinely masked (gap ≈ 1.0–1.7, i.e. **not** plaintext); **0** plaintext-hidden
  materializations
- private CE == plaintext CE: diff **0.0**
- gradA/gradB recovered (`Uᵀ·gradA_t==gradA`, `gradB_t·U==gradB`): **1.9e-16 / 1.8e-16**
- masked SGD update == plaintext SGD: **1.1e-16**
- **nonlinear_trusted_calls=0, trusted_nonlinear_ops=0, packed_update=0,
  trusted_optimizer=0**; the 5 nonlinear islands (3×rmsnorm, softmax, silu) ran
  on-accelerator; exactly **1** trusted crossing (the private CE).
- Observed facts satisfy **paper_safe** (`security_claim_allowed==True` for the contract).
- Tests: `tests/test_unified_masked_training.py` — **11/11**, incl. GQA nkv∈{1,2,4} and
  seeds {1,7,99}.

**Shared kernels (spec E):** the nonlinear islands + RoPE + GQA + masks live in one
`masked_training_kernels.py` used by the contract; the *approach* is proven, so the
real integration extracts the same kernels for the inference/training shared path
rather than forking a parallel training copy. The autograd design matches the C audit:
base weight stays frozen/detached, the **LoRA delta is folded into each `*_tilde` with
grad intact** (`_eff_WT` = `Wᵀ + scaling·A_tᵀB_tᵀ`), and the private boundary is a
surrogate `(Z_tilde·G_tilde).sum()` cut (labels never leave the boundary).

## 5. Autograd integration gaps to remove in the real path (spec E, from C audit)
1. Lift the base-weight detaches in `_extract_linear` (`llama_qwen_single_block.py:231/234/251/253`)
   **for the LoRA path only** — keep the frozen base detached; route the trainable delta
   with grad. 2. Make each `*_tilde` fold a differentiable function of `A_t,B_t` (the crux).
   3. Hold `N_res` / per-head Q/K/V / SwiGLU-perm / rank-`U` as one **compatible** mask
   bundle of constant buffers for the whole step. 4. **Preserve** the two correct cuts
   (`gate3_worker.py:234` TDX detach with surrogate loss; `:266` post-step no_grad
   diagnostic). 5. KV cache confirmed training-irrelevant (single full-seq prefill,
   `causal_offset=0`) → excluded.

## 6. Monomial O(V) logit-mask contract (spec G)
`src/pllo/masks/monomial_logit_mask.py`. `Z_tilde = Z·D·Pi` with nonzero `D` (bounded
condition number, no extreme bf16 scales), exact inverse, CE recovered at the trusted
boundary, dlogits remapped, `ignore_index`/causal-shift/padding handled;
**permutation-only** baseline (`D==1`) included; **dense vocab mask intentionally NOT
implemented**. Tests: `tests/test_monomial_logit_mask.py` — **10/10**: loss correctness,
dlogits == autograd, exact inverse (O(V) == dense), bf16-range + fp32 diagnostic, and
the leakage proxy (permutation-only **leaks** the logit value-multiset; the monomial
**hides** it). Frozen default for the first unified run: **monomial** (permutation =
baseline).

## 7. Paper-safe unified TDX config schema (spec F) — prepared, NOT started
`src/pllo/experiments/unified_training_config.py`. Frozen fields — `model_id`,
`model_sha256` (real-checkpoint placeholder), `execution_profile=paper_safe`,
`optimizer_mode=gpu_masked_sgd`, `nonlinear_backend=A_rightmul`,
`logits_mask_family=monomial`, `gradient_convention`, `masked_inter_block=true`,
`plaintext_hidden_materializations_expected=0` — with `config_digest()`,
`runtime_hash_fields()`, `report_data_fields()`, `manifest_fields()`, and a
fail-closed `validate_observed()` (backend/mask-family mismatch rejected). Carries
`requires_fresh_attestation=true`: **no reuse** of Gate-2/Gate-3 quote/nonce/ECDH/
session — a fresh quote is taken only when the real unified run starts. Tests:
`tests/test_unified_training_config.py` — **12/12**.

## 8. Files + tests
**New source:** `src/pllo/experiments/execution_profile.py`,
`src/pllo/experiments/unified_masked_training.py`,
`src/pllo/experiments/unified_training_config.py`,
`src/pllo/ops/masked_training_kernels.py`,
`src/pllo/masks/monomial_logit_mask.py`.
**New tests (49 total, all green):** `tests/test_execution_profile.py` (16),
`tests/test_monomial_logit_mask.py` (10), `tests/test_unified_masked_training.py` (11),
`tests/test_unified_training_config.py` (12).
**New reports:** this directory — `matrix.{json,md}`, `missing_integration.md`,
`unified_masked_training_cpu_contract.json`, this report.
**No existing source modified; nothing committed; the 50-step run untouched.**

## What is explicitly NOT claimed
- Not Gate-3 full PASS (waits on the 50-step completing without divergence/NaN).
- Not unified masked training on a real model/GPU (this is a CPU/fp64 contract).
- Not private-base training (Gate-3's base weights + activations are plaintext).
- Attention *scores* are revealed by the A_rightmul design (documented fingerprint
  leak) — the contract asserts no plaintext *residual hidden state*, not hidden scores.
