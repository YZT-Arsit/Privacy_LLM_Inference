# Missing integration to reach the private-base unified forward

Repo: `/Users/Hoshino/Desktop/privacy_llm_obfuscation` @ `fix/folded-remote-repeat-debug-and-guards`.
Goal: a single frozen-convention forward where the untrusted GPU sees ONLY masked tensors, in BOTH training and generation. Below: (A) what loads plaintext on the GPU and must move behind the trusted packager; (B) what already runs masked and is reusable as-is; (C) what the training path lacks vs the frozen convention.

---

## A. Modules that load plaintext on the untrusted GPU and MUST move behind the trusted packager

1. **gate3 training worker** — `scripts/gate3_worker.py:147 load_model` (+ `gate3_tdx_step.py:75`).
   - Today: `from_pretrained -> .to("cuda")` then full `model(input_ids)` forward in plaintext (`:218`), plaintext logits (`:219`).
   - Required: replace the plaintext HF forward with the masked-base forward over folded `*_tilde` operators — i.e. run training through the SAME kernels the folded inference worker uses (`_masked_attention`/`_masked_mlp`, `folded_worker.apply_folded_*`) but with the LoRA leaves differentiable. The base checkpoint must be folded ONCE in TDX (`build_qwen7b_folded_package.py`) and the training worker must receive only the folded package + masked embeddings, never `from_pretrained`.
   - This is the single change that flips the audit from FAIL to PASS.

2. **In-process generation worker** — `src/pllo/protocol/gpu_worker.py:205` (`Qwen7BGpuBackend._ensure_session`) + `qwen_masked_session.py:163 _folded_layer` / `:82 _extract_boundary`.
   - Today: `from_pretrained(device_map=cuda)` then on-the-fly extract/fold from the resident plaintext model, in the untrusted process.
   - Required: make the folded-package worker (`Qwen7BFoldedPackageGpuBackend`, `gpu_worker.py:318`) the ONLY generation backend for private-base runs; retire/guard the `MaskedQwenSession`-in-process path so it is never selected when a plaintext `model_path` is present. The in-process fold is fine only when it demonstrably runs inside TDX, which the current class does not enforce.

3. **Trusted-setup builder device guard** — `scripts/build_qwen7b_folded_package.py:75/:107`.
   - Today: defaults `--device cuda`, no binding that this executes inside the attested domain.
   - Required: bind the fold step to TDX attestation (it already can stamp `tee_type`/`mr_td`/`folding_runtime_hash` into the manifest, `folded_package_manifest.py:68-70`) and refuse to serve any package whose fold did not occur under attestation. Keep plaintext weights off any host that later runs the untrusted worker.

---

## B. Forward ops that ALREADY run masked and are directly reusable (no new mask math)

These are the shared, autograd-friendly kernels; the training path should call them instead of the HF plaintext layer:

- **RMSNorm core** — `masked_training_kernels.py:87 rmsnorm_core` / `folded_worker.py:126 _rmsnorm` (signed-perm invariant, 0 trusted calls).
- **Q/K/V/O + gate/up/down projections** — `llama_qwen_single_block.py:466-506 _masked_attention/_masked_mlp` over folded `*_tilde`; fold recipe `fold_hf_single_block_weights:397`.
- **RoPE in permuted basis** — `masked_training_kernels.py:128 apply_rope` with `cos_p/sin_p` (`unified_masked_training.py:267`).
- **GQA repeat_kv** — `masked_training_kernels.py:134` (mask-transparent).
- **Attention score + softmax island** — `llama_qwen_single_block.py:310 _sdpa`, `masked_training_kernels.py:104 softmax_island` (runs on TRUE scores by design).
- **SwiGLU island** — `masked_training_kernels.py:96 silu_island` / `right_multiply_backend.py:89` (shared channel perm).
- **Folded LM head + O(V) monomial logit mask** — `causal_lm_boundaries.py:224 fold_final_norm_lm_head_with_vocab_mask`, `monomial_logit_mask.py:98 mask_logits`.
- **Private CE boundary + surrogate cut** — `monomial_logit_mask.py:121 private_ce` returns masked dlogits `G_tilde`; the `(Z_tilde * G_tilde).sum()` cut (`unified_masked_training.py:392`, `gate3_worker.py:240`) already stitches TDX gradients into autograd with labels kept in TDX.
- **Rank-masked LoRA A/B + backward + masked SGD** — `gate3_worker.py:53 MaskedLoRALinear` (A_t=U@A0, B_t=B0 U^T), `:244 backward`, `:262 opt.step`; grad recovery `U^T gA_t == gA` (`unified_masked_training.py:398`). These are correct today; they just need to sit on top of a masked base instead of a plaintext one.
- **Masked KV cache (inference)** — append-compatible masked K/V (`llama_qwen_single_block.py:474`), reused by `folded_worker` decode.

The end-to-end proof that these compose into a masked-base training step already exists as a CPU/fp64 contract: `unified_masked_training.py run_contract` (masked forward + masked LoRA backward + masked SGD, `plaintext_hidden_materializations==0`, `nonlinear_trusted_calls==0`). It is the blueprint; it is NOT yet wired to a real model or GPU.

---

## C. What the TRAINING path lacks vs the frozen convention

| Convention element | Frozen requirement | Training path today | Gap |
|---|---|---|---|
| Base weights on GPU | GPU sees only `W_tilde=N_in^-1 W N_out` | `from_pretrained` plaintext W on CUDA (`gate3_worker.py:150`) | Fold base ONCE in TDX; feed folded shards to trainer |
| Embedding | GPU sees only `h_tilde=embed@N_0` | plaintext embedding via `model(input_ids)` | Move embedding lookup+mask to boundary (`qwen_masked_session.mask_embeddings` exists for inference; not used in training) |
| Hidden states | every residual == plaintext@N | plaintext residuals on GPU | Run training forward through masked kernels (Sec B) |
| Logits | GPU sees only `logits*M` (M=D*Pi) | plaintext `out.logits` then permute (`:219/:233`) | Fold monomial mask into head; never materialize plaintext logits on GPU. gate3 uses permutation-only (d==1); adopt full monomial D*Pi (`monomial_logit_mask`) to also hide the logit value multiset |
| Nonlinear | A_rightmul, 0 trusted calls | plaintext nonlinears on GPU (but 0 trusted crossings trivially) | Route through `masked_training_kernels` islands on masked state |
| Multi-layer | N layers threaded in N basis | plaintext HF stack | No masked-base multi-layer TRAINING forward on GPU (only 1-block CPU contract) |
| Optimizer | masked-domain | masked SGD only | AdamW/momentum/weight-decay NOT supported masked (orthogonal collapse holds for plain SGD only; MEMORY lora-lifecycle-audit) |
| KV cache / prefill / decode | masked | n/a for training | Generation-only; no training analog needed |

### Concrete gaps ranked
1. **No masked-base training forward on a real model/GPU.** The masked-base forward is proven only in `unified_masked_training.py` (CPU/fp64, 1 synthetic block). It must be scaled to the real 24/28-layer folded package and run on the H800, replacing `gate3_worker.load_model`.
2. **Training reuses the plaintext HF forward.** gate3 masks only the LoRA leaves + the (already-plaintext) logits. Base, embedding, hidden, logits all plaintext on the GPU.
3. **Logit mask is permutation-only in training** (`gate3_worker.py:20`, d==1) — leaks the logit value multiset (`monomial_logit_mask.value_multiset_preserved`). Upgrade to the monomial D*Pi already implemented.
4. **Optimizer coverage**: only plain SGD is masked-exact; AdamW/momentum need a trusted-side optimizer or a different masking scheme.
5. **No attestation binding on the fold step** for training; the trusted packager path exists for inference (`build_qwen7b_folded_package.py`) but the trainer bypasses it.

### Smallest path to PASS
Wire `gate3_worker` (training) to consume a TDX-built folded package + masked embeddings and run the forward through `folded_worker.apply_folded_prefill` + the shared `masked_training_kernels`, keeping the existing masked-LoRA leaves/backward/SGD and the `private_ce` monomial boundary. All the masked pieces (Sec B) already exist and are real-model-validated on the inference side; the work is integration, not new cryptographic design.
