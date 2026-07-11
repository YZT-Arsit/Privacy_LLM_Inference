# Missing integration — unifying Gate-3 masked-LoRA training with the A_rightmul masked inference forward

Qwen2.5-0.5B. READ-only audit; every reference is a `file:line` actually read. No GPU/TDX run.

## The starting truth

The current Gate-3 training worker (`scripts/gate3_worker.py`) does **not** run the masked forward at all. It loads a plaintext HF model (`load_model`, L147-152) and calls `out = model(input_ids=input_ids, attention_mask=attn)` (L218). The whole transformer — attention, RoPE, RMSNorm, softmax, SwiGLU, and the inter-layer residual stream — executes in HuggingFace's own plaintext kernels. The single obfuscation is `MaskedLoRALinear` (L53-85), which masks the LoRA adapters in **rank space** with a fixed orthogonal `U` (`A_t = U@A0`, `B_t = B0@U.T`, L69-70). Because `B_t @ A_t == B @ A`, the mask **cancels in the forward** (L74-76) and the visible activations are byte-identical to plaintext. `loss.backward()` (L244) then differentiates the plaintext graph, and because the trainable leaves are `A_t,B_t`, the gradients come out already rotated into the masked rank basis (`gradA_t = U gradA`, docstring L14-16). Privacy in Gate-3 is enforced only at the **logit boundary** (masked/private cross-entropy in TDX, L234-240), never in the interior.

The paper-facing A_rightmul path — `MaskedQwenSession` (`qwen_masked_session.py`), the single-block reference (`llama_qwen_single_block.py`), and the deployment worker (`folded_worker.py`) — does the opposite: it masks the *hidden state itself* (`x_tilde = x @ n_res`) and runs every operator over the masked state with compatible masks. Unifying the two means giving the training worker that masked-hidden forward while keeping the LoRA adapters trainable. The sections below enumerate exactly what that requires.

## (a) Forward modules of the inference path reusable directly for training

These are ordinary differentiable torch ops (matmul, `rsqrt`, `softmax`, `silu`, `index_select`, `cat`) and can be reused verbatim as a training forward, provided their weight inputs are made differentiable (see (b)/(d)):

- **`folded_worker.apply_folded_layer_prefill`** (`src/pllo/deployment/folded_worker.py:194`) — the clean per-layer masked prefill: `_rmsnorm` → `_masked_attention` → residual add → `_rmsnorm` → `_masked_mlp` → residual add, returning only `y_tilde` + cache. This is the right training-forward unit (no plain recompute, no `.item()`).
- **`_masked_attention`** (`src/pllo/hf_wrappers/llama_qwen_single_block.py:457`) — folded q/k/v/o projections, RoPE on masked q/k (L472-473), GQA `repeat_kv` + `_sdpa` (L479-481).
- **`_masked_mlp`** (`llama_qwen_single_block.py:491`) — shared-perm SwiGLU island (`act = runner.silu(gate) * up`, L503-504).
- **`_rmsnorm`** (`folded_worker.py:126`) and the nonlinear backend **`RightMultiplyNonlinearBackend`** (`src/pllo/nonlinear/right_multiply_backend.py:89` for `silu`/`softmax`/`rmsnorm`) — the A_rightmul island evaluators. `F.silu`/`torch.softmax`/`rsqrt` are all autograd-friendly.
- **`_linear`** (`llama_qwen_single_block.py:270`) — the folded masked matmul (with optional linear-boundary pad).
- **`apply_folded_head`** (`folded_worker.py:358`) / **`_final_head`** (`qwen_masked_session.py:181`) — `rmsnorm_core(h_tilde) @ w_lm_tilde`, the masked-logit boundary. The subsequent `recover` (`qwen_masked_session.py:150`) is the trusted vocab un-mask, which lines up with the existing Gate-3 TDX logit boundary.

**Do NOT reuse** `hf_single_block_masked_prefill` (`llama_qwen_single_block.py:510`) or `hf_causal_lm_masked_prefill` (`hf_causal_lm_skeleton.py:884`) as-is: they recompute a plaintext reference (`plain = hf_single_block_plain_prefill(...)`, L525) and build large metrics dicts with `.item()` host syncs (L539-565). Those are verification harnesses, not a training forward.

## (b) Autograd-graph breaks (detach/clone/no_grad) that must be removed

The masked forward is built on **detached** weights, so today no gradient can reach any trainable parameter. To train, these detaches must be lifted (at least for the LoRA contribution):

- **`llama_qwen_single_block.py:231`** — `_extract_linear`: `w = module.weight.detach().to(...).t().contiguous()`. The primary break — every folded projection is a constant.
- **`llama_qwen_single_block.py:234`** — `_extract_linear`: `b = module.bias.detach()...clone()` (folded biases).
- **`llama_qwen_single_block.py:251`** — `rms1 = layer.input_layernorm.weight.detach()...clone()` (pre-attn norm affine, folded as `rms1 * W`).
- **`llama_qwen_single_block.py:253`** — `rms2 = layer.post_attention_layernorm.weight.detach()...clone()` (post-attn norm affine, folded as `rms2 * W`).
- **`qwen_masked_session.py:165`** — `_folded_layer` re-invokes `extract_hf_single_block_weights`, re-detaching on every call, so `worker_prefill`/`worker_decode` are constant.

Practical resolution: keep the frozen base weight detached (it is genuinely frozen — `MaskedLoRALinear` freezes base params at `gate3_worker.py:63-64`) but route the **trainable LoRA delta** into the fold with grad intact (see (d)). Only the LoRA path needs to remain attached.

Two detaches that are correct and must be **preserved** in a unified path, not removed:

- **`gate3_worker.py:234`** — `tdx_logits_loss(..., fl.detach(), ...)`: detaches only what crosses the TDX wire; the surrogate `loss = (fl * full_dlogits).sum()` (L240) keeps the graph via the non-detached `fl`. This is the intended private-boundary cut.
- **`gate3_worker.py:266`** — `with torch.no_grad(): out2 = model(...)`: a post-`opt.step()` diagnostic, outside the trained step.

## (c) Mask state that must be saved for backward

For the masked-domain backward to collapse exactly (the whole point — grads stay in the masked basis and never reveal plaintext), the masks must be **fixed constant buffers** held for the whole step, co-located with the trainable adapters:

- **Residual signed-permutation `N_res`** — currently the trusted-only `MaskedQwenSession._n0` (`qwen_masked_session.py:108`) and the per-layer `masks.residual_masks` bundle (`hf_causal_lm_skeleton.py:288`). Must be a compatible **signed permutation** (`compatible_mask_verify.assert_signed_permutation`, L94) so `rmsnorm_core(x@N)==rmsnorm_core(x)@N` holds through the norm cores in backward as well as forward.
- **Per-head attention Q/K/V masks + `kv_index`** and the **shared SwiGLU channel `perm`** — per layer in `masks.layer_block_masks[ell]` (`hf_causal_lm_skeleton.py`), folded at `llama_qwen_single_block.py:407-411,428-430`. These are constants across the step; their compatibility (`Nq Nk^T == I`, orthogonal value masks, shared perm) is what keeps the folded operators exact.
- **Rank mask `U`** — already a per-`MaskedLoRALinear` buffer (`gate3_worker.py:67`), fixed and never updated. This is the mask that makes masked-SGD equal plaintext-SGD (`masked_gradient_lora.masked_sgd_step`, `src/pllo/ops/masked_gradient_lora.py:277`; note `masked_adamw_step_unsupported` L337 — only SGD/momentum are exact under a dense orthogonal mixer).

The gap: `N_res`/attn/SwiGLU masks live in a different module (the trusted boundary) from `U` (the training worker). A unified masked-training module must instantiate one **compatible** mask bundle and hold it as buffers alongside `A_t,B_t` for the duration of forward+backward.

## (d) Linear layers that need LoRA-aware folded weights

This is the load-bearing change. Every folded operator in `fold_hf_single_block_weights` (`llama_qwen_single_block.py:397-436`) currently folds only the frozen base weight. For LoRA training the fold must include the trainable delta so that `d logits / d(A_t,B_t) ≠ 0`. Concretely, for each targeted projection `W` (Gate-3 targets `q_proj,k_proj,v_proj,o_proj` and optionally `gate_proj,up_proj,down_proj` — `gate3_worker.py:107-108`), the folded operator must become a function of the adapter:

- `wq_tilde`, `wk_tilde`, `wv_tilde` — `n_res_inv @ (rms1 * (W + scaling·BᵀAᵀ_of_layer)) @ M{q,k,v}_block` (base fold at L420-422).
- `wo_tilde` — `sv_block_inv @ (W_o + scaling·…) @ n_res` (L423).
- `wgate_tilde`, `wup_tilde` — `n_res_inv @ (rms2 * (W + Δ)).index_select(1, perm)` (L428-429).
- `wdown_tilde` — `(W_down + Δ).index_select(0, perm) @ n_res` (L430; also `qwen_masked_session.py:199`).

The existing merge hook `folded_worker._maybe_merge_lora` (`folded_worker.py:263-277`) merges a **frozen** folded-LoRA package (`W_tilde += a_tilde @ b_tilde`) — the tensors are constants loaded from disk, not autograd leaves. **No trainable-LoRA fold exists anywhere in the repo.** The unification work is to make the LoRA contribution to each `*_tilde` a differentiable function of the masked adapters `A_t,B_t`, keeping the base term detached and frozen.

Alternatively (and probably cleaner): do **not** fold the LoRA into the weight; keep the base forward folded/frozen and apply the masked LoRA delta as a separate additive branch in the masked basis — mirroring how `MaskedLoRALinear.forward` (`gate3_worker.py:72-76`) adds `(x @ A_t.T) @ B_t.T` on top of the base output. The residual mask `N_res` on the layer boundary must be reconciled with the LoRA input/output basis (the `masked_gradient_lora` construction `A_tilde = N_xᵀ A M`, `B_tilde = Mᵀ B N_y`, `src/pllo/ops/masked_gradient_lora.py:225-239`, is the algebra for exactly this — input mask `N_x`, output mask `N_y`, rank mixer `M`).

## (e) Intermediate tensors that are inference-only

- **KV cache** — `_masked_attention` builds `past_key_rope`/`past_value` concat (`llama_qwen_single_block.py:474-478`) and returns `key_rope_full`/`value_full` (L487); threaded across steps by `apply_folded_layer_decode`/`apply_folded_decode` (`folded_worker.py:228,319`). Inference decode only.
- **Plaintext reference + metrics** — `plain = hf_single_block_plain_prefill(...)` (L525) and all `_mx()`/`allclose` metrics (L539-567), plus the whole `hf_causal_lm_masked_prefill` per-layer metrics/handoff-error machinery (`hf_causal_lm_skeleton.py:937-1022`). Validation only.
- **Negative controls** — `_negative_control_recovered_logits` (`hf_causal_lm_skeleton.py:846`) and `wrong_vocab_recovery` etc. Audit only.
- **Greedy/sampling decode loops** — `hf_causal_lm_masked_greedy_decode`/`hf_causal_lm_masked_only_decode` (`hf_causal_lm_skeleton.py:1046,1168`), `MaskedQwenSession.sample_greedy` (L153). Generation only; training needs a single prefill + loss.

A training forward keeps only `y_tilde` per layer and the final masked logits.

## (f) KV cache is training-irrelevant — confirmed

Training uses one **full-sequence prefill** with full self-attention, not incremental decode. In the masked prefill, `_masked_attention` is called with `causal_offset=0` and `past_key_rope=None` (`llama_qwen_single_block.py:529`; `apply_folded_layer_prefill` passes `causal_offset=0` at `folded_worker.py:215`), so no past cache is consumed — the K/V for the whole sequence are computed in one shot and the returned `key_rope_full`/`value_full` (L487) are simply the current-sequence K/V. The cache tensors exist only so that `apply_folded_layer_decode` (`folded_worker.py:228`) can append one token at a time during generation. For a training step there is no next-token append, so the cache is **computed-then-discarded and can be dropped entirely** from the unified training forward. (Gate-3 today already never uses a KV cache across steps — each `model(...)` call at `gate3_worker.py:218` is an independent full forward.) **KV cache excludable: confirmed.**

## Minimal integration sketch (concrete symbols)

1. New masked-training module holding: frozen folded base per layer (from `fold_hf_single_block_weights`, base term detached) + trainable masked adapters `A_t,B_t` (as in `MaskedLoRALinear`, `gate3_worker.py:53`) + a **compatible** mask bundle (signed-perm `N_res`, `pairwise_rotation` attn, shared SwiGLU `perm`) held as buffers.
2. Forward = loop over `apply_folded_layer_prefill` (`folded_worker.py:194`) with `causal_offset=0`, discarding the returned cache, then `apply_folded_head` (`folded_worker.py:358`) for masked logits — with the LoRA delta injected either into the fold (d) or as a masked additive branch.
3. Remove the extraction detaches for the LoRA path only (`llama_qwen_single_block.py:231/234/251/253`); keep the base frozen.
4. Loss/backward reuse the existing Gate-3 TDX private-CE boundary (`gate3_worker.py:234-244`, surrogate `loss = (fl * full_dlogits).sum()`), then `opt.step()` on `A_t,B_t` = exact masked-domain SGD (`masked_gradient_lora.masked_sgd_step`, `src/pllo/ops/masked_gradient_lora.py:277`).

This yields a training step whose *interior activations are masked* (matching the inference threat model) rather than the current plaintext-interior, logit-boundary-only Gate-3 step.
