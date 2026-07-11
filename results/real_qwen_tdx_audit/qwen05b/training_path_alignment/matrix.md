# Training-path alignment matrix — Qwen2.5-0.5B

**Gate-3 masked-LoRA TRAINING** forward/backward vs. **paper-facing A_rightmul masked INFERENCE** path.

Scope: READ-only provenance audit. No GPU/TDX run. Every claim cites a `file:line` actually read.

## One-line finding

Gate-3 training runs the **full plaintext HuggingFace forward** (`model(input_ids=...)`). The *only* obfuscation is a rank-space orthogonal mask `U` on the LoRA adapters, and that mask **cancels in the forward** (`B_t @ A_t == B @ A`). Training touches **none** of the A_rightmul / residual-mask / folded-weight / masked-nonlinear machinery. The paper inference path implements all of it, but over **detached constant** folded weights, so it is not differentiable w.r.t. any trainable parameter.

---

## Table 1 — Gate-3 training forward probes

| Probe | Supported | Where | Evidence |
|---|---|---|---|
| `plaintext_hf_model_path` | ✅ yes | `scripts/gate3_worker.py:218` (`load_model` L150) | `out = model(input_ids=input_ids, attention_mask=attn)` — full HF plaintext Qwen2 forward (attention/RoPE/RMSNorm/softmax/SwiGLU/residual all internal). |
| `rank_masked_lora_only` | ✅ yes | `scripts/gate3_worker.py:69` (`MaskedLoRALinear`) | `A_t = U @ A0`, `B_t = B0 @ U.T`; forward L75 `lora = (x @ A_t.T) @ B_t.T` ⇒ `B_t@A_t == B@A`. `U` is an `r×r` orthogonal buffer (L67); base weights frozen plaintext (L63-64). |
| `calls_A_rightmul` | ❌ **NOT FOUND** | — | gate3 imports only stdlib + torch + transformers + gate3_tdx_client (L37-43,148,233). grep `right_multiply\|A_rightmul\|rmsnorm_core\|masked_attention\|n_res\|folded` over the 3 gate3 files → only the word "folded" in a docstring (L32). Nonlinears run inside HF. |
| `masked_rmsnorm` | ❌ **NOT FOUND** | — | RMSNorm is HF's plaintext `Qwen2RMSNorm` inside `model(...)` (L218). No `rmsnorm_core` over a signed-permutation-masked state. |
| `post_rope_masking` | ❌ **NOT FOUND** | — | RoPE runs inside HF plaintext attention. No per-head `pairwise_rotation` Q/K masks, no `apply_rope` over masked q/k. |
| `gqa_masked_cache` | ❌ **NOT FOUND** | — | KV cache / GQA `repeat_kv` handled by HF over plaintext K/V. No `key_rope_tilde`/`value_tilde`. (Full-seq forward L218 needs no cache anyway.) |
| `swiglu_island` | ❌ **NOT FOUND** | — | MLP is HF plaintext `Qwen2MLP`. No shared-channel-perm island. With `targets=='attn_mlp'` (L199) gate/up/down just get the same cancelling rank-mask LoRA, not the perm island. |
| `inter_block_hidden_masked` | ❌ **NOT FOUND** | — | The 24-layer residual stream is plaintext HF hidden state. No `x @ n_res`, no per-layer handoff. GPU sees plaintext hidden everywhere; only rank space is masked. |
| `backward_through_masked_hidden` | ❌ no (rank-space only) | `scripts/gate3_worker.py:244` `loss.backward()` | Autograd runs through the **plaintext** HF graph; leaves are `A_t,B_t` so grads emerge in the masked rank basis (`gradA_t = U gradA`, docstring L14-16). dlogits are private at the LOGIT boundary (L234-240), but the interior activations the gradient flows through are unmasked. |

**Summary: 2 / 9 supported** (plaintext HF path, rank-masked LoRA). The 7 masked-domain-forward probes are all NOT FOUND — masking exists only in LoRA rank space, which is forward-invisible.

---

## Table 2 — Paper inference path operators (the reference to unify toward)

| Operator | Where | Mask convention | Evidence |
|---|---|---|---|
| `masked_input` | `qwen_masked_session.py:142` `mask_embeddings` | residual signed-perm `N_res` (`_n0`) | `return self.embed_plain(input_ids) @ self._n0`. Block analogue `llama_qwen_single_block.py:527` `x_tilde = x @ n_res`. |
| `residual_transport` | `hf_causal_lm_skeleton.py:296` `handoff` | `T_ell = N_ell^-1 @ N_ell+1`; shared mask ⇒ identity (L300-303) | `return self.residual_mask_inverses[ell] @ self.residual_masks[ell+1]`; applied L955-957. |
| `nonlinear_backend` | `right_multiply_backend.py:89` `silu`/`softmax`/`rmsnorm` | compatible right-multiply / permutation, 0 trusted crossings | `def silu(self,x): return self._on_accelerator(x, F.silu(x), 'silu')` (softmax L92, rmsnorm L108). Dispatched by `folded_worker.py:126` `_rmsnorm` and `llama_qwen_single_block.py:503`. |
| `q_k_v_projections` | `llama_qwen_single_block.py:466` `_masked_attention` | `wq_tilde = n_res_inv @ (rms1 * W) @ mq_block`; `wo_tilde = sv_block_inv @ W_o @ n_res` | `q_t = split_heads(_linear(r1_core_tilde, folded['wq_tilde'], ...))`. Fold L420-423. |
| `post_rope_paired_masking` | `llama_qwen_single_block.py:472` `_masked_attention` | `pairwise_rotation` per-head Q/K, RoPE-compatible + `Nq Nk^T == I` | `qr_t = apply_rope(q_t, cos, sin, position_ids=...)` on already-masked q_t. Family enforced by `compatible_mask_verify.py:59` + `assert_qk_compatible` L135-146. |
| `gqa_attention` | `llama_qwen_single_block.py:479` `_masked_attention` | value masks GQA-expanded by `kv_index`; scores are true `QK^T` | `_sdpa(qr_t, repeat_kv(kr_t,nh,nkv), repeat_kv(v_full,nh,nkv), ...)`. GQA expansion fold L410-411. |
| `swiglu_shared_perm` | `llama_qwen_single_block.py:491` `_masked_mlp` | single shared channel perm on gate/up (`index_select(1,perm)`) + down (`index_select(0,perm) @ n_res`) | `act = runner.silu(gate) ...; hidden = act * up` (L503-504). Fold L428-430. |
| `second_rmsnorm` | `folded_worker.py:219` `apply_folded_layer_prefill` | RMSNorm core over signed-perm-masked `x1_tilde` (invariant `rmsnorm_core(x@N)==rmsnorm_core(x)@N`) | `r2 = _rmsnorm(x1, eps, runner)`; block analogue L532 `r2_core_tilde = rmsnorm_core(x1_tilde, eps)`. |
| `inter_block_remask` | `folded_worker.py:314` `apply_folded_prefill` | production uses ONE shared residual basis → `h = out['y_tilde']` straight to next layer, no re-mask GEMM | `h = out['y_tilde']; kv.append(out['cache'])`. Per-layer variant would insert `masks.handoff(ell)` (skeleton L955). |
| `output_logit_boundary` | `qwen_masked_session.py:181` `_final_head` + `recover` L150 | folded `w_lm_tilde` (vocab mask baked in) + trusted `recover_vocab_logits` | `return rmsnorm_core(h_tilde, self.eps) @ self._w_lm_tilde`; worker analogue `folded_worker.py:358` `apply_folded_head`. |

---

## Table 3 — Integration gaps

| Gap | Category | Where | Detail |
|---|---|---|---|
| Training never invokes the masked forward | `reusable_forward` | `gate3_worker.py:218` | Pure ops `apply_folded_layer_prefill` (`folded_worker.py:194`), `_masked_attention` (`llama_qwen_single_block.py:457`), `_masked_mlp` (L491), `_rmsnorm` (`folded_worker.py:126`) are differentiable torch ops, reusable as a training forward — but gate3 calls HF plaintext instead. |
| Folded weights are LoRA-blind | `lora_aware_folded_weight` | `llama_qwen_single_block.py:397` `fold_hf_single_block_weights` | Every `*_tilde` folds only the FROZEN base (`wq_tilde = n_res_inv @ (rms1*q_proj_weight) @ mq_block`, L420). Need `W_tilde(θ)=fold(W_base + scaling·B(θ)A(θ))`. `folded_worker._maybe_merge_lora` (L263-277) merges only a *frozen* folded-LoRA package — constants, not leaves. **Core missing piece.** |
| Weights detached at extraction | `autograd_break` | `llama_qwen_single_block.py:231` `_extract_linear` | `w = module.weight.detach()...` (+ bias L234, layernorms L251/253) makes every folded operator an autograd constant. |
| Two mask systems not co-located for backward | `mask_state_for_backward` | `qwen_masked_session.py:108` (`_n0`) vs `gate3_worker.py:67` (`U`) | Inference `N_res`/attn/SwiGLU masks live in the trusted boundary bundle; the worker holds none. Rank mask `U` is a per-`MaskedLoRALinear` buffer. A unified path must fix a compatible `N_res` + per-layer attn/SwiGLU masks as constant buffers alongside `A_t,B_t`. |
| KV cache is inference-only | `kv_cache_excludable` | `llama_qwen_single_block.py:474` | `past_key_rope/value` concat (L474-478) + returned `key_rope_full`/`value_full` (L487) serve incremental decode. Training uses full-seq prefill (`causal_offset=0`, `past_key_rope=None`, L529) → cache computed-then-discarded, excludable. |
| Verification wrappers recompute plaintext + call `.item()` | `inference_only_tensor` | `llama_qwen_single_block.py:525` `hf_single_block_masked_prefill` | `plain = hf_single_block_plain_prefill(...)` (L525) + `_mx`/`.item()` metrics (L539-565) are validation-only. Training should call the pure worker ops (no plain recompute, no host sync). |

---

## Autograd breaks in the inference forward path

Every `detach()`/`clone()`/`no_grad()` on the inference path that a training graph would trip over:

- **`llama_qwen_single_block.py:231`** — `w = module.weight.detach().to(...).t().contiguous()`. Detaches the base Linear weight before folding → `wq/wk/wv/wo/wgate/wup/wdown_tilde` are constants. **The primary break.**
- **`llama_qwen_single_block.py:234`** — `b = module.bias.detach()...clone()`. Detaches folded biases.
- **`llama_qwen_single_block.py:251`** — `rms1 = layer.input_layernorm.weight.detach()...clone()`. Pre-attention norm affine (folded as `rms1 * W`) becomes constant.
- **`llama_qwen_single_block.py:253`** — `rms2 = layer.post_attention_layernorm.weight.detach()...clone()`. Post-attention norm affine (folded as `rms2 * W`) becomes constant.
- **`qwen_masked_session.py:165`** — `_folded_layer` calls `extract_hf_single_block_weights(...)`, re-triggering the L231/234/251/253 detaches on every worker call, so `worker_prefill`/`worker_decode` logits are constant w.r.t. any trainable parameter.

Not actually breaks (listed for completeness / to replicate correctly):

- **`gate3_worker.py:266`** — `with torch.no_grad(): out2 = model(...)` guards the POST-step next-step-logits diagnostic (after `opt.step()`), not the trained step.
- **`gate3_worker.py:234`** — `tdx_logits_loss(..., fl.detach(), ...)` correctly detaches only the tensors crossing the TDX wire; the training graph is preserved via the surrogate `loss = (fl * full_dlogits).sum()` (L240) using the non-detached `fl`. This is the intended private-boundary cut and must be replicated in a unified path.
