# Privacy LLM Obfuscation — Calibrated Workload Profile (Stage 5.0.1)

Cost model splits every method into four explicit slices: **preprocessing trusted cost** (amortised), **online boundary crossings** (true trusted↔untrusted round trips), **online trusted compute** (LayerNorm / GELU / sampling / recovery FLOPs in the TEE), and **online GPU obfuscated compute** (linear matmuls, attention, LM head). Internal Python bookkeeping such as mask-state creation is **not** counted as a boundary call.

`model_id=sshleifer/tiny-gpt2`, `batch_size=2`, `prompt_len=8`, `max_new_tokens=4`, `device=cpu`, `dtype=float32`, `use_pad=True`, `warmup=2`, `repeat=5`.

GPU-FLOPs/ms calibration constant: `2.206e+06` (derived from measured `plain_hf_gpu` wall time).

> **Warning:** simulated cost model, not real SGX.

## Method comparison
| method | impl? | wall_time_ms (measured/proj.) | boundary calls | boundary formula | trusted compute (ops) | trusted transfer (bytes) | gpu (ops) |
|---|---|---|---|---|---|---|---|
| plain_hf_gpu | true | 2.010 | 0 | 0 (no boundary) | 0 | 0 | 4434424 |
| tslp_trusted_nonlinear_baseline | false | 31.455 (proj.) | 32 | 3L + 2 = 8 per forward (LN_1 + LN_2 + GELU per layer + ln_f + LM head) | 1110230 | 4427192 | 4429848 |
| ours_current | true | 6.539 | 36 | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | 1116310 | 4428424 | 4429848 |
| ours_ideal_gpu_nonlinear | false | 31.209 (proj.) | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4422792 | 4434424 |
| ours_compatible_nonlinear_islands | false | 31.274 (proj.) | 16 | L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model) | 1105830 | 4423496 | 4434424 |
| amulet_style_reference | false | 31.209 (proj.) | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4422792 | 4434424 |

## Preprocessing (amortised; excluded from online latency)
| method | preprocessing_trusted_ops | preprocessing_transfer_bytes |
|---|---|---|
| plain_hf_gpu | 0 | 0 |
| tslp_trusted_nonlinear_baseline | 0 | 0 |
| ours_current | 403784 | 402440 |
| ours_ideal_gpu_nonlinear | 403784 | 402440 |
| ours_compatible_nonlinear_islands | 403992 | 402440 |
| amulet_style_reference | 403784 | 402440 |

## Interaction breakdown (online slice by interaction type)
### boundary calls per interaction
| interaction | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| input_masking | 0 | 0 | 0 | 4 | 4 | 4 |
| trusted_layernorm | 0 | 20 | 0 | 0 | 0 | 0 |
| trusted_gelu | 0 | 8 | 0 | 0 | 0 | 0 |
| lm_head_recovery | 0 | 4 | 4 | 4 | 4 | 4 |
| sampling | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_weight_obfuscation | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_affine_folding | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_permutation_absorption | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_norm_core_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_activation_island_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_sandwich_transition | 0 | 0 | 0 | 0 | 8 | 0 |
| security_proxy_requirements | 0 | 0 | 0 | 0 | 0 | 0 |

### transfer bytes per interaction
| interaction | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| input_masking | 0 | 0 | 0 | 176 | 176 | 176 |
| trusted_layernorm | 0 | 1760 | 0 | 0 | 0 | 0 |
| trusted_gelu | 0 | 2816 | 0 | 0 | 0 | 0 |
| lm_head_recovery | 0 | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 |
| sampling | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_weight_obfuscation | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_affine_folding | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_permutation_absorption | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_norm_core_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_activation_island_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_sandwich_transition | 0 | 0 | 0 | 0 | 704 | 0 |
| security_proxy_requirements | 0 | 0 | 0 | 0 | 0 | 0 |

### trusted compute (ops) per interaction
| interaction | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| input_masking | 0 | 0 | 0 | 0 | 0 | 0 |
| trusted_layernorm | 0 | 1760 | 1760 | 0 | 0 | 0 |
| trusted_gelu | 0 | 2816 | 2816 | 0 | 0 | 0 |
| lm_head_recovery | 0 | 1105654 | 1105654 | 1105654 | 1105654 | 1105654 |
| sampling | 0 | 1105654 | 1105654 | 1105654 | 1105654 | 1105654 |
| preprocessing_weight_obfuscation | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_affine_folding | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_permutation_absorption | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_norm_core_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_activation_island_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_sandwich_transition | 0 | 0 | 0 | 0 | 176 | 0 |
| security_proxy_requirements | 0 | 0 | 0 | 0 | 0 | 0 |

## Module breakdown (online slice by module category)

### boundary calls per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |
| layernorm | 0 | 20 | 0 | 0 | 0 | 0 |
| attention_qkv | 0 | 0 | 8 | 0 | 8 | 0 |
| attention_score | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_output | 0 | 0 | 8 | 0 | 8 | 0 |
| mlp_fc | 0 | 0 | 8 | 0 | 8 | 0 |
| activation | 0 | 8 | 0 | 0 | 0 | 0 |
| mlp_proj | 0 | 0 | 8 | 0 | 8 | 0 |
| lm_head | 0 | 4 | 4 | 4 | 4 | 4 |
| kv_cache_update | 0 | 0 | 0 | 0 | 0 | 0 |

### trusted compute (ops) per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |
| layernorm | 0 | 1760 | 1760 | 0 | 0 | 0 |
| attention_qkv | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_score | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_output | 0 | 0 | 0 | 0 | 0 | 0 |
| mlp_fc | 0 | 0 | 0 | 0 | 0 | 0 |
| activation | 0 | 2816 | 2816 | 0 | 0 | 0 |
| mlp_proj | 0 | 0 | 0 | 0 | 0 | 0 |
| lm_head | 0 | 1105654 | 1105654 | 1105654 | 1105654 | 1105654 |
| kv_cache_update | 0 | 0 | 0 | 0 | 0 | 0 |

### trusted transfer (bytes) per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |
| layernorm | 0 | 1760 | 0 | 0 | 0 | 0 |
| attention_qkv | 0 | 0 | 1408 | 0 | 1408 | 0 |
| attention_score | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_output | 0 | 0 | 704 | 0 | 704 | 0 |
| mlp_fc | 0 | 0 | 1760 | 0 | 1760 | 0 |
| activation | 0 | 2816 | 0 | 0 | 0 | 0 |
| mlp_proj | 0 | 0 | 1760 | 0 | 1760 | 0 |
| lm_head | 0 | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 |
| kv_cache_update | 0 | 0 | 0 | 0 | 0 | 0 |

### gpu ops per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 44 | 44 | 44 | 44 | 44 | 44 |
| layernorm | 1760 | 0 | 0 | 1760 | 1760 | 1760 |
| attention_qkv | 1056 | 1056 | 1056 | 1056 | 1056 | 1056 |
| attention_score | 3008 | 3008 | 3008 | 3008 | 3008 | 3008 |
| attention_output | 352 | 352 | 352 | 352 | 352 | 352 |
| mlp_fc | 1408 | 1408 | 1408 | 1408 | 1408 | 1408 |
| activation | 2816 | 0 | 0 | 2816 | 2816 | 2816 |
| mlp_proj | 1408 | 1408 | 1408 | 1408 | 1408 | 1408 |
| lm_head | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 |
| kv_cache_update | 176 | 176 | 176 | 176 | 176 | 176 |

## Compatible Nonlinear Islands Method

ours_compatible_nonlinear_islands is a projected method based on Stage 5.2a correctness probes and Stage 5.2b security proxies. It is not yet integrated into GPT-2 / BERT / T5 wrappers — Stage 5.3 is the integration step.

### Boundary Call Formulas

- `ours_current`: 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head)
- `ours_compatible_nonlinear_islands`: L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model)
- `ours_ideal_gpu_nonlinear`: 1 per forward (single fused GPU pipeline round trip)

### Trusted Compute Reduction

- vs `ours_current`: 0.94%
- vs `tslp_trusted_nonlinear_baseline`: 0.40%
- vs `ours_current` boundary call count: 55.56%

### Preprocessing Cost Increase

- Preprocessing increase vs `ours_current`: 0.05% (affine folding + permutation absorption + compatible mask generation, all amortised over many sessions).

- Preprocessing breakdown (ops):
  - base weight obfuscation: 403784
  - affine folding: 40
  - permutation absorption: 128
  - compatible mask generation: 40

### Online Extra Matmul Count

- `online_extra_matmul_count = 0`. Stage 5.2a verified this across every MLP island cell — operator-compatible mask transitions are folded into adjacent Linear weights offline and add zero online matmuls.

### Security Proxy Caveats

- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.
- This is not a real TEE measurement.
- This is not a real TEE measurement.

### Stage 5.3a Wrapper Integration Status

- `partial_implementation = True` — the GPT-2 single-block wrapper now exposes a `nonlinear_mode="compatible_islands"` feature flag, but the GPT-2 model-level wrapper, BERT, and T5 paths are not yet wired up.
- `gpt2_single_block`: `implemented`
- `gpt2_model_level`: `implemented`
- `bert`: `implemented_probe_level`
- `t5`: `implemented_probe_level`
- Default mode remains `trusted`; compatible_islands must not be enabled by default.
- GPT-2 model-level integration is available.
- BERT/T5 are probe-level integrations, not full wrappers.
- `measured_integration_scope = "cross_architecture_probe_level"`.
- `full_runtime_integrated = False`.
- `all_architecture_probe_level_implemented = True`.
- `security_profile` remains `proxy-evaluated, not formal`.

## Paper metrics

- `boundary_call_reduction_vs_tslp` = -12.50% (ours_current vs tslp)
- `trusted_transfer_reduction_vs_tslp` = -0.03%
- `online_trusted_compute_reduction_vs_tslp` = -0.55%
- `gpu_offload_ratio` (ours_current) = 79.87%
- `preprocessing_amortized` = `True`
- `boundary_calls_per_forward` =
  - `plain_hf_gpu`: 0
  - `tslp_trusted_nonlinear_baseline`: 8
  - `ours_current`: 9
  - `ours_ideal_gpu_nonlinear`: 1
  - `ours_compatible_nonlinear_islands`: 4
  - `amulet_style_reference`: 1

## Interpretation

- **Main online bottleneck (ours_current):** `lm_head`
- **Next primitive to obfuscate on GPU:** `GELU`

*Note on ours_current vs TSLP boundary calls:* ours_current crosses the boundary once per obfuscated linear (4 per layer) while TSLP crosses once per non-linear (3 per layer plus ln_f). This is an **architectural** difference, not a bookkeeping artefact. Each ours_current crossing moves a smaller activation than a TSLP LayerNorm crossing, and the measured wall time is consistent with that tradeoff.

## Method semantics & citation caveats
### `plain_hf_gpu` — Plain HuggingFace on GPU

Plaintext HF GPT-2 forward / greedy decode. No protection.

- Implemented: **True**
- Implementation note: Hand-written HF greedy loop over plain model().
- Caveat: No security; measured wall time is the GPU-only baseline.

### `tslp_trusted_nonlinear_baseline` — TSLP-style trusted non-linear baseline

Linear / attention on GPU, every LayerNorm and GELU activation makes a TEE round-trip. Modeled after the trusted non-linear split common in shielded-inference literature.

- Implemented: **False**
- Implementation note: No real implementation in this repo. Wall time is projected from op counts using a documented cost model — not a measurement of any specific published system.
- Caveat: TSLP-style is used here as a generic non-linear-in-TEE baseline. It is not a faithful re-implementation of any single published system. Adjust the cost model constants in WorkloadProfileConfig before drawing system-level conclusions.

### `ours_current` — This work — current Stage 4.9 implementation

Right-multiply mask + per-block Conv1D pad compensation, trusted LayerNorm / GELU shortcuts, diagonal vocab output mask on the LM head, internal ObfuscatedGPT2KVCache.

- Implemented: **True**
- Implementation note: Wall time is measured against the real ObfuscatedGPT2ModelWrapper generation path.
- Caveat: Trusted LayerNorm / GELU are engineering shortcuts; their TEE cost is included in proxies but their security model is unprotected non-linearity. See Stage 5.1 / 5.2 roadmap.

### `ours_ideal_gpu_nonlinear` — This work — ideal: LN / GELU on GPU in masked domain

Same wrapper but with LayerNorm and GELU executed inside the obfuscated GPU domain. The trusted side only crosses the boundary to prepare the masked input and to recover the LM head logits. Used as an upper bound, not a measured system.

- Implemented: **False**
- Implementation note: Hypothetical. Op counts come from the same model graph; LN / GELU FLOPs are reattributed from TEE to GPU. Wall time is projected, not measured.
- Caveat: Upper bound. Real obfuscated LN / GELU primitives are deferred to Stage 5.1 / 5.2 and may carry additional overhead this estimate does not capture.

### `ours_compatible_nonlinear_islands` — This work — projected: operator-compatible nonlinear islands

Modeled / projected method. RMSNorm core uses an orthogonal mask, LayerNorm core uses a mean-preserving orthogonal mask, GELU / ReLU / SiLU activations use permutation masks, and SwiGLU uses a paired permutation. Every mask transition is folded into adjacent Linear weights offline, so the masked forward executes with the same number of matmuls as the plaintext forward (Stage 5.2a verified ``online_extra_matmul_count = 0`` for every MLP island cell). Trusted shortcuts for LN and GELU are removed.

- Implemented: **False**
- Implementation note: Projected, not measured. Stage 5.2a verified the correctness probe (28 cells, all_allclose=True, max_online_extra_matmul=0). Stage 5.2b validated the security proxy (fresh permutation + dense sandwich + pad at Linear boundaries are required mitigations). Not yet integrated into the GPT-2 / BERT / T5 wrappers — Stage 5.3 is the integration step.
- Caveat: Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands. The Stage 5.2b security proxy quantified per-strategy linkability and permutation recovery, but the result is a naive-observer upper bound, NOT a formal security proof and NOT a real TEE measurement.

### `amulet_style_reference` — Amulet-style reference (cost model)

Input masking + GPU obfuscated forward + output unmasking, modeled after the high-level pattern in Amulet-style systems. Reference only, not a re-implementation.

- Implemented: **False**
- Implementation note: No implementation in this repo. Cost-model reference under the assumption that the entire obfuscated forward runs as a single GPU pipeline between trusted input masking and trusted output unmasking.
- Caveat: Amulet-style here means the abstract pattern of input mask + GPU forward + output recovery. Real Amulet systems may include primitives and overheads not captured by this proxy. Use with explicit attribution to assumptions.

## Limitations
- Simulated TEE cost model — not real SGX wall-clock.
- Wall time for tslp_baseline, ours_ideal, and amulet_style_reference is projected, not measured.
- tiny-gpt2 (n_layer=2, n_embd=2, n_head=2) is far smaller than production GPT-2.
- FLOP / byte proxies use coarse constants; absolute numbers are illustrative.
- Qwen / Llama not yet covered — see Stage 5.4 roadmap.
- amulet_style_reference is a reference cost model, not a re-implementation of any published system.

## Reproducibility

```bash
python scripts/run_workload_profile.py --batch-size 2 --prompt-len 8 --max-new-tokens 4 --warmup 2 --repeat 5 --use-pad True
```
