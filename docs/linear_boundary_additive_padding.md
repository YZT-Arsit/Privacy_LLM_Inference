# Linear-boundary additive padding (production Qwen folded path)

This documents the Linear-boundary additive input padding implemented in the
**production Qwen2.5-7B folded-package / H800 worker path** (not only the small
GPT-2 / tiny-decoder validation wrappers). It is wired through
`scripts/build_qwen7b_folded_package.py` (`--linear-boundary-pad`),
`MaskedQwenSession.export_folded_layer_tensors` / `export_folded_head_tensors`,
`src/pllo/deployment/linear_boundary_pad.py`, `src/pllo/deployment/folded_worker.py`
(`_linear` / `apply_folded_layer_prefill` / `apply_folded_layer_decode` /
`apply_folded_head`), and `Qwen7BFoldedPackageGpuBackend`.

## Method (exact invariant)

> The trusted boundary initializes the private input embedding and mask schedule
> once. The untrusted GPU executes all intermediate decoder layers. Linear
> boundaries use additive input padding: the GPU receives `X_tilde = (X − T) N`
> and applies a folded compensation `C_pad = T W N_out` so that the Linear output
> returns to the compatible masked basis `Y N_out`. Pads are boundary-local and
> do not enter RMSNorm, RoPE, softmax, or activation cores. Nonlinear operations
> are handled by compatible right-multiply or permutation masks. The trusted
> boundary is invoked again only for final logits recovery and sampling.

For a Linear `Y = X W + b` (row-vector convention `y = x @ W`):

```
X_tilde = (X − T) N_in            # GPU-visible Linear matmul operand
W_tilde = N_in^{-1} W N_out       # folded operator (unchanged)
b_tilde = b N_out                 # folded bias
C_pad   = T W N_out               # precomputed compensation (= xpad @ W_tilde)
Y_tilde = X_tilde W_tilde + b_tilde + C_pad
        = (X − T) N_in N_in^{-1} W N_out + b N_out + T W N_out
        = (X W + b) N_out = Y N_out.
```

Coverage: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
lm_head` — every Linear family in the production folded path.

### Parameterization in the masked basis

The pad is sampled directly in the masked input basis as `xpad = T N_in` (a
broadcast per-input-channel vector) and the compensation is computed once at
trusted setup as `C_pad = xpad @ W_tilde` (algebraically identical to
`T W N_out`, reusing the already-folded operator). Equivalently, the raw pad is
`T = xpad N_in^{-1}`. This needs neither `N_in` nor `N_out` at the call site and
keeps the raw mask/pad inside trusted setup.

### Boundary-local, not persistent

The pad enters ONLY the Linear matmul operand and is compensated before the
output is consumed by the next compatible GPU-side op. It is **not** persisted in
the residual stream and does **not** enter any nonlinear core:

| stage | additive pad present? |
|---|---|
| Linear matmul input view | **yes** (`(X − T) N_in`) |
| Linear output (after compensation) | no (`Y N_out`) |
| RMSNorm core | no |
| RoPE | no |
| attention softmax | no |
| SiLU / SwiGLU | no |
| residual stream (persistent) | no |

RMSNorm is not translation-invariant (`RMSNorm((H−T)N) ≠ RMSNorm(H) N`), so the
additive pad is never pushed through it. Nonlinear islands keep their existing
compatible masks: residual signed-permutation `N_res`, per-head Q/K/V masks
satisfying `Q_tilde K_tilde^T = Q K^T`, and the SwiGLU channel permutation
(`SiLU(Gate P) * (Up P) = (SiLU(Gate) * Up) P`).

## Compensation engineering (no online matmul)

`C_pad` is **precomputed at trusted setup** (one vector-matrix product per
module, reusing `W_tilde`) and stored as a folded shard tensor. At decode the
runtime cost is a fused broadcast subtract (`x − xpad`) + add (`+ C_pad`) — no
extra dense matmul:

```
c_pad_materialization      = precomputed
c_pad_runtime_cost         = fused_add_or_slice_add
online_extra_matmul_for_pad = 0
```

## GPU visibility (security)

GPU-visible artifacts are only the folded/composed offsets — never raw secrets:

| visible to GPU | hidden (trusted-only) |
|---|---|
| `*_xpad_tilde` (= `T N_in`) | raw pad `T` |
| `*_cpad_tilde` (= `T W N_out`) | raw masks `N_in`, `N_out`, `N_res` |
| `*_tilde` folded weights/biases | plaintext `X`, input ids, recovered logits |

```
raw_pad_visible_to_gpu  = false
raw_mask_visible_to_gpu = false
c_pad_visible_to_gpu    = true   # a folded compensation offset, not a secret
```

From `xpad = T N_in` and `W_tilde = N_in^{-1} W N_out` the GPU can form
`xpad @ W_tilde = C_pad`, but cannot recover `T`, `N_in`, or `N_out` (it never
holds a mask). The package writer rejects any tensor name resembling a mask /
pad-secret / plaintext / raw-LoRA artifact, so `xpad`/`cpad` are admitted only as
`*_tilde` composed offsets.

## Audit fields

Emitted by the build report, the worker `describe()` / health, and the decode
probe JSON (read back from the LOADED package, so they reflect what the worker
actually executes — not a build-time flag):

```json
{
  "qwen_production_path_uses_linear_input_pad": true,
  "linear_boundary_pad_enabled": true,
  "linear_input_form": "(X - T) N_in",
  "linear_output_form": "Y N_out",
  "linear_pad_compensation_formula": "C_pad = T W N_out",
  "pad_scope": "linear_boundary_local",
  "persistent_residual_additive_pad": false,
  "nonlinear_masking_mode": "compatible_right_multiply_or_permutation",
  "pad_enters_rmsnorm_core": false,
  "pad_enters_rope_core": false,
  "pad_enters_softmax": false,
  "pad_enters_swiglu_core": false,
  "intermediate_tee_boundary_calls_per_layer": 0,
  "semantic_input_boundary_calls": 1,
  "semantic_final_logits_boundary_calls": 1,
  "raw_pad_visible_to_gpu": false,
  "raw_mask_visible_to_gpu": false,
  "c_pad_visible_to_gpu": true,
  "online_extra_matmul_for_pad": 0,
  "linear_pad_coverage": {
    "q_proj": true, "k_proj": true, "v_proj": true, "o_proj": true,
    "gate_proj": true, "up_proj": true, "down_proj": true, "lm_head": true
  }
}
```

If any module is uncovered, the build report sets `paper_ready: false` with a
`paper_ready_blocker` naming the uncovered modules.

## Limitation (honest scope)

> We do not claim that arbitrary dense affine masks commute with nonlinear
> operations. Additive pads are used only at Linear boundaries and are compensated
> before nonlinear cores. Nonlinear islands rely on compatible mask families such
> as permutations, signed permutations, or Q/K/V masks satisfying the attention
> invariants.

The additive pad obfuscates the **Linear matmul operand view** the GPU consumes;
because it is compensated, the recovered logits and generated tokens are
identical to the mask-only folded path (verified to plaintext within fp
tolerance, exact token match). It is therefore a defense-in-depth perturbation of
the Linear input distribution, not a change to the output basis or a persistent
residual transformation. Per-token / structured pads can be sliced or fused into
the bias epilogue; the current build uses a broadcast per-channel pad fused as a
constant offset.

## Validation

- `tests/test_qwen_linear_boundary_pad.py` — algebraic correctness (multi-shape,
  ±bias), no-pad-in-nonlinear-core, no-intermediate-TEE, production audit,
  per-module coverage, no-raw-secret, recovered-logits + token-exact correctness,
  and a mask-only-vs-pad output-equivalence check.
- Regression: `tests/test_folded_package_remote_lite.py`,
  `tests/test_folded_package_remote_exec.py`,
  `tests/test_qwen7b_folded_package_1layer_probe.py`,
  `tests/test_lora_folded_e6.py` (folded-LoRA reuses the base folded path).
- Evidence: `outputs/linear_boundary_pad/{build_pad.json, probe_pad.json,
  probe_pad.md, token_correctness.json}` (dry-run tiny model; real Qwen2.5-7B
  uses the same functions with `--model-path` on the H800).
