# Evaluation E6–E8 — private LoRA (methodology + reproduction)

Extends the no-LoRA folded-package deployment with **private LoRA**: the base
model `W` is public, the user's LoRA adapter and training data are private, and
the untrusted GPU may hold only the public base folded package + **folded LoRA
operators** + masked activations/KV + public metadata. The trusted boundary (TDX
guest) holds input ids, the embedding artifact, mask secrets, the raw LoRA
adapter, optimizer state, labels/training data, and does logit recovery/sampling.

The no-LoRA path is unchanged and backward compatible: a worker started without a
folded-LoRA package reports `lora_enabled=False` and behaves exactly as before
(covered by `tests/test_lora_folded_e6.py::test_no_lora_path_unchanged`).

## LoRA folding rule (E6)

Row-vector convention, weight `W` is `[in, out]`. The base fold of module `m` is
`W_tilde = L_m (rms_m * W) R_m` (left op `L_m` = input residual-mask inverse /
value-mask inverse / SwiGLU row-perm; right op `R_m` = per-head / SwiGLU-col /
residual output mask, incl. the q/k RoPE alignment) — mirroring
`fold_layer_attention_and_up` + the folded down projection.

A LoRA branch is `ΔW = (alpha/r) · A B`, `A [in,r]`, `B [r,out]`. Folding is
linear and **factors through the rank**: `L_m (A B) R_m = (L_m A)(B R_m)`. So the
package stores low-rank folded operators

    a_tilde = (L_m A) @ Rk            # [in, r]
    b_tilde = (alpha/r) · Rk^{-1} @ (B R_m)   # [r, out]

where `Rk` is a private per-(layer, module) **rank mask** (signed permutation,
`Rk^{-1}=Rk^T`). `Rk` cancels in the product (`a_tilde @ b_tilde == folded ΔW`)
but masks the rank-`r` bottleneck. The worker merges `W_tilde += a_tilde @ b_tilde`
and runs the **existing** masked kernels unchanged.

Security: the GPU sees only `a_tilde`/`b_tilde` (folded with the `N` masks + `Rk`);
recovering raw `A`/`B` needs `N`, `rms`, `Rk` — all trusted. Validated to
`max_abs_error ≈ 3e-8` across all 7 modules
(`tests/test_lora_folded_e6.py::test_folding_matches_reference_all_modules`).

### E6 components

| stage | code | example |
| --- | --- | --- |
| builder | `scripts/build_qwen7b_lora_folded_package.py`, `src/pllo/deployment/lora_folded_package.py` | folds adapter -> shards + `manifest.json` + `lora_meta.json` (rank/alpha/target_modules/adapter_hash/base_package_manifest_hash, `contains_*=False`, `trusted_setup=True`) |
| verifier | `scripts/verify_qwen7b_lora_folded_package.py` | shard hashes, no forbidden / raw-LoRA tensor names, no optimizer/training/mask secrets, target coverage, base-manifest compatibility |
| local probe | `scripts/run_qwen7b_lora_folded_local_probe.py` | base+raw-LoRA (trusted) vs base-folded+folded-LoRA (package path): allclose/errors/top-1/top-k/next-token/tokens_exact_match + security flags |
| remote probe | `scripts/run_qwen7b_lora_folded_remote_decode_probe.py` | TDX-lite boundary + worker with base+LoRA packages; correctness vs expected ids |
| attested | `scripts/run_tee_gpu_protocol_demo.py` (folded branch attaches attestation when `--attestation-evidence` is given) | folded-LoRA metadata is covered by the attested run |

Worker: start with `--folded-lora-package-path`; the boundary side is unchanged
(it sends only masked embeddings + public metadata). Example commands are in each
script's module docstring.

```
# H800 worker (base + private LoRA)
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package \
  --folded-package-path /root/.../qwen7b_folded_full \
  --folded-lora-package-path /root/.../qwen7b_lora_folded \
  --device cuda --dtype bfloat16 --audit true
```

## E7 — minimal private LoRA update prototype

`scripts/run_lora_private_training_tiny_probe.py`,
`src/pllo/training/lora_private_trainer.py` (numpy only). A tiny synthetic linear
task (configurable independent "target modules"): the trusted side owns inputs,
labels, raw `A`/`B`, gradients (clipped), optimizer step, and loss; the GPU does
only the **masked frozen-base matmul** (`X_tilde = X @ N_in` → `(X@W) @ N_out`,
folded base public). The recorded GPU trace is audited with the existing
`pllo.protocol.lora_training_audit.audit_lora_training_trace`, confirming no raw
`A`/`B`, data, labels, gradients, optimizer state, or mask secrets crossed.
Reports `loss_before/after`, `loss_decreased`, `adapter_delta_norm`, the security
flags, and explicit `limitations`. **This is a minimal prototype, not full
fine-tuning** (see `limitations`): only the frozen-base matmul is offloaded; a
fully GPU-offloaded masked backward is future work.

## E8 — private LoRA final report

`scripts/run_e8_lora_final_report.py`, `src/pllo/experiments/e8_lora_report.py`
(pure parsing). Four tables: (1) inference correctness (local / remote / attested),
(2) security matrix (raw adapter A/B, folded adapter, optimizer state, gradients,
training data, labels) cross-checked against the audit results in the provided
outputs, (3) cost (folded-LoRA size, setup time, decode latency overhead vs
no-LoRA, memory overhead), (4) training prototype (loss before/after, update
correctness, limitations). Missing inputs are reported as not-provided.

## Examples + testing

`outputs/examples_e6_e8/` holds generated example artifacts (local probe, training
probe, E8 report). Tests (no H800/TDX/CUDA/checkpoint): `tests/test_lora_folded_e6.py`
(folding math, build/verify, local probe, remote live-worker decode, no-LoRA
backward compat) and `tests/test_lora_private_training_e7_e8.py` (E7 update + audit,
E8 tables + missing-not-assumed + leak-flips-cross-check). All new trusted-side
LoRA scripts/modules are in the Python-3.6 API scan. No real H800/TDX runs were
performed; real runs are deferred.
