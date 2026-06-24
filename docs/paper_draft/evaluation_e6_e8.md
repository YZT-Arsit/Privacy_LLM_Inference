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

## Real-run orchestration (H800 / TDX)

The expensive real runs are made cheap + reproducible by a one-command pipeline
and a set of helpers (all dry-run validated; no real H800/TDX run was performed):

| tool | role |
| --- | --- |
| `scripts/run_e6_lora_real_h800_pipeline.py` | builds + verifies the folded-LoRA package, runs the H800 local + remote LoRA probes (start-or-check the worker), optionally emits TDX-lite replay inputs, and writes ONE consolidated JSON/MD (`--plan-only` prints the exact command plan first) |
| `scripts/create_tiny_hf_lora_fixture.py` | a tiny PEFT-style adapter fixture so the real `load_hf_lora_adapter` path is exercised without a download |
| `scripts/validate_lora_effect.py` | guards against a silently-unmerged adapter (LoRA tokens identical to no-LoRA ⇒ warning) |
| `scripts/prepare_tdx_lora_lite_inputs.py` | turns an H800 reference into TDX-lite replay `input_ids` + expected tokens + artifact hashes + `run_tdx_lora_lite_decode.sh` (no model / base package / raw LoRA on the TDX side) |
| `scripts/check_tdx_measurement_coverage.py` | fails if the boundary import closure includes a trusted-side module not measured into the runtime hash |
| `scripts/package_final_artifacts.py` | bundles all no-LoRA + LoRA outputs, runtime-hash/evidence, and package manifests into one tarball with per-file hashes |

The TDX-lite boundary surface (`folded_probe_common`, `embedding_artifact`,
`causal_lm_boundaries`, `remote`, `wire`, `nonlinear_islands`,
`mitigation_bundles`) is now measured into the attestation runtime hash; the
private LoRA adds **no** new boundary file (folding/merging runs on the worker).
The full server procedure is in
[`docs/runbooks/REAL_H800_TDX_LORA_RUNBOOK.md`](../runbooks/REAL_H800_TDX_LORA_RUNBOOK.md).

The real HF/PEFT adapter is built with
`build_qwen7b_lora_folded_package.py --raw-lora-adapter-path <dir>
--adapter-format hf_peft` (rank/alpha/target_modules are read from
`adapter_config.json`).

## Examples + testing

`outputs/examples_e6_e8/` holds generated example artifacts (local probe, training
probe, E8 report). Tests (no H800/TDX/CUDA/checkpoint): `tests/test_lora_folded_e6.py`
(folding math, build/verify, local probe, remote live-worker decode, no-LoRA
backward compat) and `tests/test_lora_private_training_e7_e8.py` (E7 update + audit,
E8 tables + missing-not-assumed + leak-flips-cross-check),
`tests/test_lora_hf_adapter_fixture.py` (real PEFT-layout loader + folding vs
reference), and `tests/test_e6_lora_real_pipeline.py` (pipeline command plan,
validate_lora_effect, TDX-lite prep, measurement-coverage gap detection, artifact
packaging with missing inputs). All new trusted-side LoRA scripts/modules are in
the Python-3.6 API scan. No real H800/TDX runs were performed; real runs are
deferred (see the runbook).
