# Leakage-Aware Obfuscated Execution for Public-Weight LLM Inference and LoRA Adaptation

This repository is a CPU-local research prototype that explores
mask-compatible inference and LoRA adaptation for public-weight
large language models. Base model weights are public; user inputs,
hidden states, KV cache contents, LoRA adapters, gradients, and
logits are the *protected* objects. A simulated cloud accelerator
only sees masked tensors; a trusted side controls the masks, the
loss / logit boundary, output recovery, and audit. The artifact is
intended as a paper-grade *algebraic correctness + leakage-aware*
study, not a TEE or GPU deployment: every claim is graded under the
vocabulary documented in
[docs/PAPER_EVALUATION_MAP.md](docs/PAPER_EVALUATION_MAP.md).

The original engineering-by-stage changelog (Stages 2 through 7.6 +
5.7 + 5.8) has been moved to
[docs/STAGE_ARTIFACTS.md](docs/STAGE_ARTIFACTS.md). This page is
the project entry point; that document is the artifact reference.


## What this repository demonstrates

### 1. Mask-compatible inference for public-weight LLMs

Right-action masks and boundary pads commute with the linear layers,
attention scores, and KV-cache append used in GPT-2, BERT, T5, and
modern decoder-only models (LLaMA / TinyLlama / Qwen-shaped blocks).
The trusted side samples per-call orthogonal masks `N_x`, `N_y`,
per-head masks satisfying `N_Q N_K^T = I`, and post-RoPE per-head
masks; the accelerator only sees masked queries, keys, values, and
cache appends. RMSNorm `γ` is folded into adjacent linears so the
norm core runs in the orthogonal residual mask space. GQA / MQA work
by tying the per-K/V mask across the query heads that read it.

### 2. Leakage-aware nonlinear execution

The compatible SwiGLU island runs SiLU with per-call fresh masks
`N_in`, `Pi`, `N_out`. This preserves per-call correctness exactly,
but it also preserves per-row L1 / L2 / Linf norms, the sorted
row-multiset, and row-wise quantiles — the
**permutation-invariant statistics theorem**. Stage 5.7 measures
this directly and grades it conservatively. Stage 5.8 follows up
with a *cost proxy only* comparison against a hypothetical lookup-
style SwiGLU, motivating lookup-style protection as a future
strong-security direction rather than the default lightweight
runtime path. No secure lookup, MPC, FHE, garbled circuit, Tabula,
or FLUTE primitive is implemented.

### 3. Masked-gradient LoRA adaptation

`A_tilde = N_x^T A M`, `B_tilde = M^T B N_y`, `X_tilde = X N_x`,
`Y_tilde = X_tilde A_tilde B_tilde = X A B N_y`. The GPU never
receives plaintext `A`, `B`, plaintext LoRA gradients, masks, or
plaintext optimizer state. Masked SGD and masked momentum SGD are
algebraically equivalent to their plaintext counterparts under
trusted-side recovery (lockstep float64 verification at machine
precision in
[outputs/masked_gradient_lora_training.md](outputs/masked_gradient_lora_training.md)).
**AdamW under dense masks is unsupported** because coordinate-wise
second moments are not invariant under dense orthogonal mixing —
the module raises `DenseMaskedAdamWUnsupported` rather than
silently approximating. Cancellation-padded rank
(`A_pad = [A_real, R, -R]`, `B_pad = vstack(B_real, S, S)`) hides
`true_rank` from tensor shape; `padded_rank` remains visible.


## Security and claim scope

| Claim | Status | Evidence | Limitation |
|---|---|---|---|
| Linear / attention / KV-cache correctness | algebraically proven | [outputs/ours_runtime_api_validation.json](outputs/ours_runtime_api_validation.json), [outputs/attention_experiments.md](outputs/attention_experiments.md) | Correctness only; not a security claim |
| Modern decoder correctness (RMSNorm, post-RoPE masking, GQA, full forward / prefill / decode_step / greedy) | algebraically proven | [outputs/modern_decoder_model_wrapper_smoke.md](outputs/modern_decoder_model_wrapper_smoke.md) | Greedy generation only; no beam / top-k / top-p; no real model loading by default |
| Compatible SwiGLU island correctness per call | algebraically proven | [outputs/modern_decoder_model_wrapper_smoke.md](outputs/modern_decoder_model_wrapper_smoke.md) | Per-call only; see permutation-invariant leakage below |
| Permutation-invariant leakage (compatible island preserves row-wise norms / sorted multiset / quantiles) | algebraically proven + experimentally validated | [outputs/permutation_invariant_leakage.md](outputs/permutation_invariant_leakage.md) | Channel-index hiding, not value hiding; both Stage 5.3e mitigation bundles share this leakage |
| Named-attacker risk (linear / MLP inverter, signature permutation recovery, linkability) | proxy evaluated | [outputs/real_activation_attacks.md](outputs/real_activation_attacks.md), [outputs/real_token_activation_attacks.md](outputs/real_token_activation_attacks.md), [outputs/stronger_attackers.md](outputs/stronger_attackers.md) | Synthetic-by-default; real tokenizer / model loading is opt-in |
| Lookup nonlinear cost proxy (table size `2^(2b)`, online bandwidth) | cost proxy only | [outputs/lookup_nonlinear_cost_proxy.md](outputs/lookup_nonlinear_cost_proxy.md) | No secure lookup, MPC, FHE, Tabula, or FLUTE primitive is implemented |
| Masked-gradient LoRA SGD / momentum SGD correctness | algebraically proven | [outputs/masked_gradient_lora_training.md](outputs/masked_gradient_lora_training.md) | MSE loss only; cross-entropy / softmax loss would need a trusted loss boundary |
| GPU-visible LoRA leakage (true-rank, real-vs-dummy subspace, cross-step linkability) | proxy evaluated | [outputs/masked_gradient_lora_security_proxy.md](outputs/masked_gradient_lora_security_proxy.md) | Conservative labels: `low_proxy_risk` / `medium_proxy_risk` / `high_proxy_risk` / `needs_more_evaluation` |
| AdamW under dense masks | unsupported | [outputs/masked_gradient_lora_training.md](outputs/masked_gradient_lora_training.md) (`adamw_dense_mask_unsupported.status = "explicitly_raised_as_designed"`) | Coordinate-wise second moments not invariant under dense orthogonal mixing |
| Real TEE or GPU wall-time benchmark | unsupported | — | `wall_time_source = "projected_from_op_counts"`; only local CPU emulation in [paper_results/markdown/measured_runtime.md](paper_results/markdown/measured_runtime.md) |
| Formal / cryptographic / semantic security | unsupported | [outputs/stage_7_6_claims_consistency.md](outputs/stage_7_6_claims_consistency.md), [outputs/paper_claims_audit_v2.md](outputs/paper_claims_audit_v2.md) | `formal_security_claim = False` in every artifact that records it |
| Hardware side-channel security (cache / power / EM) | unsupported | — | Out of scope |
| Full Qwen / TinyLlama / LLaMA PEFT fine-tuning | unsupported | — | Synthetic regression tile only; PEFT / DeepSpeed / vLLM / FlashAttention not integrated |

The grading vocabulary is defined in
[docs/PAPER_EVALUATION_MAP.md](docs/PAPER_EVALUATION_MAP.md) and the
detailed theorem-to-artifact map is in
[paper_results/markdown/security_claims_table.md](paper_results/markdown/security_claims_table.md).


## Quick start

```bash
pip install -e ".[dev]"

# Full test suite (synthetic-by-default, no network, no GPU)
python -m pytest -q

# Key experiments
python scripts/run_permutation_invariant_leakage.py
python scripts/run_lookup_nonlinear_cost_proxy.py
python scripts/run_masked_gradient_lora_training.py

# Paper-side audit (lifecycle + claims consistency)
python scripts/run_masked_gradient_lora_training.py  # also emits the lifecycle + claims-consistency reports
```

Synthetic-by-default; real HuggingFace tokenizer / model loading is
opt-in via per-script `--attempt-tokenizer-load` /
`--attempt-real-model-load` flags. The remaining historical scripts
(GPT-2 wrappers, BERT / T5 probes, every Stage 7.x experiment) are
catalogued in [docs/STAGE_ARTIFACTS.md](docs/STAGE_ARTIFACTS.md).


## Key results

| Metric | Value |
|---|---|
| `python -m pytest -q` | 1307 passed / 4 skipped / 0 failed (latest local run) |
| Masked-gradient LoRA float64 error envelope | forward `≤ 4.66e-15`, loss `≤ 1.11e-16`, grad relations `≤ 2.52e-15`, SGD / momentum recovery `≤ 1.78e-15` |
| Lookup table sizes | `b=4` → 256 entries / 512 B, `b=6` → 4 096 / 8 KiB, `b=8` → 65 536 / 128 KiB |
| `formal_security_claim` | `False` everywhere it is recorded |
| `cryptographic_lookup_implemented` | `False` |
| AdamW under dense masks | unsupported (raises `DenseMaskedAdamWUnsupported`) |
| Default `nonlinear_mode` / `mitigation_bundle` / `inter_block_mask_mode` / `constant_time_decode_mode` | `"trusted"` / `"fresh_perm_only"` / `"plain_boundary"` / `"off"` (unchanged) |
| `paper_claims_audit` totals | 8 supported / 5 proxy_supported / 8 unsupported |

Detailed numerical envelopes per stage are in the corresponding
output `.md` files under [outputs/](outputs/) and the paper-side
roll-up in [paper_results/summary.md](paper_results/summary.md).


## Repository layout

```
src/pllo/ops/             # Core primitives (mask / pad / LoRA / nonlinear island / masked-gradient LoRA)
src/pllo/experiments/     # Probes, attackers, audits, cost models, paper-side aggregators
src/pllo/hf_wrappers/     # GPT-2 / modern-decoder block + model wrappers (synthetic-by-default)
src/pllo/runtime/         # TrustedController + AcceleratorBackend protocol + LocalCPUBackend
src/pllo/baselines/       # Direct prior-work primitive skeletons (Slalom / Amulet / DarKnight / ...)
scripts/                  # Runner scripts (one per experiment)
tests/                    # pytest suite (synthetic-by-default; no network; no GPU required)
outputs/                  # JSON / CSV / Markdown artifacts (no raw tensors)
docs/                     # PAPER_THEORY_OUTLINE, PAPER_EVALUATION_MAP, STAGE_ARTIFACTS, ...
paper_results/            # Paper-side CSV / Markdown / LaTeX summary tables
paper_draft/              # Paper draft sources (sections + LaTeX skeleton)
```


## Trusted-vs-untrusted boundary

`SimulatedTEE` is a Python simulation of trusted-side execution. It
holds plaintext inputs, one-time pads, masks, mask inverses, LoRA
adapters, optimizer state, compensation terms, and output recovery
state. `UntrustedGPUExecutor` only receives obfuscated inputs,
masked weights / adapters, `bias_tilde`, and compensation tensors.
The boundary is enforced by `src/pllo/runtime/TrustedController` +
`LocalCPUBackend` (the only concrete backend in this artifact); the
LoRA training-to-inference visibility lifecycle is enumerated in
[outputs/lora_training_inference_lifecycle.md](outputs/lora_training_inference_lifecycle.md).

This prototype validates algebraic correctness and proxy-evaluated
leakage only. It does not provide real security isolation,
side-channel resistance, memory isolation, attestation, or
production TEE guarantees.


## Limitations

- No real TEE isolation. `wall_time_source = "projected_from_op_counts"`; `full_runtime_integrated = False`. CPU-local emulation only.
- No real GPU wall-time benchmark. The "GPU" is a simulated cloud accelerator; the only concrete backend is `LocalCPUBackend`.
- No hardware side-channel evaluation (cache / power / EM).
- No formal, cryptographic, or semantic security is claimed.
- The compatible nonlinear island preserves permutation-invariant activation statistics (row norms, sorted multiset, quantiles). The Stage 5.3e `fresh_perm_plus_sandwich_plus_pad` bundle changes freshness / boundary / temporal posture but not single-shot value-level multiset visibility — see [outputs/permutation_invariant_leakage.md](outputs/permutation_invariant_leakage.md).
- The Stage 5.8 lookup experiment is a *cost proxy only*. No secure lookup, MPC, FHE, garbled circuit, Tabula, or FLUTE primitive is implemented. `cryptographic_lookup_implemented = False`.
- AdamW under dense orthogonal masks is unsupported; the module raises `DenseMaskedAdamWUnsupported`. Trusted-assisted AdamW, signed-permutation masks, or a specialised masked optimiser are deferred future work.
- Full Qwen / TinyLlama / LLaMA PEFT fine-tuning is not implemented. Stage 7.x exercises a synthetic LoRA tile; PEFT / DeepSpeed / vLLM / FlashAttention are not integrated.
- `padded_rank` itself is visible from tensor shape; only `true_rank` is hidden via cancellation padding. Heterogeneous `padded_rank` across modules is deferred.
- Loss and optimizer remain on the trusted side. The MSE loss boundary used by Stage 7.6 is preserved by orthogonal `N_y` exactly; cross-entropy / softmax loss would need a trusted loss boundary.
- Raw tensors, masks, permutations, adapters, gradients, and private data are NEVER exported. JSON / CSV / Markdown carry only summary scalars, shapes, and short SHA-256 fingerprints.
- No default mode is flipped by any stage: `nonlinear_mode = "trusted"`, `mitigation_bundle = "fresh_perm_only"`, `inter_block_mask_mode = "plain_boundary"`, `constant_time_decode_mode = "off"`, `implemented = False`, `security_profile = "proxy-evaluated, not formal"`.


## Citation and paper status

Paper draft in progress. Theory outline:
[docs/PAPER_THEORY_OUTLINE.md](docs/PAPER_THEORY_OUTLINE.md).
Evaluation map: [docs/PAPER_EVALUATION_MAP.md](docs/PAPER_EVALUATION_MAP.md).
Consolidated security-claims table:
[paper_results/markdown/security_claims_table.md](paper_results/markdown/security_claims_table.md).
Paper draft sources are under [paper_draft/](paper_draft/) and
paper-side summary tables under [paper_results/](paper_results/).


## Development notes

- Tests are synthetic-by-default and require no network and no GPU. Real HuggingFace tokenizer / model loading is opt-in per script.
- `python -m pytest -q` runs the full suite. The Stage 7.6 claims-consistency scanner (lexical audit of `paper_draft/`, `paper_results/`, `outputs/`, `docs/`, `README.md`) is exercised by [tests/test_lora_training_inference_lifecycle.py](tests/test_lora_training_inference_lifecycle.py) and currently records `total_unsafe_wording_present = 0`, `passes_consistency_check = True`.
- The detailed per-stage record (every Stage X.Y block, output filenames, additive `workload_profiler` / `cross_architecture_summary` metadata, explicit non-goals) is preserved in [docs/STAGE_ARTIFACTS.md](docs/STAGE_ARTIFACTS.md).
- New experiments should publish only summary scalars, shapes, and short fingerprints. Raw tensors / masks / adapters / gradients / optimizer state must never reach JSON / CSV / Markdown.
