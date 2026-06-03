# Figures

This appendix lists the canonical figures referenced in the paper body. Where an existing figure already exists under `paper_results/figures/`, we point at the path; figures that still need manual illustration are marked `TODO: draw`.

## Figure 1 — System overview

- **Description.** A block diagram of the trusted-side controller (top, in a TEE-shaped box) and the untrusted GPU (bottom). Arrows for: prompt-in to controller; per-call mask sampling inside controller; masked tensors `(X_tilde, W_tilde, A_tilde, B_tilde, K_tilde, V_tilde)` from controller to GPU; masked output `Y_tilde` and masked gradients `dA_tilde, dB_tilde` back to controller; sampler/optimizer inside controller; un-masked token-out to user.
- **Status.** `TODO: draw figure` (no existing PNG).
- **Caption draft.** *System overview. The trusted-side controller (locally emulated, not a real TEE) samples masks, holds the LoRA adapter and optimizer state, and dispatches masked dense matmuls to the untrusted GPU. The GPU never sees plain activations, plain adapter factors, or plain gradients.*

## Figure 2 — Right-masked generation and KV cache

- **Description.** A schematic of one decode_step: the residual stream at position `t` is masked with `N_in`, dispatched through the masked Linear → masked island → masked Linear pipeline, the per-head `(K_t N_K, V_t N_V)` are appended to the masked KV cache, and the recovered logits are sampled at the controller.
- **Status.** `TODO: draw figure`.
- **Caption draft.** *Right-masked decode with KV-cache append. Right masks commute with token-axis concatenation, so the masked cache append is a no-op rewriter; the attention identity `Q_tilde K_tilde^T = Q K^T` is preserved by `N_Q N_K^T = I`.*

## Figure 3 — Operator-compatible nonlinear island

- **Description.** Three side-by-side sub-panels: (a) pointwise activation island with paired permutation `P`; (b) SwiGLU paired-permutation island; (c) RMSNorm orthogonal-mask island. Each shows the input `X_tilde`, the inner masked operator, and the output `Y_tilde`, with the operator's commutation identity below.
- **Status.** `TODO: draw figure`.
- **Caption draft.** *Operator-compatible nonlinear islands. The mask family is restricted to the smallest invariance group of the inner operator: permutation for elementwise activations and SwiGLU, orthogonal for RMSNorm, mean-preserving orthogonal for LayerNorm.*

## Figure 4 — Dense sandwich and pad compensation

- **Description.** A linear → island → linear stack with: outer dense `N_in / N_out` boundaries, inner `P / orthogonal` island mask, boundary pad `T`, and the compensation term `C_T = T W N_out` returning to the trusted side.
- **Status.** `TODO: draw figure`. (`paper_results/figures/boundary_call_reduction.png` is the quantitative counterpart but is not the schematic.)
- **Caption draft.** *Dense Linear sandwich and boundary pad. The island operator commutes with a restricted mask family; the surrounding dense Linear boundaries upgrade the GPU-visible mask to a dense invertible transformation; the boundary pad `T` randomizes the centered hidden state at the island boundary, with `C_T = T W N_out` absorbed in the trusted compensation.*

## Figure 5 — LoRA private training path

- **Description.** A two-panel figure: (a) forward `Y_tilde = X_tilde W_tilde + (alpha/r) X_tilde A_tilde B_tilde + C_T + b N_out`; (b) backward `dA_tilde, dB_tilde` derived from `G_tilde = G N_out^{-T}`. Optimizer and loss inside the trusted-side box; adapter never merged into `W`.
- **Status.** `TODO: draw figure`. (`paper_results/figures/lora_training_errors.png` is the quantitative correctness counterpart but is not the schematic.)
- **Caption draft.** *Private LoRA training path. The adapter factors `(A, B)` are masked with a paired inner mask `U`; per-step gradients are masked in the same right-masked space; the optimizer state and loss closure remain trusted-side; the adapter is never merged into the public base weight `W`.*

## Figure 6 — Security risk matrix

- **Description.** A heat-map of attacker × mitigation bundle × risk level for the 14 rows in `security_proxy_summary.md`.
- **Status.** Generated. Path: `paper_results/figures/security_risk_matrix.png`.
- **Caption draft.** *Security risk matrix across the 14 proxy-attacker rows. `low` cells are bounded close to random chance in our tests under the named attacker; `needs_more_evaluation` cells are open under stronger attackers; `medium` / `high` cells are reported as residual leakage. Every row is `proxy-supported only`.*

## Figure 7 — Runtime summary

- **Description.** A bar chart of the six measured runtime primitives in `measured_runtime.md` plus the opt-in `modern_decoder_model_wrapper` placeholder.
- **Status.** Generated. Path: `paper_results/figures/measured_runtime_summary.png`.
- **Caption draft.** *Local-emulation runtime per primitive (this is **not** real TEE wall-time). `num_warmup = 2`, `num_repeats = 5`, `device = cpu`, `dtype = float64`, `wall_time_source = measured_local_emulation`.*

## Auxiliary figures already generated

The Stage 7.5 figure pipeline emits four further auxiliary plots that are useful for the evaluation section:

- `paper_results/figures/correctness_error_summary.png` — per-component correctness error.
- `paper_results/figures/boundary_call_reduction.png` — boundary-call counts across the workload-summary methods.
- `paper_results/figures/lora_training_errors.png` — Stage 7.0 → 7.4 LoRA training errors.
- `paper_results/figures/rank_inference_risk.png` — rank-inference risk across detectors.
- `paper_results/figures/timing_proxy_before_after.png` — cost-model timing classifier before vs after `proxy_equalized`.

These are referenced inline from `paper_draft/evaluation.md`.

## Figures that need manual illustration in Stage 7.6b

- Figure 1: System overview.
- Figure 2: Right-masked decode + KV cache.
- Figure 3: Nonlinear-island sub-panels.
- Figure 4: Dense sandwich + pad-compensation schematic.
- Figure 5: LoRA private training-path schematic.

(Figures 6 and 7 are already in `paper_results/figures/` and only need captioning.)
