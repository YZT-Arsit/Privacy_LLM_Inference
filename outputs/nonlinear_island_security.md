# Privacy LLM Obfuscation — Nonlinear Island Security Proxies (Stage 5.2b)

## Experiment scope

Three lightweight proxies over the operator-compatible mask scheme used by the Stage 5.2a nonlinear islands. None of these constitute a formal security proof — each is a naive-observer upper bound, recorded so the paper's security section can quote it directly.

## Threat Model for Proxy Experiments

Naive-observer adversary: sees only GPU-visible tensors produced by the masked forward (post-mask Linear inputs / outputs and the activation island's permuted input). Has prior knowledge of the plaintext channel distribution but does NOT execute adaptive or learned inversion attacks, does NOT observe trusted-side tensors, and does NOT use side channels.

## Permutation Recovery Proxy

Channel signature ``(mean, std, median, q25, q75, mean_abs)`` over a synthetic activation distribution with per-channel mean offset and scale, matched to the plaintext reference signature by greedy cosine nearest-neighbour. Random-chance top-1 baseline is ``1 / hidden_size = 0.0156``.

| strategy | top-1 | top-5 | mean rank | sig error | risk level |
|---|---|---|---|---|---|
| fixed_permutation | 0.250000 | 0.812500 | 2.516e+00 | 0.017082 | high (stable across sessions; aggregate signature is sharp) |
| permutation_pool | 0.218750 | 0.675781 | 4.176e+00 | 0.027294 | moderate-to-high (per-perm signature is sharper than fresh but weaker than fixed) |
| fresh_permutation_per_session | 0.100586 | 0.474609 | 7.194e+00 | 0.050772 | moderate (per-session recovery only; no cross-session alignment) |
| dense_sandwich_reference | 0.020508 | 0.086914 | 3.065e+01 | 0.238610 | near random chance (column statistics destroyed by dense mixing) |

- Recovery ranking (high → low): fixed_permutation > permutation_pool > fresh_permutation_per_session > dense_sandwich_reference

## Island Linkability Proxy

Same plaintext input is run repeatedly through each strategy; the pairwise cosine and L2 distance of the GPU-visible tensor across trials is reported below. Dual-view strategies record both ``boundary_input_visible`` (post-pad-and-mask Linear input) and ``activation_input_visible`` (``Z P``, no pad) — see notes.

| strategy | view | mean cos | mean L2 | expected linkability |
|---|---|---|---|---|
| fixed_perm_no_pad | activation_input_visible | 1.000e+00 | 0 | highest |
| fixed_perm_with_linear_boundary_pad | boundary_input_visible | -4.013e-04 | 6.321e+01 | boundary low / activation high |
| fixed_perm_with_linear_boundary_pad | activation_input_visible | 1.000e+00 | 0 | boundary low / activation high |
| fresh_perm_with_linear_boundary_pad | boundary_input_visible | 0.001349 | 6.353e+01 | low on both views |
| fresh_perm_with_linear_boundary_pad | activation_input_visible | 0.024573 | 4.392e+01 | low on both views |
| dense_to_perm_to_dense_sandwich | pre_island_dense_visible | 0.003852 | 4.439e+01 | lowest |
| dense_to_perm_to_dense_sandwich | island_visible | 0.021314 | 4.400e+01 | lowest |
| dense_to_perm_to_dense_sandwich | post_island_dense_visible | 0.002079 | 4.443e+01 | lowest |

- Main-metric linkability ranking (high → low): fixed_perm_no_pad > fixed_perm_with_linear_boundary_pad > fresh_perm_with_linear_boundary_pad > dense_to_perm_to_dense_sandwich
- Activation island permutation preserves the coordinate-value multiset.
- Boundary pad does not protect the activation input when P is fixed.
- Dense sandwich is the strongest of the four under naive observers.

## Mask Family Security Accounting

| mask family | used for | preserved statistics | gpu-visible leakage | mitigation | strength vs dense |
|---|---|---|---|---|---|
| dense_invertible | Linear / Attention / KV cache boundaries | rank, dimension, algebraic relations under same mask | channel identity hidden via dense mixing; algebraic structure preserved only under reuse | fresh mask across sessions; pad at Linear boundaries | baseline (this IS dense) |
| orthogonal | RMSNorm core | row L2 norm (||X N||_2 = ||X||_2); pairwise dot products if the same mask is reused | row L2 norm preserved by design; coordinate identity hidden under rotation | restrict island lifetime; sandwich with dense masks at Linear boundaries | weaker (row L2 norm always preserved) |
| mean_preserving_orthogonal | LayerNorm core | row mean (X N · 1 = X · 1); row centered L2 norm | row mean and centered norm preserved by design | sandwich with dense masks at Linear boundaries; avoid reuse across sessions | weaker (mean + centered norm always preserved) |
| permutation | GELU / ReLU / SiLU activation island | coordinate-value multiset; per-token sorted values | channel identity hidden if P is secret; multiset always leaks | fresh permutation per session; permutation pool; dense sandwich at Linear boundaries | weaker (multiset and sorted values always preserved) |
| paired_permutation | SwiGLU up / gate branches (shared P) | coordinate-value multiset; paired (up, gate) alignment per channel | same as permutation, plus paired alignment exposes that up and gate share P | fresh paired permutation; branch-consistency checks; dense sandwich | weaker (paired multiset always preserved) |

- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Orthogonal masks preserve row norms by design.
- Mean-preserving orthogonal masks preserve row means and centered norms by design.
- Dense sandwiching and pad at Linear boundaries are required mitigations.

## Interpretation

- Permutation recovery: fixed top-1 = **0.250** vs fresh top-1 = **0.101** vs sandwich top-1 = **0.021** (random chance = **0.0156**). Fixed permutation enables cross-session signature accumulation; dense sandwich destroys per-channel statistics.
- Island linkability: `fixed_perm_no_pad` mean cos = **1.0000** vs `fresh_perm_with_linear_boundary_pad` = **0.0246** vs `dense_to_perm_to_dense_sandwich` = **0.0021**. Note: under `fixed_perm_with_linear_boundary_pad`, the boundary view is well-protected (fresh pad / mask), but the activation view ``Z P`` remains fully linkable because P is fixed — pad at the Linear boundary does NOT protect the activation island for a fixed permutation.
- Mask family accounting (Proxy 3) catalogues per-family preserved invariants and leakage — *what is preserved by design*, not what an adversary can or cannot recover beyond that.

## Limitations

- These experiments are security proxies, not formal security proofs.
- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Orthogonal masks preserve row norms by design.
- Mean-preserving orthogonal masks preserve row means and centered norms by design.
- Fresh permutation reduces stable statistical recovery but does not provide dense linear mixing.
- Dense sandwiching and pad at Linear boundaries are required mitigations.
- This stage does not implement adaptive learned inversion attacks.
- This stage does not implement real TEE isolation.
- This stage does not prove semantic security.

## Next Stage Plan

- **Stage 5.2c** — Workload profiler integration: extend the Stage 5.0.1 cost model to count the trusted-side norm / activation operations replaced by the Stage 5.2a islands, and compare the three architectures' boundary-call counts under ``ours_with_islands`` vs ``ours_current``.
- **Stage 5.3** — Wrapper selective integration: replace the trusted LayerNorm / GELU shortcut with the Stage 5.2a islands in the GPT-2 / BERT / T5 wrappers behind a feature flag, gated on the Stage 5.2b linkability + recovery results documented above.
- **Stage 5.4** — Adaptive / learned-inverter attacker that goes beyond per-channel statistics (the Stage 5.2b proxies are naive-observer bounds only).
