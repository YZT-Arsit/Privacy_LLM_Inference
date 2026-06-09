# System Model and Threat Model

## Entities

* **User.** Holds the prompt tokens and (optionally) a private LoRA adapter trained on confidential data. Receives final generated tokens. Trusts the controller.
* **Trusted controller (TEE-like domain).** Holds per-session secrets: the residual-stream orthogonal masks `Q_l`, the per-head KV masks `N_K`, `N_V`, the per-head RoPE-plane block-diagonal rotations `B_Q`, `B_K`, the SwiGLU paired permutation `P`, the LM-head vocab mask `N_vocab`, the per-call boundary masks `M`, the fresh per-call additive pads `T`, and the trusted randomness `ρ` for sampling. Performs trusted-side embedding lookup, per-call mask / pad sampling, boundary-table compilation, logits recovery, and generation-time logit processors (greedy, temperature, top-k, top-p, repetition penalty, stop / bad-words / forced tokens).
* **Untrusted accelerator (GPU-like domain).** Holds the *public base model* weights as transformed boundary tables (`W_tilde = M^{-1} W N`, bias_tilde, `C_linear = T W N`, and per-(layer, head) folded versions for the QKV / o / up / gate / down projections and the LM head). Executes the forward pass on masked tensors. Observes the values it computes through (the accelerator transcript).
* **Public base model.** Architecture, hyperparameters (`L, d, h, h_kv, d_h, V, intermediate_size`), and *weight values* are public. The base model's weights are *not* the protection target.
* **Optional private LoRA adapter.** `A in R^{d_in × r}`, `B in R^{r × d_out}` per insertion site. Adapter values are user-supplied (treated as protection target).

## Public information

The following are *not* protected by the protocol — they may appear in plaintext at the accelerator boundary, in the network transcript, or in the system metadata:

* Base model architecture and hyperparameters (`L, d, h, h_kv, d_h, intermediate_size, V, num_layers`).
* Base model weight values (`W` of every linear projection, `gamma` of every RMSNorm).
* Sequence length policy (prompt length and number of new tokens), unless an additional length-hiding policy is layered on top.
* Sliding-window attention window size policy `w`, when sliding-window mode is in use.
* High-level protocol metadata: number of layers, RoPE base, dtype.
* The fact that a LoRA adapter is in use (the existence of `(A_tilde, B_tilde)` boundary tables is observable).
* The padded inner rank `r_pad` of any LoRA adapter (the *true* rank `r` is hidden by zero-padding inside the trusted side).

## Protection targets

The protocol aims to keep the following tensors *out of accelerator-visible plaintext*, under the leakage budgets declared by the active mode knobs:

* **Tokenised user prompt and embeddings.** The trusted controller does the embedding lookup; the accelerator only ever sees the masked layer-entry tensor `H_hat_0 = H_0 Q_0` (or a per-row variant under chunk / token granularity).
* **Per-layer hidden states.** All `H_l` for `l = 1, ..., L` are observed only as `H_l Q_l` (under sequence granularity) or under per-row variants (`H_hat_{l, i} = H_{l, i} Q_{l, c(i)}`).
* **KV cache contents.** Per session per layer per head, the cache holds `K_tilde = K N_K`, `V_tilde = V N_V`. Cache *length* is public; cache *values* are not.
* **Per-head Q / K / V values inside each attention block.** Under the RoPE-safe mode the qkv-projection output is masked *directly* at the accelerator (`qkv_projection_outputs_masked_directly = true`, `rope_transient_plain_qk_visible = false`).
* **Attention values (the post-softmax `P V` output).** Per Q head, the accelerator sees `attn_out_tilde = attn_out_plain @ N_V[kv_head]`, never `attn_out_plain` itself.
* **Pre-sampling logits.** The accelerator sees `z_tilde = z N_vocab`; the trusted controller recovers `z = z_tilde N_vocab^{-1}` before any sampling decision.
* **Private LoRA adapter contributions.** The accelerator sees `A_tilde = M^{-1} A R`, `B_tilde = R^{-1} B N_out`. The plaintext `A`, `B` are never materialised on the accelerator.
* **Sampling candidate sets** (top-k indices, top-p nucleus, bad-words mask, forced-token id, stop-token id). These are policy inputs to the trusted-side processor and are *not* exposed to the accelerator.
* **Boundary-fingerprint linkability across sessions.** Per-session orthogonal masks make the masked layer-entry tensors fingerprint-distinguishable across sessions even for identical prompts.

## Non-targets

We *do not* aim to protect the following; treating them as "protected" in any paper claim would be unsafe wording:

* **Public base model weights.** Not a protection target. The accelerator legitimately reads transformed weight tables to compute the forward.
* **Output text shown to the user.** The user-facing channel is plaintext by definition; downstream uses of the generated text are out of scope.
* **Output length / stop timing side-channel.** Observable unless the trusted side separately pads or batches. The current protocol does *not* hide it; see the explicit unsupported claim `output_length_side_channel_not_hidden_unless_separately_padded`.
* **Timing and memory-access side channels.** Window-size policy, paged-cache block-table allocator behaviour, page-fault timing, RDMA traffic patterns, and GPU power are not evaluated.
* **Hardware side channels.** No timing / power / electromagnetic / cache side-channel evaluation is performed.
* **Active malicious accelerator.** The probabilistic spot-check prototype (`outputs/integrity_spotcheck.json`) is a proxy, not a verifiable-computation primitive; an active adversary can mount denial-of-service or corrupt UN-checked items.
* **Formal cryptographic / semantic / differential-privacy security.** We do not analyse pad-scale distinguishability, mask reuse, or any indistinguishability game.
* **Compromised TEE.** If the trusted controller leaks its secrets, every protection collapses.
* **Multi-modal positional encodings (M-RoPE), Mixture-of-Experts, Multi-Head Latent Attention, speculative decoding, LoRA training.** Listed as unsupported with explicit remaining blockers.

## Adversary

The primary adversary is an **honest-but-curious accelerator** that observes the accelerator-visible persistent tensors and the protocol transcript at the TEE-accelerator boundary. The adversary

* sees every transformed weight table, every boundary table (`A`, `C_T`, `W_tilde`, `b_tilde`, `C_linear`), and every accelerator-side computation result;
* knows the public hyperparameters and the *fact* of which protocol mode is active;
* does *not* see trusted-side secrets (`Q_l`, `N_K`, `N_V`, `B_Q`, `B_K`, `P`, `N_vocab`), trusted randomness `ρ`, or recovered logits.

An **active tampering adversary** is considered only via the probabilistic spot-check (`outputs/integrity_spotcheck.json`). Detection probability scales linearly with the trusted-side `checked_fraction`; this is *not* verifiable computation, and we mark the corresponding claim as `proxy_supported` only.

A **compromised TEE** is out of scope; we trust the controller for its declared role.

## Mode-specific declared leakage surfaces

The protocol exposes the following mode knobs, each with a declared leakage surface and an artifact path. Choosing a more permissive mode is *not* a bug; it is a stated trade-off.

* `exact_visible_attention` (default attention privacy mode). The QK invariant `B_Q B_K^T = I` preserves scores by construction; therefore the score matrix `S = Q K^T / sqrt(d_h) + causal mask` and the post-softmax probabilities are visible on the accelerator. Evidence: `outputs/attention_privacy_modes.json`. Trade-off: one TEE-accelerator round trip per decode step, attention map *not* hidden.
* `trusted_softmax_attention`. The accelerator ships masked Q / K / V to the trusted side per attention block; the trusted side runs the softmax in trusted memory and returns a masked attention output. Trade-off: attention map hidden from the accelerator transcript, but the protocol gains `1 + L` round trips per decode step.
* `score_blinding_experimental`. Adds a row-constant additive shift `c_i` to the score matrix. Softmax is invariant under row-constant shifts, but ranking and relative margins are preserved; the attention pattern is *not* hidden. Reported as `privacy_gain = none_against_relative_attention_observer`.
* `norm_mask_granularity = sequence`. One per-layer `Q_l` shared across all token rows. Full Gram matrix `H_hat H_hat^T = H Q Q^T H^T = H H^T` is preserved exactly; token-pair similarity structure is visible at the layer-entry boundary. Evidence: `outputs/norm_granularity_low_interaction.json`.
* `norm_mask_granularity = chunk(k)`. Per-chunk orthogonal `Q_chunk` of chunk size `k`. Within-chunk Gram preserved; cross-chunk Gram disrupted.
* `norm_mask_granularity = token`. Per-row orthogonal `Q_i`. Off-diagonal Gram disrupted; per-row L2 norms remain preserved (a mathematical requirement of RMSNorm correctness). The row norm is *not* hidden.
* RoPE-plane mask: per-RoPE-pair 2D norms are preserved by construction (the mask is a 2D rotation in each RoPE plane). This is unavoidable for RoPE correctness.
* `vocab_permutation_mask`. Logit *multiset* preserved exactly; the sorted-logits vector is observable at the LM-head boundary. The mapping from token-index to logit value is hidden.
* `block_diagonal_vocab_mask(b)`. Within-block masking; block membership of each vocab index is observable unless the block partition is itself permuted.
* Sliding window attention. Window size `w` is *public*; per-(layer, head) masked KV invariant is preserved over the rolling window.

Every leakage surface above is enumerated in the claims audit (`outputs/paper_claims_audit_v2.json`) with safe / unsafe wording. The threat model is therefore conservative: any tensor or mapping *not* explicitly listed under "Protection targets" is *not* protected.
