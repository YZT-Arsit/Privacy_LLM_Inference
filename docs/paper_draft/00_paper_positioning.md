# Paper Positioning

## Thesis (one paragraph)

We study privacy-preserving inference for *public-weight* decoder-only large language models executed across a trusted controller (a TEE-like domain) and an untrusted accelerator (a GPU-like domain). The public base weights are *not* the protection target; the protection targets are the user prompt, the per-layer hidden states, the KV cache contents, optional private LoRA adapter contributions, the pre-sampling logits, and selected accelerator-transcript surfaces. We propose *operator-compatible padded masked execution*: every linear boundary uses a fresh additive pad and an invertible right mask, while every nonlinear / structure-sensitive operator (RMSNorm, RoPE, GQA softmax, SwiGLU, KV append) is paired with a *mask family it commutes with*, not with arbitrary dense randomness. The main protocol attains exact greedy / sampling equivalence under one online TEE-accelerator round trip per decode step, and exposes a small number of declared mode knobs (norm-mask granularity, attention-map hiding) that control unavoidable trade-offs. The accompanying CPU local-emulation suite validates algebraic correctness, leakage accounting, and a paper-claims audit; it does *not* constitute real GPU/TEE deployment or formal cryptographic security.

## What this paper solves

* A protocol whereby a public-weight decoder-only LLM can run its forward pass on an untrusted accelerator while keeping user-supplied tensors (prompt embeddings, hidden states, attention values, KV cache, recovered logits) hidden from accelerator-visible plaintext, **without** re-entering the trusted side at every nonlinear operator.
* A *decoder-operator coverage framework* that enumerates which mainstream decoder components (RMSNorm, standard 1D RoPE, GQA / MQA causal attention, SwiGLU, KV cache, paged KV, sliding window attention, LM head, LoRA inference, trusted-side generation processors) admit a mask family that preserves exactness, and which do not.
* A paper-ready CPU local-emulation experiment suite plus a per-claim audit that distinguishes algebraically *supported*, *proxy-supported*, and *unsupported* claims, so reviewers can read every claim against an artifact path.

## What this paper does NOT solve

* Real wall-clock on a real H100 confidential-computing GPU, real SGX / Gramine / Occlum TEE, or any real serving runtime (vLLM, TRT-LLM, FlashAttention kernel).
* Formal cryptographic, semantic, or differential-privacy security. The protocol relies on operator-compatible orthogonal masking; we make *no* indistinguishability claim against a polynomially bounded adversary, and explicitly do not analyse pad scale, randomness reuse, or distinguishing attacks.
* Hardware side channels: timing, memory-access patterns, page faults, RDMA, GPU power. The window-size policy and the cache length are public by construction.
* Active malicious accelerator under the full sense. A probabilistic spot-check prototype is provided as a proxy, not as verifiable computation.
* Multi-modal / multi-axis positional encodings (M-RoPE), Mixture-of-Experts routing, Multi-Head Latent Attention, speculative decoding, LoRA *training* (backward pass), or full Qwen / LLaMA model loading. These are listed as future work with explicit remaining blockers.

## Main novelty

* The protocol is *operator-compatible* rather than *operator-agnostic*: instead of trying to mask through a nonlinear core with arbitrary dense random matrices and then re-entering the TEE to recover correctness, every nonlinear operator is paired with a *structured* mask family (RMSNorm with row-wise orthogonal, RoPE with RoPE-plane block-diagonal rotation, GQA softmax via the QK invariant `B_Q B_K^T = I`, SwiGLU with paired permutation, KV cache append with per-head right mask). The pad enters only the linear boundary and is *cancelled* by a precomputed `C = TWN` term so that the nonlinear core sees a clean transformed activation.
* The price of operator-compatibility is paid as *declared mode knobs*, not as silent leakage: norm-mask granularity (sequence / chunk / token) trades off Gram-leakage vs per-token transition cost, attention privacy mode (exact-visible vs trusted-softmax) trades off attention-map hiding vs the one-round-trip property, LM-head masking (dense vs permutation vs block) trades off masking strength vs storage scalability.
* A boundary-transition trick lets the trusted side hand a *compatible* state (an `H_l Q_l` that already lives in the residual-stream basis) to the accelerator, then transition it to the padded boundary basis on the accelerator using precomputed `(A_i, C_i)` tables — without re-sampling pads or re-entering the TEE between layers.

## Why this is not merely "TEE + GPU"

A naïve TEE+GPU split runs *either* the entire forward inside the TEE (no GPU benefit) or it streams every nonlinear activation back to the TEE for correction (re-entry per RMSNorm / RoPE / softmax / SwiGLU). The latter forfeits the GPU bandwidth advantage on every decode step. Our protocol *eliminates* intermediate TEE re-entry in the exact mode (`intermediate_tee_reentry = false`, `online_boundary_round_trips_per_decode_step = 1`) by giving each nonlinear operator a mask family it commutes with at the algebraic level. The accelerator computes the full forward over masked tensors; the TEE only sees a per-step entry mask and exit logits.

## Why this is not merely "OTP masking"

A one-time-pad style additive mask `x + r` works for one linear `Wx`, but breaks under RMSNorm (changes the row L2 norm), RoPE (the 2D rotation does not commute with arbitrary additive randomness), softmax (`softmax(s + R) ≠ softmax(s)` for non-row-constant R), and SwiGLU (element-wise nonlinearity does not absorb additive pads). Our padded boundary `(x - T) M` cancels the pad at the next linear via the precomputed `C = T W N`, never letting the pad enter a nonlinear core. The flags `pad_enters_rmsnorm_core = pad_enters_rope_core = pad_enters_swiglu_core = pad_enters_softmax = false` are reported per call and per mode in the CPU emulation reports.

## Why arbitrary dense masks cannot be used through all decoder operators

* RMSNorm: only orthogonal right-action preserves the per-row L2 norm; a generic dense mask changes the norm and breaks the post-norm computation.
* RoPE: the rotate-half RoPE acts as a block-diagonal 2D rotation in `(j, j + d_h/2)` pairs; it commutes only with masks that share the same 2D rotation structure within each RoPE pair.
* GQA softmax: the score matrix `Q K^T` is preserved by Q/K mask pairs satisfying `B_Q B_K^T = I`. Arbitrary masks change the scores and therefore the post-softmax probabilities.
* SwiGLU: the gate–up element-wise product `a ⊙ silu(b)` admits a *shared permutation* but not an arbitrary mask, because element-wise nonlinearities only commute with operations that permute axes.
* Causal softmax: `softmax(S + c_i 1) = softmax(S)` for row-constant `c_i`, but for non-row-constant `R` the softmax output changes; therefore additive score blinding cannot achieve attention-map privacy while preserving exact softmax.

## Why operator-compatible mask families are necessary

The above structural constraints reduce the design problem to picking, *for each operator*, a mask family that the operator commutes with at the algebraic level. The protocol composes one such family per operator and pays the cost of operator-compatibility as a *declared* leakage / round-trip mode knob: norm-mask granularity, attention-map hiding, LM-head scalability, paged KV abstraction. Every mode knob comes with a paper-ready safe-wording / unsafe-wording entry in the claims audit.

## Three contributions

* **C1.** A low-interaction padded masked execution protocol for public-weight decoder-only LLMs, using fresh linear-boundary pads and operator-compatible masks, attaining exact greedy / sampling equivalence under one online TEE-accelerator round trip per decode step in the main mode.

* **C2.** A decoder-operator coverage framework covering RMSNorm with sequence / chunk / token granularity, RoPE-safe pre-masking via block-diagonal rotations aligned with RoPE planes, GQA / MQA via the QK invariant, SwiGLU via paired permutation, KV cache append and paged / sliding-window variants, scalable LM-head alternatives, LoRA inference at every supported insertion site, and trusted-side generation processors.

* **C3.** A CPU local-emulation experiment suite and per-claim audit validating algebraic correctness (logit recovery error at float64 machine precision, greedy match 1.0 across all supported modes), leakage accounting (Gram leakage per granularity, attention-map exposure per privacy mode), scalability trade-offs (dense vs permutation vs block LM-head at `V ∈ {97, 1024, 4096, 16k, 50k}`), and a 26-row claims audit with safe / unsafe wording and remaining blockers per claim.

## Explicit non-claims

* This is **not** a real TEE / GPU deployment.
* This is **not** a formal cryptographic security analysis.
* This is **not** a full Qwen / LLaMA deployment; the artifacts use a synthetic tiny modern-decoder surrogate that mirrors the LLaMA / Qwen forward graph.

Evidence: every contribution above is backed by an artifact under `outputs/`. The claim audit in `outputs/paper_claims_audit_v2.json` enumerates 26 claims (15 supported, 1 proxy-supported, 10 explicitly unsupported with remaining blockers).
