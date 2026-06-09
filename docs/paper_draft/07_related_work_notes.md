# Related Work Notes

These are *notes for the final related-work section*, not the polished prose. We organise prior work by category and, for each, list (i) what prior work generally does, (ii) how this work differs, (iii) what overlap exists, and (iv) what *not* to claim. Placeholder citations are marked `[CITE: ...]`; concrete bibliography entries are to be inserted before submission.

## 1. TEE-assisted LLM inference

**What prior work generally does.** Place either the full LLM forward pass inside a TEE (CPU-based SGX / SEV-SNP, or H100 confidential-computing GPU), or place only sensitive sub-components (embeddings, LM head, attention softmax) inside the TEE while leaving dense matmul outside. Examples: confidential-GPU offloading papers, GPU-TEE bridging frameworks, TEE-inside-vLLM-style deployments. `[CITE: TEE-LLM-1, TEE-LLM-2, confidential-GPU-LLM]`.

**How this work differs.** We do *not* require the dense matmul to run inside the TEE; we run it on the untrusted accelerator with masked tensors. We do *not* re-enter the TEE at every nonlinear operator; the operator-compatible mask families let the nonlinear cores execute on the accelerator at the algebraic level. The protocol's main mode attains `one online round trip per decode step` and `intermediate_tee_reentry = false`. We do *not* assume a confidential-GPU hardware feature; the protocol works under a software-only operator-compatibility argument (though it would compose with hardware confidentiality to give defence in depth).

**Overlap.** The end-state — a TEE-resident trusted controller plus a GPU-resident accelerator domain — is shared with TEE+GPU offloading frameworks. We share the protection target (user-derived tensors, not public weights).

**Not to claim.** "Real TEE / GPU performance." "Compromised TEE protection." "Defence against host-side hypervisor compromise that breaks the TEE."

## 2. Model obfuscation / masked execution

**What prior work generally does.** Apply linear mask transformations to obfuscate intermediate activations during outsourced inference. Variants: per-layer orthogonal masks for matmul-only inference; pad-based additive masks; permutation-based masks for embedding-space outputs; arbitrary invertible masks for fully connected networks. `[CITE: masked-inference-1, OTP-inference, obfuscation-MM]`.

**How this work differs.** Prior masked execution typically targets feed-forward networks or single linear layers; it does not address the structure-sensitive operators of modern decoder-only LLMs (RMSNorm row L2, RoPE plane rotations, GQA softmax invariant, SwiGLU element-wise nonlinearity, KV append, paged storage). We pair each operator with a specific *mask family* that the operator commutes with (orthogonal for RMSNorm, RoPE-plane block rotation for RoPE, `B_Q B_K^T = I` for GQA, paired permutation for SwiGLU). We also distinguish *operator-compatible mask families* from *one-time-pad additive masks*: the pad enters only at linear boundaries and is cancelled by a precomputed `C = T W N`, never entering a nonlinear core.

**Overlap.** Linear-layer padded-boundary algebra `(x - T) M W M^{-1} W N + T W N = x W N` is the same algebraic identity used by prior pad-based masked-matmul schemes.

**Not to claim.** "Cryptographic security." "Indistinguishability against an adaptive adversary." "Defence against pad-scale distinguishing attacks."

## 3. Secure transformer inference with MPC / FHE

**What prior work generally does.** Run the entire transformer (or selected layers / heads) under secure multi-party computation, fully homomorphic encryption, or hybrid MPC + FHE protocols. Use polynomial approximation for nonlinearities (softmax, GELU, LayerNorm), garbled circuits for comparisons (top-k, top-p), or arithmetic shares for matmuls. `[CITE: Iron, CrypTen-LLM, MPC-Transformer, FHE-Transformer, sigma]`.

**How this work differs.** We do *not* use MPC or FHE. The protocol is software algebraic obfuscation with operator-compatible masks, not secret-sharing. Computational cost is one matmul-equivalent per linear plus per-operator masking overhead, not the orders-of-magnitude slowdown of MPC / FHE softmax / GELU. We *do not* claim cryptographic security; we claim *algebraic correctness and declared leakage*.

**Overlap.** The high-level system goal (privacy-preserving transformer inference) is shared. The trade-off framing — exactness vs hiding the attention map — recurs in MPC softmax design.

**Not to claim.** "Secure multi-party computation." "Cryptographic privacy." "Polynomial softmax approximation." "Garbled-circuit comparisons."

## 4. KV-cache privacy

**What prior work generally does.** Address KV cache leakage in long-context LLMs, especially under shared-prefix caching across users, via cache compression, cache scrubbing, per-tenant cache isolation, or trusted-cache enclaves. `[CITE: KV-leakage-1, prefix-cache-attack, KV-isolation]`.

**How this work differs.** We define a per-(session, layer, head) masked KV invariant `K_tilde = K N_K, V_tilde = V N_V` that holds across standard, paged, and rolling-window storage layouts. Cross-session block sharing is *off* by default; if enabled via an explicit `public_prefix` flag it is reported as a leakage surface, not silently allowed. We do *not* compress, scrub, or evict the cache for privacy reasons — only sliding window evicts, and the window policy is *public*.

**Overlap.** Per-tenant isolation is the same goal. The "prefix sharing is a leakage surface" framing is shared with prefix-cache attack papers.

**Not to claim.** "Cache-timing side channels evaluated." "Cryptographic isolation of paged blocks." "Prefix sharing is private when enabled."

## 5. GPU confidential computing

**What prior work generally does.** Hardware-supported GPU TEE (NVIDIA H100 CC, AMD Secure Encrypted Virtualization) provides an attested execution domain on the GPU itself. The bus between CPU TEE and GPU TEE is encrypted; GPU memory is encrypted; the host hypervisor cannot read GPU memory in plaintext. `[CITE: H100-CC, NVIDIA-CC-arch]`.

**How this work differs.** Our protocol does *not* require GPU confidential computing. We give a *software-only* operator-compatibility argument over masked tensors; the accelerator can be a plain GPU (or, in the artifact, a plain CPU). Under H100 CC, the protocol composes additively: H100 CC protects the bus and GPU memory under a hardware threat model; the protocol protects against accelerator-transcript inspection under a software threat model. We do not claim any hardware support.

**Overlap.** The threat model (untrusted accelerator host) is the same. The protection target (user-derived tensors) is the same.

**Not to claim.** "H100 confidential computing required." "Hardware attestation provided." "GPU-side TEE evaluation."

## 6. Private LoRA / adapter protection

**What prior work generally does.** Protect privately fine-tuned adapter parameters when serving on a shared base model. Variants: TEE-resident adapters, encrypted adapter shipping, adapter-only secret sharing, federated adapter training. `[CITE: private-LoRA-serving, adapter-TEE, LoRA-MPC]`.

**How this work differs.** We integrate LoRA *inference* directly into the operator-compatible padded boundary at every supported insertion site (`q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj`) with site-appropriate `N_out`. The adapter factors `A`, `B` are zero-padded to `r_pad` and masked by a rank-space invertible `R`; the accelerator sees `A_tilde = M^{-1} A_pad R`, `B_tilde = R^{-1} B_pad N_out`, never plaintext `A`, `B`. The true rank `r` is hidden by the zero pad; the padded rank `r_pad` is observable.

**Overlap.** The "protect adapter against accelerator" goal is shared. Rank padding (hiding `r`) is a natural design point.

**Not to claim.** "LoRA training (backward) supported." "Adapter values cryptographically hidden." "Padded rank cryptographically hides true rank."

## 7. LLM serving systems and KV-cache management

**What prior work generally does.** Build production-grade serving runtimes that manage paged KV cache, continuous batching, prefix caching, scheduler-aware request mixing, speculative decoding, and quantised inference (vLLM, TensorRT-LLM, TGI, SGLang). `[CITE: vLLM, TRT-LLM, SGLang, ContinuousBatching]`.

**How this work differs.** We provide *algebraic abstractions* (paged KV invariant, sliding-window invariant, multi-session boundary fingerprint isolation) that a real serving runtime could *adopt*, but we do not implement a serving runtime. No scheduler, no real batching engine, no real prefix-cache de-duplication. The protocol *composes* with these systems if the masked KV invariant is preserved by their storage layouts.

**Overlap.** The KV cache abstraction (per-token append, paged storage, sliding window) is the same. The multi-session isolation framing is shared.

**Not to claim.** "Real vLLM serving support." "Real continuous batching." "Real prefix-cache deduplication with privacy."

## 8. Quantised LLM inference

**What prior work generally does.** Quantise LLM weights and activations to int8 / int4 / fp8 / bf16 with weight-only, activation-quantised, or smooth-quant flavours, using calibration data to pick per-channel scales. `[CITE: AWQ, GPTQ, SmoothQuant, FP8-LLM]`.

**How this work differs.** We *simulate* quantisation on CPU (fp16 / bf16 / int8 round-trip casts on float64 storage; int4 symbolic only) to characterise how the *protocol's* mask-induced error interacts with reduced precision. We report per-mask-family error vs condition number and recommend well-conditioned mask families (orthogonal, permutation, RoPE-plane block rotation, block-diagonal) for low-precision deployment. We do *not* claim real-hardware quantised inference; the unsafe wording "real quantized model deployment" is forbidden.

**Overlap.** Per-channel symmetric quantisation is the same arithmetic. Condition-number analysis of mask matrices generalises the well-known sensitivity of low-precision inverse-matrix algebra.

**Not to claim.** "Real GPU int8 / int4 kernels." "AWQ / GPTQ benchmark performance." "Quantised model deployed in TEE-GPU split."

## Notes on bibliography to insert before submission

* Every `[CITE: ...]` placeholder above must be replaced with a concrete reference. The placeholders are intentional so we do *not* invent citations in the draft.
* The related work section in the final paper should follow the same category structure, but compressed to one paragraph per category. The "How this work differs" sentences are the ones to keep.
* The "Not to claim" lists must align with the unsafe-wording list in the per-claim audit (`outputs/paper_claims_audit_v2.json`); any new related-work claim must add a corresponding entry to the audit.
