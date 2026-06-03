# 10. Related Work

Citations below are placeholders; concrete references will be inserted in the LaTeX pass (Stage 7.6b). The intent of this section is to position the paper against the categories of related work, not to substitute a literature review.

## 10.1 TEE-assisted ML inference

A body of work runs ML inference end-to-end inside a TEE (SGX, TDX, SEV-SNP), trading throughput for confidentiality. These designs are sufficient at small scale but are limited by TEE-resident dense-linear-algebra throughput; LLM-scale serving is impractical without GPU acceleration. Hybrid TEE+GPU systems push a fraction of the computation to the GPU but typically expose plain activations at the boundary. **Difference.** Our scheme keeps a TEE-like trusted controller and the loss/optimizer trusted-side while pushing *masked* dense matmuls to the untrusted GPU, retaining GPU throughput without exposing plain activations.

## 10.2 Privacy-preserving LLM inference

Several recent systems target LLM inference privacy through model splitting, prompt encryption, or partial homomorphic / partial-TEE deployments. Most do not address generation specifically: they either focus on encoder-only embeddings (good for retrieval but not for autoregressive decoding) or assume a non-streaming inference pattern. **Difference.** We explicitly target generation-compatible inference: right masking commutes with token-axis KV cache append, RoPE blocks, and grouped-query attention, so the decode step preserves the cache identity without re-masking the prefix.

## 10.3 Amulet-style / matrix obfuscation methods

The Amulet line of work (and the broader family of matrix-mask offload schemes for deep learning) introduced the boundary-mask pattern we build on: `(X N) (N^{-1} W) = X W` over Linear layers. These designs do not natively pass through pointwise nonlinearities (`phi(MX) != phi(X)M` for generic dense `M`), through RMSNorm / LayerNorm (different invariance groups), through SwiGLU (paired permutation required), through RoPE (head-block structure), or through LoRA personalization (rank leakage). **Difference.** Operator-compatible nonlinear islands plus a dense Linear sandwich plus a boundary pad extend the Amulet skeleton to a full decoder-only stack and to a LoRA personalization path.

## 10.4 Secure GPU offloading

A line of work treats the GPU as an untrusted accelerator and uses cryptographic or obfuscation techniques (homomorphic encryption, garbled circuits, secret-sharing) to keep the operands hidden. These provide strong formal-security properties but at large compute overheads, and most do not support autoregressive generation natively. **Difference.** We trade formal-security guarantees for *proxy-evaluated security under named attackers* in exchange for keeping plain GPU dense matmul kernels on the hot path. Our security claim is therefore an empirical bound, not a cryptographic one — and our Limitations make that explicit.

## 10.5 FHE / MPC based private inference

Fully homomorphic encryption and secure multi-party computation give formal guarantees of input privacy at the cost of one to several orders of magnitude overhead and significant restrictions on the operator set. Recent FHE-friendly transformer designs reduce overhead but are still well below GPU dense-matmul throughput. **Difference.** Our approach is *not* a cryptographic alternative to FHE/MPC. It offers a different point on the spectrum — proxy security with near-GPU-native throughput — and is best read as complementary to, not a replacement for, formal-security work.

## 10.6 KV cache privacy

KV cache leakage has been demonstrated as an end-to-end channel for prompt-content recovery. Mitigations proposed in the literature include cache eviction, cache encryption at rest, and cache partitioning across tenants. **Difference.** Our scheme keeps the KV cache in *masked* space across the entire decode trajectory: per-step append concatenates `K_t N_K` and `V_t N_V` onto a cache that is never recovered to plain space inside the GPU; the attention identity is preserved by `N_Q N_K^T = I`.

## 10.7 LoRA personalization security

A growing body of work shows that LoRA factor publication leaks training-data signatures and enables membership inference, prompt-style fingerprinting, and partial training-data reconstruction. Mitigations include differential privacy at the optimizer, gradient clipping, and adapter watermarking. **Difference.** Our LoRA path masks `(A, B)` with a trusted inner mask `U` such that `A_tilde = N_in^{-1} A U` and `B_tilde = U^{-1} B N_out`, keeps the optimizer and loss trusted-side, and adds rank padding with stronger dummy distributions. We never merge the adapter into the public base weight `W`.

## 10.8 Gradient leakage and membership inference

Deep Leakage from Gradients, iDLG, GradInversion, and follow-up work show that per-step gradients are highly informative about training data. Defenses range from gradient clipping to local SGD to differential privacy. **Difference.** We do not propose a differential-privacy bound. We mask the per-step gradients in the same right-masked space as the forward path, so that the GPU-visible `dA_tilde, dB_tilde` are not the plain gradients; the membership-style linkability AUC is brought close to random chance under our proxy attackers in the tested configurations. This is reported as `proxy_supported`, not `provably private`.

## 10.9 Side-channel defenses

Constant-time programming, branch-balanced code, padded data flow, and traffic shaping are standard side-channel defenses. Our `proxy_equalized` constant-time-emulated training mode is a cost-model proxy, not a real wall-time defense. **Difference.** We make no hardware-side-channel claim. The cost-model timing proxy is reported as `proxy-supported` only and only at the trusted-controller cost model.

## 10.10 Positioning summary

Our work sits at the intersection of (a) hybrid TEE+GPU systems that need a clean abstraction for the masked boundary, (b) matrix-obfuscation methods that need an extension to nonlinear and to autoregressive paths, and (c) LoRA personalization that needs a tractable adapter-and-gradient mask. The principal *new* ingredients we contribute are: generation-compatible right masking; operator-compatible nonlinear islands; private LoRA forward + backward + rank-padded training path; and an artifact-backed, claim-audited proxy security evaluation. The principal *deliberate non-contributions* are: no cryptographic / formal / semantic security claim, no real-TEE deployment, no full production fine-tune, no PEFT / vLLM / DeepSpeed / FlashAttention integration.
