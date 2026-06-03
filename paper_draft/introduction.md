# 1. Introduction

## 1.1 Privacy in modern LLM deployment

LLM-as-a-service deployments now mediate a large fraction of sensitive user workflows: legal drafting, medical summarization, source-code refactoring, customer-support routing, and personalized fine-tuning on enterprise data. Even when the *base* model weights are public, the runtime data flowing through a serving GPU is not:

- **Prompt privacy.** The token embedding of the user's prompt is the first activation entering layer 0; an attacker who can read activations recovers the prompt with high fidelity.
- **Hidden-state / KV-cache privacy.** Every subsequent layer's residual stream and every attention layer's `K` / `V` cache are derived from the prompt and from any retrieved context.
- **Generation-trace privacy.** During autoregressive decoding, the sequence of decoded tokens is itself sensitive (e.g., a partial draft of a confidential contract).
- **LoRA personalization privacy.** When a user fine-tunes a LoRA adapter `(A, B)` on private data, the adapter factors and their per-step gradients are derived from that data and have been shown to enable membership inference, training-data extraction, and downstream prompt reconstruction.
- **Gradient and adapter leakage.** Gradient tensors crossing a TEE-GPU boundary expose mini-batch composition; published adapter factors expose statistical signatures of the fine-tuning corpus.

In short: a deployment that treats a GPU as fully trusted ships the user's data straight to a component that is the *least* trusted in the trust chain.

## 1.2 Why naive TEE partitioning is not enough

The natural defense is to run the model inside a confidential computing TEE (SGX, TDX, SEV-SNP, H100 Confidential Computing). Two problems prevent this from being a complete answer:

- **TEEs are unsuited to LLM-scale dense linear algebra.** Effective TEE memory and compute throughput are an order of magnitude (or more) below datacenter GPUs for the kinds of dense, large-tile matmuls that dominate LLM inference and training. A pure-TEE deployment is dramatically slower and harder to scale.
- **GPU acceleration is required, but the GPU is outside the TEE.** Even "Confidential GPU" deployments still expose a transcript at the GPU device interface that adversarial code colocated with the workload, a compromised host, or a curious cloud operator can observe.

So the engineering problem is concrete: *we must keep dense linear algebra on the GPU while keeping the user's data out of any plain-text representation the GPU can observe.*

## 1.3 Why prior obfuscation/Amulet-style schemes do not cover generative LLMs

Boundary-masking schemes (Amulet-style matrix obfuscation, masked offload to untrusted accelerators) provide a partial answer for encoder-style and feed-forward workloads. They typically apply an invertible left-mask to the activation tensor and an inverse right-mask to the weight matrix, so the GPU performs `(M^{-1} X) (M W) = X W` without observing `X` or `W` in the clear. Three properties of generative LLMs make a direct port unsatisfactory:

- **Decoder-only autoregressive generation needs a stable token axis.** During decode, the KV cache is *appended* token-by-token. A left mask along the token dimension breaks the cache identity: the masked entries no longer concatenate to a contiguous masked sequence, and the mask has to be re-sampled and re-applied on every step.
- **Nonlinear layers do not commute with arbitrary dense masks.** `phi(MX) != phi(X)M` for a generic dense `M` and a pointwise activation `phi`. Modern decoder blocks also include RMSNorm, SwiGLU, and rotary positional embedding (RoPE) inside grouped-query attention (GQA); each has its own structural symmetries that any "compatible" mask family must respect.
- **LoRA personalization introduces structural leakage that linear masking alone does not address.** The LoRA factors `(A, B)` reveal the rank `r` from their tensor shape; the per-step gradients `dA`, `dB` reveal mini-batch identity; and a low-rank adapter is itself an extraction-friendly target.

A practically deployable scheme for generative LLMs has to solve all three simultaneously.

## 1.4 Our approach

We design and prototype a generation-compatible masked execution layer with four ingredients:

- **Generation-compatible right masking.** Masks act on the *feature* dimension of the activation rather than the token dimension. The KV cache append, RoPE rotation, and attention dot-product identities are preserved exactly because the right mask commutes with the token-axis concatenation and with the per-head feature blocks.
- **Operator-compatible nonlinear islands.** Around each nonlinear layer we use a *family-restricted* mask (channel permutation for elementwise activations and paired permutation for SwiGLU; orthogonal mask for RMSNorm; mean-preserving orthogonal mask for LayerNorm). Inside each island the mask family commutes with the operator, while outside the island we recover the dense Linear regime.
- **Dense sandwich plus boundary pad.** Each nonlinear island is sandwiched by fully-dense Linear masks; a *boundary pad* (a trusted-side translation `T`) is added before the island and absorbed back into the next Linear's bias, so an attacker who linearly inverts the island sees a randomized translation rather than the centered activation.
- **Private LoRA training path.** The LoRA factors `(A, B)` are masked with a trusted inner mask `U` such that `A_tilde = N_in^{-1} A U` and `B_tilde = U^{-1} B N_out`. The factor product is unchanged, the per-step backward gradient identity holds in masked space, and a rank pad with stronger dummy distributions hides the true rank from the tensor shape (the padded rank is still visible, see Limitations).

## 1.5 Contributions

We make four contributions, each backed by an explicit artifact in `paper_results/`:

- **Contribution 1: generation-compatible masked execution for decoder-only LLMs.** A right-masked wrapper that supports prefill, decode_step, greedy generation, KV cache append, RoPE, and grouped-query attention, and that reproduces the plain reference output token-for-token on GPT-2 and on a modern decoder-only wrapper in our tested configurations.
- **Contribution 2: operator-compatible nonlinear islands with rigorous correctness conditions.** Paired channel permutations for GELU / ReLU and for SwiGLU; orthogonal masks for RMSNorm; mean-preserving orthogonal masks for LayerNorm; each surrounded by a dense Linear sandwich and a boundary pad whose compensation algebra is closed in trusted space.
- **Contribution 3: a private LoRA personalization path.** Masked LoRA forward, masked LoRA backward, rank padding with five stronger dummy distributions (zero, paired-cancellation, Gaussian-matched, spectrum-matched, mixed-ensemble), and a multi-layer LoRA training step. Loss and optimizer remain trusted-side throughout; the adapter is never merged into the public base weight.
- **Contribution 4: an artifact-backed evaluation.** Six paper tables (artifact inventory, correctness, security proxy, workload, LoRA training, limitations), a measured local-emulation runtime evaluation (`time.perf_counter`), seven figures, and a `claims_mapping` document that labels every claim as `supported`, `proxy_supported`, or `unsupported`.

## 1.6 What this paper is not

This paper is **not** a formal-security paper. We do not prove cryptographic indistinguishability, semantic security, or differential privacy of the masked transcript; we do not measure real TEE wall-time; we do not implement a full Qwen / TinyLlama / LLaMA LoRA fine-tune; we do not claim PEFT / DeepSpeed / vLLM / FlashAttention compatibility; we do not claim that the padded LoRA rank itself is hidden. The Limitations and the `claims_mapping` document make every such boundary explicit.
