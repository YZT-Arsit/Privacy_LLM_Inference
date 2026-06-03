# 3. System and Threat Model

## 3.1 System model

The system has two parties and a one-way data plane.

**Trusted-side controller (TEE-like, locally emulated).** Holds: the user prompt and any retrieved private context; the user's LoRA adapter `(A, B)`; the optimizer state (SGD momentum / AdamW moments); the loss closure; the sampler (greedy / top-k / top-p); the mask sampler that draws fresh masks `N_in, N_out, N_Q, N_K, N_V, P, U` per call; the boundary pad `T`; and the trusted compensation tensors `C_T` and `C1`. The controller embeds the prompt, applies right masking to each Linear boundary, dispatches every dense matmul to the GPU, recovers plain-space results when an inter-block boundary requires it, samples the next token, and orchestrates the LoRA backward / optimizer step entirely on the trusted side. In our prototype the controller is **emulated as a local trusted runtime; it is not a real TEE.**

**Untrusted GPU.** Performs: masked Linear matmuls; masked nonlinear layers when an operator-compatible mask family is in use (paired permutations, orthogonal mask, mean-preserving orthogonal mask); masked attention `Q_tilde K_tilde^T = Q K^T` when `N_Q N_K^T = I`; KV cache reads and per-step appends in masked space; LoRA forward `X (W + (alpha/r) A_pad B_pad) N_out`; and the masked backward matmuls. The GPU observes every input/output tensor it sees, every kernel argument shape, and the elapsed real time it spends per call.

**Data plane.** The controller pushes masked tensors `X_tilde, W_tilde, A_tilde, B_tilde, K_tilde, V_tilde` and reads back masked outputs `Y_tilde, dA_tilde, dB_tilde`. The controller never pushes a plain `X, A, B, dA, dB` to the GPU.

The full system is sketched in **Figure 1 (system overview)**.

## 3.2 Threat model

### Trusted side
- Holds user input.
- Generates masks and pads.
- Holds and updates the LoRA adapter.
- Holds and updates the optimizer state.
- Holds the loss closure.
- Performs recovery (un-masking) and sampling.
- Currently emulated as a local trusted runtime; **not** a real TEE in this paper.

### Untrusted GPU / cloud
- Sees masked activations across every Linear and nonlinear boundary.
- Sees masked weights and masked LoRA factors.
- Sees the masked KV cache and the masked attention transcript.
- Sees the full GPU-visible execution transcript (kernel order, shapes, dtype).
- May observe tensor shapes, sequence length, per-call timing as a cost-model proxy, and the *output tokens themselves* (because the user receives them and they are not, in general, a secret of the system).
- May colocate adversarial code and record everything.

### Protected assets
- Prompt / private input tensors.
- Hidden states / residual stream between Linear layers.
- KV cache entries `(K, V)` in plain space.
- Pre-sampling logits.
- LoRA adapter factors `A`, `B` (per-user, per-task secret).
- Per-step LoRA gradients `dA`, `dB`.
- Private training data inside the LoRA training loop.
- Optimizer state.

### Allowed leakage
- Model architecture (layer counts, head counts, dtype).
- The public base-model weights (assumed public by the problem statement).
- Tensor shapes, unless a separate pad hides a particular dimension (e.g., rank padding hides `r` from shape but not `r_pad`).
- The padded LoRA rank `r_pad` — see Limitations.
- The output text/token, which the user receives and the system does not try to hide from the GPU.
- Sequence length, unless a separate sequence-length pad is engaged (not implemented in this paper).
- Local-runtime metadata exposed by our local-emulation harness — this is a prototype artifact, not a real-TEE measurement.

### Out of scope
The following are not covered by this paper, and the design does not claim to defend against them:

- **A compromised TEE.** We assume the controller is honest; a compromised controller breaks every guarantee we evaluate.
- **Hardware side-channels.** Cache, power, EM, microarchitectural transient execution: not evaluated.
- **Formal / semantic / cryptographic security.** We make no formal indistinguishability or semantic-security claim; every security number in this paper is a *proxy* under a specific, named attacker.
- **Real TEE wall-time.** Our runtime numbers are local-emulation measurements (`time.perf_counter`), not real TEE wall-time.
- **Full production Qwen / TinyLlama / LLaMA LoRA fine-tuning.** Our LoRA evaluation uses synthetic single-linear and synthetic multi-layer tiles only.
- **PEFT / vLLM / DeepSpeed / FlashAttention integration.** Our LoRA primitives are a stand-alone functional API; we do not integrate with these frameworks.
- **Padded-rank hiding.** Stage-7.2 / 7.3 / 7.4 hide the *true* rank from the tensor shape, but `r_pad` is itself visible.
- **Fully outsourced loss / optimizer.** Loss and optimizer remain trusted-side; only forward / backward matmuls cross the boundary.

### Adversary capabilities exercised in evaluation
- Ridge / small-MLP activation inverters with adaptive training on `(X_tilde, Y_tilde)` pairs.
- Signature / Sinkhorn permutation-recovery attackers against permutation islands.
- Membership-style linkability AUC over per-call masked traces.
- LoRA adapter extraction proxies and gradient extraction proxies.
- Spectral-cliff, energy, elbow, and ensemble rank-inference proxies against rank-padded LoRA.
- A cost-model timing classifier that distinguishes execution profiles when constant-time mode is off.
- A black-box query attacker over (output token, summary-logits) traces.

A summary of the threat model and leakage boundary is enumerated in **Table 1**.
