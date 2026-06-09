# Limitations and Scope

We list the boundaries of this work explicitly. None of the items below is hidden in a limitations footnote; each one is enumerated in the per-claim audit (`outputs/paper_claims_audit_v2.json`) with a corresponding *unsafe wording* that the paper must never use. The pytest enforces that the unsafe-wording list is non-empty and contains the canonical forbidden phrases.

## What this work does not do

1. **CPU local emulation only.** Every experiment runs on CPU at float64. There is no execution on a real GPU, no real TEE platform, and no real serving runtime. The CPU emulation is faithful at the *algebraic* level — pads, masks, and per-call boundary tables flow through every linear and operator-compatible nonlinear core — but it is *not* a deployment artefact. Marked: `no_real_gpu_or_tee_wall_clock` (status: *unsupported*).

2. **No real GPU / TEE wall-clock.** The cost model (`outputs/paper_cost_model.json`) provides symbolic and tiny-/real-config FLOP, transfer, and storage *estimates* for every protocol mode. These are *complexity-model evidence*, not measurements. We do not report latency, throughput, memory-bandwidth, or kernel-launch numbers. Marked: `no_real_gpu_or_tee_wall_clock`.

3. **No real CUDA / FlashAttention / vLLM backend.** The QKV / o / up / gate / down projections, RoPE, softmax, KV cache append, and LM-head linear are reference-implementation PyTorch on CPU. We do not integrate a real CUDA kernel, FlashAttention variant, paged-attention kernel, or any vLLM-style serving engine. Marked: `real_vllm_or_flashattention_backend` (status: *unsupported*).

4. **No hardware side-channel evaluation.** Timing, memory-access pattern, page-fault timing, paged-cache block-allocator races, RDMA traffic, GPU power, EM radiation, and cache side channels are *not* evaluated. The window-size policy under sliding-window attention and the KV cache length are *public* by construction. Marked: `no_hardware_side_channel_evaluation` (status: *unsupported*).

5. **No formal cryptographic security.** The protocol uses operator-compatible orthogonal / permutation / RoPE-plane-rotation / paired-permutation masks. We make *no* indistinguishability-against-chosen-message-attack claim, *no* semantic-security claim, *no* hybrid-argument argument over mask reuse, and *no* analysis of pad-scale distinguishability. Marked: `no_formal_cryptographic_security` (status: *unsupported*).

6. **No semantic security.** We do not show that an adversary cannot recover meaningful semantic information from the masked tensors. We show *algebraic* leakage characterisations per mode (Theorem 12).

7. **No differential privacy.** There is no privacy budget `(epsilon, delta)`, no Laplace / Gaussian noise calibration, and no output perturbation. Trusted-side processors run on exact recovered logits.

8. **No compromised TEE protection.** If the trusted controller is itself compromised — its session masks `(Q_l, N_K, N_V, B_K, P, N_vocab)`, per-call masks `M`, pads `T`, recovered logits, or trusted randomness `rho` leak — every protection collapses. We trust the controller for its declared role.

9. **Honest-but-curious accelerator is the primary adversary.** Active malicious accelerator is addressed *only* by a probabilistic spot-check prototype (`outputs/integrity_spotcheck.json`). The corresponding claim is *proxy_supported*, not *supported*, and explicitly *not* a verifiable-computation primitive. An adaptive adversary that observes which items are spot-checked can lower the effective detection rate. Marked: `active_adversary_integrity_supported = "probabilistic spot-check only"`, `full_verifiable_computation = false`, `malicious_accelerator_privacy_not_addressed = true`. The unsafe wording "active malicious accelerator fully handled" is in the audit's forbidden list.

10. **`exact_visible_attention` mode exposes attention maps.** The QK invariant `B_Q B_K^T = I` intentionally preserves `S = Q K^T / sqrt(d_h) + causal_mask` and `P = softmax(S)`. In this mode, the score matrix, the probability matrix, the per-query entropy, the top-1 index, the top-k indices, the relative margins, and the attention-map fingerprint are *all* available to the accelerator. The unsafe wording "attention maps hidden in exact low-interaction mode" is forbidden.

11. **`trusted_softmax_attention` hides attention but increases interaction.** The mode adds `L` extra TEE round trips per decode step (one per layer attention block). It is *not* the one-round-trip main path. The unsafe wording "trusted-softmax baseline preserves one round trip" is forbidden.

12. **Output length / stop timing is not hidden.** Generated token count, EOS emission timing, and per-step latency are observable on the user-facing channel and (where it crosses the boundary) on the protocol channel. We do *not* implement output-length padding or batching-based length hiding. Marked: `output_length_side_channel_not_hidden_unless_separately_padded` (status: *unsupported*). The unsafe wording "output length hidden" is forbidden.

13. **Dense vocab masking is not scalable.** For real LLM vocab sizes (`V >= 16k`) the dense orthogonal `[V, V]` mask costs `O(V^2)` storage and `O(B s V^2)` per-call FLOPs, which is infeasible. The protocol provides permutation and block-diagonal alternatives with declared leakage; the unsafe wording "dense vocab mask is scalable" is forbidden.

14. **Real quantised kernels are not evaluated.** fp16, bf16, and int8 are *simulated* via CPU round-trip casts on float64 storage; int4 is *symbolic only*. We do not load a real quantised model, do not measure real GPU fp16 / bf16 / int8 / int4 tensor-core matmul, and do not analyse mixed-precision accumulator types. Marked: `fp16_bf16_int8_int4_simulated_only_not_real_kernels` and `quantized_real_model_deployment_unsupported_without_real_backend` (both *unsupported*).

15. **M-RoPE, MoE, MLA, speculative decoding are not supported.** Multi-axis / multimodal positional encodings (M-RoPE) require an extension of the RoPE-plane analysis to multiple position dimensions and are not addressed. Mixture-of-Experts router / expert dispatch reveals expert selection by construction and would need a trusted-routing or masked-dispatch protocol. Multi-Head Latent Attention changes the (Q, K, V) algebra and the QK invariant must be re-derived. Speculative decoding's draft / target verification protocol crosses the TEE boundary in a way that the current invariant does not cover. Marked: `m_rope_multimodal_unsupported`, `moe_unsupported`, `speculative_decoding_unsupported` (all *unsupported*).

16. **LoRA training (backward) is not supported.** The protocol covers LoRA *inference* (forward pass) at every supported insertion site. The backward pass requires masked gradient flow through the boundary tables and is not implemented or analysed. Marked: `lora_training_backward_supported = false`.

17. **Full Qwen / LLaMA deployment is not supported.** The artefacts run a synthetic tiny modern-decoder surrogate that mirrors the LLaMA / Qwen forward graph (RMSNorm, rotate-half RoPE, GQA, SwiGLU, KV cache). We do *not* load real Qwen / LLaMA weights, do *not* use the real tokenizer, and do *not* support real model loaders. Marked: `no_full_qwen_or_llama_deployment_unless_real_wrapper`.

## How these limitations can be addressed

We list the concrete blockers that would lift the corresponding limitation. None of these is undertaken in this work; they are the obvious next milestones.

* **Real GPU / TEE backend.** Provision a real H100 confidential-computing GPU (or AMD MI series equivalent) and a real TEE platform (SGX, Gramine, Occlum). Port the boundary tables and trusted-side compilation into the TEE; port the masked forward into a CUDA implementation. Measure wall-clock under realistic prompt and decode lengths.

* **Fused confidential attention kernel.** Implement a fused CUDA kernel that holds the score matrix and the post-softmax probabilities only in registers / ephemeral memory inside the kernel boundary, never spilling them to global memory. This would change the threat model from "accelerator persistent transcript" to "accelerator persistent transcript modulo kernel-internal registers" and would let `exact_visible_attention` become attention-hidden under a *different* threat assumption. Marked symbolically in `outputs/attention_privacy_modes.json` as `fused_kernel_transcript_hiding`.

* **Secure softmax primitive.** Implement an MPC / FHE / trusted-coprocessor softmax that the accelerator can call without ever reading the scores. This is what the `trusted_softmax_attention` mode approximates symbolically; a real implementation would replace the TEE-resident softmax with a true secure primitive.

* **Real quantised deployment.** Integrate a real GPU int8 / int4 kernel (e.g. CUTLASS, Marlin, AWQ kernels) and verify that orthogonal / permutation masks survive int8 weight-only quantisation with acceptable error. Compare against the simulated bounds in `outputs/precision_quantization_stability.json`.

* **Real model wrapper.** Implement a loader for real Qwen / LLaMA safetensors weights, a real BPE tokenizer integration, and per-architecture gamma / RoPE / GQA configuration probes. Re-run the test suite against the real graph.

* **Hardware side-channel evaluation platform.** Co-locate the protocol with a hardware side-channel testbed (timing, power, EM); evaluate window-size leakage under sliding-window mode, paged-cache allocator timing, and KV cache length under multi-tenant batching.

* **Verifiable computation for malicious accelerator.** Replace the probabilistic spot-check with an authenticated dataflow primitive (e.g. a SNARK on the forward pass, or a zk-SNARK on the LM head). The current `proxy_supported` status would then upgrade to *supported* against active malicious accelerator integrity (not privacy).

* **M-RoPE / MoE / MLA / speculative decoding.** Derive per-component invariants. For M-RoPE, generalise the RoPE-plane block-diagonal rotation analysis to multi-axis rotations. For MoE, propose either a trusted-router (router runs in TEE) or a masked-expert-dispatch (route on a one-hot vector that the accelerator cannot decode). For MLA, re-derive the QK invariant under the latent-K / latent-V compression. For speculative decoding, propose a masked draft model + masked verification.

* **LoRA backward.** Derive the gradient flow through the padded boundary; verify that gradient masks `(M^{-T}, R^{-T}, N^{-T})` are consistent with the forward `(M, R, N)`; address optimiser state masking.

* **Output-length / stop-timing hiding.** Implement a trusted-side padded-generation policy (always emit `max_new_tokens` and post-truncate by EOS) or a batching policy that hides individual sequence completion times. Update the unsafe-wording list to reflect the new supported claim.

Every blocker above corresponds to a specific *unsupported* claim in `outputs/paper_claims_audit_v2.json`. Lifting a blocker means promoting the corresponding claim from *unsupported* to *supported*; doing so without the corresponding evidence is disallowed by the audit.
