# 9. Limitations

This section enumerates every limitation that the artifacts surface and that the claims audit pins. None of the items below are buried; each is a first-class result of the work.

1. **No formal / cryptographic / semantic security.** Every security number in this paper is a *proxy* under a specific, named attacker. We do not prove cryptographic indistinguishability, semantic security, or differential privacy of the masked transcript. Every artifact's `security_profile` field is `proxy-evaluated, not formal`. Affected claims: any sentence that would otherwise read "secure"; we reword to "bounded under the named proxy attacker in our tested configurations".

2. **Trusted runtime is emulated, not a real TEE.** The trusted-side controller is implemented as a local Python runtime that holds masks, pads, adapters, optimizer state, and the loss closure. It is *not* an SGX / TDX / SEV-SNP / H100-Confidential-Compute deployment. Real-TEE deployment, attestation, sealed storage, and remote attestation flows are out of scope. Affected claims: any sentence about TEE-level isolation, attestation, or hardware-rooted trust.

3. **Measured runtime is local emulation, not real TEE wall-time.** `time.perf_counter` on a small CPU configuration; `num_warmup=2`, `num_repeats=5`; `dtype=float64`; `device=cpu`. Workload tiles are small for test stability. The `modern_decoder_model_wrapper` benchmark is opt-in and skipped by default. Reports publish timing statistics only; raw tensors, adapters, gradients, dense masks, and pads are never emitted. Affected claims: any sentence about deployment wall-time; we report `wall_time_source = measured_local_emulation` for measured rows and `projected_from_op_counts` for projected rows, and never claim TEE wall-time.

4. **Hardware side-channels are not evaluated.** Cache, power, EM, microarchitectural transient execution (Spectre, Foreshadow, MDS, downfall, etc.) are explicitly out of scope. Only a cost-model timing *proxy* is evaluated. Affected claims: any sentence about side-channel resistance; we say "we do not evaluate hardware side-channels."

5. **Full Qwen / TinyLlama / LLaMA LoRA fine-tuning is not implemented.** Our LoRA evaluation uses synthetic single-linear tiles and a synthetic multi-layer (2-layer, 14-module) decoder configuration. No production LoRA fine-tune is run. Stage 7.7 is deferred. Affected claims: any sentence that would advertise "private fine-tuning of LLaMA / Qwen / TinyLlama".

6. **Loss and optimizer remain trusted-side.** Only forward / backward matmuls cross the boundary. The optimizer state (SGD momentum, AdamW moments) is held trusted-side and never exported to JSON / CSV / Markdown. Affected claims: any sentence that would imply optimizer is run on the GPU; we say "loss + optimizer remain trusted-side".

7. **Padded LoRA rank `r_pad` is visible to the GPU.** Stage 7.2 / 7.3 / 7.4 hide the *true* rank `r` from the tensor shape, but the padded rank `r_pad` is the published shape and is visible. We do not claim `r_pad` is hidden. Stage 7.6 (heterogeneous padded rank with per-module independent `r_pad`) is deferred.

8. **The output text / token itself is visible.** By the threat model, the user receives the decoded tokens and the GPU observes them on the output path. The system does not attempt to hide the output text from the GPU. Sensitive applications must therefore not assume the *answer* itself is private — only the prompt / hidden state / cache / adapter / gradient pipeline.

9. **Sequence length and tensor shapes may leak.** We do not currently pad the sequence-length dimension. Per-layer tensor shapes are public. Mitigation (length padding, batching, dummy traffic) is out of scope.

10. **Stronger dummy distributions reduce tested spectral risk but do not prove rank hiding.** The Stage 7.4 ensemble keeps cross-layer linkage `low` but the worst-case spectral / gradient rank inference risk remains `high` under the stronger detectors. The dummy-strategy classifier itself reaches `0.476` (chance `0.143`), risk `medium`. We do not claim the rank-hardening is sufficient; we report the residual risk faithfully.

11. **PEFT / DeepSpeed / vLLM / FlashAttention compatibility is not implemented.** The LoRA primitives are a stand-alone functional API. We do not implement, test, or claim compatibility with any of these frameworks. Stage 7.8 (real TEE + Qwen LoRA fine-tune integrated with a real framework) is deferred.

12. **Compromised TEE is out of scope.** We assume the controller is honest. A compromised controller breaks every guarantee we evaluate. Roll-back attacks, replay across sessions, and malicious-mask sampling are not part of the evaluated threat model.

13. **The base model weights are assumed public.** Our scheme protects the user-side runtime data (prompt, hidden states, KV cache, adapter, gradients) but does *not* hide the base-model weights from the GPU. Deployments where the base model itself is a secret will need additional weight-side mechanisms not evaluated here.

14. **Distributed training is not implemented.** Single-process trusted runtime only. Multi-GPU and multi-node configurations are out of scope.

15. **Adapter merging into `W` is intentionally not supported.** The Stage 7.0 contract is that the LoRA adapter is *never* merged into the public base weight; doing so would publish a derived weight that no longer admits the masked forward identity of Theorem 7.

The aggregated Stage 1 → 7.4 limitation index (`paper_results/markdown/limitations_summary.md`) contains 161 per-stage limitation rows feeding into the items above. The recurring themes are exactly the ones enumerated here: no formal / cryptographic / semantic security; no real TEE wall-time; padded rank still visible; loss / optimizer trusted-side; no PEFT integration; no hardware side-channel evaluation; no full Qwen / TinyLlama / LLaMA LoRA fine-tuning.
