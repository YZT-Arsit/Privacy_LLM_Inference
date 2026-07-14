# Isolated implementation plan

1. Freeze paper, threat, ownership, optimizer, and checkpoint contracts.
2. Implement orthogonal-coordinate trainable linear and LoRA primitives under
   `src/pllo/baselines/obfuscatune_lora_v2/`.
3. Prove forward and backward against a plaintext oracle in FP32, then define a
   BF16 tolerance profile.
4. Implement optimizer semantics as an explicit adaptation. Test zero-gradient
   decay, one step, ten steps, and restart. Do not claim coordinatewise AdamW
   invariance under a dense rotation.
5. Build Qwen attention/MLP/block training wrappers with trusted nonlinearities.
6. Implement versioned checkpoint and adapter packages with base/config/target/
   rank/alpha/source bindings and negative tests.
7. Add tiny training and generation/reload tests.
8. Only after all mathematical gates pass, add standalone A10 and TDX runtimes
   and exact future commands. They must carry a baseline runtime identifier and
   reject G2 shortcut flags.
9. Run legacy and v2 regressions and emit readiness status. No hardware launch
   occurs in this implementation session.
