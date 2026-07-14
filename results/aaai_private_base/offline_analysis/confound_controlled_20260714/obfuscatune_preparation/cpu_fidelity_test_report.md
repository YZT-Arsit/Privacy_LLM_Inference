# ObfuscaTune CPU Fidelity Test Report

The existing offline ObfuscaTune suite was run without pytest cache creation or GPU access:

```text
52 passed in 3.24s
```

Coverage includes matrix construction/conditioning, linear equivalence, Qwen RMSNorm and
attention/MLP boundaries, Qwen KV-cache correctness, generation smoke behavior, experiment
registry checks, and GPT-2 wrapper correctness.

This validates the current CPU simulator tests only. It does not close the missing real LoRA
training path, full frozen-checkpoint fidelity gate, or real TEE cost/runtime gate; GPU launch
therefore remains prohibited.
