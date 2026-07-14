# ObfuscaTune-style LoRA v2 fidelity-gap analysis

| Component | Paper requires | Existing code | Required work | Fidelity |
|---|---|---|---|---|
| Model-weight transformation | Yes, Eq. 1-6 | Orthogonal/input/output primitives tested | Reuse in isolated v2 API | FAITHFUL |
| Input/output transformation | Yes | CPU simulator | Add trainable/autograd path | FAITHFUL |
| Residual representation | Plaintext/trusted in Fig. 2 | Qwen simulator | Preserve trusted ownership | ADAPTED_WITH_JUSTIFICATION |
| Q/K/V/O | Yes | Qwen forward tested | Add saved tensors/backward | ADAPTED_WITH_JUSTIFICATION |
| RoPE | Not discussed for GPT-2 | Qwen plaintext trusted helper | Gradient and shape tests | ADAPTED_WITH_JUSTIFICATION |
| GQA | Not discussed | `repeat_kv` and cache tests | Training backward tests | ADAPTED_WITH_JUSTIFICATION |
| KV cache | Not specified | Plaintext exposed cache simulator | Honest lifecycle labels | AMBIGUOUS_IN_PAPER |
| RMSNorm | General normalization trusted | Qwen RMSNorm tests | Backward and dtype tests | ADAPTED_WITH_JUSTIFICATION |
| SwiGLU | General nonlinear trusted | Qwen forward tests | Gate/up/down backward tests | ADAPTED_WITH_JUSTIFICATION |
| LoRA A/B placement | Outside TEE | Plaintext GPT-2 smoke only | Qwen all-seven trainable path | MISSING |
| LoRA forward | Required empirically | Not protected | Implement equations and ownership | MISSING |
| LoRA backward | Required empirically | None | Implement/test boundary Jacobians | MISSING |
| Gradient representation | Not specified | None | Conservative versioned contract | AMBIGUOUS_IN_PAPER |
| Optimizer update | Mentioned, placement absent | None | Freeze adapted optimizer contract | AMBIGUOUS_IN_PAPER |
| Trusted state | Low-param model and transforms | Simulator only | Explicit inventory and byte counters | MISSING |
| Checkpoint/restart | Not described | None | Repository adaptation + fail closed | MISSING |
| Adapter export | Not described | None | Only supported-coordinate package | MISSING |
| Protected inference handoff | Not described | None | Mark adaptation/limitation | MISSING |
| Authentication/attestation | Authentication assumed | No baseline real runtime | Reuse no G2 shortcut; baseline binding | MISSING |
| Communication counters | Efficiency requirement | Simulated proxy only | Runtime measured counters | MISSING |
| Memory counters | Efficiency requirement | Parameter ratio only | Runtime peak/state bytes | MISSING |
| Generation evaluation | Inference supported | CPU Qwen greedy tests | E2E 500-example path | MISSING |

## Existing inventory findings

The existing package contains a correct inference-side algebra simulator, not a
training system. `protocol.py` reports `full_system_reproduced=False`, while its
legacy declaration also reports `supports_lora_training=True`. The Qwen-specific
security schema correctly says `protected_lora=not_implemented`. V2 treats the
latter and executable behavior as authoritative and does not alter the shared
registry during active experiments.

Formal classification of the legacy flag:
`LEGACY_DECLARATION_NOT_BACKED_BY_REAL_QWEN_TRAINING_RUNTIME`.

The 52 legacy tests validate transforms, GPT-2/Qwen inference, condition numbers,
RMSNorm, attention, SwiGLU, GQA/cache, and generation. They do not validate LoRA
gradients, optimizer state, checkpointing, real TDX execution, or measured
boundary traffic.
