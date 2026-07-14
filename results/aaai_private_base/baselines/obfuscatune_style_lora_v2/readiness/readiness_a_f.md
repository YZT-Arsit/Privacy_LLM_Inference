# Phases A-F readiness decision

Status: `OBFUSCATUNE_IMPLEMENTATION_PARTIAL`

Phases A-D are frozen and the Phase E minimal oracle passes. Candidate 2 is
mathematically consistent and sufficiently distinct from G2 to continue toward
a Qwen implementation in a later authorized phase.

## Gates passed

- Canonical paper identified and hashed with page-level traceability.
- Legacy `supports_lora_training=True` corrected in audit without registry edit.
- Three candidate LoRA semantics derived.
- SGD/momentum/Adam/AdamW equivariance classified with numerical counterexample.
- Primary transformed-external contract frozen.
- Minimal forward/backward/optimizer/checkpoint implementation isolated.
- 16/16 v2 tests and 52/52 legacy regressions pass.

## Fidelity limitations

The selected contract is a repository adaptation because the paper omits factor
coordinates, backward protocol and optimizer placement. Its AdamW trajectory is
not plaintext equivalent. Therefore it is suitable for a paper table only as a
clearly labeled adapted-design baseline; it cannot be called official, fully
faithful at system level, or precision/optimizer-matched to G1/G2.

The Qwen/TDX runner remains prohibited and absent. A real paper-facing result
requires Qwen operator/backward tests, BF16 policy, process restart, runtime
instrumentation, and a baseline-specific A10/TDX implementation with no G2
shortcut.
