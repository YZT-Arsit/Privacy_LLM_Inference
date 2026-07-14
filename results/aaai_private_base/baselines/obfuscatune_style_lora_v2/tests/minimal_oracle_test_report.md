# Minimal oracle test report

Date: 2026-07-14

## Commands and status

```text
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/obfuscatune_v2
16 passed in 0.99s

PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/obfuscatune_v2 tests/obfuscatune tests/test_baselines_obfuscatune.py
68 passed in 2.63s
```

No test was weakened or deleted. The 52 legacy tests still pass independently.

## Measured FP64 oracle errors

Across seeds 1, 7 and 23 and both input/output target directions:

| Quantity | Maximum absolute error |
|---|---:|
| LoRA forward vs canonical plaintext function | 7.105427357601002e-15 |
| Manual dA* vs autograd | 0.0 |
| Manual dB* vs autograd | 1.7763568394002505e-15 |
| Restored input gradient vs canonical plaintext gradient | 1.0658141036401503e-14 |
| One-step stored-coordinate AdamW vs PyTorch AdamW | 1.1102230246251565e-16 |
| Ten-step stored-coordinate AdamW vs PyTorch AdamW | 1.1102230246251565e-16 |

The deliberate AdamW rotation counterexample differs by
`0.09999999516666691`, so the unsupported plaintext-trajectory equivalence
claim fails by a large margin rather than being hidden by tolerance.

## Covered gates

- Dense input/output transformed forward and backward.
- LoRA forward for input- and output-transformed targets.
- dA*, dB*, and input gradient against autograd/canonical oracles.
- One and ten transformed-coordinate AdamW steps.
- Zero-gradient decoupled weight decay.
- Checkpoint round trip, stale transform-generation rejection, and corrupted
  tensor-hash rejection.
- Negative AdamW rotation-equivariance test.

## Not covered yet

BF16 tolerance, full Qwen attention/MLP/block backward, decoder generation,
real checkpoint restart in a second process, Intel TDX transport, attestation,
and measured runtime counters remain outside the approved A-F scope.
