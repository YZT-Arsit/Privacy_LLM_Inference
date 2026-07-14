# Baseline threat-model contract

## Parties and adversary

The model provider owns a proprietary plaintext model. The data owner owns
private training/inference inputs. The cloud provider is honest-but-curious: it
executes the protocol correctly but observes all state outside the trusted
component. Authentication restricts queries to the data owner.

## Protected and exposed state

- Plaintext base weights enter and remain in the trusted provisioning path;
  high-parameter linear weights leave only after orthogonal transformation.
- Input/output text, token embedding/output layers, normalization, nonlinear
  operations, residual bookkeeping, and loss are trusted.
- The untrusted accelerator observes transformed base weights, transformed
  residual embeddings at boundaries, true Q/K/V, attention intermediates,
  MLP pre-nonlinearity intermediates, tensor shapes, and plaintext KV cache.
- The paper places LoRA parameters outside the TEE. V2 therefore must not claim
  that the untrusted accelerator cannot inspect their stored representation.
- The paper does not define private from-scratch versus public backbones; its
  stated target is a proprietary secret backbone.

## Non-goals

This contract does not claim malicious-GPU integrity, formal confidentiality,
KV-cache protection, label-only black-box behavior, or equivalence to the G2
threat model. Intel TDX attestation is an implementation adaptation, not a paper
security theorem.
