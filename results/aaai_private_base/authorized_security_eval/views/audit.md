# View-enforcement audit

- Every attacker-facing record requires an explicit `V0`, `V1`, `V2`, or `V3`.
- Construction rejects keys outside the exact allowlist; field reads are checked again.
- V0 rejects logits, timing, hidden states, attention, package tensors, and gradients.
- V2 rejects plaintext weights/adapters, labels, references, raw evaluator inputs,
  mask/inverse/gamma/rank/TDX secrets, and live model/oracle objects.
- Corpus manifests bind ordered deterministic sample IDs, file hashes, byte sizes,
  and record counts; a changed frozen manifest is rejected rather than overwritten.
- V3 is restricted to trusted positive controls and is not mixed into primary V2 results.

The enforcement test result is recorded separately in `enforcement_tests.json`.
