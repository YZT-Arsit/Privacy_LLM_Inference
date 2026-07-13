# Known-plaintext stress test

Condition: `KNOWN_PLAINTEXT_STRESS_TEST`  
View: `V3`  
Status relative to primary threat model: **outside scope / positive control**

The trusted evaluator explicitly supplied paired plaintext and transformed q/k/v
weights for layers 0, 12, and 23 of the real Qwen2.5-0.5B experimental proxy.
Multi-Gram diagonal fingerprints recovered all 896 residual-basis permutation
coordinates. Off-diagonal Gram relationships then recovered every sign, up to the
irrelevant global sign.

- permutation accuracy: **1.0000**;
- sign accuracy up to global flip: **1.0000**;
- basis relative error up to global flip: **0.0000**;
- mean Gram reconstruction relative error: **1.19e-8**;
- positive-control gate: **PASS**.

This is a material limitation: if paired plaintext/transformed weights become
available, the signed-permutation residual basis is recoverable under this tested
condition. It is not evidence available to the primary V2 adversary and is not
mixed into the V2-versus-V0 result.
