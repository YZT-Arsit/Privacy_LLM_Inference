# Attention/runtime leakage pilot

All attacker features come from the immutable V0/V2 view manifest. Labels are joined
by the trusted evaluator and are not present in either attacker corpus.

| View | Category accuracy | Category macro-F1 | Length MAE | Exact length | Within 1 token |
|---|---:|---:|---:|---:|---:|
| V0 | 0.9617 | 0.9579 | 2.2470 | 0.1433 | 0.4467 |
| V2_without_attention | 1.0000 | 1.0000 | 0.0958 | 0.9800 | 0.9983 |
| V2_with_attention | 1.0000 | 1.0000 | 0.0273 | 1.0000 | 1.0000 |

This is a measured pilot, not the complete attention-ablation phase. Repeated-input, membership, relation, and surrogate-fidelity cells remain pending.
