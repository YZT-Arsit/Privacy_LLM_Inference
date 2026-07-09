# V2_ffn_perm_to_token_recovery

- **threat_model**: public weights; attacker GIVEN ground-truth FFN neuron permutation
- **attacker_observation**: masked residual state H_obs = H Q; folded FFN weights
- **attacker_knowledge**: plaintext weights + TRUE FFN neuron permutation; NOT Q
- **attack_objective**: recover input prompt tokens from H_obs using the FFN permutation
- **metric**: token_recovery_top1/top5 (NN vs wte+wpe), exact_prompt_match, norm leak
- **result**: with the TRUE FFN permutation, token top-1 = 0.00e+00 (~random 1.99e-05); with Q known it would be 1.000. FFN permutation does not unlock Q.
- **claim_survives**: True

## Token recovery with TRUE FFN permutation

- random baseline top1: 1.99e-05
- masked token top1: 0.00e+00
- masked token top5: 0.00e+00
- upper bound if Q known: 1.000
- **norm leak** (rank corr H vs H_obs): 1.000000

per-token norm is PRESERVED by orthogonal Q (||H Q|| = ||H||): norm rank-corr(H, H_obs) = 1.000000 -- this IS leaked and is NOT hidden. FFN neuron permutation is weight-alignment leakage.
