# V4_forward_matching_oracle

- **threat_model**: public weights; attacker runs candidates through the plaintext model
- **attacker_observation**: masked hidden state H_obs = H Q (Q secret, possibly fresh)
- **attacker_knowledge**: full plaintext model; a candidate set containing the true prompt
- **attack_objective**: identify which candidate produced H_obs (no Q recovery needed)
- **metric**: candidate_top1/rank, sensitive_token_top1, Procrustes residual margin
- **result**: A: true ranks #1/6 (residual 9.59e-15 vs margin 1.34e-03); B top1 YES; C sensitive-token top1 1.00; D combined rank #1. Fresh Q does NOT help (residual invariant to Q).
- **claim_survives**: False

- **invariant**: orthogonal Procrustes residual is 0 iff token-token Gram H H^T matches; (H Q)(H Q)^T = H H^T for ANY orthogonal Q, so Q (fresh or static) is irrelevant to this attack.

## A. small candidate set
- true rank #1/6, true residual 9.59e-15, margin 1.34e-03

## B. large candidate set
- true rank #1/21, top1=True

## C. partial-prompt / sensitive-token (top1=1.00)

| template | true | recovered | correct | true_resid | nearest_false |
|---|---|---|---|---|---|
| The password is{} | red | red | True | 1.18e-14 | 1.50e-03 |
| Patient diagnosis is{} | red | red | True | 5.58e-15 | 1.10e-03 |
| The phone code is{} | red | red | True | 1.03e-14 | 2.08e-03 |

## D. multi-layer (fresh Q per layer)
- single-layer true ranks: {3: 0, 6: 0, 9: 0}
- combined true rank: #1

## Fresh-Q ablation
- residual under Q1: 4.84e-15, under Q2: 5.79e-15 (identical -> Q irrelevant)

**oracle_succeeds = True**
