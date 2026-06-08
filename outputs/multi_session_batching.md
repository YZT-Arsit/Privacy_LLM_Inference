# Multi-Session / Continuous-Batching Simulation

_Stage 7.7d: simulate multiple sessions with independent masks, ragged prompt / decode lengths, and per-session boundary-fingerprint isolation._

## Configuration

| Field | Value |
|---|---|
| num_sessions | 3 |
| prompt_lens | [4, 5, 7] |
| max_new_tokens | [2, 3, 2] |
| num_layers | 1 |
| mask_seeds | [2031, 2032, 2033] |

## Per-Session Results

| sid | prompt_len | max_new | greedy_match | seq_exact | lm_head_recovery_max | h_hat_max |
|---|---|---|---|---|---|---|
| 0 | 4 | 2 | 1.0 | True | 8.786e-16 | 2.220e-15 |
| 1 | 5 | 3 | 1.0 | True | 8.604e-16 | 3.109e-15 |
| 2 | 7 | 2 | 1.0 | True | 8.882e-16 | 2.665e-15 |

## Fingerprint Isolation (Same Prompt, Different Sessions)

- fingerprints_differ_per_session: `True`
- session_0 layer_entry fp: `373180222f71cc75520df561a838acb9731c6404dda6e44501597bc5bf706b37`
- session_1 layer_entry fp: `f18ffee1ea0f92dc366d595c64aa853321952d7a06cd423a94753629bb48857d`

## Batching Equivalence

| Field | Value |
|---|---|
| checked | False |
| reason | ragged prompt lengths in this config |

## Policy Flags

| Field | Value |
|---|---|
| multi_session_supported | True |
| continuous_batching_simulated | True |
| cross_session_mask_isolation | True |
| cross_session_prefix_sharing_default | False |
| ragged_lengths_handled | True |
| timing_side_channel_not_evaluated | True |

## Limitations

- CPU local emulation only; no real serving scheduler or continuous-batching engine.
- Sessions are simulated independently with fresh per-session masks; the wrapper does NOT yet batch their boundaries into a single low-interaction call.
- Cross-session prefix sharing is OFF by default; any future support must be reported as a leakage surface.
- Timing / memory side channels NOT evaluated.
- Not formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> Per-session orthogonal masks are sampled independently and produce per-session boundary fingerprints; the same prompt under two different sessions yields different masked boundary tensors. Cross-session prefix sharing is disabled by default.

## Unsafe Wording to Avoid

- Continuous batching is cryptographically isolated.
- Cross-session sharing is private.
- Timing side channels evaluated.
- This is formal cryptographic security.

