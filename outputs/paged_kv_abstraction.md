# Paged KV-Cache Abstraction

_Stage 7.7c: verify the per-(session, layer, head) masked KV invariant under a synthetic paged cache._

## Configuration

| Field | Value |
|---|---|
| num_sessions | 3 |
| block_size | 4 |
| num_layers | 2 |
| num_kv_heads | 2 |
| head_dim | 16 |
| max_tokens_per_session | 13 |

## Per-Session Audit

| session_id | per_block_inv_max | full_cache_inv_max | num_tokens | num_blocks_used |
|---|---|---|---|---|
| 0 | 4.441e-16 | 8.882e-16 | 13 | 4 |
| 1 | 4.441e-16 | 8.882e-16 | 13 | 4 |
| 2 | 4.441e-16 | 8.882e-16 | 13 | 4 |

## Cross-Session Mask Isolation

isolated = `True`

## No-Plaintext-Block Check

min distance between any masked block row and any plain row = `2.691e+00`

## Policy Flags

| Field | Value |
|---|---|
| paged_kv_supported | True |
| private_cache_mode | True |
| prefix_cache_sharing_default | False |
| cross_user_cache_sharing_allowed | False |
| timing_side_channel_not_evaluated | True |

## Prefix-Cache Sharing

- prefix_cache_sharing_enabled: `False`
- public_prefix_token_count: `0`
- leakage_note: Cross-session prefix sharing requires an explicit public-prefix flag; enabling it intentionally exposes the shared prefix's K_tilde / V_tilde rows across sessions.

## Limitations

- CPU local emulation only; no real GPU paged attention kernel.
- Block-table abstraction is in-memory Python; no real memory allocator or page-fault behaviour.
- Cross-session block sharing is disabled by default; prefix sharing requires explicit public-prefix flag and is reported as a leakage surface when enabled.
- Timing / memory side channels (page-fault timing, block allocator races, evictions) are NOT evaluated.
- Not formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> The masked KV invariant ``K_tilde = K @ N_K`` and ``V_tilde = V @ N_V`` is preserved under a CPU synthetic paged cache: per-session N_K and N_V are sampled independently, each physical block of a session is masked by the same per-(layer, head) mask, and the logical-to-physical mapping is reconstructed by walking the block table. Cross-session block sharing is disabled by default.

## Unsafe Wording to Avoid

- Paged cache is cryptographically isolated.
- Cross-user cache sharing is private.
- Timing side channels evaluated.
- This is formal cryptographic security.

