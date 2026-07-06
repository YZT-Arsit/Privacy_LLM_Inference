# Block-C completeness (real Qwen-7B GPU), base decode 15.97 ms/token

## (b) correctness — ours signed-perm residual fold vs plaintext
- max|y'−y| = **3.10e-06**, rel-L2 = 4.09e-07 → **lossless** (fp32).
- y=(W·N)(Nᵀ·x) reproduces y=W·x exactly. STIP/ObfuscaTune not tested (also exact linear).

## (c) boundary cost per token — consistent convention
| op | ms/token |
|---|---|
| static fold apply (folded into weights) | 0.000 |
| fresh: generate signed-perm | 0.1002 |
| fresh: apply mask (residual→GPU) | 0.0210 |
| fresh: unmask (result→back) | 0.0225 |
| **fresh total** | **0.1438** (0.90% of decode) |

Even counting the boundary apply+unmask the e2e omitted, fresh stays well under ObfuscaTune's 1.55 ms (224 in-TEE nonlinearities) — the earlier +0.9% was gen-only but the corrected total is still an order below ObfuscaTune.

## (a) mask-buffer memory (folded weights add 0 — same size as plaintext)
| mask | MB |
|---|---|
| static signed-perm (O(D)) | 30697.506 |
| fresh signed-perm (O(D)) | 30697.645 |
| dense fresh DxD (O(D^2)) | 30802.4 |

