# Corrupted G2 Seed-1234 Copy Notice

The historical workspace path
`results/aaai_private_base/generative_lora/generations/g2_protected_e2e_L12_s1234_full.jsonl`
is a truncated, non-parseable copy. It is preserved unchanged.

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Damaged copy | `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_L12_s1234_full.jsonl` | 524,288 | `c3aa66cbbe899885f29b24e25082c1bcc24f2625c791160319c43ef073380ec4` |
| Canonical verified copy | `results/aaai_private_base/offline_analysis/confound_controlled_20260714/source_cache/g2_protected_e2e_L12_s1234_full.jsonl` | 1,010,375 | `d3b5ad9820c13e473ee7a995480a9ea6b5b3f5a786a4ba3276774cf116e46e31` |

The canonical copy exactly matches the size and SHA-256 recorded for the 500-row generation
artifact in `results/aaai_private_base/generative_lora/monitor/e2e_L12_s1234_full/completion_manifest.json`
(manifest SHA-256 `8fcc155bf34c7c818573025bbfb856e7f7b859fd5ca960bae2ce23cbcef46612`).
That manifest match—not path preference or parse success alone—is why the cache copy is canonical.

Both copies are immutable inputs. New v2 analysis scripts load the verified path only through the
canonical registry, verify its size and SHA-256 before parsing, and fail closed on any mismatch.
