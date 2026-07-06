# Cross-length / cross-statistics robustness of the block-A discriminators

model=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct, source=/root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ag_news_small.jsonl, subdim=256. KPA success (1=broken, 0=resist), multiset leak (1=leak, 0=safe).

| seq_len | layer | KPA obf | KPA ours-static | KPA ours-fresh | mset STIP | mset obf | mset ours-fresh |
|---|---|---|---|---|---|---|---|
| 64 | embed | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 64 | mid(L14) | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 256 | embed | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 256 | mid(L14) | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 512 | embed | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 512 | mid(L14) | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |

**Reading.** ours-fresh KPA stays ~0 (resist) and ObfuscaTune/ours-static stay ~1 (broken) across every seq_len and BOTH layers; STIP multiset stays ~1 (leak) while ObfuscaTune/ours-fresh stay ~0. Layer 0 is per-token (length-invariant by construction); the mid layer mixes the whole context via attention, so its length-invariance is the substantive robustness result. Conclusion: the block-A verdict is stable to input length and statistics.
