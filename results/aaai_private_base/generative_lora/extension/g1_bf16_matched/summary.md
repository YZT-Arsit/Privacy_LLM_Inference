# G1 BF16 Matched — three-seed completion

All three precision-matched plaintext controls completed 750/750 finite training steps and 500/500 generations with 500 unique sample IDs. This control matches the executed G2 numerical profile documented in `precision_matching_audit.md`; it is not a replacement for G1 FP32.

| Seed | BLEU | chrF | ROUGE-L | Invalid | Repetition | tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 65.2371 | 71.2679 | 62.1114 | 0.0% | 2.7860% | 31.8158 |
| 7 | 63.8695 | 72.0124 | 60.7878 | 0.0% | 3.2888% | 32.4698 |
| 2025 | 64.3521 | 70.6242 | 60.2031 | 0.0% | 2.8746% | 32.0317 |
| Mean ± sample SD | 64.4862 ± 0.6936 | 71.3015 ± 0.6947 | 61.0341 ± 0.9777 | 0.0% | 2.9831 ± 0.2684% | 32.1057 ± 0.3332 |

No equivalence claim is made here. Paired corpus-level bootstrap and hierarchical analyses belong to the offline analysis handoff.
