# Claim-evidence matrix

| Claim | Evidence | Verdict |
|---|---|---|
| The evaluated private backbone was trained completely from scratch | Frozen registry, checkpoint inventory and loading code identify public pretrained Qwen2.5-0.5B | **Contradicted** |
| The transformed package corresponds to a pinned checkpoint | Checkpoint SHA `88c142…342`; package root `bfd578…cde1` | Supported, but for a public pretrained source checkpoint |
| B2 is empirically black-box-like relative to B0 | No valid from-scratch B0/B2 corpora or attacks | Not evaluated |
| The system is formally/universally black-box secure | No formal proof; empirical stage stopped | Unsupported and prohibited |
