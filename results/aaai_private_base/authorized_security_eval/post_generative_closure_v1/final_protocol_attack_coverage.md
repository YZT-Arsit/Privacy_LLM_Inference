# Final-Protocol Security Coverage Audit

This is an authorized defensive evaluation of researcher-owned assets. The final transformed package
root is `bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1`; the final LoRA target
set is q/k/v/o/gate/up/down.

| Experiment | Asset/view | Compatibility | Classification | Representative rerun? |
|---|---|---|---|---|
| Representation inversion | transformed hidden; strict/known-pair | No | diagnostic + positive control | Yes, fixed-length final V2 |
| Transformed-adapter recovery | transformed factors/product | No | diagnostic + positive control | Yes, actual rank-8 seven-target artifact |
| Output/logit leakage | masked-logit statistics | Partial | diagnostic | Yes, fixed-length final V2 |
| Gradient inversion | training-gradient proxy | No | diagnostic + positive control | No for inference phases; retain for optional fusion |
| KV leakage | transformed KV | No | diagnostic + positive control | Yes, fixed-length final V2 |
| Attention leakage | V0/V2 statistics | Partial | diagnostic | Yes, remove length/shape metadata |
| Known-plaintext basis recovery | actual package + plaintext pairs | Yes | known-plaintext positive control only | No |
| Membership inference | corrected same-pool design | Design only | pending | Yes, only after shadows exist |
| Artifact resistance R3–R5 | runtime/session/binding controls | Yes | final-protocol controls | Only R0–R2 missing |

Files are not rerun merely because names differ. Reruns are limited to material protocol gaps: actual
R0–R2 package/runtime behavior, the 1% surrogate-fidelity pilot, and the metadata-free fixed-length
attention/KV cohort. The historical source-confounded membership AUC is excluded.
