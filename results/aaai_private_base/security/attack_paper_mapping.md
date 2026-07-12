# Attack → published-methodology mapping (private-base security section)

All references below are real, published works. No fabricated citations.

| Attack | Our experiment | Published methodology | Venue / year | What we adapt |
|---|---|---|---|---|
| Representation inversion | S1 (A1/A2/A3) | Mahendran & Vedaldi, *Understanding Deep Image Representations by Inverting Them* | CVPR 2015 | Recover an input/representation consistent with an observed feature vector; we invert masked hidden states `h_tilde → h`. |
| Model inversion (confidence) | S1 | Fredrikson, Jha & Ristenpart, *Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures* | CCS 2015 | Reconstruct private inputs from model outputs/representations; here from masked representations. |
| LoRA structure | S2 | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen, *LoRA: Low-Rank Adaptation of Large Language Models* | ICLR 2022 | The low-rank adapter factorization `ΔW = A B` we attempt to recover from transformed `A_tilde,B_tilde`. |
| Logit-masking design | S3 | Our `design_spec.md §F` (monomial `logits·D·Pi`) vs permutation-only baseline | internal frozen spec | Rank/multiset/confidence leakage under permutation-only vs monomial. |
| Gradient inversion | S4 | Zhu, Liu & Han, *Deep Leakage from Gradients* | NeurIPS 2019 | Optimize dummy inputs so their gradients match observed gradients; applied to transformed LoRA gradients. |
| Gradient inversion (label) | S4 | Zhao, Mopuri & Bilen, *iDLG: Improved Deep Leakage from Gradients* | arXiv 2020 | Analytic label extraction + improved DLG init. |
| Membership inference | S5 (deferred, MEDIUM) | Shokri, Stronati, Song & Shmatikov, *Membership Inference Attacks Against Machine Learning Models* | IEEE S&P 2017 | Shadow-model membership classifier. |

## Honest scoping of each mapping
- **S1**: The cited inversion attacks assume access to the feature extractor (weights)
  or paired data. Under our frozen threat model the strict attacker has neither, so we
  report (a) the strict result and (b) a *generous* known-pairs upper bound that
  quantifies how much the mask alone protects. We do **not** claim the original
  attacks fail because "inversion is impossible" — only that they lack the inputs they
  require under this observation model, and that the orthogonal mask is linearly
  invertible once paired plaintext leaks.
- **S2**: LoRA factorization is inherently non-unique (`A B = (A G)(G^{-1} B)`), so
  "recover A,B" is ill-posed even without masking; we measure recovery of the
  *functional* `ΔW` (well-defined) and of individual factors (expected to fail).
- **S3**: permutation-only preserves the value multiset and full ranking (documented);
  the monomial `D` perturbs values. We quantify the difference; we do not claim the
  monomial hides the top-1 ranking (it does not, by construction of a positive `D`).
- **S4**: DLG/iDLG were demonstrated on small vision batches and shallow nets; we run
  the positive control on plaintext LoRA gradients first, then the transformed case.
