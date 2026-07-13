# Result-to-claim finding

- `claim_supported`: partial
- `confidence`: high

## Supported

This exact L12 implementation completed one real E2E NLG run: 750 finite protected training steps with an authoritative FP32 optimizer/master state in TDX and BF16 A10 runtime copies, verified attestation and package-root binding, followed by fresh-process protected generation of 500 valid aligned outputs. The recorded counters are internally consistent: 750 AdamW steps, 1,501 CE batches, 15,694 decode calls, no silent fallback, and no prohibited master/moment materialization. Completion-manifest and artifact hashes verify collection integrity.

Quality is competitive for this run. G2 obtains 65.88 BLEU and 61.96 ROUGE-L versus same-seed G1 at 63.98 and 61.40, with zero invalid outputs and lower measured repetition. G2 chrF is lower by 2.22.

## Not supported

The result does not establish strict numeric equivalence to plaintext G1, statistically significant quality improvement, seed robustness, or generalization beyond one G2 seed and one dataset/model/configuration. The paired sentence-BLEU bootstrap interval crosses zero; G2 outputs are shorter and chrF is lower.

The security evidence supports the implemented honest-but-curious-accelerator threat model and instrumented fail-closed invariants, not comprehensive malicious-GPU computation integrity. The HMAC key is shared with A10, does not prove an honest A10, and does not verify accelerator computation integrity. Restart reproducibility, transformed-adapter equivalence, and cost competitiveness are also unproven.

## Revised claim

On one E2E NLG run with seed 1234, the exact L12 A10+TDX path completed 750-step protected LoRA training and fresh-process protected generation for 500 examples with verified attestation, artifact integrity, and no violations in the instrumented fail-closed counters. Its generation quality was competitive with plaintext G1 on BLEU and ROUGE-L, while chrF was lower. This demonstrates feasibility under the stated honest-but-curious-accelerator threat model, not strict numeric equivalence, statistically proven superiority, or broad generality.

## Missing evidence and next experiments

Run at least two more G2 seeds, add a matched-precision G1/G2 control, test another dataset or model, evaluate the transformed adapter and checkpoint/restart path, run targeted security ablations and negative controls, and produce a consolidated performance/cost report. Re-run the claim gate after those results.
