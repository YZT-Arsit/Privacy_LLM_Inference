# EXTENDED_GENERATIVE_LORA_PARTIAL

Updated: 2026-07-14 after protected T3/T4 artifact closure.

## Newly completed and verified

- T3 MLP-only protected: 750/750 finite steps, 500/500 generations, real A10
  plus Intel TDX, attestation success, no silent fallback, metrics and hashes
  closed.
- T4 all-seven protected: 750/750 finite steps, 500/500 generations, real A10
  plus Intel TDX, attestation success, no silent fallback, metrics and hashes
  closed.
- Protected T1-T4 target matrix: `PROTECTED_T1_T4_COMPLETE_VERIFIED`.

## Protected target quality, seed 1234

| Cell | BLEU | chrF | ROUGE-L | invalid% | repetition% |
|---|---:|---:|---:|---:|---:|
| T1 q+v | 64.08 | 68.34 | 60.28 | 0.0 | 2.25 |
| T2 q+k+v+o | 64.13 | 70.63 | 60.67 | 0.0 | 3.04 |
| T3 MLP | 65.45 | 70.60 | 61.77 | 0.0 | 2.73 |
| T4 all-seven | 65.88 | 70.56 | 61.96 | 0.0 | 2.79 |

T4 has the highest BLEU and ROUGE-L. T3 has the highest chrF by 0.04. These
are single-seed comparisons and do not establish statistical superiority.

## Stage items not closed by this operation

- Full plaintext T1-T4 matrix still requires the separately tracked plaintext
  T2 and T3 cells if they have not been completed elsewhere.
- ObfuscaTune-style paper-facing GPU result is not complete; its v2
  implementation is currently at the mathematical-contract/oracle stage.
- Fresh-process handoff and unified whole-stage cost-table status were not
  re-audited in this T3/T4 closure.

No training cell was rerun. No frozen G0/G1/G2 artifact was overwritten. No
code was committed.
