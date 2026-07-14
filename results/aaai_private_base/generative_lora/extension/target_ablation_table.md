# Protected target-set ablation (seed 1234)

All rows are real A10 + Intel TDX protected runs with 750 finite steps and 500 direct generations.

| Cell | Targets | BLEU | chrF | ROUGE-L | invalid% | repetition% | avg len |
|---|---|---:|---:|---:|---:|---:|---:|
| T1 | q_proj+v_proj | 64.08 | 68.34 | 60.28 | 0.0 | 2.25 | 29.9 |
| T2 | q_proj+k_proj+v_proj+o_proj | 64.13 | 70.63 | 60.67 | 0.0 | 3.04 | 31.3 |
| T3 | gate_proj+up_proj+down_proj | 65.45 | 70.6 | 61.77 | 0.0 | 2.73 | 31.1 |
| T4 | all-seven | 65.88 | 70.56 | 61.96 | 0.0 | 2.79 | 30.4 |

T4 has the highest BLEU and ROUGE-L; T3 has the highest chrF by 0.04. This is a single-seed quality comparison, not a significance claim.
