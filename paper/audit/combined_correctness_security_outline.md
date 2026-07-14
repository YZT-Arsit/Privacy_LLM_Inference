# Compact Combined Correctness and Security Outline

Target: **0.80--1.05 compiled page** in the seven-page main paper. Planned
source after the writing pass: `paper/sections/06_correctness_security.tex`,
replacing the current separate Correctness draft and Security TODO. Complete
proofs remain in `paper_supplement/sections/05_complete_correctness_proofs.tex`.

## Main-Paper Sequence and Budget

| Block | Target | Main-paper content | Claim IDs |
|---|---:|---|---|
| Scope and convention | 0.05 page | exact arithmetic versus floating-point validation; functionality is not confidentiality | C-COR-CONV, C-NONCLAIM-01 |
| Theorem 1 | 0.10 page | masked linear execution, private learned bias, pad-compensation condition | C-COR-01 |
| Theorem 2 | 0.14 page | post-RoPE masking, Q/K compatibility, GQA/MQA tying, session-fixed KV masks | C-COR-02/03/04/05 |
| Theorem 3 | 0.14 page | rank-padded LoRA forward/backward/update equivalence; trusted optimizer | C-LORA-FWD/BWD/RANK, C-OPT-O1C |
| Security claim boundary | 0.18 page | evaluated private-base attacker; transformed transcript and admitted leakage; proxy-only non-recovery | C-THREAT-02/03, C-SEC-S1..S6, C-NONCLAIM-06 |
| Residual leakage and failure cases | 0.16 page | norms/Gram/multiset/rank/attention scores remain visible; known-pair degradation; blocked attacks are not wins | C-SEC-LEAK/PIINV/BLOCKED |
| Numerical/security evidence bridge | 0.12 page | compact error range and attack-proxy headline; defer configurations/tables | C-COR-01..05, C-LORA-FWD/BWD, C-SEC-S1..S6 |
| Explicit limitations | 0.08 page | no cryptographic security, nonlinear masked backward, bf16/GPU error bound, or non-greedy guarantee | C-NONCLAIM-01/04/05/06 |
| **Total target** | **0.97 page** | | |

## Required Theorem Statements

1. **Masked linear execution.** State invertibility, compatible dimensions,
   exact recovery, and the requirement that pad compensation precede nonlinear
   operators.
2. **Attention and cache.** Group post-RoPE ordering,
   $N_QN_K^{\mathsf T}=I$, per-group GQA/MQA mask tying, and session-fixed
   $N_K,N_V$ into one theorem.
3. **LoRA adaptation.** State $A_{pad}B_{pad}=AB$, invertible rank mixer and
   boundary masks, compensated factor/input-gradient recovery, and identical
   trusted optimizer state/update.

Each theorem receives at most two sentences of intuition. Derivations,
dimensions, and complete failure-condition proofs stay in the Supplement.

## Required Security Boundary

- Name the result as **empirical resistance for the evaluated private-base GPU
  attacker**, never formal security.
- State what the GPU observes and the admitted metadata before reporting an
  attack number.
- Report preserved invariants as leakage, not as protected signals.
- Keep checkpoint-disclosure attacks outside the paper's canonical register.
- Label setup-blocked and unimplemented attacks as unevaluated, not defeated.
- Distinguish synthetic/CPU proxy evidence from real-system measurements.

## Main-Paper Self-Containment Check

The combined section must allow a reviewer to recover the theorem assumptions,
the exact-versus-floating distinction, the adversary being evaluated, the
residual leakage, and the limitations without opening the Supplement. The
Supplement supplies verification depth only.
