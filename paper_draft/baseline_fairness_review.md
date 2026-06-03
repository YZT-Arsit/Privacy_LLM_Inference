# Baseline Fairness Review

Reviewer-risk pass focused on the RQ13 prior-work comparison. Goal: ensure the comparison is framed as primitive-functional rather than as a security or runtime ranking, and that cryptographic baselines are explicitly complementary, not replaced.

## Risk items

- **Risk ID:** `PWR-01`
  - Severity: **high**
  - Dimension: `prior_work_comparison_risk`
  - Location: sec:eval:rq13 (Cross-cutting observations final paragraph)
  - Risky wording or missing explanation: '... highlights which primitives extend to generative LLMs (ours) and which do not (Slalom, DarKnight, Amulet static masking, CryptoNets arithmetic skeleton).' Reads as 'ours is better than them'.
  - Why a reviewer may object: Each baseline solves a different problem under a different threat model. Saying 'they do not extend to generative LLMs' invites the rebuttal 'because they were not designed for generative LLMs, and you do not provide their formal-security property in return.'
  - Recommended revision: Re-word as: 'The direct prior-work primitives we implement target different problems and threat models. Their primitive surface does not cover decoder-only generation, KV-cache append, or LoRA personalization, which is the surface our scheme targets; conversely, our scheme does not provide the formal cryptographic guarantees that CryptoNets / Gazelle / Delphi / SecureML / MiniONN provide.' Drop the implicit 'ours wins' framing.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `PWR-02`
  - Severity: **medium**
  - Dimension: `prior_work_comparison_risk`
  - Location: sec:eval:rq13 (Setup paragraph)
  - Risky wording or missing explanation: 'We implement the primitive(s) of each named prior work directly from its paper formula and execute them under the same CPU local runtime API as our scheme.' Good. But the sentence is dense and could be misread as 'we reproduced these papers.'
  - Why a reviewer may object: Reviewer may quote this sentence in isolation and claim we are overstating reproduction.
  - Recommended revision: Add the disclaimer in the *same* sentence: 'We implement the named primitive of each prior work (NOT the full system) directly from its paper formula and execute that single primitive under the same CPU local runtime API as our scheme.' Then list which papers and which primitive.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `PWR-03`
  - Severity: **medium**
  - Dimension: `prior_work_comparison_risk`
  - Location: sec:eval:rq13 + sec:related (Privacy-preserving ML inference)
  - Risky wording or missing explanation: Related work says 'They give formal-security guarantees but pay one to several orders of magnitude in overhead.' RQ13 reports local CPU runtime for ours and explicitly refuses runtime for cost-model rows. A reviewer may still try to compute an unfair ratio.
  - Why a reviewer may object: Reviewers can mis-attribute a comparison if both 'we are faster' and 'they are formal' appear within a few pages of each other.
  - Recommended revision: Add an explicit sentence to the RQ13 setup: 'Runtime numbers in the cost-model rows are deliberately left blank (NaN). Any ratio computed against an imputed runtime would be meaningless because the cryptographic protocols are not executed.' Mirror this in sec:related.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `PWR-04`
  - Severity: **low**
  - Dimension: `prior_work_comparison_risk`
  - Location: sec:eval:rq13 (Result) + tables/direct_prior_work_comparison.tex
  - Risky wording or missing explanation: Result paragraph mentions Slalom's 'Freivalds-style randomised verification check' which sounds like a security claim; we should anchor it to integrity, not confidentiality.
  - Why a reviewer may object: A cryptographic reviewer who reads only the Result paragraph may flag this as confusing the integrity property with a privacy property.
  - Recommended revision: Re-word as 'a Freivalds-style randomised integrity check (not a privacy primitive)'.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

- **Risk ID:** `BAS-01`
  - Severity: **high**
  - Dimension: `baseline_fairness_risk`
  - Location: sec:eval:rq13 + sec:related
  - Risky wording or missing explanation: Ours rows have full_system_reproduced=True; prior-work rows have False. Comparing 'we, full system' to 'prior work, primitive only' is unfair if not framed.
  - Why a reviewer may object: This is the single most likely reject reason from a systems reviewer: 'unfair comparison.'
  - Recommended revision: Add an upfront paragraph to sec:eval:rq13: 'We compare *primitives*. For our scheme the primitive *is* the full generation path; for the prior-work rows the primitive is one of: delegated linear (Slalom), additive sharing (DarKnight), static PHQ (Amulet), polynomial activation skeleton (CryptoNets), or a cost-model row (Gazelle / Delphi / SecureML / MiniONN). Direct runtime / threat-model comparison across rows is not valid; the table reports each row in its own scope.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `BAS-02`
  - Severity: **medium**
  - Dimension: `baseline_fairness_risk`
  - Location: sec:eval:rq13 (Result) + sec:related (FHE and MPC for private inference)
  - Risky wording or missing explanation: Related work says 'Our approach is complementary, not a replacement.' Good. RQ13 result paragraph could echo this more loudly, especially next to the CryptoNets / Gazelle / Delphi / SecureML / MiniONN rows.
  - Why a reviewer may object: Reviewer may otherwise read the table as 'we beat HE/MPC.'
  - Recommended revision: Add one sentence at the end of RQ13 Result: 'Cryptographic baselines provide formal-security guarantees that our scheme does not provide; the comparison is *not* a security ranking.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `BAS-03`
  - Severity: **medium**
  - Dimension: `baseline_fairness_risk`
  - Location: sec:eval:rq9 (Baseline comparison)
  - Risky wording or missing explanation: RQ9 compares ten *internal* variants. The word 'baseline' may be misread as 'external system' if the reader is in a hurry.
  - Why a reviewer may object: Reviewer who jumps to RQ9 first may expect Slalom / Amulet rows and not find them; the external comparison is in RQ13.
  - Recommended revision: Rename RQ9 prose 'mitigation-configuration comparison' or 'internal ablation baseline comparison' and add a one-line forward reference: 'External prior-work primitives are compared in RQ13.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

