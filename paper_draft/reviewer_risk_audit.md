# Reviewer Risk Audit -- Stage 7.6c

_Static audit of paper_draft + paper_results. No experiment was run, no outputs/ or paper_results/ file was modified. Wording classification is lexical; risk-catalogue entries are encoded from the Stage 7.6c reading of the paper body._

## 1. Executive Summary

- Total risk items: **31**.
- Severity distribution: `high`=**10**, `low`=**5**, `medium`=**16**.
- Dimension coverage: `baseline_fairness_risk`=3, `correctness_proof_risk`=4, `evaluation_sufficiency_risk`=4, `novelty_risk`=3, `prior_work_comparison_risk`=4, `runtime_deployment_risk`=4, `security_claim_risk`=4, `threat_model_risk`=4, `wording_risk`=1.
- Simulated reviewer questions answered: 10/12 (partial: 2, missing: 0).
- Wording scan counts: `risky`=58, `safe`=35.

Bottom line: the paper body is heavily hedged. The remaining risks are wording overclaim, comparability framing (RQ13), local-emulation runtime framing (RQ7 / RQ12 / RQ14), and a few correctness-proof gaps. None of the P0 items requires a new experiment.

## 2. Top-10 Reviewer Risks

### Rank 1: `BAS-01` -- high -- `baseline_fairness_risk`

- Location: sec:eval:rq13 + sec:related
- Risky wording or missing explanation: Ours rows have full_system_reproduced=True; prior-work rows have False. Comparing 'we, full system' to 'prior work, primitive only' is unfair if not framed.
- Why a reviewer may object: This is the single most likely reject reason from a systems reviewer: 'unfair comparison.'
- Recommended revision: Add an upfront paragraph to sec:eval:rq13: 'We compare *primitives*. For our scheme the primitive *is* the full generation path; for the prior-work rows the primitive is one of: delegated linear (Slalom), additive sharing (DarKnight), static PHQ (Amulet), polynomial activation skeleton (CryptoNets), or a cost-model row (Gazelle / Delphi / SecureML / MiniONN). Direct runtime / threat-model comparison across rows is not valid; the table reports each row in its own scope.'
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 2: `EVA-01` -- high -- `evaluation_sufficiency_risk`

- Location: sec:eval:rq1 / rq5 / rq8 / rq11 + sec:limitations item 5
- Risky wording or missing explanation: Every correctness / LoRA / toy-task / stability RQ uses 'synthetic' or 'tiny-HF' configurations. The paper does name this as a limitation, but the evaluation section reads as 'we evaluated' rather than 'we evaluated on synthetic tiles only'.
- Why a reviewer may object: Reviewer will say 'this is not a real LLM evaluation.' We cannot add new experiments under Stage 7.6c; we must surface the limitation in every RQ result paragraph rather than only in sec:limitations.
- Recommended revision: Append 'on synthetic / tiny-HF configurations only' to the Result line of RQ1, RQ5, RQ8, RQ11 so a reviewer skimming Result paragraphs sees the scope immediately, not only in the Limitation paragraph.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 3: `NOV-01` -- high -- `novelty_risk`

- Location: sec:introduction (Our approach) + sec:related (Amulet-style and matrix obfuscation methods)
- Risky wording or missing explanation: The four ingredients in 'Our approach' read close to 'Amulet right-mask + LoRA'. Related work distinguishes Amulet but the contrast is one paragraph and uses 'Amulet-style' phrasing that may be skimmed as 'we are an Amulet variant'.
- Why a reviewer may object: A security reviewer who reads only the intro and related work may conclude the system is Amulet + LoRA + island wrappers, i.e. an engineering extension rather than a separate point on the design space.
- Recommended revision: Add a one-paragraph 'What is new vs. Amulet' explicit contrast in sec:related enumerating: (i) generation-compatible right-masking (Amulet KV-append counterexample, RQ13); (ii) operator-compatible nonlinear islands with proved invariance groups (Theorems 2-5); (iii) private LoRA backward + rank padding (Theorems 7-9); (iv) deployable trusted-controller / accelerator-backend split (sec:design:backend, RQ14). Cross-reference each item to its theorem and RQ.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 4: `PWR-01` -- high -- `prior_work_comparison_risk`

- Location: sec:eval:rq13 (Cross-cutting observations final paragraph)
- Risky wording or missing explanation: '... highlights which primitives extend to generative LLMs (ours) and which do not (Slalom, DarKnight, Amulet static masking, CryptoNets arithmetic skeleton).' Reads as 'ours is better than them'.
- Why a reviewer may object: Each baseline solves a different problem under a different threat model. Saying 'they do not extend to generative LLMs' invites the rebuttal 'because they were not designed for generative LLMs, and you do not provide their formal-security property in return.'
- Recommended revision: Re-word as: 'The direct prior-work primitives we implement target different problems and threat models. Their primitive surface does not cover decoder-only generation, KV-cache append, or LoRA personalization, which is the surface our scheme targets; conversely, our scheme does not provide the formal cryptographic guarantees that CryptoNets / Gazelle / Delphi / SecureML / MiniONN provide.' Drop the implicit 'ours wins' framing.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 5: `RUN-01` -- high -- `runtime_deployment_risk`

- Location: sec:design:backend (last sentence) + sec:eval:rq14 (Result + Limitation)
- Risky wording or missing explanation: 'the interface is backend-ready, not backend-deployed' / 'tee_gpu_ready_interface=True means the protocol logic is backend-agnostic; it does NOT mean hardware isolation has been deployed.' Wording is correct, but the *name* 'tee_gpu_ready_interface' invites confusion.
- Why a reviewer may object: A skim reviewer who reads only a table cell labelled 'tee_gpu_ready_interface = True' may conclude TEE / GPU deployment has happened.
- Recommended revision: Rename in *prose* to 'backend-agnostic interface flag' or 'interface-ready flag' in every sentence referencing it; keep the raw artifact key in the table caption with an explicit gloss: 'tee_gpu_ready_interface = True means the protocol logic is backend-agnostic; no TEE or GPU has been deployed in this artifact.' Optionally rename the column in the rendered table as a P1 follow-up.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 6: `RUN-02` -- high -- `runtime_deployment_risk`

- Location: sec:eval:rq7 (Result) + tables/measured_runtime.tex + figures/measured_runtime_summary.png caption
- Risky wording or missing explanation: RQ7 reports 'Mean times: plain_synthetic_linear 0.002 ms, ..., multi_layer_lora_training_step 4.684 ms.' Numbers in ms invite the reading 'these are real-system latencies'.
- Why a reviewer may object: A reviewer pulled to the runtime table first may assume the numbers are comparable to production deployments.
- Recommended revision: Prepend every ms number with 'local-emulation ' in prose; ensure each runtime table / figure caption begins with 'Local-emulation runtime on tiny tiles; not real TEE wall-time, not GPU throughput.' Already partly present; audit for consistency.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 7: `SEC-01` -- high -- `security_claim_risk`

- Location: sec:security (Transcript obfuscation, Activation recovery and linkability)
- Risky wording or missing explanation: Repeated phrase 'Bounded near random chance in our evaluated setting.' / 'keeps the worst-case attacker close to random chance'. The italicised wording 'Bounded' may be read as 'we prove a bound.'
- Why a reviewer may object: Cryptographic reviewer will say 'bounded' implies a proof; you only show empirical accuracy near 0.5.
- Recommended revision: Replace 'bounded' with 'the measured attacker accuracy stayed close to random chance' or 'did not deviate significantly from random chance in our tests' throughout sec:security.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 8: `THR-01` -- high -- `threat_model_risk`

- Location: sec:threat (Allowed leakage) + sec:limitations item 9
- Risky wording or missing explanation: Sequence-length leakage is mentioned once ('unless a separate sequence-length pad is engaged (not implemented)'). Reviewer may underweight this and miss that we do not currently pad seq-length.
- Why a reviewer may object: Sequence-length is a classical side-channel for prompt-content recovery in LLM serving. A reviewer who sees no explicit treatment will demand it.
- Recommended revision: Promote the seq-length leak into its own bullet in 'Allowed leakage' and reference sec:limitations item 9 explicitly. State that batch shape, per-layer tensor shape, and seq-len are visible by default.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 9: `THR-02` -- high -- `threat_model_risk`

- Location: sec:threat (Trusted-side controller) + sec:abstract + figures/captions
- Risky wording or missing explanation: Trusted-side controller is correctly described as 'TEE-like, locally emulated' in sec:threat, but the qualifier is not always co-located with 'trusted controller' in figures and captions.
- Why a reviewer may object: A skim reviewer who only reads a figure caption may not notice the 'locally emulated' qualifier and conclude that a real TEE was deployed.
- Recommended revision: Audit every figure caption (fig:system-overview, fig:right-masked-decode, fig:dense-sandwich, fig:nonlinear-island, fig:lora-training, fig:measured-runtime-summary, fig:security-risk-matrix) to confirm each occurrence of 'trusted controller' is co-located with 'locally emulated' or an equivalent disclaimer.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

### Rank 10: `WRD-01` -- high -- `wording_risk`

- Location: all sections (automated wording scan, see unsafe_wording_review.md)
- Risky wording or missing explanation: Automated scan of the LaTeX body for dangerous words (secure, guarantee, private, protect, hide, outperform, SOTA, TEE-ready, GPU-ready, real-time, production, full system, reproduced).
- Why a reviewer may object: Any single occurrence outside a 'we do not claim' clause is a quotable overclaim.
- Recommended revision: See unsafe_wording_review.md for per-occurrence classification (safe / risky / unsafe) and per-occurrence revision suggestion.
- New experiment needed: **no**; wording fix enough: **yes**; priority: `P0`

## 3. Novelty Risk

- **Risk ID:** `NOV-01`
  - Severity: **high**
  - Dimension: `novelty_risk`
  - Location: sec:introduction (Our approach) + sec:related (Amulet-style and matrix obfuscation methods)
  - Risky wording or missing explanation: The four ingredients in 'Our approach' read close to 'Amulet right-mask + LoRA'. Related work distinguishes Amulet but the contrast is one paragraph and uses 'Amulet-style' phrasing that may be skimmed as 'we are an Amulet variant'.
  - Why a reviewer may object: A security reviewer who reads only the intro and related work may conclude the system is Amulet + LoRA + island wrappers, i.e. an engineering extension rather than a separate point on the design space.
  - Recommended revision: Add a one-paragraph 'What is new vs. Amulet' explicit contrast in sec:related enumerating: (i) generation-compatible right-masking (Amulet KV-append counterexample, RQ13); (ii) operator-compatible nonlinear islands with proved invariance groups (Theorems 2-5); (iii) private LoRA backward + rank padding (Theorems 7-9); (iv) deployable trusted-controller / accelerator-backend split (sec:design:backend, RQ14). Cross-reference each item to its theorem and RQ.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `NOV-02`
  - Severity: **medium**
  - Dimension: `novelty_risk`
  - Location: sec:introduction (Why prior obfuscation does not cover generative LLMs)
  - Risky wording or missing explanation: Cites 'Amulet-style matrix obfuscation' and 'secret-shared offload' together. The LoRA / KV-append / RoPE / GQA novelty list is good but the public-base-model vs. user-data distinction is implicit, not stated.
  - Why a reviewer may object: Reviewers from the cryptographic privacy community routinely ask 'why are you protecting the activations instead of the model?' If the public-base-model assumption is not stated up-front, the contribution looks confused.
  - Recommended revision: Add one sentence to the introduction: 'Our threat model assumes the base-model weights are public; we protect the *user's* runtime data (prompt, hidden states, KV cache, LoRA adapter, gradients). Model weight extraction is out of scope.' Cross-reference sec:threat 'Allowed leakage'.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `NOV-03`
  - Severity: **medium**
  - Dimension: `novelty_risk`
  - Location: sec:design:backend + sec:eval:rq14
  - Risky wording or missing explanation: RQ14 introduces the runtime API but does not explicitly position it as a *deployment-readiness primitive*, separate from the masking primitives. Reviewers may read it as plumbing rather than contribution.
  - Why a reviewer may object: Systems reviewers may say 'the backend split is engineering, not novelty'. The argument that the split makes future TEE/GPU swap possible without protocol changes should be made explicit, not implicit.
  - Recommended revision: Add one paragraph at the end of sec:design:backend explaining that this boundary makes the protocol logic and the hardware substrate orthogonal -- the same TrustedController is reused across LocalCPUBackend and any future TEE / GPU backend without touching the masking code. Keep the 'not deployed' caveat.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

## 4. Threat Model Risk

- **Risk ID:** `THR-01`
  - Severity: **high**
  - Dimension: `threat_model_risk`
  - Location: sec:threat (Allowed leakage) + sec:limitations item 9
  - Risky wording or missing explanation: Sequence-length leakage is mentioned once ('unless a separate sequence-length pad is engaged (not implemented)'). Reviewer may underweight this and miss that we do not currently pad seq-length.
  - Why a reviewer may object: Sequence-length is a classical side-channel for prompt-content recovery in LLM serving. A reviewer who sees no explicit treatment will demand it.
  - Recommended revision: Promote the seq-length leak into its own bullet in 'Allowed leakage' and reference sec:limitations item 9 explicitly. State that batch shape, per-layer tensor shape, and seq-len are visible by default.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `THR-02`
  - Severity: **high**
  - Dimension: `threat_model_risk`
  - Location: sec:threat (Trusted-side controller) + sec:abstract + figures/captions
  - Risky wording or missing explanation: Trusted-side controller is correctly described as 'TEE-like, locally emulated' in sec:threat, but the qualifier is not always co-located with 'trusted controller' in figures and captions.
  - Why a reviewer may object: A skim reviewer who only reads a figure caption may not notice the 'locally emulated' qualifier and conclude that a real TEE was deployed.
  - Recommended revision: Audit every figure caption (fig:system-overview, fig:right-masked-decode, fig:dense-sandwich, fig:nonlinear-island, fig:lora-training, fig:measured-runtime-summary, fig:security-risk-matrix) to confirm each occurrence of 'trusted controller' is co-located with 'locally emulated' or an equivalent disclaimer.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `THR-03`
  - Severity: **medium**
  - Dimension: `threat_model_risk`
  - Location: sec:threat (Adversary capabilities exercised in evaluation)
  - Risky wording or missing explanation: Lists ridge / MLP / signature / Sinkhorn / linkability / DLG / spectral / ensemble / cost-model attackers. Does not say what *stronger* attacker is out of scope (e.g. learning-based inverters with substantially more data, white-box weight access).
  - Why a reviewer may object: Reviewer will ask 'have you tried a stronger attacker?' We should pre-empt by naming the explicit stronger-attacker categories that we do not evaluate.
  - Recommended revision: Add one sentence to 'Adversary capabilities' enumerating attackers we do not evaluate: large-scale learning-based inverters trained on millions of (masked, plain) pairs; white-box-weight inverters; hardware-side-channel attackers; multi-tenant trace correlation attackers.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `THR-04`
  - Severity: **medium**
  - Dimension: `threat_model_risk`
  - Location: sec:threat (Out of scope) + sec:design:backend
  - Risky wording or missing explanation: Out-of-scope correctly lists compromised TEE, HW side-channels, formal security, real-TEE wall-time, full fine-tune, framework integration, padded-rank hiding, outsourced loss/optimizer. The deployable-runtime boundary contract (Stage 7.5c) is not enumerated here.
  - Why a reviewer may object: A reviewer scanning the threat model only will not realise that the runtime boundary is interface-only; the qualifier currently lives only in sec:design:backend and sec:eval:rq14.
  - Recommended revision: Add one out-of-scope bullet: 'Real-hardware deployment of the trusted controller / accelerator backend split. The runtime API is interface-ready (see sec:design:backend); only the LocalCPUBackend is implemented.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

## 5. Prior-Work Comparison Risk

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

## 6. Runtime Deployment Risk

- **Risk ID:** `RUN-01`
  - Severity: **high**
  - Dimension: `runtime_deployment_risk`
  - Location: sec:design:backend (last sentence) + sec:eval:rq14 (Result + Limitation)
  - Risky wording or missing explanation: 'the interface is backend-ready, not backend-deployed' / 'tee_gpu_ready_interface=True means the protocol logic is backend-agnostic; it does NOT mean hardware isolation has been deployed.' Wording is correct, but the *name* 'tee_gpu_ready_interface' invites confusion.
  - Why a reviewer may object: A skim reviewer who reads only a table cell labelled 'tee_gpu_ready_interface = True' may conclude TEE / GPU deployment has happened.
  - Recommended revision: Rename in *prose* to 'backend-agnostic interface flag' or 'interface-ready flag' in every sentence referencing it; keep the raw artifact key in the table caption with an explicit gloss: 'tee_gpu_ready_interface = True means the protocol logic is backend-agnostic; no TEE or GPU has been deployed in this artifact.' Optionally rename the column in the rendered table as a P1 follow-up.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `RUN-02`
  - Severity: **high**
  - Dimension: `runtime_deployment_risk`
  - Location: sec:eval:rq7 (Result) + tables/measured_runtime.tex + figures/measured_runtime_summary.png caption
  - Risky wording or missing explanation: RQ7 reports 'Mean times: plain_synthetic_linear 0.002 ms, ..., multi_layer_lora_training_step 4.684 ms.' Numbers in ms invite the reading 'these are real-system latencies'.
  - Why a reviewer may object: A reviewer pulled to the runtime table first may assume the numbers are comparable to production deployments.
  - Recommended revision: Prepend every ms number with 'local-emulation ' in prose; ensure each runtime table / figure caption begins with 'Local-emulation runtime on tiny tiles; not real TEE wall-time, not GPU throughput.' Already partly present; audit for consistency.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `RUN-03`
  - Severity: **medium**
  - Dimension: `runtime_deployment_risk`
  - Location: sec:eval:summary (Cross-cutting observations)
  - Risky wording or missing explanation: 'so a future TEE or GPU deployment only needs to swap the backend object.' Reads as 'deployment is one engineering step away.'
  - Why a reviewer may object: Reviewer may say 'this trivialises confidential-computing deployment; attestation, sealed memory, page-table side-channels are not just plumbing.'
  - Recommended revision: Soften to 'so the protocol logic does not block a future TEE or GPU deployment; we do not claim that the remaining engineering (attestation, sealed memory, side-channel hardening, multi-tenant isolation) is trivial.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `RUN-04`
  - Severity: **medium**
  - Dimension: `runtime_deployment_risk`
  - Location: docs/runtime_boundary.md sections 4-5
  - Risky wording or missing explanation: The 'How to plug in a TEE backend' section reads like a how-to. Reviewers may read it as 'they shipped a TEE backend.'
  - Why a reviewer may object: Same skim risk as RUN-01.
  - Recommended revision: Prefix the section header with '(future work)' and add a banner sentence at the top: 'No TEE backend exists in this artifact; the snippet below is the contract a future deployment must satisfy.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

## 7. Security Claim Risk

- **Risk ID:** `SEC-01`
  - Severity: **high**
  - Dimension: `security_claim_risk`
  - Location: sec:security (Transcript obfuscation, Activation recovery and linkability)
  - Risky wording or missing explanation: Repeated phrase 'Bounded near random chance in our evaluated setting.' / 'keeps the worst-case attacker close to random chance'. The italicised wording 'Bounded' may be read as 'we prove a bound.'
  - Why a reviewer may object: Cryptographic reviewer will say 'bounded' implies a proof; you only show empirical accuracy near 0.5.
  - Recommended revision: Replace 'bounded' with 'the measured attacker accuracy stayed close to random chance' or 'did not deviate significantly from random chance in our tests' throughout sec:security.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `SEC-02`
  - Severity: **medium**
  - Dimension: `security_claim_risk`
  - Location: sec:security:adapter, sec:security:gradient
  - Risky wording or missing explanation: Reports '\Delta\text{AUC} = +0.463' and '+0.478' but the baseline (fixed_masks_fixed_u) is mentioned only once. Reviewer may compute the post-mitigation AUC themselves and dispute the +0.463 phrasing if they cannot quickly find the baseline.
  - Why a reviewer may object: Numbers without their baseline are easy to challenge.
  - Recommended revision: Add 'AUC reduced from 0.963 to 0.500 (\Delta\text{AUC} = +0.463 vs the fixed_masks_fixed_u baseline)' style absolute numbers, or reference the specific table cell in lora_training_summary.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `SEC-03`
  - Severity: **medium**
  - Dimension: `security_claim_risk`
  - Location: sec:security:rank + sec:security:boundary
  - Risky wording or missing explanation: Claims boundary correctly says 'we do *not* claim ... padded LoRA rank is hidden'. But sec:security:rank uses 'hides the true rank' which is ambiguous: it could be read 'hides rank in general'.
  - Why a reviewer may object: Two adjacent sentences using 'hides' / 'is not hidden' with different subjects can confuse a reviewer in a hurry.
  - Recommended revision: Be uniform: 'Rank padding removes the *true* rank r_true from tensor shape but leaves the *padded* rank r_pad visible.' Use 'removes from shape' rather than 'hides'.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `SEC-04`
  - Severity: **low**
  - Dimension: `security_claim_risk`
  - Location: sec:security:timing
  - Risky wording or missing explanation: 'Risk level: low.' for cost-model timing classifier. 'low' is the safe label, but a reviewer may still read this as 'timing is solved.'
  - Why a reviewer may object: The cost-model proxy is not a side-channel evaluation; saying 'low' invites misreading.
  - Recommended revision: Append 'under our cost-model proxy only; hardware timing side-channels are out of scope' explicitly at the end of sec:security:timing.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

## 8. Correctness Proof Risk

- **Risk ID:** `COR-01`
  - Severity: **medium**
  - Dimension: `correctness_proof_risk`
  - Location: sec:correctness (Theorem 7 LoRA masked forward, Theorem 8 LoRA backward)
  - Risky wording or missing explanation: Theorem 7 statement omits the row-vector convention 'X in R^{T x d_in} acts from the left'. Theorem 8 proof sketch is terse and may not convince a reviewer that the U^{-1} cancellation is total.
  - Why a reviewer may object: Backward-pass identities in masked space are a classical place for sign / transpose errors. Reviewer will want a full derivation.
  - Recommended revision: Add a one-paragraph row-vector convention statement at the top of sec:correctness. Expand Theorem 8 proof to spell out the U^{-1} and N_out^{-1} substitutions explicitly. Cross-reference the artifact row lora_backward_experiments with max-grad-error ~ 1.3e-15.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `COR-02`
  - Severity: **medium**
  - Dimension: `correctness_proof_risk`
  - Location: sec:correctness (Theorem 2 pointwise activation permutation island)
  - Risky wording or missing explanation: Theorem 2 covers GELU/ReLU/SiLU pointwise. The 'dense mask does not commute' *counterexample* is mentioned in sec:design but not proved in sec:correctness.
  - Why a reviewer may object: A reviewer may ask 'why a permutation island and not a dense one?' The counterexample is the answer and should appear as a remark or lemma.
  - Recommended revision: Add a 'Remark (non-commutation under dense masks)' after Theorem 2: for generic dense invertible N and pointwise phi, phi(XN) != phi(X) N. Cite an explicit two-line counterexample (phi = ReLU, N a 2x2 rotation).
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

- **Risk ID:** `COR-03`
  - Severity: **low**
  - Dimension: `correctness_proof_risk`
  - Location: sec:correctness (Theorem 6 attention and KV cache invariant)
  - Risky wording or missing explanation: Theorem 6 covers Q K^T = Q\tilde K\tilde^T. The V mask absorption claim ('absorbed by the trailing output projection') is in the design section but not in the theorem.
  - Why a reviewer may object: Reviewer who only reads the theorems may miss the V mask story.
  - Recommended revision: Promote 'and the V mask N_V is absorbed into the trailing output projection' into Theorem 6's statement.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

- **Risk ID:** `COR-04`
  - Severity: **low**
  - Dimension: `correctness_proof_risk`
  - Location: sec:correctness (Theorem 9 rank padding factor-product equality)
  - Risky wording or missing explanation: States two cases (A_dummy B_dummy = 0 vs tracked delta) but the proof sketch covers only case 1.
  - Why a reviewer may object: Reviewer may say 'where is the delta-folding proof?'
  - Recommended revision: Add a one-line proof sketch for the tracked-correction case: when A_dummy B_dummy = Delta, A_pad B_pad = A B + Delta; the trusted side subtracts Delta at recovery time, restoring A B.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P3`

## 9. Evaluation Sufficiency Risk

- **Risk ID:** `EVA-01`
  - Severity: **high**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq1 / rq5 / rq8 / rq11 + sec:limitations item 5
  - Risky wording or missing explanation: Every correctness / LoRA / toy-task / stability RQ uses 'synthetic' or 'tiny-HF' configurations. The paper does name this as a limitation, but the evaluation section reads as 'we evaluated' rather than 'we evaluated on synthetic tiles only'.
  - Why a reviewer may object: Reviewer will say 'this is not a real LLM evaluation.' We cannot add new experiments under Stage 7.6c; we must surface the limitation in every RQ result paragraph rather than only in sec:limitations.
  - Recommended revision: Append 'on synthetic / tiny-HF configurations only' to the Result line of RQ1, RQ5, RQ8, RQ11 so a reviewer skimming Result paragraphs sees the scope immediately, not only in the Limitation paragraph.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `EVA-02`
  - Severity: **medium**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq3 (Workload profile)
  - Risky wording or missing explanation: RQ3 names 'tslp_trusted_nonlinear_baseline' and 'amulet_style_reference' as cost-model baselines but uses the word 'amulet_style' which is exactly the vague label the Stage 7.5c spec ruled out.
  - Why a reviewer may object: Stage 7.5c spec forbids 'Amulet-style' phrasing. The RQ3 cost-model table still uses the legacy label.
  - Recommended revision: Rename 'amulet_style_reference' to 'amulet_static_phq_cost_model' in the narrative paragraph (the artifact key may stay for backward compatibility, but the prose should be precise).
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `EVA-03`
  - Severity: **medium**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq6 (Rank-inference and timing)
  - Risky wording or missing explanation: Reports four 'high' risk rows (gradient rank inference, stronger-dummy spectral, stronger-dummy gradient) and one 'medium' (dummy-strategy classifier). The Result paragraph states the numbers but does not tie each 'high' to a future-work item.
  - Why a reviewer may object: Reviewer who sees 'high' without a next-step plan may ask 'so why is this a contribution?'
  - Recommended revision: After each 'high' or 'medium' status add a sentence: 'This is reported as open; heterogeneous padded rank (sec:conclusion item c) is the proposed follow-up.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `EVA-04`
  - Severity: **low**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq2 + sec:limitations item 9 (sequence length leak)
  - Risky wording or missing explanation: RQ2 covers KV cache append invariant but does not measure prompt-length leakage in the decode trace.
  - Why a reviewer may object: Reviewer may ask 'what about prompt-length recovery?' We must point to sec:limitations not promise an experiment we did not run.
  - Recommended revision: Append 'we do not measure prompt-length or seq-length recovery; this is explicit in sec:limitations item 9' to RQ2 Limitation paragraph.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

## 10. Baseline Fairness Risk

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

## 11. Wording Risk

- **Risk ID:** `WRD-01`
  - Severity: **high**
  - Dimension: `wording_risk`
  - Location: all sections (automated wording scan, see unsafe_wording_review.md)
  - Risky wording or missing explanation: Automated scan of the LaTeX body for dangerous words (secure, guarantee, private, protect, hide, outperform, SOTA, TEE-ready, GPU-ready, real-time, production, full system, reproduced).
  - Why a reviewer may object: Any single occurrence outside a 'we do not claim' clause is a quotable overclaim.
  - Recommended revision: See unsafe_wording_review.md for per-occurrence classification (safe / risky / unsafe) and per-occurrence revision suggestion.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

## 12. Simulated Reviewer Questions

### Q1: Why is this not just Amulet with LoRA?

- Answer status: **answered**
- Where answered: sec:introduction (Why prior obfuscation does not cover generative LLMs); sec:related (Amulet-style and matrix obfuscation methods); sec:eval:rq13 (Amulet static PHQ + KV-append counterexample)
- Answer summary: Three new ingredients: generation-compatible right masking (RQ13 KV-append counterexample shows Amulet's static left mask breaks), operator-compatible nonlinear islands with proved invariance groups (Theorems 2-5), and a private LoRA training path with masked backward (Theorems 7-9). The new deployable runtime API (sec:design:backend, RQ14) is a fourth ingredient.

### Q2: Why not use Slalom / DarKnight?

- Answer status: **answered**
- Where answered: sec:related (TEE-assisted ML inference); sec:eval:rq13 (Slalom delegated linear primitive, DarKnight k=2 sharing skeleton)
- Answer summary: Slalom's delegated linear is implemented directly under our runtime API (RQ13) and is correctness-equivalent on the linear primitive but does not cover generation, KV append, nonlinear islands, or LoRA. DarKnight's k=2 additive sharing also covers only the linear primitive.

### Q3: Why not use HE/MPC?

- Answer status: **answered**
- Where answered: sec:related (FHE and MPC for private inference); sec:eval:rq13 (CryptoNets / Gazelle / Delphi / SecureML / MiniONN rows)
- Answer summary: HE/MPC give formal-security guarantees but pay one to several orders of magnitude in overhead and restrict the operator set; ours sits at a different point on the spectrum (proxy-evaluated, near-GPU-native dense matmul). RQ13 records the cryptographic baselines as cost-model rows that deliberately do not produce a measured runtime.

### Q4: What exactly does the GPU see?

- Answer status: **answered**
- Where answered: sec:threat (Untrusted GPU paragraph + Allowed leakage); sec:design:linear, sec:design:right-mask
- Answer summary: X-tilde, W-tilde, A-tilde, B-tilde, K-tilde, V-tilde, Y-tilde; shapes; dtypes; per-call timing; the public base-model weight W; padded LoRA rank r_pad; output tokens (by design).

### Q5: What does the TEE do?

- Answer status: **answered**
- Where answered: sec:threat (Trusted-side controller); sec:design:runtime; sec:design:backend
- Answer summary: Holds user prompt and private context; the LoRA adapter (A, B); optimizer state; loss closure; sampler; the mask sampler and pad sampler; performs mask transform and recovery; runs the LoRA backward / optimizer step; publishes a sanitised RuntimeTranscript.

### Q6: Is the TEE implemented?

- Answer status: **answered**
- Where answered: sec:threat (Trusted-side controller -- 'emulated as a local trusted runtime; it is not a real TEE'); sec:design:backend; sec:eval:rq14; sec:limitations item 2 and item 18-19; docs/runtime_boundary.md
- Answer summary: No. The trusted controller is locally emulated. The deployable runtime API (TrustedController + AcceleratorBackend) ships exactly one backend, LocalCPUBackend. No real TEE, no real GPU, no attestation.

### Q7: Are the prior systems fully reproduced?

- Answer status: **answered**
- Where answered: sec:eval:rq13 (Setup + Limitation); tables/direct_prior_work_comparison.tex; sec:limitations items 16-17; appendix B (Stage 7.5c safe wordings)
- Answer summary: No. Each prior-work row implements only the named primitive (delegated linear, k=2 additive sharing, static PHQ, polynomial-activation skeleton) or is cost-model-only (Gazelle, Delphi, SecureML, MiniONN). full_system_reproduced is False for every non-ours row. Arrow is recorded as missing_paper_formula.

### Q8: Are runtime numbers real GPU/TEE numbers?

- Answer status: **answered**
- Where answered: sec:eval:rq3, sec:eval:rq7, sec:eval:rq12, sec:eval:rq14, sec:limitations item 3; figures/measured_runtime_summary.png caption; paper_results/summary.md sections 4 and 13
- Answer summary: No. Local-emulation latencies via time.perf_counter on small CPU tiles; wall_time_source is either projected_from_op_counts or measured_local_emulation. Not real TEE wall-time, not GPU throughput.

### Q9: What leaks through shape, length, rank, timing?

- Answer status: **partial**
- Where answered: sec:threat (Allowed leakage); sec:design:rank-pad; sec:security:rank, sec:security:timing; sec:limitations items 7, 9, 10
- Answer summary: Per-layer tensor shapes are public; padded LoRA rank r_pad is visible; sequence length is not currently padded; cost-model timing is reported near random chance under proxy_equalized but real hardware-side-channel timing is out of scope.

### Q10: Are proxy attacks enough?

- Answer status: **answered**
- Where answered: sec:security (every subsection labels itself proxy_supported); sec:security:boundary; sec:limitations items 1, 10; appendix B (unsupported claims U1)
- Answer summary: Explicitly no in the formal sense: no cryptographic or semantic-security claim is made anywhere. The proxy results are reported with 'in our tested configurations' hedge and labelled needs_more_evaluation when a stronger attacker may succeed.

### Q11: Does this work on real LLMs?

- Answer status: **partial**
- Where answered: sec:eval:rq1 (GPT-2 model wrapper + modern decoder wrapper); sec:limitations items 5, 8; sec:conclusion (next-step a, b)
- Answer summary: GPT-2 (tiny) reproduces token-for-token; the modern decoder wrapper (RMSNorm + SwiGLU + RoPE + GQA) reproduces token-for-token in synthetic or tiny-HF configurations. Full Qwen / TinyLlama / LLaMA fine-tune is explicitly out of scope.

### Q12: What claim remains if no formal security is provided?

- Answer status: **answered**
- Where answered: sec:introduction (What this paper is not); sec:security:boundary; sec:conclusion; appendix B (claims mapping)
- Answer summary: The paper makes only correctness claims (S1-S8) and proxy-supported claims (P1-P5). The contribution is: a generation-compatible masked-execution boundary that *reproduces* the plain reference for the tested decoder-only LLMs and the synthetic LoRA training path, together with a deployable runtime API and an artifact-backed claim audit. No security theorem is claimed.

## 13. Revision Priority Plan

Detailed priority bands live in `paper_draft/revision_plan.md`. Counts:
- `P0`: 10
- `P1`: 14
- `P2`: 6
- `P3`: 1

No risk item requires a new experiment. Every P0 / P1 / P2 / P3 fix is a *wording or structural* change to the LaTeX body. Stage 7.6d (paper revision patch) will execute the P0 / P1 plan.

