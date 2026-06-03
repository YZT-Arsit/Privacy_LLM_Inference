# Revision Plan -- Stage 7.6c -> Stage 7.6d

_Priority bands for the paper revision patch. No new experiment is required at any band; every item is a wording or structural LaTeX edit._

## P0 -- must fix before submission

- `NOV-01` (high, `novelty_risk`)
  - Location: sec:introduction (Our approach) + sec:related (Amulet-style and matrix obfuscation methods)
  - Recommended revision: Add a one-paragraph 'What is new vs. Amulet' explicit contrast in sec:related enumerating: (i) generation-compatible right-masking (Amulet KV-append counterexample, RQ13); (ii) operator-compatible nonlinear islands with proved invariance groups (Theorems 2-5); (iii) private LoRA backward + rank padding (Theorems 7-9); (iv) deployable trusted-controller / accelerator-backend split (sec:design:backend, RQ14). Cross-reference each item to its theorem and RQ.

- `THR-01` (high, `threat_model_risk`)
  - Location: sec:threat (Allowed leakage) + sec:limitations item 9
  - Recommended revision: Promote the seq-length leak into its own bullet in 'Allowed leakage' and reference sec:limitations item 9 explicitly. State that batch shape, per-layer tensor shape, and seq-len are visible by default.

- `THR-02` (high, `threat_model_risk`)
  - Location: sec:threat (Trusted-side controller) + sec:abstract + figures/captions
  - Recommended revision: Audit every figure caption (fig:system-overview, fig:right-masked-decode, fig:dense-sandwich, fig:nonlinear-island, fig:lora-training, fig:measured-runtime-summary, fig:security-risk-matrix) to confirm each occurrence of 'trusted controller' is co-located with 'locally emulated' or an equivalent disclaimer.

- `PWR-01` (high, `prior_work_comparison_risk`)
  - Location: sec:eval:rq13 (Cross-cutting observations final paragraph)
  - Recommended revision: Re-word as: 'The direct prior-work primitives we implement target different problems and threat models. Their primitive surface does not cover decoder-only generation, KV-cache append, or LoRA personalization, which is the surface our scheme targets; conversely, our scheme does not provide the formal cryptographic guarantees that CryptoNets / Gazelle / Delphi / SecureML / MiniONN provide.' Drop the implicit 'ours wins' framing.

- `RUN-01` (high, `runtime_deployment_risk`)
  - Location: sec:design:backend (last sentence) + sec:eval:rq14 (Result + Limitation)
  - Recommended revision: Rename in *prose* to 'backend-agnostic interface flag' or 'interface-ready flag' in every sentence referencing it; keep the raw artifact key in the table caption with an explicit gloss: 'tee_gpu_ready_interface = True means the protocol logic is backend-agnostic; no TEE or GPU has been deployed in this artifact.' Optionally rename the column in the rendered table as a P1 follow-up.

- `RUN-02` (high, `runtime_deployment_risk`)
  - Location: sec:eval:rq7 (Result) + tables/measured_runtime.tex + figures/measured_runtime_summary.png caption
  - Recommended revision: Prepend every ms number with 'local-emulation ' in prose; ensure each runtime table / figure caption begins with 'Local-emulation runtime on tiny tiles; not real TEE wall-time, not GPU throughput.' Already partly present; audit for consistency.

- `SEC-01` (high, `security_claim_risk`)
  - Location: sec:security (Transcript obfuscation, Activation recovery and linkability)
  - Recommended revision: Replace 'bounded' with 'the measured attacker accuracy stayed close to random chance' or 'did not deviate significantly from random chance in our tests' throughout sec:security.

- `EVA-01` (high, `evaluation_sufficiency_risk`)
  - Location: sec:eval:rq1 / rq5 / rq8 / rq11 + sec:limitations item 5
  - Recommended revision: Append 'on synthetic / tiny-HF configurations only' to the Result line of RQ1, RQ5, RQ8, RQ11 so a reviewer skimming Result paragraphs sees the scope immediately, not only in the Limitation paragraph.

- `BAS-01` (high, `baseline_fairness_risk`)
  - Location: sec:eval:rq13 + sec:related
  - Recommended revision: Add an upfront paragraph to sec:eval:rq13: 'We compare *primitives*. For our scheme the primitive *is* the full generation path; for the prior-work rows the primitive is one of: delegated linear (Slalom), additive sharing (DarKnight), static PHQ (Amulet), polynomial activation skeleton (CryptoNets), or a cost-model row (Gazelle / Delphi / SecureML / MiniONN). Direct runtime / threat-model comparison across rows is not valid; the table reports each row in its own scope.'

- `WRD-01` (high, `wording_risk`)
  - Location: all sections (automated wording scan, see unsafe_wording_review.md)
  - Recommended revision: See unsafe_wording_review.md for per-occurrence classification (safe / risky / unsafe) and per-occurrence revision suggestion.

## P1 -- should fix before submission

- `NOV-02` (medium, `novelty_risk`)
  - Location: sec:introduction (Why prior obfuscation does not cover generative LLMs)
  - Recommended revision: Add one sentence to the introduction: 'Our threat model assumes the base-model weights are public; we protect the *user's* runtime data (prompt, hidden states, KV cache, LoRA adapter, gradients). Model weight extraction is out of scope.' Cross-reference sec:threat 'Allowed leakage'.

- `THR-03` (medium, `threat_model_risk`)
  - Location: sec:threat (Adversary capabilities exercised in evaluation)
  - Recommended revision: Add one sentence to 'Adversary capabilities' enumerating attackers we do not evaluate: large-scale learning-based inverters trained on millions of (masked, plain) pairs; white-box-weight inverters; hardware-side-channel attackers; multi-tenant trace correlation attackers.

- `THR-04` (medium, `threat_model_risk`)
  - Location: sec:threat (Out of scope) + sec:design:backend
  - Recommended revision: Add one out-of-scope bullet: 'Real-hardware deployment of the trusted controller / accelerator backend split. The runtime API is interface-ready (see sec:design:backend); only the LocalCPUBackend is implemented.'

- `PWR-02` (medium, `prior_work_comparison_risk`)
  - Location: sec:eval:rq13 (Setup paragraph)
  - Recommended revision: Add the disclaimer in the *same* sentence: 'We implement the named primitive of each prior work (NOT the full system) directly from its paper formula and execute that single primitive under the same CPU local runtime API as our scheme.' Then list which papers and which primitive.

- `PWR-03` (medium, `prior_work_comparison_risk`)
  - Location: sec:eval:rq13 + sec:related (Privacy-preserving ML inference)
  - Recommended revision: Add an explicit sentence to the RQ13 setup: 'Runtime numbers in the cost-model rows are deliberately left blank (NaN). Any ratio computed against an imputed runtime would be meaningless because the cryptographic protocols are not executed.' Mirror this in sec:related.

- `RUN-03` (medium, `runtime_deployment_risk`)
  - Location: sec:eval:summary (Cross-cutting observations)
  - Recommended revision: Soften to 'so the protocol logic does not block a future TEE or GPU deployment; we do not claim that the remaining engineering (attestation, sealed memory, side-channel hardening, multi-tenant isolation) is trivial.'

- `RUN-04` (medium, `runtime_deployment_risk`)
  - Location: docs/runtime_boundary.md sections 4-5
  - Recommended revision: Prefix the section header with '(future work)' and add a banner sentence at the top: 'No TEE backend exists in this artifact; the snippet below is the contract a future deployment must satisfy.'

- `SEC-02` (medium, `security_claim_risk`)
  - Location: sec:security:adapter, sec:security:gradient
  - Recommended revision: Add 'AUC reduced from 0.963 to 0.500 (\Delta\text{AUC} = +0.463 vs the fixed_masks_fixed_u baseline)' style absolute numbers, or reference the specific table cell in lora_training_summary.

- `SEC-03` (medium, `security_claim_risk`)
  - Location: sec:security:rank + sec:security:boundary
  - Recommended revision: Be uniform: 'Rank padding removes the *true* rank r_true from tensor shape but leaves the *padded* rank r_pad visible.' Use 'removes from shape' rather than 'hides'.

- `COR-01` (medium, `correctness_proof_risk`)
  - Location: sec:correctness (Theorem 7 LoRA masked forward, Theorem 8 LoRA backward)
  - Recommended revision: Add a one-paragraph row-vector convention statement at the top of sec:correctness. Expand Theorem 8 proof to spell out the U^{-1} and N_out^{-1} substitutions explicitly. Cross-reference the artifact row lora_backward_experiments with max-grad-error ~ 1.3e-15.

- `EVA-02` (medium, `evaluation_sufficiency_risk`)
  - Location: sec:eval:rq3 (Workload profile)
  - Recommended revision: Rename 'amulet_style_reference' to 'amulet_static_phq_cost_model' in the narrative paragraph (the artifact key may stay for backward compatibility, but the prose should be precise).

- `EVA-03` (medium, `evaluation_sufficiency_risk`)
  - Location: sec:eval:rq6 (Rank-inference and timing)
  - Recommended revision: After each 'high' or 'medium' status add a sentence: 'This is reported as open; heterogeneous padded rank (sec:conclusion item c) is the proposed follow-up.'

- `BAS-02` (medium, `baseline_fairness_risk`)
  - Location: sec:eval:rq13 (Result) + sec:related (FHE and MPC for private inference)
  - Recommended revision: Add one sentence at the end of RQ13 Result: 'Cryptographic baselines provide formal-security guarantees that our scheme does not provide; the comparison is *not* a security ranking.'

- `BAS-03` (medium, `baseline_fairness_risk`)
  - Location: sec:eval:rq9 (Baseline comparison)
  - Recommended revision: Rename RQ9 prose 'mitigation-configuration comparison' or 'internal ablation baseline comparison' and add a one-line forward reference: 'External prior-work primitives are compared in RQ13.'

## P2 -- nice to fix

- `NOV-03` (medium, `novelty_risk`)
  - Location: sec:design:backend + sec:eval:rq14
  - Recommended revision: Add one paragraph at the end of sec:design:backend explaining that this boundary makes the protocol logic and the hardware substrate orthogonal -- the same TrustedController is reused across LocalCPUBackend and any future TEE / GPU backend without touching the masking code. Keep the 'not deployed' caveat.

- `PWR-04` (low, `prior_work_comparison_risk`)
  - Location: sec:eval:rq13 (Result) + tables/direct_prior_work_comparison.tex
  - Recommended revision: Re-word as 'a Freivalds-style randomised integrity check (not a privacy primitive)'.

- `SEC-04` (low, `security_claim_risk`)
  - Location: sec:security:timing
  - Recommended revision: Append 'under our cost-model proxy only; hardware timing side-channels are out of scope' explicitly at the end of sec:security:timing.

- `COR-02` (medium, `correctness_proof_risk`)
  - Location: sec:correctness (Theorem 2 pointwise activation permutation island)
  - Recommended revision: Add a 'Remark (non-commutation under dense masks)' after Theorem 2: for generic dense invertible N and pointwise phi, phi(XN) != phi(X) N. Cite an explicit two-line counterexample (phi = ReLU, N a 2x2 rotation).

- `COR-03` (low, `correctness_proof_risk`)
  - Location: sec:correctness (Theorem 6 attention and KV cache invariant)
  - Recommended revision: Promote 'and the V mask N_V is absorbed into the trailing output projection' into Theorem 6's statement.

- `EVA-04` (low, `evaluation_sufficiency_risk`)
  - Location: sec:eval:rq2 + sec:limitations item 9 (sequence length leak)
  - Recommended revision: Append 'we do not measure prompt-length or seq-length recovery; this is explicit in sec:limitations item 9' to RQ2 Limitation paragraph.

## P3 -- appendix / artifact only

- `COR-04` (low, `correctness_proof_risk`)
  - Location: sec:correctness (Theorem 9 rank padding factor-product equality)
  - Recommended revision: Add a one-line proof sketch for the tracked-correction case: when A_dummy B_dummy = Delta, A_pad B_pad = A B + Delta; the trusted side subtracts Delta at recovery time, restoring A B.

