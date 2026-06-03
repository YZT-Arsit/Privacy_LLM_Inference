# Threat Model Review

Reviewer-risk pass focused on the system / threat model. Goal: ensure every leakage channel is named, every TEE / GPU / real-hardware boundary is co-located with its 'not deployed' qualifier, and that the deployable runtime API is enumerated in the out-of-scope list.

## Risk items

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

