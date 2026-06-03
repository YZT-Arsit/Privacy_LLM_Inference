# Novelty Positioning Review

Reviewer-risk pass focused on novelty positioning. Goal: ensure the four contributions (generation-compatible right masking, operator-compatible nonlinear islands, private LoRA training, deployable runtime API) are not dismissable as 'Amulet plus LoRA' or as engineering plumbing.

## Risk items

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

