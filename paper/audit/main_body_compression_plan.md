# Main-Body Compression Plan

Audit date: 2026-07-14. Metric source: `paper/build/main.pdf` (6 pages), measured from PDF text and bounding-box coordinates rather than source lines. Fractions below use the two AAAI columns and the usable vertical text extent on each page.

## 1. Current page occupancy

| Material | PDF location | Approx. occupancy |
|---|---:|---:|
| Abstract + Introduction + Related Work placeholders | page 1 before §3 | 1.20 pages combined, mostly drafting placeholders |
| Problem Formulation and Threat Model | page 1, mid-left column through page 3, top-left | **1.80 pages** |
| Method | page 3, upper-left through page 6, lower-left | **3.36 pages** |
| Correctness, Security, Evaluation, Limitations, Conclusion placeholders | remainder of page 6 | 0.64 page |
| Appendices | page 6 after the main-body placeholders | shares the final page because all are placeholders |

The current paper through Method consumes about 5.36 pages of the eight-page body, leaving only about 2.64 pages before accounting for the still-unwritten main sections. This fails the required 4.5-page reserve.

## 2. Repeated material

- The Method lifecycle is described in the overview prose, Figure 1, Algorithm 1, the Runtime Protocol prose, and Algorithm 2.
- Tensor ownership is repeated in the Threat Model partition, Method overview, Runtime Protocol prose, ownership table, and both algorithms.
- Mask freshness is repeated in the partition definition, attacker taxonomy, Runtime Protocol, attention decode prose, and lifecycle algorithm.
- Prefill/decode ordering is repeated in the overview, attention subsection, and two algorithms.
- LoRA forward/backward transitions appear in overview prose, the full derivation, optimizer prose, and both algorithms.
- Security goals are split into seven objectives whose training/runtime/generation entries restate the same per-asset confidentiality goal.

## 3. Material required in the main paper

- Three administrative parties and the TEE/GPU component split inside the Cloud Service.
- Protected assets, the honest-but-curious GPU view, admitted metadata, and explicit non-goals.
- One system schematic and one compact end-to-end lifecycle algorithm.
- The transformed linear construction and trusted recovery.
- The attention compatibility condition, post-RoPE execution order, GQA/MQA tying, and session-consistent masked cache.
- Rank-padded LoRA factor transformation, separate base/adapter branches, and trusted loss/gradient recovery/optimizer locus.

## 4. Material moved to appendices

### Appendix A — Detailed Protocol and Implementation Correspondence

- Full tensor-set partition and transcript definition.
- Expanded adversary observations, freshness schedule, and non-goal list.
- Offline preparation, communication sequence, boundary crossings, tensor ownership/lifetime table, and complete unified lifecycle.
- Detailed prefill/decode ordering and implementation-path correspondence.

### Appendix B — Operator Construction

- Full linear dimensions, compensation construction, and recovery map.
- RoPE/GQA/MQA execution details; masked cache append mechanics.
- Trusted normalization and residual handling; accelerator softmax and activation handling.
- Full rank-padding, LoRA forward/backward, pad-gradient compensation, gradient recovery, and optimizer interface.

### Appendix C — Complete Proofs

- Algebraic derivations for linear execution, attention/cache composition, and LoRA forward/backward/update equivalence.
- Operator lemmas and failure conditions.

## 5. Canonical-model contradictions

- The current text defines five parties; the canonical model has Model Provider, User, and Cloud Service only.
- TEE/controller and GPU/accelerator are incorrectly presented as administrative parties rather than Cloud Service components.
- A separate LoRA owner conflicts with User ownership of training data, adapter, prompts, and results.
- Returned output tokens are currently public metadata, but generated token identities and the final result must be hidden from the GPU and returned only to the User.
- Learned biases are called public although the implementation can transform them with the output mask; they belong to the private model parameters.
- The integrity objective promises bounded detection while the implementation supplies only a limited probabilistic spot-check. Integrity and availability must be non-goals in the confidentiality model.

## 6. Unsupported or unnecessarily strong statements

- The previous confidentiality objective says the transcript gives no recovery advantage beyond public information. The evidence is proxy-evaluated, not a formal non-recovery or indistinguishability result.
- “For every input” and unrestricted output-sequence equivalence are too broad relative to floating-point validation and greedy-only generation evidence.
- The full nonlinear masked backward is not integrated; LoRA backward claims must be scoped to delegated linear boundaries with trusted upstream gradients.
- Dense-masked AdamW is not an accelerator-side exact update; the canonical interface keeps coordinatewise optimizer state and updates in the TEE.
- Integrity beyond the implemented spot-check, cryptographic/semantic indistinguishability, malicious-TEE protection, side-channel protection, and availability are unsupported.

## 7. Proposed eight-page allocation

| Section | Target pages |
|---|---:|
| Abstract | 0.20–0.25 |
| Introduction | 0.70–0.85 |
| Related Work / Background | 0.60–0.75 |
| Problem + Threat Model | 0.65–0.85 |
| Method | 1.35–1.65 |
| Correctness + Security | 1.20–1.50 |
| Evaluation | 1.60–1.90 |
| Limitations + Conclusion | 0.45–0.60 |

The immediate compiled gate is Threat Model ≤0.9 page, Method ≤1.75 pages, and at least 4.5 pages reserved after Method. No format manipulation is permitted.

## 8. Post-rewrite gate measurement

The final measurement uses `paper/build/main.pdf` bounding boxes under the
unchanged AAAI 2027 submission style. Problem/Threat begins on page 1 in the
left column and ends on page 1 in the right column when Method begins, occupying
approximately **0.69 page**. Method begins on page 1 in the right column and
ends on page 2 in the right column when Correctness begins, occupying
approximately **0.84 page**. The paper through Method therefore consumes about
**1.80 pages**, leaving about **6.20 pages** under the eight-page body limit.
The Phase 1 gate passes: latexmk exits 0; there are no undefined references,
duplicate labels, or overfull boxes; and no float blocks a section boundary.

## 9. Phase 2 result

Because the gate passed, the compact Correctness draft was authored. It occupies
approximately **0.62 page**, contains the three requested grouped theorems, and
distinguishes exact algebra from finite-precision validation. Complete
derivations and failure conditions are in Appendix C.
