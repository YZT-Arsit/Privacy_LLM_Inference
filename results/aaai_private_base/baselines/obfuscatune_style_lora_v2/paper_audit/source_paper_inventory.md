# ObfuscaTune source-paper inventory

Audit date: 2026-07-14

## Canonical source

- Repository file: `papers/ObfuscaTune-AAAI.pdf`
- SHA-256: `c747ff601b62053241cc7a9a6f70da48de146aaef552b6789dbcb2142dd99b16`
- Size: 512,498 bytes; 9 pages
- Title: *ObfuscaTune: Obfuscated Offsite Finetuning and Inference of Proprietary LLMs on Private Datasets*
- Authors: Ahmed Frikha, Nassim Walha, Ricardo Mendes, Krishna Kanth Nakka, Xue Jiang, and Xuebing Zhou
- Identifier: arXiv:2407.02960v2 [cs.CR], dated 2025-01-12
- Venue marking in the PDF: PPAI-25, The 6th AAAI Workshop on Privacy-Preserving Artificial Intelligence

No other ObfuscaTune-named PDF is present under `papers/`; this file is therefore
the canonical local specification. Existing code comments and v1 audit files are
secondary evidence only.

## Sections used

- Section 2: stakeholders, honest-but-curious cloud provider, confidentiality,
  utility, and efficiency requirements.
- Section 4: TEE split, model/data flow, equations (1)-(6), Figures 1-2,
  authentication assumption, random-matrix refresh, and orthogonal transforms.
- Section 5: GPT-2/nanoGPT experiments, LoRA placement, two-device TEE
  simulation, utility, parameter split, and slowdown.
- Appendix A: training and LoRA hyperparameters.
- Appendix B: prescribed-condition-number matrix construction.

## Traceable claim inventory

| Page | Location | Paraphrased claim | Implementation implication | Basis |
|---:|---|---|---|---|
| 2 | Sec. 2, first paragraph | Cloud provider is honest-but-curious. | Correct execution is assumed; malicious-GPU integrity is out of scope. | Explicit |
| 2 | Sec. 4, model-protection paragraph | Attention/MLP weights are transformed in TEE and placed outside; small/nonlinear layers stay trusted. | Provision plaintext weights only in trusted runtime; export transformed big linears. | Explicit |
| 3 | Fig. 1 and data-protection paragraph | Private data enters TEE; text and output remain trusted. | Never send token IDs or generated text to untrusted runtime. | Explicit |
| 3-4 | Sec. 4, Eq. 1-6 and Fig. 2 | Input transform cancels for Q/K/V; output projection returns a transformed result for trusted restoration. | Untrusted Q/K/V are plaintext; o/down outputs cross in transformed coordinates. | Explicit |
| 4 | Paragraph below Fig. 2 | Output-projection bias is trusted; nonlinearities require de-obfuscated values. | Bias, norm, softmax and activation execute trusted. | Explicit |
| 4-5 | Security analysis | Authentication prevents arbitrary attacker queries; Q/K/V are exceptions to otherwise transformed external tensors. | Authentication is an assumption; do not claim label-only view or hidden Q/K/V. | Explicit |
| 5 | Numerical-error paragraph | Orthogonal matrices have condition number one and inverse by transpose. | Use orthogonal transforms for the primary baseline. | Explicit |
| 5 | Sec. 5, experiment setup | LoRA is applied to all linear/attention layers; factors are randomly initialized outside TEE. | Qwen all-seven is a justified mapping; external factor ownership is primary. | Explicit |
| 5 | Sec. 5, experiment setup | A second GPU simulates TEE. | Real Intel TDX is a repository adaptation, not an official reproduction. | Explicit |
| 9 | Appendix A | Paper uses 10 epochs, LR 3e-5, batch 1, rank 16, alpha 32, dropout .05. | Current rank/LR/batch recipe is a task-matched adaptation and must be labeled. | Explicit |
| 3 | End of feedforward paragraph | Loss is computed in TEE and backpropagation/updates occur for finetuning. | A backward path is required, but placement and messages are not specified. | Explicit statement, protocol details absent |

The paper contains no algorithm block for backward, optimizer placement,
checkpointing, adapter export, or restart. Any such behavior is an adaptation.

## Scope cautions

The paper demonstrates GPT-2 LoRA finetuning on two GPUs, one of which simulates
the TEE. It does not specify an Intel TDX network protocol, attestation wire
format, checkpoint schema, adapter export format, or fresh-process handoff.
Those features are repository-specific adaptations if implemented.
