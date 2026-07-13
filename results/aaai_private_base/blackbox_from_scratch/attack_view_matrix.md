# Attack-view matrix — not instantiated

Phase 0 failed before attack interfaces or immutable corpora could be instantiated.

| View | Intended access | Stage status |
|---|---|---|
| B0 label-only | Final label or generated tokens only | NOT RUN |
| B1 output-rich | Only additional values exposed by the deployed API | NOT RUN |
| B2 transformed transcript | All accelerator-visible transformed tensors and transcripts; no TDX/plaintext values | NOT RUN |
| B3 plaintext white-box | Plaintext weights, activations, logits, gradients and LoRA state | NOT RUN |

No attack script or corpus manifest was created because doing so after the mandatory provenance failure would imply that the frozen from-scratch threat model remained valid.
