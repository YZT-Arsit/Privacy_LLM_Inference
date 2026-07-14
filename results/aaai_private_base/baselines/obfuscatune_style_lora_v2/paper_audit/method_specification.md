# Paper-derived method specification

This document treats `papers/ObfuscaTune-AAAI.pdf` as the primary specification.

## Explicitly described

1. The model provider sends the proprietary plaintext model into a TEE.
2. Attention and MLP linear weights are obfuscated inside the TEE and then
   placed outside it. Embedding, output, normalization, dropout, and other
   low-parameter/nonlinear layers remain in the TEE.
3. The data owner sends encrypted data to the TEE. Tokenization may occur at
   the owner or in the TEE. Input and output text remain inside the TEE.
4. For an input-side projection, with math-layout weight `W`, the paper uses
   `X* = X R_a` and `W* = R_a^{-1} W`, so `X* W* = X W`. Consequently the
   untrusted device sees true Q, K, and V.
5. For an output projection, `W_o* = W_o R_b`, the untrusted device produces
   `O* = H W_o*`, and the TEE restores `O = O* R_b^{-1}`. Output bias is added
   after restoration in the TEE.
6. Nonlinear operations cannot be evaluated in a general transformed basis.
   Layer normalization, attention softmax, and MLP activation therefore use
   de-obfuscated/plaintext values in the TEE.
7. Orthogonal random matrices are used so the condition number is one and the
   inverse is the transpose. The paper permits per-block matrices and periodic
   refresh.
8. Experiments use LoRA on all linear and attention layers. LoRA parameters are
   randomly initialized and placed outside the TEE.
9. The final output layer and finetuning loss are evaluated in the TEE. The
   paper states that backpropagation and parameter updates are performed for
   finetuning, but does not give their operator/message decomposition.

## Source locations

| Specification item | Paper location | Status |
|---|---|---|
| Stakeholders and honest-but-curious cloud | p.2, Sec. 2 | PAPER_EXPLICIT |
| TEE provisioning and big-linear transformation | p.2, Sec. 4 | PAPER_EXPLICIT |
| Private input/output lifecycle | p.3, Fig. 1 and following paragraph | PAPER_EXPLICIT |
| Input/output linear algebra | pp.3-4, Eq. 1-6 | PAPER_EXPLICIT |
| Per-block trusted/untrusted split | p.4, Fig. 2 | PAPER_EXPLICIT |
| Bias/nonlinearity trusted placement | p.4, paragraph below Fig. 2 | PAPER_EXPLICIT |
| Orthogonal matrices and refresh | p.5, Sec. 4 | PAPER_EXPLICIT |
| LoRA outside TEE and target scope | p.5, Sec. 5 | PAPER_EXPLICIT |
| Backpropagation and updates exist | p.3, end of feedforward paragraph | PAPER_EXPLICIT_BUT_UNDERSPECIFIED |
| LoRA coordinate equations | absent | UNSUPPORTED |
| Optimizer location/state | absent | UNSUPPORTED |
| Checkpoint/export/handoff | absent | UNSUPPORTED |

## Inferable from equations and figures

- The first projection in attention or MLP cancels the input transform, so its
  output is plaintext on the untrusted device before a TEE nonlinearity.
- Residual additions require compatible plaintext coordinates and are shown in
  the trusted portion of Figure 2.
- The second projection consumes a plaintext nonlinear result and produces a
  transformed output returned to the TEE.
- Backpropagation must reverse each forward boundary and apply the corresponding
  transpose Jacobians. This is mathematically inferable, but the paper does not
  define batching, message aggregation, or ownership of saved activations.

## Absent or ambiguous

- The transformed representation of LoRA A and B is not given.
- The paper does not state whether Adam/AdamW moments live in the TEE or outside,
  nor whether updates occur in plaintext or transformed coordinates.
- Dense orthogonal changes of basis do not generally commute with coordinatewise
  Adam/AdamW. Therefore transformed-coordinate AdamW must not be described as
  plaintext-trajectory equivalent without a separate proof.
- Gradient clipping, mixed-precision/master-state policy, checkpoint schema,
  adapter export, restart, and post-training handoff are absent.
- The paper assumes authentication but does not specify remote attestation.
- KV-cache protection and decoder-specific cache lifecycle are not discussed.

## Repository-specific Qwen adaptation

- RMSNorm remains trusted; variance is accumulated in FP32 before cast-back.
- RoPE is applied to plaintext Q/K in the trusted runtime.
- GQA repeats K/V after RoPE; the cache holds pre-repeat plaintext K/V, matching
  the paper's stated exposure of intermediate Q/K/V rather than adding a new
  protection claim.
- SwiGLU gate/up projections execute outside, while SiLU and elementwise product
  execute trusted; down projection executes outside and returns transformed
  output.
- All-seven Qwen LoRA targets are the closest mapping to the paper's “all linear
  and attention layers”, subject to correctness tests.

## Conservative training interpretation

The v2 implementation will distinguish paper facts from adaptation. LoRA factors
are owned outside as the paper states. Their exact storage coordinates and
optimizer semantics will be frozen in a versioned contract. If the chosen
outside-coordinate optimizer differs from plaintext AdamW, the result is an
empirical ObfuscaTune-style recipe, not strict numerical equivalence.

## Legacy repository correction

The existing `protocol.py` declaration `supports_lora_training=True` is labeled
`LEGACY_DECLARATION_NOT_BACKED_BY_REAL_QWEN_TRAINING_RUNTIME`. It describes the
paper's claimed capability, not an executable repository feature. The shared
registry is intentionally unchanged during active experiments.
