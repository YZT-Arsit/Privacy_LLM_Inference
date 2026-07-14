# Selected conservative training contract

Contract ID: `OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1`

## Selection

Candidate 2 is primary: LoRA factors are stored and optimized outside the TEE in
the coordinates of each obfuscated linear layer.

| Element | Contract | Evidence class |
|---|---|---|
| Orthogonal base/activation transformation | Paper Eq. 1-6 | PAPER_EXPLICIT |
| LoRA on all linear/attention layers | Map to Qwen all-seven | PAPER_EXPLICIT plus architecture adaptation |
| LoRA factors outside TEE | GPU owns A*/B* and moments | PAPER_EXPLICIT |
| Input target factor map | `A*=R_i^T A`, `B*=B` | PAPER_INFERRED from Eq. 1-3 |
| Output target factor map | `A*=A`, `B*=B R_o` | PAPER_INFERRED from Eq. 5-6 |
| LoRA backward | Gradient equations in candidate contract | REPOSITORY_ADAPTATION |
| Optimizer | AdamW on stored coordinates | REPOSITORY_ADAPTATION |
| Optimizer equivalence claim | Explicitly none for AdamW | REPOSITORY_ADAPTATION |
| Checkpoint | A*/B*/m*/v*, transform-generation binding | REPOSITORY_ADAPTATION |
| Canonical plaintext adapter export | Not provided | UNSUPPORTED_AND_OMITTED |
| Trusted canonical optimizer | Not used in primary | UNSUPPORTED_AND_OMITTED |

## Optimizer contract

- Type: explicit AdamW over stored transformed-coordinate tensors.
- Default matched task recipe: LR 2e-4, betas (.9,.999), epsilon 1e-8,
  weight decay .01, constant LR, no warmup, no clipping.
- Authoritative parameters and m/v: external FP32.
- Runtime compute precision will be separately configured; no mixed-precision
  claim is part of the minimal oracle.
- Update order: increment step; update m; update v; bias correct; apply
  decoupled decay and adaptive update.
- Semantics: internally consistent transformed-coordinate trajectory, not
  plaintext AdamW trajectory.

## Checkpoint and inference

The checkpoint stores only transformed factors and transformed-coordinate
optimizer state plus direction, dimensions, rank/alpha, base/config/source
hashes, and transform-generation ID. It does not store R or canonical A/B.
Post-training inference can reuse the same transformed domain. Transform refresh
requires an authenticated trusted rebase and is deferred from the minimum run.

## Why this baseline is not our G2 path

G2 uses a protocol-specific split of authoritative plaintext-equivalent FP32
optimizer state and selective trusted correction. This baseline keeps every
LoRA factor and every optimizer moment on the untrusted accelerator and accepts
the non-equivalent transformed-coordinate AdamW trajectory. It exposes the
paper's true Q/K/V and plaintext KV cache and places all nonlinearities in the
trusted runtime. It neither imports nor calls the G2 runner or optimizer.

## Honest label

`faithful ObfuscaTune-style Qwen LoRA adaptation (transformed-coordinate optimizer)`.
It is not an official reproduction and not strict optimizer equivalence.
