"""TEE boundary runtime (Stage 8.3).

A *thin* trusted runtime that wraps an untrusted masked decoder. Only the
privacy-critical boundary operations live here (embedding+masking, logit
recovery, greedy sampling); the transformer, KV cache, LM head, and all large
GEMMs stay on the untrusted side. The package depends on **numpy only** -- no
torch, no transformers, no GPU -- so it can run inside a small TEE (e.g. an
Intel TDX guest on Alibaba Cloud) and be unit-tested without ML dependencies.

No semantic, cryptographic, or formal security is claimed. Attention scores and
sequence length remain visible to the untrusted side; the residual/vocab masks
are orthogonal (signed-permutation) + positive scaling, which is weaker than
dense masking. See ``docs/tee_boundary_design.md``.
"""

from __future__ import annotations

from pllo.tee.runtime_api import (
    AttestationReport,
    MaskedEmbeddingPacket,
    MaskedLogitsPacket,
    SamplingResult,
    TEEConfig,
    TrustedRuntime,
    apply_vocab_logit_mask,
    make_runtime,
)

__all__ = [
    "AttestationReport",
    "MaskedEmbeddingPacket",
    "MaskedLogitsPacket",
    "SamplingResult",
    "TEEConfig",
    "TrustedRuntime",
    "apply_vocab_logit_mask",
    "make_runtime",
]
