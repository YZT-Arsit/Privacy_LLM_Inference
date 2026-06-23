"""TEE-boundary <-> untrusted-GPU-worker protocol (Stage 8.5).

A message protocol + reference orchestration that drives the thin trusted
boundary runtime (:mod:`pllo.tee`) against an *untrusted* GPU worker. The
trusted boundary owns the raw prompt, tokenization, ``input_ids``, mask secrets,
embedding masking, logit recovery, greedy selection, and the generated tokens.
The GPU worker receives only masked embeddings + public metadata + the folded
LM head, and returns only masked logits. The model (Qwen / decoder / attention /
MLP / KV cache / LM head) is NEVER placed inside the TEE -- ``tee_used`` on the
GPU side is always ``False``.

numpy only (the qwen7b GPU backend imports torch lazily, on the GPU server).
See :mod:`pllo.protocol.security_audit` for the confidentiality checks.
"""

from __future__ import annotations

from pllo.protocol.attestation import (
    AttestationEvidence,
    attest_boundary,
    compute_runtime_hash,
    runtime_report_data_hex,
    verify_evidence,
)
from pllo.protocol.gpu_worker import (
    GpuBackend,
    LocalGpuWorker,
    MockGpuBackend,
    Qwen7BGpuBackend,
    make_gpu_backend,
)
from pllo.protocol.orchestrator import (
    fold_lm_head,
    run_protocol,
    trusted_tokenize,
)
from pllo.protocol.security_audit import (
    assert_no_gpu_visible_plaintext,
    assert_no_mask_secret_leak,
    assert_wrong_mask_recovery_fails,
)
from pllo.protocol.tee_gpu_messages import (
    BoundaryInitRequest,
    BoundaryInitResponse,
    MaskedDecodeRequest,
    MaskedDecodeResponse,
    MaskedPrefillRequest,
    MaskedPrefillResponse,
    ProtocolTrace,
    RecoveredTokenResponse,
)

__all__ = [
    "AttestationEvidence",
    "attest_boundary",
    "compute_runtime_hash",
    "runtime_report_data_hex",
    "verify_evidence",
    "GpuBackend",
    "LocalGpuWorker",
    "MockGpuBackend",
    "Qwen7BGpuBackend",
    "make_gpu_backend",
    "fold_lm_head",
    "run_protocol",
    "trusted_tokenize",
    "assert_no_gpu_visible_plaintext",
    "assert_no_mask_secret_leak",
    "assert_wrong_mask_recovery_fails",
    "BoundaryInitRequest",
    "BoundaryInitResponse",
    "MaskedDecodeRequest",
    "MaskedDecodeResponse",
    "MaskedPrefillRequest",
    "MaskedPrefillResponse",
    "ProtocolTrace",
    "RecoveredTokenResponse",
]
