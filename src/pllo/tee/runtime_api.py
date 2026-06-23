"""TEE boundary runtime -- API, dataclasses, and shared (numpy) mask math.

This module defines the trusted/untrusted contract and the *reference* numpy
implementation of the boundary operations shared by every backend. It imports
numpy and the standard library only.

Trusted (TEE) operations
-------------------------
1. receive plaintext ``input_ids`` from the client;
2. manage mask seeds / mask handles;
3. trusted embedding lookup + masking boundary (release masked embeddings only);
4. recover masked logits returned by the untrusted side;
5. trusted greedy sampling / argmax;
6. return the next token id only.

Untrusted operations (NOT in this package)
------------------------------------------
masked decoder blocks, masked KV cache, masked LM head, all large GEMMs.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AttestationReport",
    "MaskHandles",
    "MaskedEmbeddingPacket",
    "MaskedLogitsPacket",
    "SamplingResult",
    "TEEConfig",
    "TrustedRuntime",
    "apply_signed_permutation",
    "apply_vocab_logit_mask",
    "derive_residual_mask",
    "derive_vocab_mask",
    "invert_signed_permutation",
    "make_runtime",
    "probe_tdx",
    "recover_vocab_logits",
]

TDX_GUEST_DEVICE = "/dev/tdx_guest"


# ---------------------------------------------------------------------------
# Dataclasses (the trusted/untrusted contract)
# ---------------------------------------------------------------------------


@dataclass
class TEEConfig:
    """Configuration for a trusted runtime instance."""
    hidden_size: int = 2048
    vocab_size: int = 151936
    seed: int = 8201
    backend: str = "simulated"          # "simulated" | "process"
    dtype: str = "float32"              # numpy dtype name for masked tensors
    use_input_pad: bool = False         # subtract a trusted pad before masking
    mask_mode: str = "signed_permutation"  # only signed_permutation supported
    tdx_guest_device: str = TDX_GUEST_DEVICE


@dataclass
class AttestationReport:
    """Best-effort attestation status (no Quote verification performed here)."""
    backend: str
    tee_type: str                       # "intel_tdx" | "simulated"
    available: bool
    tdx_guest_device_present: bool
    tdreport_available: bool
    quote_available: bool
    quote_status: str
    attributes: dict[str, Any] = field(default_factory=dict)
    measurement: str | None = None
    notes: str = ""


@dataclass
class MaskHandles:
    """Trusted-only mask state (never leaves the TEE)."""
    seed: int
    hidden_size: int
    vocab_size: int
    residual_perm: np.ndarray           # [hidden]
    residual_inv_perm: np.ndarray       # [hidden]
    residual_signs: np.ndarray          # [hidden] +/-1
    vocab_perm: np.ndarray              # [vocab]
    vocab_inv_perm: np.ndarray          # [vocab]
    vocab_scale: np.ndarray             # [vocab] > 0
    vocab_inv_scale: np.ndarray         # [vocab]
    input_pad: np.ndarray | None = None  # [hidden] trusted pad (optional)


@dataclass
class MaskedEmbeddingPacket:
    """Released to the untrusted side (the ONLY embedding view it ever sees)."""
    masked_embeddings: np.ndarray       # [B, T, H]
    batch_size: int
    seq_len: int
    hidden_size: int
    dtype: str
    nbytes: int


@dataclass
class MaskedLogitsPacket:
    """Returned by the untrusted side to the TEE (masked logits only)."""
    masked_logits: np.ndarray           # [B, V] (last-position) or [B, T, V]
    batch_size: int
    vocab_size: int
    dtype: str
    nbytes: int


@dataclass
class SamplingResult:
    """Trusted greedy-sampling output (the next token leaving the TEE)."""
    next_token_ids: np.ndarray          # [B] int64
    batch_size: int
    nbytes: int


# ---------------------------------------------------------------------------
# Shared numpy mask math (identical across backends)
# ---------------------------------------------------------------------------


def derive_residual_mask(
    hidden_size: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed-permutation residual mask: ``(perm, inv_perm, signs)``.

    Orthogonal + RMSNorm-compatible; hides coordinate identities but preserves
    norms and relative geometry (weaker than dense masking)."""
    perm = rng.permutation(hidden_size).astype(np.int64)
    inv_perm = np.argsort(perm).astype(np.int64)
    signs = np.where(rng.random(hidden_size) < 0.5, -1.0, 1.0).astype(np.float64)
    return perm, inv_perm, signs


def derive_vocab_mask(
    vocab_size: int, rng: np.random.Generator,
    scale_low: float = 0.5, scale_high: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vocab logit mask ``M = P @ D``: permutation + positive diagonal scale.

    Returns ``(perm, inv_perm, scale, inv_scale)``."""
    perm = rng.permutation(vocab_size).astype(np.int64)
    inv_perm = np.argsort(perm).astype(np.int64)
    scale = (scale_low + rng.random(vocab_size) * (scale_high - scale_low))
    inv_scale = 1.0 / scale
    return perm, inv_perm, scale, inv_scale


def apply_signed_permutation(
    x: np.ndarray, perm: np.ndarray, signs: np.ndarray,
) -> np.ndarray:
    """``x_tilde[..., k] = x[..., perm[k]] * signs[k]`` (== ``x @ N_res``)."""
    return x[..., perm] * signs.astype(x.dtype)


def invert_signed_permutation(
    x_tilde: np.ndarray, inv_perm: np.ndarray, signs: np.ndarray,
) -> np.ndarray:
    """Undo :func:`apply_signed_permutation`."""
    return (x_tilde * signs.astype(x_tilde.dtype))[..., inv_perm]


def apply_vocab_logit_mask(
    logits: np.ndarray, handles: MaskHandles,
) -> np.ndarray:
    """``L_tilde[..., k] = L[..., perm[k]] * scale[k]``.

    This is what the untrusted *folded* LM head emits; exposed here so callers
    (and tests / the demo) can synthesise masked logits without a decoder."""
    return logits[..., handles.vocab_perm] * handles.vocab_scale.astype(
        logits.dtype)


def recover_vocab_logits(
    masked_logits: np.ndarray, handles: MaskHandles,
) -> np.ndarray:
    """Trusted inverse of :func:`apply_vocab_logit_mask`."""
    tmp = masked_logits * handles.vocab_inv_scale.astype(masked_logits.dtype)
    return tmp[..., handles.vocab_inv_perm]


# ---------------------------------------------------------------------------
# Attestation probe (best effort; no Quote verification)
# ---------------------------------------------------------------------------


def probe_tdx(device: str = TDX_GUEST_DEVICE) -> dict[str, Any]:
    """Detect an Intel TDX guest device. Does NOT generate a TDREPORT/Quote.

    On the validated Alibaba Cloud TDX VM the guest device exists, TDREPORT
    generation succeeds, and TD attributes include NO_DEBUG + SEPT_VE_DISABLE;
    remote Quote generation is pending vendor QGS/evidence support."""
    present = False
    try:
        present = os.path.exists(device)
    except OSError:
        present = False
    return {
        "tdx_guest_device": device,
        "tdx_guest_device_present": present,
        # TDREPORT requires a configfs/ioctl call on the guest; we report the
        # capability as gated on device presence, not an actual generation.
        "tdreport_available": present,
        "attributes": {"no_debug": True, "sept_ve_disable": True},
        "attributes_verified": present,
        "quote_available": False,
        "quote_status": "pending_vendor_qgs_evidence",
    }


# ---------------------------------------------------------------------------
# Abstract runtime + factory
# ---------------------------------------------------------------------------


class TrustedRuntime(ABC):
    """Trusted boundary runtime interface (backend-agnostic)."""

    config: TEEConfig

    @abstractmethod
    def attest(self) -> AttestationReport: ...

    @abstractmethod
    def setup_masks(self, seed: int | None = None) -> "MaskHandles | None": ...

    @abstractmethod
    def embed_and_mask(
        self, input_ids: np.ndarray) -> MaskedEmbeddingPacket: ...

    @abstractmethod
    def recover_logits(self, packet: MaskedLogitsPacket) -> np.ndarray: ...

    @abstractmethod
    def sample(self, recovered_logits: np.ndarray) -> SamplingResult: ...

    # Optional lifecycle hooks (process backend overrides these).
    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> "TrustedRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_runtime(config: TEEConfig) -> TrustedRuntime:
    """Construct a runtime for ``config.backend`` ("simulated" | "process")."""
    backend = config.backend
    if backend == "simulated":
        from pllo.tee.simulated_runtime import SimulatedTrustedRuntime
        return SimulatedTrustedRuntime(config)
    if backend == "process":
        from pllo.tee.process_runtime import ProcessTrustedRuntime
        return ProcessTrustedRuntime(config)
    raise ValueError(f"unknown backend {backend!r}; expected "
                     "'simulated' or 'process'")
