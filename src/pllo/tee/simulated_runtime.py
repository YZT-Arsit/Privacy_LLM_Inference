"""In-process (simulated) trusted runtime -- numpy reference implementation.

This is the ground-truth backend: every other backend must reproduce its
numbers exactly. It performs only the trusted boundary operations; there is no
transformer, KV cache, LM head, or GEMM here. numpy only.
"""

from __future__ import annotations

import numpy as np

from pllo.tee.runtime_api import (
    AttestationReport,
    MaskedEmbeddingPacket,
    MaskedLogitsPacket,
    MaskHandles,
    SamplingResult,
    TEEConfig,
    TrustedRuntime,
    apply_signed_permutation,
    derive_residual_mask,
    derive_vocab_mask,
    probe_tdx,
    recover_vocab_logits,
)

__all__ = ["SimulatedTrustedRuntime", "build_embedding_table"]


def build_embedding_table(
    vocab_size: int, hidden_size: int, seed: int, dtype: np.dtype,
) -> np.ndarray:
    """Deterministic trusted embedding table ``E`` ``[vocab, hidden]``.

    Derived from ``seed`` so every backend/process builds the identical table.
    Scaled by ``1/sqrt(hidden)`` like a real embedding init."""
    rng = np.random.default_rng([seed, 0xE10BED])
    e = rng.standard_normal((vocab_size, hidden_size)) * (1.0 / hidden_size ** 0.5)
    return e.astype(dtype)


class SimulatedTrustedRuntime(TrustedRuntime):
    """Reference trusted runtime (in-process)."""

    def __init__(self, config: TEEConfig) -> None:
        self.config = config
        self._np_dtype = np.dtype(config.dtype)
        self._handles: MaskHandles | None = None
        self._embed: np.ndarray | None = None      # lazy [vocab, hidden]
        self.setup_masks(config.seed)

    # -- attestation ----------------------------------------------------
    def attest(self) -> AttestationReport:
        tdx = probe_tdx(self.config.tdx_guest_device)
        present = bool(tdx["tdx_guest_device_present"])
        return AttestationReport(
            backend=self.config.backend,
            tee_type="intel_tdx" if present else "simulated",
            available=present,
            tdx_guest_device_present=present,
            tdreport_available=bool(tdx["tdreport_available"]),
            quote_available=bool(tdx["quote_available"]),
            quote_status=str(tdx["quote_status"]),
            attributes=dict(tdx["attributes"]),
            measurement=None,
            notes="TDREPORT capability gated on guest-device presence; remote "
                  "Quote verification pending vendor QGS/evidence support. No "
                  "Quote is generated or verified by this runtime.",
        )

    # -- mask management -------------------------------------------------
    def setup_masks(self, seed: int | None = None) -> MaskHandles:
        seed = self.config.seed if seed is None else int(seed)
        rng = np.random.default_rng([seed, 0x5A5E])
        rp, rip, rs = derive_residual_mask(self.config.hidden_size, rng)
        vp, vip, vs, vis = derive_vocab_mask(self.config.vocab_size, rng)
        pad = None
        if self.config.use_input_pad:
            pad = (rng.standard_normal(self.config.hidden_size)
                   * (1.0 / self.config.hidden_size ** 0.5)).astype(
                       self._np_dtype)
        self._handles = MaskHandles(
            seed=seed, hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size, residual_perm=rp,
            residual_inv_perm=rip, residual_signs=rs, vocab_perm=vp,
            vocab_inv_perm=vip, vocab_scale=vs, vocab_inv_scale=vis,
            input_pad=pad)
        return self._handles

    @property
    def handles(self) -> MaskHandles:
        if self._handles is None:
            self.setup_masks(self.config.seed)
        assert self._handles is not None
        return self._handles

    def _embedding_table(self) -> np.ndarray:
        if self._embed is None:
            self._embed = build_embedding_table(
                self.config.vocab_size, self.config.hidden_size,
                self.config.seed, self._np_dtype)
        return self._embed

    # -- trusted embedding + masking boundary ---------------------------
    def embed_and_mask(self, input_ids: np.ndarray) -> MaskedEmbeddingPacket:
        ids = np.asarray(input_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("input_ids must be an integer array")
        if ids.ndim == 1:
            ids = ids[None, :]
        h = self.handles
        x = self._embedding_table()[ids]                 # [B, T, H]
        if h.input_pad is not None:
            x = x - h.input_pad
        x_tilde = apply_signed_permutation(
            x, h.residual_perm, h.residual_signs).astype(self._np_dtype)
        return MaskedEmbeddingPacket(
            masked_embeddings=x_tilde, batch_size=int(x_tilde.shape[0]),
            seq_len=int(x_tilde.shape[1]), hidden_size=int(x_tilde.shape[2]),
            dtype=str(self._np_dtype), nbytes=int(x_tilde.nbytes))

    # -- trusted logit recovery -----------------------------------------
    def recover_logits(self, packet: MaskedLogitsPacket) -> np.ndarray:
        return recover_vocab_logits(np.asarray(packet.masked_logits),
                                    self.handles)

    # -- trusted greedy sampling ----------------------------------------
    def sample(self, recovered_logits: np.ndarray) -> SamplingResult:
        logits = np.asarray(recovered_logits)
        if logits.ndim == 3:                             # [B, T, V] -> last
            logits = logits[:, -1, :]
        if logits.ndim == 1:                             # [V] -> [1, V]
            logits = logits[None, :]
        tokens = logits.argmax(axis=-1).astype(np.int64)
        return SamplingResult(next_token_ids=tokens,
                              batch_size=int(tokens.shape[0]),
                              nbytes=int(tokens.nbytes))
