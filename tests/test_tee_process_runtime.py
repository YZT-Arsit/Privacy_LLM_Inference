"""TEE process-backend tests (must match the simulated backend; numpy only).

Run: python -m pytest tests/test_tee_process_runtime.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pllo.tee import (
    MaskedEmbeddingPacket,
    MaskedLogitsPacket,
    SamplingResult,
    TEEConfig,
    apply_vocab_logit_mask,
    make_runtime,
)
from pllo.tee.process_runtime import ProcessTrustedRuntime
from pllo.tee.runtime_api import AttestationReport, MaskHandles


def _cfg(backend, hidden=64, vocab=2000, seed=321):
    return TEEConfig(hidden_size=hidden, vocab_size=vocab, seed=seed,
                     backend=backend)


@pytest.fixture()
def both():
    sim = make_runtime(_cfg("simulated"))
    proc = make_runtime(_cfg("process"))
    assert isinstance(proc, ProcessTrustedRuntime)
    yield sim, proc
    proc.close()


# 1.
def test_process_embed_matches_simulated(both) -> None:
    sim, proc = both
    ids = np.random.default_rng(0).integers(0, 2000, (8, 16), dtype=np.int64)
    a = sim.embed_and_mask(ids)
    b = proc.embed_and_mask(ids)
    assert np.array_equal(a.masked_embeddings, b.masked_embeddings)
    assert (a.batch_size, a.seq_len, a.hidden_size, a.nbytes) == \
           (b.batch_size, b.seq_len, b.hidden_size, b.nbytes)


# 2.
def test_process_recover_and_sample_match_simulated(both) -> None:
    sim, proc = both
    L = np.random.default_rng(1).standard_normal((8, 2000)).astype(np.float32)
    pkt = MaskedLogitsPacket(apply_vocab_logit_mask(L, sim.handles), 8, 2000,
                             "float32", 0)
    rec_s = sim.recover_logits(pkt)
    rec_p = proc.recover_logits(pkt)
    assert np.array_equal(rec_s, rec_p)
    res_p = proc.sample(rec_p)
    assert isinstance(res_p, SamplingResult)
    # full round trip recovers the plaintext argmax
    assert np.array_equal(res_p.next_token_ids, L.argmax(axis=-1))
    assert np.array_equal(sim.sample(rec_s).next_token_ids,
                          res_p.next_token_ids)


# 3.
def test_process_attest(both) -> None:
    _, proc = both
    rep = proc.attest()
    assert rep.backend == "process"
    assert rep.quote_status == "pending_vendor_qgs_evidence"
    assert rep.attributes["no_debug"] is True


# 4.
def test_process_setup_masks_returns_none(both) -> None:
    """Mask handles stay inside the worker (trusted domain) -- the untrusted
    parent never receives them."""
    _, proc = both
    assert proc.setup_masks(321) is None


# 5.
def test_process_wrong_mask_fails(both) -> None:
    sim, _ = both
    L = np.random.default_rng(2).standard_normal((8, 2000)).astype(np.float32)
    pkt = MaskedLogitsPacket(apply_vocab_logit_mask(L, sim.handles), 8, 2000,
                             "float32", 0)
    wrong = make_runtime(_cfg("process", seed=99999))
    try:
        rec = wrong.recover_logits(pkt)
        assert np.abs(rec - L).max() > 1.0
        assert (wrong.sample(rec).next_token_ids != L.argmax(axis=-1)).any()
    finally:
        wrong.close()


class _RecordConn:
    """Wraps a Pipe connection to record everything that crosses the boundary."""

    def __init__(self, real):
        self._real = real
        self.sent = []
        self.received = []

    def send(self, obj):
        self.sent.append(obj)
        return self._real.send(obj)

    def recv(self):
        r = self._real.recv()
        self.received.append(r)
        return r

    def close(self):
        return self._real.close()


# 7.
def test_process_sends_only_boundary_messages() -> None:
    """The untrusted parent must transmit/receive ONLY boundary payloads:
    method tags + input_ids / masked embeddings / masked logits / recovered
    logits / next token / attestation. Never mask handles, model weights, an
    embedding table, or a decoder/model object."""
    rt = make_runtime(_cfg("process", hidden=32, vocab=2000))
    rt._parent_conn = _RecordConn(rt._parent_conn)
    allowed_methods = {"attest", "setup_masks", "embed_and_mask",
                       "recover_logits", "sample", "shutdown"}
    try:
        rt.attest()
        rt.setup_masks(321)
        ids = np.random.default_rng(0).integers(0, 2000, (4, 8), dtype=np.int64)
        emb = rt.embed_and_mask(ids)
        L = np.random.default_rng(1).standard_normal((4, 2000)).astype(
            np.float32)
        pkt = MaskedLogitsPacket(apply_vocab_logit_mask(L, make_runtime(
            _cfg("simulated")).handles), 4, 2000, "float32", 0)
        rt.recover_logits(pkt)
        rt.sample(L)
    finally:
        sent = list(rt._parent_conn.sent)
        received = list(rt._parent_conn.received)
        rt.close()

    weight_shape = (rt.config.vocab_size, rt.config.hidden_size)
    weight_shape_t = (rt.config.hidden_size, rt.config.vocab_size)

    # --- outbound (parent -> worker): tags + boundary inputs only ----------
    for method, payload in sent:
        assert method in allowed_methods, f"unexpected method {method!r}"
        assert not isinstance(payload, MaskHandles), "mask handles sent out!"
        if method == "embed_and_mask":
            assert isinstance(payload, np.ndarray)
            assert np.issubdtype(payload.dtype, np.integer)  # input_ids
        elif method == "recover_logits":
            assert isinstance(payload, MaskedLogitsPacket)
        elif method == "sample":
            assert isinstance(payload, np.ndarray)
        elif method == "setup_masks":
            assert payload is None or isinstance(payload, int)
        else:  # attest / shutdown
            assert payload is None
        if isinstance(payload, np.ndarray):
            assert payload.shape not in (weight_shape, weight_shape_t)

    # --- inbound (worker -> parent): boundary results only -----------------
    for status, result in received:
        assert status == "ok"
        assert isinstance(result, (AttestationReport, MaskedEmbeddingPacket,
                                   SamplingResult, np.ndarray, type(None))), \
            f"forbidden result type {type(result)!r}"
        assert not isinstance(result, MaskHandles), "mask handles returned!"
        if isinstance(result, np.ndarray):
            # recovered logits [B, V] are fine; an embedding table / weight is
            # never returned.
            assert result.shape not in (weight_shape, weight_shape_t)
        if isinstance(result, MaskedEmbeddingPacket):
            # only masked embeddings cross; never the plaintext table
            assert result.masked_embeddings.shape == (4, 8, 32)


# 6.
def test_process_close_idempotent_and_context_manager() -> None:
    with make_runtime(_cfg("process")) as rt:
        ids = np.zeros((2, 4), dtype=np.int64)
        assert rt.embed_and_mask(ids).batch_size == 2
    # closing again is safe
    rt.close()
    # using after close raises
    with pytest.raises(RuntimeError):
        rt.embed_and_mask(np.zeros((1, 1), dtype=np.int64))
