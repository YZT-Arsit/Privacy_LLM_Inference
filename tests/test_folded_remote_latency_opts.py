"""Correctness of the folded_remote latency optimisations.

Two opt-in optimisations must NOT change any result:

* ``precompute_masked_embed`` -- precompute ``E_masked = E @ N_0`` so the per-token
  input mask is a row lookup instead of a per-token ``[hidden x hidden]`` matmul.
  Gathering rows of ``E @ N_0`` equals ``(gathered rows of E) @ N_0`` element for
  element, so the masked embedding a decode step sends must be BIT-IDENTICAL.
* ``worker_persistent_conn`` -- reuse one keep-alive TCP connection across decode
  steps. The HTTP bytes, the audited masked payloads, and therefore the decoded
  logits are identical whether or not the connection is reused. We prove it end to
  end against an in-process HTTP worker serving the deterministic MockGpuBackend.

Run: python -m pytest tests/test_folded_remote_latency_opts.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.folded_probe_common import LiteBoundary  # noqa: E402
from pllo.ops.causal_lm_boundaries import (  # noqa: E402
    VocabLogitMask, recover_vocab_logits)
from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker  # noqa: E402
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest, MaskedDecodeRequest)
from pllo.protocol.wire import decode_value, encode_value  # noqa: E402


def _tiny_boundary(precompute: bool, *, vocab=16, hidden=8):
    torch.manual_seed(0)
    embed = torch.randn(vocab, hidden, dtype=torch.float32)
    # N_0 as a random orthogonal-ish square (any square works for bit-identity)
    n0 = torch.randn(hidden, hidden, dtype=torch.float32)
    perm = torch.randperm(vocab)
    inv = torch.argsort(perm)
    scale = torch.rand(vocab) + 0.5
    vm = VocabLogitMask(permutation=perm, inverse_permutation=inv,
                        scale=scale, inverse_scale=1.0 / scale)
    return LiteBoundary(embed, n0, vm, {"rms_norm_eps": 1e-6, "vocab_size": vocab},
                        device="cpu", precompute_masked_embed=precompute)


def test_precompute_masked_embed_is_bit_identical():
    base = _tiny_boundary(precompute=False)
    fast = _tiny_boundary(precompute=True)
    assert fast._embed_masked is not None and base._embed_masked is None
    ids = torch.tensor([[0, 5, 15, 3, 5]])
    # prompt masking
    assert torch.equal(base.mask_embeddings(ids), fast.mask_embeddings(ids))
    # per-token masking (every vocab id)
    for t in range(16):
        tok = torch.tensor([t])
        assert torch.equal(base.mask_token_embedding(tok),
                           fast.mask_token_embedding(tok)), t


def _decode_series(url, persistent):
    w = RemoteGpuWorker(url, "mock", persistent=persistent)
    folded_head = np.random.default_rng(0).standard_normal(
        (8, 16)).astype(np.float32)   # [H, V] identity-decoder head
    w.init(BoundaryInitRequest(
        session_id="t", hidden_size=8, vocab_size=16, num_layers=1,
        dtype="float32", gpu_backend="mock", folded_lm_head=folded_head))
    outs = []
    rng = np.random.default_rng(1)
    for step in range(6):
        emb = rng.standard_normal((1, 1, 8)).astype(np.float32)
        r = w.decode(MaskedDecodeRequest(
            session_id="t", masked_embedding=emb, position=step, step=step))
        outs.append(np.asarray(r.masked_logits).copy())
    w.close()
    return outs


def test_persistent_connection_gives_identical_results():
    server = GpuWorkerServer(host="127.0.0.1", port=0, backend_name="mock")
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port
    try:
        non_persistent = _decode_series(url, persistent=False)
        persistent = _decode_series(url, persistent=True)
    finally:
        server.shutdown()
    assert len(non_persistent) == len(persistent) == 6
    for a, b in zip(non_persistent, persistent):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# T1a: native bf16 logits on the wire (half the bytes, bit-identical)
# --------------------------------------------------------------------------- #

def test_wire_bf16_roundtrip_is_bit_identical():
    torch.manual_seed(3)
    t = torch.randn(1, 257, dtype=torch.float32).to(torch.bfloat16)
    enc = encode_value(t)
    assert enc["dtype"] == "bfloat16" and enc["shape"] == [1, 257]
    back = decode_value(enc)
    assert isinstance(back, torch.Tensor) and back.dtype == torch.bfloat16
    # raw bf16 bit patterns are preserved exactly
    assert torch.equal(back.view(torch.uint16), t.view(torch.uint16))
    # and the fp32 upcast (what recovery uses) is exactly the fp32 upcast payload
    assert torch.equal(back.float(), t.float())


def test_native_bf16_wire_recover_is_bit_identical():
    torch.manual_seed(4)
    vocab = 512
    logits_bf16 = torch.randn(1, vocab, dtype=torch.float32).to(torch.bfloat16)
    perm = torch.randperm(vocab)
    vm = VocabLogitMask(permutation=perm, inverse_permutation=torch.argsort(perm),
                        scale=(torch.rand(vocab) + 0.5),
                        inverse_scale=1.0 / (torch.rand(vocab) + 0.5))
    # PATH A (historical wire): worker upcasts bf16 -> fp32 numpy, boundary
    # upcasts to fp32 and recovers.
    a_wire = decode_value(encode_value(
        logits_bf16.float().numpy()))                   # fp32 numpy across wire
    rec_a = recover_vocab_logits(
        torch.as_tensor(np.asarray(a_wire)).float(), vm)
    # PATH B (native bf16 wire): worker sends bf16, boundary upcasts to fp32.
    b_wire = decode_value(encode_value(logits_bf16))    # bf16 tensor across wire
    rec_b = recover_vocab_logits(b_wire.float(), vm)
    assert torch.equal(rec_a, rec_b)
    assert int(rec_a.argmax(-1)) == int(rec_b.argmax(-1))
