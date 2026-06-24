"""Tiny-random-Qwen2 CPU integration test for the qwen7b worker bridge.

Validates the boundary-split masked Qwen path end-to-end at tiny scale (no CUDA,
no checkpoint): the trusted boundary masks the embeddings, the untrusted
``Qwen7BGpuBackend`` returns masked logits via ``MaskedQwenSession``, and the
boundary recovers logits that match the validated plaintext reference. Also
checks the backend exposes no plaintext and reports ``tee_used=False``.

Run: python -m pytest tests/test_qwen7b_bridge.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    masked_prefill_full_logits,
)
from pllo.protocol.gpu_worker import Qwen7BGpuBackend  # noqa: E402
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest,
    MaskedDecodeRequest,
    MaskedPrefillRequest,
    MaskedPrefillResponse,
)

SEED = 2035
T = 6


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _cfg(mc):
    return MemoryOptimizedConfig(
        num_layers=mc.num_hidden_layers, batch_size=1, seq_len=T,
        max_new_tokens=2, device="cpu", dtype="float32", folding_dtype="float32",
        folded_weight_device="cpu", mlp_down_chunk_size=64, seed=SEED)


def _backend(model, mc):
    be = Qwen7BGpuBackend(
        model=model, model_config=mc, device="cpu", dtype="float32",
        seq_len=T, num_layers=mc.num_hidden_layers, folded_weight_device="cpu",
        mlp_down_chunk_size=64, seed=SEED)
    be.init(BoundaryInitRequest(
        session_id="s", hidden_size=mc.hidden_size, vocab_size=mc.vocab_size,
        num_layers=mc.num_hidden_layers, dtype="float32", gpu_backend="qwen7b",
        public_metadata={"max_new_tokens": 2}))
    return be


def test_bridge_prefill_recovers_plaintext_reference() -> None:
    model, mc = _tiny_model()
    cfg = _cfg(mc)
    boundary = MaskedQwenSession(model, mc, cfg)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, mc.vocab_size, (1, T), generator=g)

    # trusted boundary: mask embeddings (the only thing the GPU sees)
    h_tilde = boundary.mask_embeddings(ids)
    assert h_tilde.shape == (1, T, mc.hidden_size)

    # untrusted worker: masked logits only
    be = _backend(model, mc)
    resp = be.prefill(MaskedPrefillRequest(
        session_id="s", masked_embeddings=h_tilde.detach().numpy(),
        positions=list(range(T)), batch_size=1, seq_len=T))
    assert isinstance(resp, MaskedPrefillResponse)
    assert be.tee_used is False

    # trusted boundary: recover logits, compare to the validated plain reference
    recovered_last = boundary.recover(torch.as_tensor(resp.masked_logits))
    plain_logits, recovered_full = masked_prefill_full_logits(model, mc, ids, cfg)
    ref_last = plain_logits[:, -1, :]

    # bridge recovered == validated masked path recovered (same kernels)
    assert torch.allclose(recovered_last, recovered_full[:, -1, :], atol=1e-4)
    # and both match the plaintext reference top-1
    assert int(recovered_last.argmax(-1)) == int(ref_last.argmax(-1))
    assert float((recovered_last - ref_last).abs().max()) < 1e-2


def test_bridge_decode_step_runs_and_recovers() -> None:
    model, mc = _tiny_model()
    cfg = _cfg(mc)
    boundary = MaskedQwenSession(model, mc, cfg)
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, mc.vocab_size, (1, T), generator=g)
    be = _backend(model, mc)

    resp = be.prefill(MaskedPrefillRequest(
        session_id="s", masked_embeddings=boundary.mask_embeddings(ids).numpy(),
        positions=list(range(T)), batch_size=1, seq_len=T))
    tok = boundary.recover(torch.as_tensor(resp.masked_logits)).argmax(-1)

    x_next = boundary.mask_token_embedding(tok)          # [1,1,H]
    dresp = be.decode(MaskedDecodeRequest(
        session_id="s", masked_embedding=x_next.detach().numpy(),
        position=T, step=1))
    rec2 = boundary.recover(torch.as_tensor(dresp.masked_logits))
    assert rec2.shape[-1] == mc.vocab_size
    assert torch.isfinite(rec2).all()
    assert dresp.kv_cache_len == T + 1


def test_bridge_response_carries_only_masked_logits() -> None:
    # The worker response exposes masked logits + public ints only -- no plaintext
    # hidden, no input ids, no recovered logits.
    model, mc = _tiny_model()
    cfg = _cfg(mc)
    boundary = MaskedQwenSession(model, mc, cfg)
    ids = torch.zeros((1, T), dtype=torch.long)
    be = _backend(model, mc)
    resp = be.prefill(MaskedPrefillRequest(
        session_id="s", masked_embeddings=boundary.mask_embeddings(ids).numpy(),
        positions=list(range(T)), batch_size=1, seq_len=T))
    # masked logits differ from the recovered (plaintext-equivalent) logits
    masked = np.asarray(resp.masked_logits)
    recovered = boundary.recover(torch.as_tensor(masked)).numpy()
    assert masked.shape == recovered.shape
    assert float(np.abs(masked - recovered).max()) > 1e-3   # genuinely masked


def test_qwen7b_backend_requires_model_or_checkpoint() -> None:
    be = Qwen7BGpuBackend(model_path=None, device="cpu")
    be.init(BoundaryInitRequest(
        session_id="s", hidden_size=8, vocab_size=8, num_layers=1,
        dtype="float32", gpu_backend="qwen7b"))
    with pytest.raises(RuntimeError):
        be.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=np.zeros((1, 2, 8), np.float32),
            positions=[0, 1], batch_size=1, seq_len=2))
