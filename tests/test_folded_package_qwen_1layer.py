"""1-layer folded Qwen package: build -> load -> masked-prefill correctness.

Proves the next incremental folded-package step on a tiny Qwen2 (CPU, no
checkpoint): the untrusted worker loads a 1-layer folded package (no masks) and
its single masked-block prefill matches the in-process folded path exactly, the
package verifies, and the worker reports it holds no mask secrets.

Run: python -m pytest tests/test_folded_package_qwen_1layer.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pllo.deployment import (  # noqa: E402
    FoldedPackageWriter,
    build_manifest,
    forbidden_tensor_names,
    verify_package,
    write_manifest,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    _masked_block_prefill_chunked,
)
from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend  # noqa: E402
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest  # noqa: E402


def _tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=128,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _session(model, mc, n_layers=1, seq_len=6):
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=seq_len, max_new_tokens=1,
        device="cpu", dtype="float32", folding_dtype="float32",
        folded_weight_device="cpu", mlp_down_chunk_size=64, seed=2035)
    return MaskedQwenSession(model, mc, cfg), cfg


def _build_1layer_package(tmp_path, session):
    writer = FoldedPackageWriter(tmp_path / "pkg")
    writer.add_shard("layer_000", session.export_folded_layer_tensors(0))
    writer.add_shard("head", session.export_folded_head_tensors())
    manifest = build_manifest(
        package_type="base_model", model_name="tiny-qwen", model_path_or_id=None,
        num_layers=1, dtype="float32", nonlinear_backend="current",
        created_by="test", shard_index=writer.shard_index, hidden_size=128,
        vocab_size=256, mask_schedule_id="t-0",
        created_at="2026-06-24T00:00:00Z")
    write_manifest(manifest, tmp_path / "pkg")
    return tmp_path / "pkg"


def test_1layer_package_verifies_and_has_no_mask_secrets(tmp_path) -> None:
    model, mc = _tiny()
    session, _ = _session(model, mc, n_layers=1)
    pkg = _build_1layer_package(tmp_path, session)
    rep = verify_package(pkg)
    assert rep["package_valid"] is True
    assert rep["forbidden_fields_found"] == []
    # exported layer tensors are all folded operators, no mask names
    names = list(session.export_folded_layer_tensors(0).keys())
    assert forbidden_tensor_names(names) == []
    assert all(n.endswith("_tilde") for n in names)


def test_worker_1layer_prefill_matches_in_process(tmp_path) -> None:
    model, mc = _tiny()
    session, cfg = _session(model, mc, n_layers=1, seq_len=6)
    pkg = _build_1layer_package(tmp_path, session)

    ids = torch.randint(0, mc.vocab_size, (1, 6))
    h_tilde = session.mask_embeddings(ids)            # boundary-masked input

    # in-process folded reference (uses masks internally)
    folded, down_info, cfg_c = session._folded_layer(0)
    ref = _masked_block_prefill_chunked(
        h_tilde, folded, down_info, cfg_c, session._cos, session._sin,
        session.chunk)["y_tilde"]

    # worker path: load the 1-layer package (NO masks) + run masked prefill
    backend = Qwen7BFoldedPackageGpuBackend(folded_package_path=str(pkg),
                                            device="cpu", dtype="float32")
    resp = backend.init(BoundaryInitRequest(
        session_id="s", hidden_size=128, vocab_size=256, num_layers=1,
        dtype="float32", gpu_backend="qwen7b_folded_package"))
    assert resp.tee_used_on_gpu is False
    d = backend.describe()
    assert d["folded_package_loaded"] is True
    assert d["worker_has_mask_secrets"] is False
    assert d["package_valid"] is True

    out = backend.run_single_layer_prefill(
        h_tilde, 0, cfg_c, session._cos, session._sin, session.eps)
    # worker (folded weights only) == in-process protected path
    assert torch.allclose(out["y_tilde"], ref, atol=1e-5, rtol=1e-4)


def test_full_decode_still_todo(tmp_path) -> None:
    model, mc = _tiny()
    session, _ = _session(model, mc, n_layers=1)
    pkg = _build_1layer_package(tmp_path, session)
    backend = Qwen7BFoldedPackageGpuBackend(folded_package_path=str(pkg),
                                            device="cpu", dtype="float32")
    backend.init(BoundaryInitRequest(
        session_id="s", hidden_size=128, vocab_size=256, num_layers=1,
        dtype="float32", gpu_backend="qwen7b_folded_package"))
    from pllo.protocol.tee_gpu_messages import MaskedPrefillRequest
    with pytest.raises(NotImplementedError):
        backend.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=[[0.0] * 128], positions=[0],
            batch_size=1, seq_len=1))
