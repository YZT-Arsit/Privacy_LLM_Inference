"""TEE runtime API tests (simulated backend; numpy only, no torch/transformers).

Run: python -m pytest tests/test_tee_runtime_api.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pllo.tee import (
    AttestationReport,
    MaskedEmbeddingPacket,
    MaskedLogitsPacket,
    SamplingResult,
    TEEConfig,
    apply_vocab_logit_mask,
    make_runtime,
)
from pllo.tee.runtime_api import make_runtime as api_make_runtime

REPO_ROOT = Path(__file__).resolve().parents[1]
TEE_DIR = REPO_ROOT / "src" / "pllo" / "tee"


def _cfg(backend="simulated", hidden=64, vocab=2000, seed=123):
    return TEEConfig(hidden_size=hidden, vocab_size=vocab, seed=seed,
                     backend=backend)


def _masked_logits(rt, logits):
    Lt = apply_vocab_logit_mask(logits, rt.handles)
    return MaskedLogitsPacket(Lt, logits.shape[0], logits.shape[-1],
                              str(Lt.dtype), int(Lt.nbytes))


# 1.
def test_dataclasses_construct() -> None:
    cfg = TEEConfig()
    assert cfg.backend == "simulated"
    assert cfg.hidden_size == 2048 and cfg.vocab_size == 151936
    rep = AttestationReport(backend="simulated", tee_type="simulated",
                            available=False, tdx_guest_device_present=False,
                            tdreport_available=False, quote_available=False,
                            quote_status="pending_vendor_qgs_evidence")
    assert rep.attributes == {}


# 2.
def test_attest_report_fields() -> None:
    rt = make_runtime(_cfg())
    rep = rt.attest()
    assert isinstance(rep, AttestationReport)
    assert rep.quote_available is False
    assert rep.quote_status == "pending_vendor_qgs_evidence"
    assert isinstance(rep.tdx_guest_device_present, bool)
    assert rep.attributes["no_debug"] is True
    assert rep.attributes["sept_ve_disable"] is True
    # tee_type reflects the actual device presence
    assert rep.tee_type in ("intel_tdx", "simulated")
    assert (rep.tee_type == "intel_tdx") == rep.tdx_guest_device_present


# 3.
def test_setup_deterministic_same_seed() -> None:
    ids = np.random.default_rng(0).integers(0, 2000, (4, 8))
    a = make_runtime(_cfg(seed=55)).embed_and_mask(ids).masked_embeddings
    b = make_runtime(_cfg(seed=55)).embed_and_mask(ids).masked_embeddings
    assert np.array_equal(a, b)


# 4.
def test_different_seed_changes_masking() -> None:
    ids = np.random.default_rng(0).integers(0, 2000, (4, 8))
    a = make_runtime(_cfg(seed=1)).embed_and_mask(ids).masked_embeddings
    b = make_runtime(_cfg(seed=2)).embed_and_mask(ids).masked_embeddings
    assert not np.array_equal(a, b)


# 5.
def test_embedding_mask_is_signed_permutation_roundtrip() -> None:
    from pllo.tee.runtime_api import (apply_signed_permutation,
                                      invert_signed_permutation)
    rt = make_runtime(_cfg())
    h = rt.handles
    x = np.random.default_rng(3).standard_normal((2, 5, 64)).astype(np.float32)
    xt = apply_signed_permutation(x, h.residual_perm, h.residual_signs)
    back = invert_signed_permutation(xt, h.residual_inv_perm, h.residual_signs)
    assert np.allclose(back, x, atol=1e-6)


# 6.
def test_vocab_logit_recover_roundtrip_deterministic() -> None:
    rt = make_runtime(_cfg())
    L = np.random.default_rng(4).standard_normal((4, 2000)).astype(np.float32)
    pkt = _masked_logits(rt, L)
    rec = rt.recover_logits(pkt)
    assert np.allclose(rec, L, atol=1e-4)
    # recovered argmax equals plaintext argmax
    assert np.array_equal(rt.sample(rec).next_token_ids, L.argmax(axis=-1))


# 7.
def test_wrong_mask_recovery_fails_numerically() -> None:
    good = make_runtime(_cfg(seed=10))
    L = np.random.default_rng(5).standard_normal((8, 2000)).astype(np.float32)
    pkt = _masked_logits(good, L)            # masked with seed=10
    wrong = make_runtime(_cfg(seed=20))      # recover with the WRONG seed
    rec_wrong = wrong.recover_logits(pkt)
    assert np.abs(rec_wrong - L).max() > 1.0
    wrong_tokens = wrong.sample(rec_wrong).next_token_ids
    assert (wrong_tokens != L.argmax(axis=-1)).any()


# 8.
def test_sample_greedy_argmax_and_shapes() -> None:
    rt = make_runtime(_cfg())
    logits = np.random.default_rng(6).standard_normal((3, 2000)).astype(
        np.float32)
    res = rt.sample(logits)
    assert isinstance(res, SamplingResult)
    assert res.next_token_ids.shape == (3,)
    assert np.array_equal(res.next_token_ids, logits.argmax(axis=-1))
    # 3-D [B,T,V] -> uses last position
    logits3 = np.random.default_rng(7).standard_normal((3, 5, 2000)).astype(
        np.float32)
    res3 = rt.sample(logits3)
    assert np.array_equal(res3.next_token_ids, logits3[:, -1, :].argmax(-1))


# 9.
def test_packet_schema_stable() -> None:
    rt = make_runtime(_cfg())
    ids = np.zeros((2, 4), dtype=np.int64)
    emb = rt.embed_and_mask(ids)
    assert isinstance(emb, MaskedEmbeddingPacket)
    assert set(vars(emb)) == {"masked_embeddings", "batch_size", "seq_len",
                              "hidden_size", "dtype", "nbytes"}
    assert emb.batch_size == 2 and emb.seq_len == 4 and emb.hidden_size == 64
    assert emb.nbytes == emb.masked_embeddings.nbytes
    L = np.zeros((2, 2000), dtype=np.float32)
    pkt = _masked_logits(rt, L)
    assert set(vars(pkt)) == {"masked_logits", "batch_size", "vocab_size",
                              "dtype", "nbytes"}
    res = rt.sample(rt.recover_logits(pkt))
    assert set(vars(res)) == {"next_token_ids", "batch_size", "nbytes"}


# 10.
def test_embed_and_mask_rejects_non_integer_ids() -> None:
    rt = make_runtime(_cfg())
    with pytest.raises(TypeError):
        rt.embed_and_mask(np.zeros((2, 4), dtype=np.float32))


# 11.
def test_make_runtime_unknown_backend() -> None:
    with pytest.raises(ValueError):
        api_make_runtime(TEEConfig(backend="bogus"))


# 12.
def test_no_torch_or_transformers_required() -> None:
    """The TEE runtime must import with numpy only -- no torch/transformers.

    Checked two ways: (a) a clean subprocess import leaves torch/transformers
    out of sys.modules; (b) no TEE source file imports them."""
    code = (
        "import sys; import pllo.tee; "
        "import pllo.tee.simulated_runtime, pllo.tee.process_runtime; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'transformers' not in sys.modules, 'transformers imported'; "
        "print('clean')"
    )
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout

    import re
    forbidden = re.compile(
        r"^\s*(?:import|from)\s+(?:torch|transformers)\b", re.MULTILINE)
    for f in TEE_DIR.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        assert not forbidden.search(src), f"{f.name} imports torch/transformers"


# 13.
def test_no_decoder_attention_mlp_model_class_in_tee() -> None:
    """No decoder / attention / MLP / LM-head / transformer / model class or
    function may exist inside src/pllo/tee/* -- those run on the untrusted side.

    Uses AST so the prose in docstrings (which legitimately *mentions* these
    untrusted components) does not trigger false positives; only actual
    class/function definition names and import targets are inspected."""
    import ast

    forbidden_defs = ("attention", "attn", "decoder", "mlp", "transformer",
                      "lmhead", "lm_head", "feedforward", "feed_forward",
                      "self_attn", "selfattn", "layernorm", "rmsnorm")
    forbidden_class_extra = ("model", "layer", "block")
    forbidden_imports = ("torch", "transformers")

    tee_files = list(TEE_DIR.glob("*.py"))
    assert tee_files, "no TEE source files found"
    for f in tee_files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                low = node.name.lower()
                assert not any(b in low for b in forbidden_defs), \
                    f"{f.name}: forbidden function {node.name!r}"
            elif isinstance(node, ast.ClassDef):
                low = node.name.lower()
                bad = forbidden_defs + forbidden_class_extra
                assert not any(b in low for b in bad), \
                    f"{f.name}: forbidden class {node.name!r}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] not in forbidden_imports, \
                        f"{f.name}: forbidden import {a.name!r}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in forbidden_imports, \
                    f"{f.name}: forbidden import-from {node.module!r}"
    # the only model-derived tensor allowed in the TEE is the embedding table
    # (the embedding boundary); assert no LM-head/decoder weight attributes leak
    import pllo.tee.simulated_runtime as sim
    rt = sim.SimulatedTrustedRuntime(_cfg())
    rt.embed_and_mask(np.zeros((1, 2), dtype=np.int64))  # force table build
    attr_names = {a.lower() for a in vars(rt)}
    for bad in ("lm_head", "decoder", "attn", "mlp", "head_weight",
                "head", "layers"):
        assert not any(bad in a for a in attr_names), \
            f"runtime holds forbidden state matching {bad!r}: {attr_names}"
