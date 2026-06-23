"""Stage 8.2 tests -- real ModelScope checkpoint probe.

No ModelScope network, no real checkpoint, no CUDA required. Heavy paths are
exercised only via clean-skip statuses and pure mask/dtype helpers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pllo.experiments import modelscope_real_checkpoint_probe as M
from pllo.experiments.modelscope_real_checkpoint_probe import (
    ModelScopeRealCheckpointProbeConfig,
    resolve_dtype,
    run_modelscope_real_checkpoint_probe,
)
from pllo.hf_wrappers.hf_causal_lm_skeleton import (
    HFCausalLMSkeletonConfig,
    generate_hf_causal_lm_masks,
    make_residual_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "run_modelscope_real_checkpoint_probe.py"


# 1.
def test_config_defaults() -> None:
    c = ModelScopeRealCheckpointProbeConfig()
    assert c.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert c.dtype == "bfloat16"
    assert c.mask_mode == "signed_permutation"
    assert c.residual_mask_strategy == "shared"
    assert c.prefill_seq_len == 16 and c.decode_steps == 8
    assert c.max_report_mb == 10


# 2.
def test_signed_permutation_mask_roundtrip() -> None:
    g = torch.Generator().manual_seed(0)
    h = 128
    m, m_inv = make_residual_mask(h, "signed_permutation", torch.float32,
                                  torch.device("cpu"), g)
    # exact orthogonality + roundtrip (no float error for ±1/0 entries)
    assert torch.allclose(m @ m_inv, torch.eye(h), atol=1e-12)
    x = torch.randn(2, 3, h)
    assert torch.allclose((x @ m) @ m_inv, x, atol=1e-12)
    # exactly one nonzero per row/col, values in {-1, +1}
    nz = (m != 0)
    assert nz.sum().item() == h
    assert set(m[nz].abs().unique().tolist()) == {1.0}


# 3.
def test_block_orthogonal_mask_roundtrip_small() -> None:
    g = torch.Generator().manual_seed(1)
    h = 96
    m, m_inv = make_residual_mask(h, "block_orthogonal", torch.float64,
                                  torch.device("cpu"), g, block_size=32)
    assert torch.allclose(m @ m_inv, torch.eye(h, dtype=torch.float64),
                          atol=1e-10)
    # block-diagonal structure: zero outside 32-blocks
    assert m[:32, 32:].abs().max().item() < 1e-12


# 4.
def test_dense_mask_rejected_for_large_hidden_without_override() -> None:
    g = torch.Generator().manual_seed(2)
    with pytest.raises(ValueError):
        make_residual_mask(2048, "dense_orthogonal", torch.float32,
                           torch.device("cpu"), g)
    # override allows it (small enough to actually build here -> use 1100)
    m, _ = make_residual_mask(1100, "dense_orthogonal", torch.float32,
                              torch.device("cpu"), g, allow_dense_large=True)
    assert m.shape == (1100, 1100)


# 5.
def test_probe_skips_cleanly_without_modelscope(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    monkeypatch.setattr(M, "has_transformers", lambda: True)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    assert r["status"] == "skipped_modelscope_unavailable"
    assert "caveats" in r and r["caveats"]


# 6.
def test_probe_skips_cleanly_without_transformers(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_transformers", lambda: False)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    assert r["status"] == "skipped_transformers_unavailable"


# 7.
def test_report_schema_compact(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    for key in ("stage", "config", "status", "resolved_dtype", "environment",
                "required_statement", "caveats"):
        assert key in r, key
    assert r["stage"] == "8.2_modelscope_real_checkpoint"
    assert r["status"].startswith("skipped")


# 8.
def test_report_has_no_tensor_dumps(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    text = json.dumps(r, default=str)
    assert "tensor(" not in text
    import re
    assert re.search(r"(-?\d+\.\d+\s*,\s*){50,}", text) is None
    assert len(text) < 200_000


# 9.
def test_real_checkpoint_cli_help() -> None:
    proc = subprocess.run([sys.executable, str(CLI), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "model-id" in proc.stdout


# 10.
def test_dtype_selection_bfloat16_or_float16() -> None:
    allowed = {torch.bfloat16, torch.float16, torch.float32}
    for name in ("bfloat16", "float16", "float32"):
        for dev in ("cuda", "cpu"):
            dt = resolve_dtype(name, dev)
            assert dt in allowed
            assert dt is not torch.float64
    # explicit fp16 is always honored
    assert resolve_dtype("float16", "cuda") is torch.float16
    assert resolve_dtype("float16", "cpu") is torch.float16
    # explicit fp32 honored; bf16 never degrades to float64
    assert resolve_dtype("float32", "cpu") is torch.float32


# ---------------------------------------------------------------------------
# QR-safety regression tests (Stage 8.2 bf16 CUDA fix)
# ---------------------------------------------------------------------------


def _tiny_skeleton(mask_mode: str, strategy: str, dtype=torch.float32):
    """Build a tiny random Qwen2 + extract + return (weights, configs, cfg)."""
    pytest.importorskip("transformers")
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        extract_hf_causal_lm_skeleton_weights, make_random_tiny_hf_causal_lm)
    cfg = HFCausalLMSkeletonConfig(
        model_family="qwen2", max_layers=2, prefill_seq_len=8, decode_steps=4,
        max_vocab_size=256, dtype=dtype, device="cpu", seed=5,
        mask_mode=mask_mode, residual_mask_strategy=strategy)
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    w, lc, _ = extract_hf_causal_lm_skeleton_weights(
        model, mc, max_layers=2, dtype=dtype, device="cpu")
    return w, lc, cfg


# 11.
def test_signed_permutation_real_mask_does_not_call_qr(monkeypatch) -> None:
    w, lc, cfg = _tiny_skeleton("signed_permutation", "shared")

    def _boom(*a, **k):
        raise AssertionError("torch.linalg.qr must not be called for "
                             "signed_permutation")

    monkeypatch.setattr(torch.linalg, "qr", _boom)
    masks = generate_hf_causal_lm_masks(w, lc, cfg)  # must not raise
    assert masks.metadata["qr_free_residual_mask"] is True
    assert masks.metadata["materialized_dense_from_signed_perm"] is True


# 12.
def test_block_orthogonal_qr_runs_cpu_float32(monkeypatch) -> None:
    seen: list[tuple] = []
    real_qr = torch.linalg.qr

    def _spy(inp, *a, **k):
        seen.append((inp.dtype, inp.device.type))
        return real_qr(inp, *a, **k)

    monkeypatch.setattr(torch.linalg, "qr", _spy)
    g = torch.Generator().manual_seed(3)
    # bf16 target => QR must be CPU float32, output cast to bf16.
    m, _inv = make_residual_mask(96, "block_orthogonal", torch.bfloat16,
                                 torch.device("cpu"), g, block_size=32)
    assert seen, "QR should have run for block_orthogonal"
    assert all(dt == torch.float32 and dev == "cpu" for dt, dev in seen)
    assert m.dtype == torch.bfloat16


# 13.
def test_bfloat16_cuda_signed_permutation_mask_generation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    g = torch.Generator(device="cuda").manual_seed(0)
    m, inv = make_residual_mask(128, "signed_permutation", torch.bfloat16,
                                torch.device("cuda"), g)
    assert m.dtype == torch.bfloat16 and m.device.type == "cuda"
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16, device="cuda")
    assert torch.allclose((x @ m) @ inv, x, atol=1e-2)


# 14.
def test_dense_orthogonal_large_hidden_rejected_without_override() -> None:
    g = torch.Generator().manual_seed(4)
    with pytest.raises(ValueError):
        make_residual_mask(2048, "dense_orthogonal", torch.bfloat16,
                           torch.device("cpu"), g)
    m, _ = make_residual_mask(1100, "dense_orthogonal", torch.bfloat16,
                              torch.device("cpu"), g, allow_dense_large=True)
    assert m.shape == (1100, 1100) and m.dtype == torch.bfloat16


# 15.
def test_stage8_2_config_signed_permutation_is_default() -> None:
    c = ModelScopeRealCheckpointProbeConfig()
    assert c.mask_mode == "signed_permutation"
    assert c.residual_mask_strategy == "shared"
