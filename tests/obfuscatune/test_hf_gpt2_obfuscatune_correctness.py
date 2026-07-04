"""HF GPT-2 wrapper correctness (tiny random model, no download).

Verifies that under orthogonal obfuscation the full-model logits reproduce the
unprotected reference to floating-point tolerance, that the naive random variant
stays finite, and that the structural facts (plaintext Q/K/V + MLP intermediate
exposed, non-linearities in the TEE) hold.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.hf_gpt2_obfuscatune import (  # noqa: E402
    gpt2_block_obfuscated_forward,
    load_gpt2,
    make_tiny_gpt2_config,
    run_mode,
)
from pllo.baselines.obfuscatune.config import ObfuscaTuneConfig  # noqa: E402


def _tiny_model(dtype=torch.float64):
    cfg = make_tiny_gpt2_config(vocab_size=64, n_positions=32, n_embd=32,
                                n_layer=3, n_head=4)
    return load_gpt2(tiny_random_config=cfg, seed=0, dtype=dtype)


def test_unprotected_is_exact_reference():
    model = _tiny_model()
    ids = torch.randint(0, 64, (12,))
    r = run_mode(model, ids, "unprotected", dtype=torch.float64)
    assert r["correctness"]["max_abs_error"] == 0.0
    assert r["nan_or_inf_count"] == 0


def test_orthogonal_reproduces_reference_fp64():
    model = _tiny_model(torch.float64)
    ids = torch.randint(0, 64, (12,))
    r = run_mode(model, ids, "obfuscatune_orthogonal", dtype=torch.float64)
    assert r["correctness"]["max_abs_error"] < 1e-8
    assert r["argmax_match_rate"] == 1.0
    assert r["nan_or_inf_count"] == 0


def test_orthogonal_reproduces_reference_fp32():
    model = _tiny_model(torch.float32)
    ids = torch.randint(0, 64, (12,))
    r = run_mode(model, ids, "obfuscatune_orthogonal", dtype=torch.float32)
    assert r["correctness"]["max_abs_error"] < 1e-3
    assert r["argmax_match_rate"] == 1.0


def test_random_variant_is_finite():
    model = _tiny_model(torch.float64)
    ids = torch.randint(0, 64, (12,))
    r = run_mode(model, ids, "obfuscatune_random", dtype=torch.float64)
    assert r["nan_or_inf_count"] == 0
    # random matrix has higher error than orthogonal
    r_orth = run_mode(model, ids, "obfuscatune_orthogonal", dtype=torch.float64)
    assert r["correctness"]["max_abs_error"] >= r_orth["correctness"]["max_abs_error"]


def test_block_exposes_plaintext_qkv_and_tee_nonlinears():
    model = _tiny_model()
    block = model.transformer.h[0]
    hidden = torch.randn(10, 32, dtype=torch.float64)
    cfg = ObfuscaTuneConfig(dtype="float64", random_matrix_type="orthogonal")
    _, audit, metrics = gpt2_block_obfuscated_forward(block, hidden, cfg)
    exposed = audit["exposed_plaintext_tensors"]
    assert {"Q_plaintext", "K_plaintext", "V_plaintext", "mlp_intermediate_plaintext"} <= set(exposed)
    assert set(audit["tee_nonlinear_ops"]) == {"layernorm", "softmax", "gelu"}
    assert metrics.boundary_calls > 0        # non-linearities force TEE crossings


def test_cond_sweep_mode_runs():
    model = _tiny_model()
    ids = torch.randint(0, 64, (12,))
    prev = -1.0
    for kappa in (1.0, 32.0, 160.0):
        r = run_mode(model, ids, "obfuscatune_cond_sweep", dtype=torch.float64,
                     condition_number=kappa)
        assert r["nan_or_inf_count"] == 0
        assert r["correctness"]["max_abs_error"] >= prev
        prev = r["correctness"]["max_abs_error"]
