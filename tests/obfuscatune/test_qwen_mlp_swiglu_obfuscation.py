"""ObfuscaTune-Qwen SwiGLU MLP correctness (gate/up/down; SiLU in plaintext)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.config import ObfuscaTuneConfig  # noqa: E402
from pllo.baselines.obfuscatune.qwen_config import (  # noqa: E402
    load_qwen, make_tiny_qwen_config)
from pllo.baselines.obfuscatune.qwen_modules import qwen_mlp_obfuscated  # noqa: E402
from pllo.baselines.obfuscatune.random_matrices import orthogonal_matrix  # noqa: E402


def _model(dtype=torch.float64):
    cfg = make_tiny_qwen_config(hidden_size=32, intermediate_size=64)
    return load_qwen(tiny_random_qwen=cfg, seed=0, dtype=dtype)


def test_mlp_orthogonal_matches_reference():
    model = _model()
    mlp = model.model.layers[0].mlp
    x = torch.randn(1, 6, model.config.hidden_size, dtype=torch.float64)
    cfg = ObfuscaTuneConfig(dtype="float64", random_matrix_type="orthogonal")
    out, audit, _m = qwen_mlp_obfuscated(x, mlp, cfg)
    ref = mlp(x)          # HF SwiGLU: down(silu(gate(x)) * up(x))
    assert torch.allclose(out, ref, atol=1e-9)
    assert "mlp_gate_plaintext" in audit["exposed_plaintext_tensors"]


def test_swiglu_not_obfuscation_safe():
    # silu(gate(X R)) != silu(gate(X)) applied through the mask: activation must
    # run on plaintext, never in the obfuscated domain.
    torch.manual_seed(0)
    d = 32
    x = torch.randn(4, d, dtype=torch.float64)
    r, _ = orthogonal_matrix(d, seed=1, dtype=torch.float64)
    lhs = torch.nn.functional.silu(x @ r)
    rhs = torch.nn.functional.silu(x) @ r
    assert not torch.allclose(lhs, rhs, atol=1e-3)


def test_mlp_condition_number_trend():
    model = _model(dtype=torch.float32)
    mlp = model.model.layers[0].mlp
    x = torch.randn(1, 6, model.config.hidden_size, dtype=torch.float32)
    ref = mlp(x)
    errs = {}
    for kind, kappa in [("orthogonal", 1.0), ("cond", 128.0), ("random", None)]:
        cfg = ObfuscaTuneConfig(dtype="float32", random_matrix_type=kind,
                                condition_number=kappa or 1.0)
        out, _a, _m = qwen_mlp_obfuscated(x, mlp, cfg)
        errs[kind] = float((out - ref).abs().max().item())
    assert errs["orthogonal"] <= errs["cond"]
    assert torch.isfinite(torch.tensor(errs["random"]))
