"""Tests for the selectable nonlinear backends (Line A: current; Line B: Amulet).

Verifies the registry, that both backends compute correct outputs for all ops,
the trusted/accelerator accounting differs as designed, and the Amulet security
status is explicitly NOT claimed. torch + numpy.

Run: python -m pytest tests/test_nonlinear_backend_registry.py -q
"""

from __future__ import annotations

import pytest
import torch

from pllo.nonlinear.backends import (
    OP_NAMES,
    reference_gelu,
    reference_layernorm,
    reference_rmsnorm,
    reference_silu,
    reference_softmax,
)
from pllo.nonlinear.registry import (
    available_backends,
    backend_security_claim_status,
    backend_security_status,
    make_nonlinear_backend,
)

BACKENDS = ["current", "amulet_migrated"]


def _input(seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 16, generator=g).to(dtype)
    w = torch.randn(16, generator=g).to(dtype)
    b = torch.randn(16, generator=g).to(dtype)
    return x, w, b


def test_registry_lists_both_backends() -> None:
    assert set(available_backends()) == {
        "current", "amulet_migrated", "compatible_right_multiply"}
    for name in BACKENDS:
        be = make_nonlinear_backend(name)
        assert be.name == name


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        make_nonlinear_backend("nope")


def test_unknown_op_raises() -> None:
    be = make_nonlinear_backend("current")
    with pytest.raises(ValueError):
        be.run("not_an_op", torch.zeros(2, 2))


@pytest.mark.parametrize("name", BACKENDS)
def test_all_ops_match_reference(name) -> None:
    be = make_nonlinear_backend(name)
    x, w, b = _input(dtype=torch.float64)
    refs = {
        "gelu": reference_gelu(x), "silu": reference_silu(x),
        "softmax": reference_softmax(x, -1),
        "layernorm": reference_layernorm(x, w, b, 1e-5),
        "rmsnorm": reference_rmsnorm(x, w, 1e-6),
    }
    for op in OP_NAMES:
        kw = {}
        if op in ("layernorm", "rmsnorm"):
            kw = {"weight": w, "eps": 1e-5 if op == "layernorm" else 1e-6}
            if op == "layernorm":
                kw["bias"] = b
        res = be.run(op, x, **kw)
        # functionally exact (migration preserves correctness)
        assert torch.allclose(res.output.to(torch.float64), refs[op],
                              atol=1e-10, rtol=1e-8), f"{name}/{op}"


@pytest.mark.parametrize("name", BACKENDS)
def test_tee_used_on_gpu_is_false(name) -> None:
    be = make_nonlinear_backend(name)
    x, w, b = _input()
    for op in OP_NAMES:
        kw = {"weight": w, "bias": b} if op == "layernorm" else (
            {"weight": w} if op == "rmsnorm" else {})
        assert be.run(op, x, **kw).tee_used_on_gpu is False


def test_current_runs_in_trusted_boundary() -> None:
    be = make_nonlinear_backend("current")
    x, w, b = _input()
    for op in OP_NAMES:
        kw = {"weight": w, "bias": b} if op == "layernorm" else (
            {"weight": w} if op == "rmsnorm" else {})
        res = be.run(op, x, **kw)
        assert res.trusted_calls >= 1            # evaluated in the trusted boundary
        assert res.gpu_bytes == 0                # nothing sent to the accelerator
        assert res.trusted_bytes > 0


def test_amulet_migrates_off_trusted_boundary() -> None:
    be = make_nonlinear_backend("amulet_migrated")
    x, w, b = _input()
    # activations: fully migrated (no trusted crossing, accelerator bytes > 0)
    for op in ("gelu", "silu"):
        res = be.run(op, x)
        assert res.trusted_calls == 0
        assert res.gpu_bytes > 0
        assert res.extra["location"] == "untrusted_accelerator"
    # softmax / norms: elementwise migrated, only a small trusted reduction stat
    for op, kw in (("softmax", {}), ("layernorm", {"weight": w, "bias": b}),
                   ("rmsnorm", {"weight": w})):
        res = be.run(op, x, **kw)
        assert res.trusted_calls == 1
        assert res.gpu_bytes > 0
        assert res.trusted_bytes < res.gpu_bytes


def test_amulet_softmax_preserves_top1() -> None:
    be = make_nonlinear_backend("amulet_migrated")
    x, _, _ = _input(dtype=torch.float32)
    out = be.run("softmax", x).output
    ref = reference_softmax(x, -1)
    assert torch.equal(out.to(torch.float64).argmax(-1), ref.argmax(-1))


def test_security_status_amulet_not_claimed() -> None:
    sec = backend_security_status()
    assert sec["current"] == "trusted_boundary"
    assert sec["amulet_migrated"] == "not_formally_claimed"
    amulet = make_nonlinear_backend("amulet_migrated")
    assert "not" in amulet.security_note.lower()
    # explicit: no proven-secure claim for amulet
    assert amulet.security_status in {"not_formally_claimed", "under_discussion"}


def test_security_claim_status_amulet_under_discussion() -> None:
    claim = backend_security_claim_status()
    assert claim["amulet_migrated"] == "under_discussion"
    assert claim["current"] == "established"
    # describe() surfaces the paper-facing claim status
    assert make_nonlinear_backend("amulet_migrated").describe()[
        "security_claim_status"] == "under_discussion"


def test_amulet_lift_k_validation() -> None:
    with pytest.raises(ValueError):
        make_nonlinear_backend("amulet_migrated", lift_k=1)
