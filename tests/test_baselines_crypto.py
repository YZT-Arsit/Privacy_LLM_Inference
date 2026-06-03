"""Stage 7.5c tests for the CryptoNets skeleton + crypto cost-model rows."""

from __future__ import annotations

import torch

from pllo.baselines import (
    CryptoNetsArithmeticSkeleton,
    CryptoNetsConfig,
    DelphiCostModel,
    GazelleCostModel,
    MiniONNCostModel,
    SecureMLCostModel,
    UnsupportedResult,
)


def test_cryptonets_arithmetic_skeleton_runs() -> None:
    cn = CryptoNetsArithmeticSkeleton(CryptoNetsConfig(dtype="float64"))
    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(4, 8, dtype=torch.float64, generator=gen)
    w1 = torch.randn(8, 8, dtype=torch.float64, generator=gen)
    w2 = torch.randn(8, 4, dtype=torch.float64, generator=gen)
    res = cn.polynomial_forward(x, [w1, w2])
    assert res["output"].shape == (4, 4)
    assert res["approx_multiplicative_depth"] == 2
    # Plaintext-only: the magnitudes are finite.
    for m in res["per_layer_max_magnitude"]:
        assert m == m  # not nan


def test_cryptonets_does_not_claim_crypto_protocol() -> None:
    cn = CryptoNetsArithmeticSkeleton()
    assert cn.exact_crypto_protocol_implemented is False
    assert cn.declare.arithmetic_skeleton_only is True
    assert cn.declare.full_system_reproduced is False
    assert cn.requires_he_library is False


def test_cryptonets_decode_unsupported() -> None:
    cn = CryptoNetsArithmeticSkeleton()
    res = cn.decode_step()
    assert isinstance(res, UnsupportedResult)
    assert "polynomial" in res.mathematical_reason.lower() or "polynomial" in res.reason.lower()


def test_crypto_cost_models_do_not_emit_fake_runtime() -> None:
    for model_cls in (GazelleCostModel, DelphiCostModel, SecureMLCostModel, MiniONNCostModel):
        inst = model_cls()
        res = inst.forward()
        assert res["directly_comparable_on_runtime"] is False
        # "runtime_ms" must not be present, or if present must be absent/None.
        assert "runtime_ms" not in res, f"{model_cls.__name__} leaked a runtime"
        # Cost dimensions must be present.
        cost = res["cost_model"]
        assert cost["protocol_rounds"]
        assert cost["threat_model"]


def test_crypto_cost_models_decode_and_train_unsupported() -> None:
    for model_cls in (GazelleCostModel, DelphiCostModel, SecureMLCostModel, MiniONNCostModel):
        inst = model_cls()
        for op in (inst.decode_step, inst.train_step):
            res = op()
            assert isinstance(res, UnsupportedResult)
