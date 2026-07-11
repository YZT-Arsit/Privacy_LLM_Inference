"""Gate 1.5-E — orthogonal masked SGD/momentum exactness (CPU fp64) + mode fail-closed."""

from __future__ import annotations

import pytest
import torch

from pllo.experiments.masked_sgd_audit import (
    invocation_counts,
    orthogonal_collapse_check,
    run_alignment,
)
from pllo.experiments.real_tdx_attestation import digest_config
from pllo.experiments.real_tdx_training_service import FailClosed, TrustedTrainingService

DT = torch.float64


# ---- exactness under fixed orthogonal / permutation masks ----
def test_orthogonal_collapse():
    c = orthogonal_collapse_check()
    assert c["orthogonal_forms_agree"]      # N_in^T gradA U^-T == N_in^-1 gradA U (orthogonal)
    assert c["gl_forms_differ"]             # not so for general GL


def test_pure_sgd_orthogonal_100_steps():
    r = run_alignment(optimizer="sgd", mask_kind="orthogonal", steps=100)
    assert r["aligned"] and r["worst"]["A_err"] < 1e-9 and r["worst"]["dW_err"] < 1e-9


def test_momentum_sgd_orthogonal_100_steps():
    r = run_alignment(optimizer="momentum_sgd", momentum=0.9, lr=0.003,
                      mask_kind="orthogonal", steps=100)
    assert r["aligned"] and r["worst"]["momentum_err"] < 1e-9


def test_permutation_masks_pass():
    for opt, mom in (("sgd", 0.0), ("momentum_sgd", 0.9)):
        r = run_alignment(optimizer=opt, momentum=mom, lr=0.003,
                          mask_kind="permutation", steps=40)
        assert r["aligned"]


def test_nesterov_supported():
    r = run_alignment(optimizer="momentum_sgd", momentum=0.9, lr=0.003, nesterov=True,
                      mask_kind="orthogonal", steps=50)
    assert r["aligned"]                     # Nesterov is proven, not rejected


def test_weight_decay_coupled_and_decoupled():
    for mode in ("coupled", "decoupled"):
        r = run_alignment(optimizer="sgd", weight_decay=0.01, wd_mode=mode,
                          mask_kind="orthogonal", steps=50)
        assert r["aligned"], mode


def test_gradient_accumulation():
    for accum in (1, 2, 4):
        r = run_alignment(optimizer="momentum_sgd", momentum=0.9, lr=0.003, accum=accum,
                          mask_kind="orthogonal", steps=30)
        assert r["aligned"], accum


# ---- negative controls (must NOT align) ----
def test_mask_refresh_without_reencoding_fails():
    r = run_alignment(optimizer="sgd", mask_kind="orthogonal", refresh_masks=True, steps=20)
    assert not r["aligned"]


def test_dense_non_orthogonal_direct_sgd_fails():
    r = run_alignment(optimizer="sgd", mask_kind="gl", steps=20)
    assert not r["aligned"]


# ---- invocation counts ----
def test_invocation_counts_two_vs_three():
    sgd = invocation_counts("gpu_masked_sgd", 32)
    ada = invocation_counts("trusted_adamw", 32)
    assert sgd["total_trusted_invocations"] == 2
    assert sgd["packed_update_invocation"] == 0 and sgd["optimizer_trusted_invocation"] == 0
    assert sgd["worker_to_trusted_returns_after_input"] == 1
    assert ada["total_trusted_invocations"] == 3 and ada["packed_update_invocation"] == 1


# ---- service-side fail-closed for the new mode ----
def _svc(optimizer_mode, optimizer):
    cfg = {"optimizer_mode": optimizer_mode, "optimizer": optimizer, "lr": 1e-3,
           "dtype": "float64", "gradient_convention": "nout_dual"}
    cfg["config_digest"] = digest_config(dict(cfg))
    svc = TrustedTrainingService(run_id="r", config=cfg, mode="cpu_contract", dtype=DT, seed=1)
    init = svc.init_session({"run_id": "r", "config_digest": cfg["config_digest"],
                             "lora_manifest": [{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                             "vocab_size": 8, "labels_by_step": {}})
    return svc, cfg, init


def test_gpu_masked_sgd_rejects_adamw_optimizer():
    with pytest.raises(FailClosed, match="rejects optimizer|requires SGD"):
        _svc("gpu_masked_sgd", "adamw")


def test_gpu_masked_sgd_uses_orthogonal_masks():
    svc, cfg, init = _svc("gpu_masked_sgd", "sgd")
    N_in, R, N_out = svc._layer_masks("L0")
    for M in (N_in, R, N_out):
        assert (M @ M.T - torch.eye(M.shape[0], dtype=DT)).abs().max() < 1e-6
    assert init["verification"]["optimizer_mode"] == "gpu_masked_sgd"


def test_packed_update_rejected_in_masked_sgd_mode():
    svc, cfg, init = _svc("gpu_masked_sgd", "momentum_sgd")
    req = {"run_id": "r", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": init["next_nonce"], "manifest": {"layers": {}}, "packed_grads": {}}
    with pytest.raises(FailClosed, match="packed_update not allowed"):
        svc.packed_update(req)


def test_trusted_adamw_still_allows_packed_update():
    svc, cfg, init = _svc("trusted_adamw", "adamw")
    N_in, R, N_out = svc._layer_masks("L0")
    d_in, r = N_in.shape[0], R.shape[0]; d_out = N_out.shape[0]
    packed = {"L0": {"gradA": torch.zeros(d_in, r, dtype=DT), "gradB": torch.zeros(r, d_out, dtype=DT)}}
    man = {"L0": {"gradA": {"shape": [d_in, r], "dtype": "float64"},
                  "gradB": {"shape": [r, d_out], "dtype": "float64"}}}
    resp = svc.packed_update({"run_id": "r", "step_id": 0, "config_digest": cfg["config_digest"],
                              "nonce": init["next_nonce"], "manifest": {"layers": man},
                              "packed_grads": packed})
    assert resp["verification"]["trusted_adamw_executed"]


# ---- Gate 2 §1-B/§1-C: no implicit mode/convention switching ----
def test_per_request_optimizer_mode_switch_rejected():
    svc, cfg, init = _svc("gpu_masked_sgd", "sgd")
    ml = torch.randn(2, 8, dtype=DT)
    req = {"run_id": "r", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": init["next_nonce"],
           "manifest": {"masked_logits": {"shape": [2, 8], "dtype": "float64"}},
           "masked_logits": ml, "optimizer_mode": "trusted_adamw"}   # attempt switch
    with pytest.raises(FailClosed, match="optimizer_mode switch"):
        svc.logits_loss(req)


def test_per_request_gradient_convention_switch_rejected():
    svc, cfg, init = _svc("gpu_masked_sgd", "sgd")
    ml = torch.randn(2, 8, dtype=DT)
    req = {"run_id": "r", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": init["next_nonce"],
           "manifest": {"masked_logits": {"shape": [2, 8], "dtype": "float64"}},
           "masked_logits": ml, "gradient_convention": "independent_mout"}
    with pytest.raises(FailClosed, match="gradient_convention switch"):
        svc.logits_loss(req)


def test_missing_gradient_convention_in_config_fails_closed():
    cfg = {"optimizer_mode": "trusted_adamw", "optimizer": "adamw", "lr": 1e-3,
           "dtype": "float64"}                          # no gradient_convention
    cfg["config_digest"] = digest_config(dict(cfg))
    svc = TrustedTrainingService(run_id="r", config=cfg, mode="cpu_contract", dtype=DT, seed=1)
    with pytest.raises(FailClosed, match="gradient_convention"):
        svc.init_session({"run_id": "r", "config_digest": cfg["config_digest"],
                          "lora_manifest": [{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                          "vocab_size": 8, "labels_by_step": {0: [1, 2]}})


def test_unimplemented_convention_fails_closed():
    cfg = {"optimizer_mode": "trusted_adamw", "optimizer": "adamw", "lr": 1e-3,
           "dtype": "float64", "gradient_convention": "independent_mout"}
    cfg["config_digest"] = digest_config(dict(cfg))
    svc = TrustedTrainingService(run_id="r", config=cfg, mode="cpu_contract", dtype=DT, seed=1)
    with pytest.raises(FailClosed, match="not implemented"):
        svc.init_session({"run_id": "r", "config_digest": cfg["config_digest"],
                          "lora_manifest": [{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                          "vocab_size": 8, "labels_by_step": {0: [1, 2]}})
