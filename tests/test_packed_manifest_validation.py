"""Gate 1 — packed manifest / shape / dtype validation (CPU contract)."""

from __future__ import annotations

import pytest
import torch

from pllo.experiments.real_tdx_attestation import digest_config
from pllo.experiments.real_tdx_training_service import FailClosed, TrustedTrainingService

DT = torch.float64


def _svc_init():
    cfg = {"model_id": "syn", "lr": 1e-3, "dtype": "float64"}
    cfg["config_digest"] = digest_config({k: v for k, v in cfg.items()})
    svc = TrustedTrainingService(run_id="r1", config=cfg, mode="cpu_contract",
                                 dtype=DT, seed=1)
    init = svc.init_session({"run_id": "r1", "config_digest": cfg["config_digest"],
                             "lora_manifest": [{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3},
                                               {"layer_id": "L1", "d_in": 6, "d_out": 8, "rank": 3}],
                             "vocab_size": 8, "labels_by_step": {0: [1, 2]}})
    return svc, cfg, init


def _good_packed(svc):
    out = {}
    man = {}
    for lid in ("L0", "L1"):
        N_in, R, N_out = svc._layer_masks(lid)
        d_in, r = N_in.shape[0], R.shape[0]; d_out = N_out.shape[0]
        gA = torch.zeros(d_in, r, dtype=DT); gB = torch.zeros(r, d_out, dtype=DT)
        out[lid] = {"gradA": gA, "gradB": gB}
        man[lid] = {"gradA": {"shape": [d_in, r], "dtype": "float64"},
                    "gradB": {"shape": [r, d_out], "dtype": "float64"}}
    return out, man


def _update(svc, cfg, nonce, packed, man):
    return svc.packed_update({"run_id": "r1", "step_id": 0, "config_digest": cfg["config_digest"],
                              "nonce": nonce, "manifest": {"layers": man}, "packed_grads": packed})


def test_good_manifest_accepted():
    svc, cfg, init = _svc_init()
    packed, man = _good_packed(svc)
    resp = _update(svc, cfg, init["next_nonce"], packed, man)
    assert set(resp["updated_masked_adapters"].keys()) == {"L0", "L1"}


def test_missing_layer_rejected():
    svc, cfg, init = _svc_init()
    packed, man = _good_packed(svc)
    del packed["L1"]; del man["L1"]
    with pytest.raises(FailClosed, match="layer set"):
        _update(svc, cfg, init["next_nonce"], packed, man)


def test_extra_layer_rejected():
    svc, cfg, init = _svc_init()
    packed, man = _good_packed(svc)
    packed["L2"] = {"gradA": torch.zeros(6, 3, dtype=DT), "gradB": torch.zeros(3, 6, dtype=DT)}
    man["L2"] = {"gradA": {"shape": [6, 3], "dtype": "float64"},
                 "gradB": {"shape": [3, 6], "dtype": "float64"}}
    with pytest.raises(FailClosed, match="layer set"):
        _update(svc, cfg, init["next_nonce"], packed, man)


def test_gradA_shape_mismatch_rejected():
    svc, cfg, init = _svc_init()
    packed, man = _good_packed(svc)
    packed["L0"]["gradA"] = torch.zeros(6, 99, dtype=DT)   # wrong rank
    with pytest.raises(FailClosed, match="gradA shape"):
        _update(svc, cfg, init["next_nonce"], packed, man)


def test_gradA_dtype_mismatch_rejected():
    svc, cfg, init = _svc_init()
    packed, man = _good_packed(svc)
    man["L0"]["gradA"]["dtype"] = "float32"                # manifest lies about dtype
    with pytest.raises(FailClosed, match="gradA dtype"):
        _update(svc, cfg, init["next_nonce"], packed, man)


def test_masked_logits_shape_mismatch_rejected():
    svc, cfg, init = _svc_init()
    ml = torch.randn(2, 8, dtype=DT)
    req = {"run_id": "r1", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": init["next_nonce"],
           "manifest": {"masked_logits": {"shape": [3, 8], "dtype": "float64"}},  # wrong rows
           "masked_logits": ml}
    with pytest.raises(FailClosed, match="masked_logits shape"):
        svc.logits_loss(req)
