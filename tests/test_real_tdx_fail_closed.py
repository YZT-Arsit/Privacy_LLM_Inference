"""Gate 1 — fail-closed behavior of the trusted training service (CPU contract).

Verifies the service REFUSES to compute on: bad nonce, replayed nonce, run_id
mismatch, config_digest mismatch, and (in real_tdx mode) missing attestation.
No silent fallback, no plain-CPU fallback, no dummy AdamW.
"""

from __future__ import annotations

import pytest
import torch

from pllo.experiments.real_tdx_attestation import (
    AttestationFailClosed,
    digest_config,
)
from pllo.experiments.real_tdx_training_client import (
    TrainingProtocolError,
    TrustedBoundaryClient,
)
from pllo.experiments.real_tdx_training_service import FailClosed, TrustedTrainingService

DT = torch.float64


def _svc(require_real_tdx=False, mode="cpu_contract"):
    cfg = {"model_id": "syn", "lr": 1e-3, "dtype": "float64"}
    cfg["config_digest"] = digest_config({k: v for k, v in cfg.items()})
    return TrustedTrainingService(run_id="r1", config=cfg, mode=mode,
                                  require_real_tdx=require_real_tdx, dtype=DT, seed=1), cfg


def _init(svc, cfg, vocab=8):
    return svc.init_session({"run_id": "r1", "config_digest": cfg["config_digest"],
                             "lora_manifest": [{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                             "vocab_size": vocab, "labels_by_step": {0: [1, 2]}})


def test_require_real_tdx_without_guest_fails_closed():
    # real_tdx mode + require_real_tdx on a CPU host (no /dev/tdx_guest) must refuse.
    svc, cfg = _svc(require_real_tdx=True, mode="real_tdx")
    with pytest.raises((AttestationFailClosed, FailClosed)):
        _init(svc, cfg)


def test_bad_nonce_rejected():
    svc, cfg = _svc(); _init(svc, cfg)
    ml = torch.randn(2, 8, dtype=DT)
    req = {"run_id": "r1", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": "deadbeefdeadbeef", "manifest": {"masked_logits": {"shape": [2, 8], "dtype": "float64"}},
           "masked_logits": ml}
    with pytest.raises(FailClosed, match="nonce"):
        svc.logits_loss(req)


def test_replayed_nonce_rejected():
    svc, cfg = _svc(); init = _init(svc, cfg)
    nonce = init["next_nonce"]
    ml = torch.randn(2, 8, dtype=DT)
    good = {"run_id": "r1", "step_id": 0, "config_digest": cfg["config_digest"],
            "nonce": nonce, "manifest": {"masked_logits": {"shape": [2, 8], "dtype": "float64"}},
            "masked_logits": ml}
    svc.logits_loss(good)                    # consumes the nonce, rotates
    with pytest.raises(FailClosed, match="nonce"):
        svc.logits_loss(good)                # same (now stale) nonce -> reject


def test_run_id_mismatch_rejected():
    svc, cfg = _svc(); init = _init(svc, cfg)
    ml = torch.randn(2, 8, dtype=DT)
    req = {"run_id": "WRONG", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": init["next_nonce"], "manifest": {"masked_logits": {"shape": [2, 8], "dtype": "float64"}},
           "masked_logits": ml}
    with pytest.raises(FailClosed, match="run_id"):
        svc.logits_loss(req)


def test_config_digest_mismatch_rejected():
    svc, cfg = _svc(); init = _init(svc, cfg)
    ml = torch.randn(2, 8, dtype=DT)
    req = {"run_id": "r1", "step_id": 0, "config_digest": "0" * 64,
           "nonce": init["next_nonce"], "manifest": {"masked_logits": {"shape": [2, 8], "dtype": "float64"}},
           "masked_logits": ml}
    with pytest.raises(FailClosed, match="config_digest"):
        svc.logits_loss(req)


def test_client_surfaces_failclosed_as_protocol_error():
    svc, cfg = _svc()
    cfg_digest = cfg["config_digest"]
    client = TrustedBoundaryClient(service=svc)
    client.init_session(run_id="r1", config_digest=cfg_digest,
                        lora_manifest=[{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                        vocab_size=8, labels_by_step={0: [1, 2]})
    client._force_nonce("badnoncebadnonce")     # corrupt the nonce
    with pytest.raises(TrainingProtocolError):
        client.logits_loss(0, torch.randn(2, 8, dtype=DT))


def test_no_dummy_adamw_on_bad_manifest():
    # a packed_update whose manifest omits a registered layer must NOT partially/dummy-update
    svc, cfg = _svc(); init = _init(svc, cfg)
    A0, B0 = svc.plaintext_adapter("L0")
    A0, B0 = A0.clone(), B0.clone()
    bad = {"run_id": "r1", "step_id": 0, "config_digest": cfg["config_digest"],
           "nonce": init["next_nonce"], "manifest": {"layers": {}},  # empty manifest
           "packed_grads": {}}
    with pytest.raises(FailClosed):
        svc.packed_update(bad)
    A1, B1 = svc.plaintext_adapter("L0")
    assert (A1 - A0).abs().max() == 0 and (B1 - B0).abs().max() == 0   # unchanged
