"""Gate 2 §4/§6 — end-to-end attested-session wire path (cpu_contract, real HTTP).

Exercises /train/challenge -> external verify -> /train/handshake -> AEAD-wrapped
/train/init + /train/logits_loss over a real localhost HTTP server. cpu_contract mode
produces NO real quote (honest); we pass allow_any_mr_td + a stub chain_verifier so
the handshake completes and the AEAD compute plane can be tested. A real quote +
real DCAP appraisal is Gate 2 on the guest, not here.
"""

from __future__ import annotations

import threading

import pytest
import torch

from pllo.experiments.real_tdx_attestation import (
    MeasurementPolicy,
    compute_service_runtime_hash,
    digest_config,
)
from pllo.experiments.real_tdx_training_client import (
    TrainingProtocolError,
    TrustedBoundaryClient,
)
from pllo.experiments.real_tdx_training_service import TrustedTrainingService, serve_http

DT = torch.float32     # the wire/safe-codec path is float32 (fp64 exactness was Gate 1)


def _serve():
    cfg = {"model_id": "synthetic", "lr": 1e-3, "beta1": 0.9, "beta2": 0.999,
           "eps": 1e-8, "weight_decay": 0.0, "dtype": "float32",
           "optimizer_mode": "trusted_adamw", "gradient_convention": "nout_dual"}
    cfg["config_digest"] = digest_config(dict(cfg))
    svc = TrustedTrainingService(run_id="rw", config=cfg, mode="cpu_contract",
                                 require_real_tdx=False, dtype=DT, seed=1)
    # force the authenticated-session wire path even in cpu_contract
    httpd = serve_http(svc, host="127.0.0.1", port=0, require_authenticated_session=True)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    return svc, cfg, httpd, port


def _establish(client, cfg, require_attested=False, policy=None, chain=lambda e: True):
    rh = compute_service_runtime_hash(cfg["config_digest"])
    return client.establish_attested_session(
        run_id="rw", config_digest=cfg["config_digest"], model_id="synthetic",
        gradient_convention="nout_dual", optimizer_mode="trusted_adamw",
        expected_runtime_hash=rh,
        policy=policy or MeasurementPolicy(allow_any_mr_td=True),
        chain_verifier=chain, require_attested=require_attested)


def test_attested_session_end_to_end_over_http():
    svc, cfg, httpd, port = _serve()
    try:
        client = TrustedBoundaryClient(base_url=f"http://127.0.0.1:{port}")
        res = _establish(client, cfg, require_attested=True)
        assert res["attestation_verified"] is True and client.attested
        client.init_session(run_id="rw", config_digest=cfg["config_digest"],
                            lora_manifest=[{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                            vocab_size=8, labels_by_step={0: [1, 2]})
        # masked logits -> the service recovers logits, computes CE, returns masked grad.
        # float32 on the wire (the safe codec excludes float64 by policy).
        N_head, _ = svc._head_masks()
        logits = torch.randn(2, 8, dtype=DT)
        ml = (logits @ N_head).to(torch.float32)
        resp = client.logits_loss(0, ml)
        assert "loss" in resp and resp["verification"]["trusted_loss_executed"]
    finally:
        httpd.shutdown()


def test_compute_endpoint_rejected_without_session():
    svc, cfg, httpd, port = _serve()
    try:
        client = TrustedBoundaryClient(base_url=f"http://127.0.0.1:{port}")
        # no handshake -> the server requires an authenticated session
        with pytest.raises(TrainingProtocolError):
            client.init_session(run_id="rw", config_digest=cfg["config_digest"],
                                lora_manifest=[{"layer_id": "L0", "d_in": 6, "d_out": 6, "rank": 3}],
                                vocab_size=8, labels_by_step={0: [1, 2]})
    finally:
        httpd.shutdown()


def test_client_fails_closed_when_attestation_unverified():
    svc, cfg, httpd, port = _serve()
    try:
        client = TrustedBoundaryClient(base_url=f"http://127.0.0.1:{port}")
        # chain_verifier returns False -> attestation_verified False -> refuse (fail-closed)
        with pytest.raises(TrainingProtocolError, match="attestation not verified"):
            _establish(client, cfg, require_attested=True, chain=lambda e: False)
    finally:
        httpd.shutdown()
