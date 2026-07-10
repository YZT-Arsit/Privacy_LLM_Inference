"""Gate 1 — trusted training protocol CONTRACT tests (CPU, synthetic tensors).

SERVICE VALIDATION ONLY. These use synthetic protocol tensors on CPU float64 and
run the trusted service in ``cpu_contract`` mode (tee_type='cpu_contract_test',
guest_verified=False). They are NOT a real TDX experiment and NOT a real Qwen
experiment; that is Gate 2 (real guest) and Gate 3 (real model).

They prove: (1) the two boundaries exist and round-trip; (2) in-TDX-side unmask +
real AdamW + remask reproduces a PLAINTEXT AdamW trajectory exactly; (3) CE loss and
masked logit gradient are correct; (4) verification flags are set.
"""

from __future__ import annotations

import torch

from pllo.experiments.real_tdx_attestation import digest_config
from pllo.experiments.real_tdx_training_client import TrustedBoundaryClient
from pllo.experiments.real_tdx_training_service import TrustedTrainingService

DT = torch.float64
torch.manual_seed(0)


def _make_service(layers, vocab, labels_by_step, lr=1e-3):
    cfg = {"model_id": "synthetic", "lr": lr, "beta1": 0.9, "beta2": 0.999,
           "eps": 1e-8, "weight_decay": 0.0, "dtype": "float64"}
    cfg["config_digest"] = digest_config({k: v for k, v in cfg.items()})
    svc = TrustedTrainingService(run_id="run_gate1", config=cfg, mode="cpu_contract",
                                 require_real_tdx=False, dtype=DT, seed=1)
    manifest = [{"layer_id": lid, "d_in": d_in, "d_out": d_out, "rank": r}
                for (lid, d_in, d_out, r) in layers]
    client = TrustedBoundaryClient(service=svc)
    client.init_session(run_id="run_gate1", config_digest=cfg["config_digest"],
                        lora_manifest=manifest, vocab_size=vocab,
                        labels_by_step=labels_by_step)
    return svc, client, cfg


def _plain_adamw(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] = st.get("t", 0) + 1
    st["m"] = b1 * st.get("m", torch.zeros_like(p)) + (1 - b1) * g
    st["v"] = b2 * st.get("v", torch.zeros_like(p)) + (1 - b2) * g * g
    mh = st["m"] / (1 - b1 ** st["t"]); vh = st["v"] / (1 - b2 ** st["t"])
    return p - lr * mh / (vh.sqrt() + eps), st


def test_boundaries_exist_and_verification_flags():
    svc, client, _ = _make_service([("L0", 8, 8, 4)], vocab=16,
                                    labels_by_step={0: [0, 1, 2]})
    v = client.verification
    assert v["tee_type"] == "cpu_contract_test"      # never claims real TDX
    assert v["guest_verified"] is False
    m = torch.randn(3, 16, dtype=DT)
    N_head, M_head = svc._head_masks()
    r1 = client.logits_loss(0, m @ N_head)
    assert "masked_logit_gradient" in r1 and r1["verification"]["trusted_loss_executed"]


def test_ce_loss_and_logit_gradient_correct():
    svc, client, _ = _make_service([("L0", 8, 8, 4)], vocab=10,
                                    labels_by_step={0: [3, 7]})
    logits = torch.randn(2, 10, dtype=DT)
    N_head, M_head = svc._head_masks()
    resp = client.logits_loss(0, logits @ N_head)
    # CE matches plaintext CE on the true logits
    labels = torch.tensor([3, 7])
    ce = torch.nn.functional.cross_entropy(logits, labels)
    assert abs(resp["loss"] - ce.item()) < 1e-9
    # returned masked gradient recovers the true dlogits
    recovered = resp["masked_logit_gradient"] @ torch.linalg.inv(M_head)
    p = torch.softmax(logits, -1); dl = p.clone(); dl[torch.arange(2), labels] -= 1; dl /= 2
    assert (recovered - dl).abs().max() < 1e-9


def test_packed_update_matches_plaintext_adamw_trajectory():
    layers = [("q_proj", 12, 12, 4), ("gate_proj", 12, 20, 4)]
    svc, client, cfg = _make_service(layers, vocab=8, labels_by_step={}, lr=5e-3)
    g = torch.Generator().manual_seed(9)
    # plaintext reference starts from the service's initial plaintext adapters
    ref = {lid: {"A": svc.plaintext_adapter(lid)[0].clone(),
                 "B": svc.plaintext_adapter(lid)[1].clone(),
                 "sA": {}, "sB": {}} for lid, *_ in layers}
    # fixed synthetic activations per layer/step (shared masked + plaintext)
    steps = 6
    for step in range(steps):
        packed = {}
        for (lid, d_in, d_out, r) in layers:
            X = torch.randn(5, d_in, generator=g, dtype=DT)
            U = torch.randn(5, d_out, generator=g, dtype=DT)      # upstream grad
            N_in, R, N_out = svc._layer_masks(lid)
            A_t, B_t = client.masked_adapters[lid]
            # GPU computes masked grads in the MASKED domain only:
            X_t = X @ N_in
            U_t = U @ torch.linalg.inv(N_out).T
            P_t = X_t @ A_t
            gA_t = X_t.T @ (U_t @ B_t.T)
            gB_t = P_t.T @ U_t
            packed[lid] = {"gradA": gA_t, "gradB": gB_t}
            # plaintext reference grads on the same synthetic data
            A, B = ref[lid]["A"], ref[lid]["B"]
            gradA = X.T @ (U @ B.T)
            gradB = (X @ A).T @ U
            ref[lid]["A"], ref[lid]["sA"] = _plain_adamw(A, gradA, ref[lid]["sA"], cfg["lr"])
            ref[lid]["B"], ref[lid]["sB"] = _plain_adamw(B, gradB, ref[lid]["sB"], cfg["lr"])
        resp = client.packed_update(step, packed)
        assert resp["verification"]["trusted_adamw_executed"]
        assert resp["verification"]["plaintext_adapter_exported"] is False
        # after each step, service plaintext adapters == plaintext reference
        for lid, *_ in layers:
            A_svc, B_svc = svc.plaintext_adapter(lid)
            assert (A_svc - ref[lid]["A"]).abs().max() < 1e-9, f"{lid} A step {step}"
            assert (B_svc - ref[lid]["B"]).abs().max() < 1e-9, f"{lid} B step {step}"
            # DeltaW = AB matches
            dW_svc = A_svc @ B_svc; dW_ref = ref[lid]["A"] @ ref[lid]["B"]
            assert (dW_svc - dW_ref).abs().max() < 1e-9
            # Adam m/v match
            adam = svc.adam_state(lid)
            assert (adam["mA"] - ref[lid]["sA"]["m"]).abs().max() < 1e-9
            assert (adam["vA"] - ref[lid]["sA"]["v"]).abs().max() < 1e-9


def test_recovered_masked_adapters_match_plaintext():
    """The updated MASKED adapters returned to the GPU un-mask to the plaintext ones."""
    layers = [("L0", 10, 10, 4)]
    svc, client, cfg = _make_service(layers, vocab=8, labels_by_step={}, lr=1e-2)
    g = torch.Generator().manual_seed(3)
    for step in range(3):
        lid, d_in, d_out, r = layers[0]
        N_in, R, N_out = svc._layer_masks(lid)
        A_t, B_t = client.masked_adapters[lid]
        X = torch.randn(4, d_in, generator=g, dtype=DT)
        U = torch.randn(4, d_out, generator=g, dtype=DT)
        X_t = X @ N_in; U_t = U @ torch.linalg.inv(N_out).T
        gA_t = X_t.T @ (U_t @ B_t.T); gB_t = (X_t @ A_t).T @ U_t
        client.packed_update(step, {lid: {"gradA": gA_t, "gradB": gB_t}})
    # unmask returned adapters: A = N_in A_t R^-1 ; B = R B_t N_out^-1
    A_t, B_t = client.masked_adapters[lid]
    A_un = N_in @ A_t @ torch.linalg.inv(R)
    B_un = R @ B_t @ torch.linalg.inv(N_out)
    A_svc, B_svc = svc.plaintext_adapter(lid)
    assert (A_un - A_svc).abs().max() < 1e-9
    assert (B_un - B_svc).abs().max() < 1e-9


def test_http_transport_round_trip():
    """Gate-2 readiness: the SAME service+client work over HTTP (localhost)."""
    import threading

    from pllo.experiments.real_tdx_training_service import serve_http

    svc, _, cfg = _make_service([("L0", 8, 8, 4)], vocab=10,
                                labels_by_step={0: [3, 7]}, lr=1e-2)
    # rebuild a fresh service to serve over HTTP (init happens via the client)
    from pllo.experiments.real_tdx_training_service import TrustedTrainingService
    # HTTP-served service must operate in a wire dtype (safe codec rejects float64)
    svc2 = TrustedTrainingService(run_id="run_http", config=cfg, mode="cpu_contract",
                                  dtype=torch.float32, seed=2)
    httpd = serve_http(svc2, host="127.0.0.1", port=18097)
    th = threading.Thread(target=httpd.serve_forever, daemon=True); th.start()
    try:
        # HTTP uses the SAFE codec (no pickle); wire dtype is float32 (fp64 is not a
        # whitelisted wire dtype). Exact-fp64 equivalence is covered by in-process tests.
        client = TrustedBoundaryClient(base_url="http://127.0.0.1:18097")   # codec='safe'
        client.init_session(run_id="run_http", config_digest=cfg["config_digest"],
                            lora_manifest=[{"layer_id": "L0", "d_in": 8, "d_out": 8, "rank": 4}],
                            vocab_size=10, labels_by_step={0: [3, 7]})
        assert client.verification["tee_type"] == "cpu_contract_test"
        logits = torch.randn(2, 10, dtype=torch.float32)
        N_head, M_head = svc2._head_masks()
        resp = client.logits_loss(0, logits @ N_head)
        ce = torch.nn.functional.cross_entropy(logits, torch.tensor([3, 7]))
        assert abs(resp["loss"] - ce.item()) < 1e-4
        assert isinstance(resp["masked_logit_gradient"], torch.Tensor)
        # a packed_update round-trips too (float32 wire)
        N_in, R, N_out = svc2._layer_masks("L0")
        A_t, B_t = client.masked_adapters["L0"]
        X = torch.randn(4, 8, dtype=torch.float32); U = torch.randn(4, 8, dtype=torch.float32)
        X_t = X @ N_in; U_t = U @ torch.linalg.inv(N_out).T
        gA_t = X_t.T @ (U_t @ B_t.T); gB_t = (X_t @ A_t).T @ U_t
        up = client.packed_update(0, {"L0": {"gradA": gA_t, "gradB": gB_t}})
        assert up["verification"]["trusted_adamw_executed"]
    finally:
        httpd.shutdown()
