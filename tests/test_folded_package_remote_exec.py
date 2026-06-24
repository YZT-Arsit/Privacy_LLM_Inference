"""Remote (HTTP) package-backed prefill+decode over the TEE<->GPU protocol.

Builds a tiny dry-run folded package, starts the untrusted GPU worker as a real
stdlib HTTP server, and drives package-backed prefill + decode from a trusted
boundary (tiny MaskedQwenSession that owns the masks). Asserts:

* the remote package-backed tokens match the trusted in-process folded reference,
* no plaintext / mask secret crossed to the worker (audit on the recorded
  traffic + the server's own forbidden-field rejection),
* ``tee_used_on_gpu`` / ``worker_has_mask_secrets`` are False.

Run: python -m pytest tests/test_folded_package_remote_exec.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.folded_probe_common import (  # noqa: E402
    folded_exec_metadata, tiny_model)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402
from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker  # noqa: E402
from pllo.protocol.security_audit import (  # noqa: E402
    assert_no_gpu_visible_plaintext, assert_no_mask_secret_leak)
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest,
    ProtocolTrace)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


SEED = 2035
N_LAYERS = 4
SEQ_LEN = 8
N_NEW = 4


@pytest.fixture()
def pkg4(tmp_path):
    builder = _load("buildpkg_remote", "scripts/build_qwen7b_folded_package.py")
    pkg = tmp_path / "pkg4"
    assert _main(builder, ["prog", "--dry-run", "--output-dir", str(pkg),
                           "--num-layers", str(N_LAYERS), "--seed", str(SEED),
                           "--write-manifest", "true"]) == 0
    return pkg


def _build_session():
    model, mc = tiny_model()
    torch.manual_seed(7)
    ids = torch.randint(0, mc.vocab_size, (1, SEQ_LEN))
    cfg = MemoryOptimizedConfig(
        num_layers=N_LAYERS, batch_size=1, seq_len=SEQ_LEN, max_new_tokens=N_NEW,
        device="cpu", dtype="float32", folding_dtype="float32",
        folded_weight_device="cpu", seed=SEED)
    session = MaskedQwenSession(model, mc, cfg)
    return session, mc, ids


def _greedy(rec):
    return int(rec.argmax(-1).item())


def _ref_tokens(session, h_tilde):
    out = session.worker_prefill(h_tilde)
    tok = _greedy(session.recover(out["logits_tilde"][:, -1, :]))
    toks, kv, pos = [tok], out["kv"], SEQ_LEN
    for _ in range(N_NEW - 1):
        x = session.mask_token_embedding(torch.tensor([tok]))
        out = session.worker_decode(x, kv, pos)
        kv = out["kv"]
        tok = _greedy(session.recover(out["logits_tilde"][:, -1, :]))
        toks.append(tok)
        pos += 1
    return toks


def test_remote_package_backed_decode_matches_reference(pkg4) -> None:
    session, mc, ids = _build_session()
    h_tilde = session.mask_embeddings(ids)
    ref_tokens = _ref_tokens(session, h_tilde)

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg4), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port

    trace = ProtocolTrace(boundary_backend="process",
                          gpu_backend="qwen7b_folded_package",
                          max_new_tokens=N_NEW, tee_used_on_gpu=False)

    def _record(direction, method, msg):
        (trace.record_gpu_inbound if direction == "inbound"
         else trace.record_gpu_outbound)(msg)

    def _to_np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    def _recover_token(masked_logits):
        rec = session.recover(torch.as_tensor(np.asarray(masked_logits)).to(
            session.compute_device, session.fdtype))
        return _greedy(rec)

    try:
        meta = folded_exec_metadata(
            session, model_name="tiny", num_layers=N_LAYERS, seq_len=SEQ_LEN,
            max_new_tokens=N_NEW, vocab_size=int(mc.vocab_size))
        worker = RemoteGpuWorker(url, "qwen7b_folded_package", recorder=_record)
        assert worker.health()["tee_used_on_gpu"] is False
        init_resp = worker.init(BoundaryInitRequest(
            session_id="folded-0", hidden_size=int(mc.hidden_size),
            vocab_size=int(mc.vocab_size), num_layers=N_LAYERS, dtype="float32",
            gpu_backend="qwen7b_folded_package", folded_lm_head=None,
            public_metadata=meta))
        assert init_resp.tee_used_on_gpu is False

        pre = worker.prefill(MaskedPrefillRequest(
            session_id="folded-0", masked_embeddings=_to_np(h_tilde),
            positions=list(range(SEQ_LEN)), batch_size=1, seq_len=SEQ_LEN))
        tok = _recover_token(pre.masked_logits)
        pkg_tokens, pos = [tok], SEQ_LEN
        for step in range(N_NEW - 1):
            x = session.mask_token_embedding(torch.tensor([tok]))
            dec = worker.decode(MaskedDecodeRequest(
                session_id="folded-0", masked_embedding=_to_np(x),
                position=pos, step=step + 1))
            tok = _recover_token(dec.masked_logits)
            pkg_tokens.append(tok)
            pos += 1
        worker.close()
    finally:
        server.shutdown()

    assert pkg_tokens == ref_tokens
    assert len(pkg_tokens) == N_NEW

    # audit the exact recorded GPU traffic
    plaintext = assert_no_gpu_visible_plaintext(
        trace, raw_prompt="tiny prompt",
        input_ids=ids.detach().to("cpu").numpy(),
        generated_token_ids=np.asarray([pkg_tokens], dtype=np.int64),
        raise_on_fail=False)
    secrets = assert_no_mask_secret_leak(trace, None, raise_on_fail=False)
    assert plaintext == []
    assert secrets == []
    assert trace.tee_used_on_gpu is False


def test_server_rejects_forbidden_field(pkg4) -> None:
    """The untrusted worker must refuse a body carrying a forbidden plaintext/
    secret field (defense-in-depth, independent of the typed message check)."""
    import json
    import urllib.error
    import urllib.request

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg4), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d/init" % server.port
    try:
        body = json.dumps({"__msgtype__": "BoundaryInitRequest",
                           "session_id": "x", "seed": 2035}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"},
            method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 400
        detail = json.loads(ei.value.read().decode("utf-8"))
        assert detail["error"] == "forbidden_field"
        assert any("seed" in f for f in detail["fields"])
    finally:
        server.shutdown()
