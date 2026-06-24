"""TDX-lite remote package-backed decode (no full model, no local package).

Builds a tiny folded package AND a tiny trusted boundary embedding artifact from
the same model + seed, starts the untrusted GPU worker over HTTP, and drives
package-backed prefill+decode from a LITE boundary that holds ONLY the embedding
artifact (embed table + N_0 + vocab mask) -- not the full model, not the package.
Correctness is checked against expected token ids from a full in-process
reference. Asserts no plaintext / mask secret crosses, tee_used_on_gpu=False.

Run: python -m pytest tests/test_folded_package_remote_lite.py -q
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

from pllo.experiments.folded_probe_common import LiteBoundary, tiny_model  # noqa: E402
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402
from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker  # noqa: E402
from pllo.protocol.security_audit import (  # noqa: E402
    assert_no_gpu_visible_plaintext, assert_no_mask_secret_leak)
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest,
    ProtocolTrace)

SEED = 2035
N_LAYERS = 4
SEQ_LEN = 8
N_NEW = 4


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


@pytest.fixture()
def pkg_and_artifact(tmp_path):
    """A tiny folded package + a matching tiny boundary embedding artifact."""
    builder = _load("buildpkg_lite", "scripts/build_qwen7b_folded_package.py")
    pkg = tmp_path / "pkg4"
    assert _main(builder, ["prog", "--dry-run", "--output-dir", str(pkg),
                           "--num-layers", str(N_LAYERS), "--seed", str(SEED),
                           "--write-manifest", "true"]) == 0
    embuild = _load("buildemb_lite",
                    "scripts/build_qwen7b_embedding_artifact.py")
    art = tmp_path / "boundary_art"
    assert _main(embuild, ["prog", "--dry-run", "--output-dir", str(art),
                           "--seed", str(SEED)]) == 0
    assert (art / "boundary_meta.json").exists()
    return pkg, art


def _greedy(rec):
    return int(rec.argmax(-1).item())


def _reference_tokens(ids):
    """Full in-process folded reference tokens for the SAME tiny model + ids."""
    model, mc = tiny_model()
    cfg = MemoryOptimizedConfig(
        num_layers=N_LAYERS, batch_size=1, seq_len=SEQ_LEN, max_new_tokens=N_NEW,
        device="cpu", dtype="float32", folding_dtype="float32",
        folded_weight_device="cpu", seed=SEED)
    session = MaskedQwenSession(model, mc, cfg)
    h_tilde = session.mask_embeddings(ids)
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


def test_lite_boundary_remote_decode_matches_expected(pkg_and_artifact) -> None:
    pkg, art = pkg_and_artifact

    torch.manual_seed(11)
    ids = torch.randint(0, 256, (1, SEQ_LEN))
    expected = _reference_tokens(ids)

    # LITE boundary: only the small artifact (no full model, no local package).
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    assert boundary.meta["num_layers"] == N_LAYERS
    h_tilde = boundary.mask_embeddings(ids)
    meta = boundary.exec_metadata(seq_len=SEQ_LEN, max_new_tokens=N_NEW)
    assert "seed" not in meta              # mask seed must never be in exec meta

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
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
        rec = boundary.recover(torch.as_tensor(np.asarray(masked_logits)).to(
            boundary.compute_device, boundary.fdtype))
        return _greedy(rec)

    try:
        worker = RemoteGpuWorker(url, "qwen7b_folded_package", recorder=_record)
        init_resp = worker.init(BoundaryInitRequest(
            session_id="folded-0", hidden_size=int(meta["hidden_size"]),
            vocab_size=int(meta["vocab_size"]), num_layers=N_LAYERS,
            dtype="float32", gpu_backend="qwen7b_folded_package",
            folded_lm_head=None, public_metadata=meta))
        assert init_resp.tee_used_on_gpu is False

        pre = worker.prefill(MaskedPrefillRequest(
            session_id="folded-0", masked_embeddings=_to_np(h_tilde),
            positions=list(range(SEQ_LEN)), batch_size=1, seq_len=SEQ_LEN))
        tok = _recover_token(pre.masked_logits)
        pkg_tokens, pos = [tok], SEQ_LEN
        for step in range(N_NEW - 1):
            x = boundary.mask_token_embedding(torch.tensor([tok]))
            dec = worker.decode(MaskedDecodeRequest(
                session_id="folded-0", masked_embedding=_to_np(x),
                position=pos, step=step + 1))
            tok = _recover_token(dec.masked_logits)
            pkg_tokens.append(tok)
            pos += 1
        worker.close()
    finally:
        server.shutdown()

    # the lite (artifact-only) boundary reproduces the full reference exactly
    assert pkg_tokens == expected
    assert len(pkg_tokens) == N_NEW

    plaintext = assert_no_gpu_visible_plaintext(
        trace, raw_prompt="tdx prompt",
        input_ids=ids.detach().to("cpu").numpy(),
        generated_token_ids=np.asarray([pkg_tokens], dtype=np.int64),
        raise_on_fail=False)
    secrets = assert_no_mask_secret_leak(trace, None, raise_on_fail=False)
    assert plaintext == []
    assert secrets == []
    assert trace.tee_used_on_gpu is False


def test_demo_lite_end_to_end(pkg_and_artifact, tmp_path) -> None:
    """Drive the demo script's lite path end-to-end against a live HTTP worker."""
    pkg, art = pkg_and_artifact
    torch.manual_seed(11)
    ids = torch.randint(0, 256, (1, SEQ_LEN))
    expected = _reference_tokens(ids)
    ids_csv = ",".join(str(int(x)) for x in ids.reshape(-1).tolist())

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port
    demo = _load("demo_lite", "scripts/run_tee_gpu_protocol_demo.py")
    js = tmp_path / "tdx.json"
    try:
        rc = _main(demo, [
            "prog", "--mode", "boundary_client", "--gpu-worker-url", url,
            "--gpu-backend", "qwen7b_folded_package",
            "--embedding-path", str(art), "--skip-reference", "true",
            "--expected-token-ids", ",".join(str(t) for t in expected),
            "--input-ids", ids_csv, "--seq-len", str(SEQ_LEN),
            "--max-new-tokens", str(N_NEW), "--dtype", "float32",
            "--device", "cpu", "--audit", "true",
            "--output-json", str(js), "--output-md", str(tmp_path / "tdx.md")])
    finally:
        server.shutdown()

    assert rc == 0
    import json
    r = json.loads(js.read_text())
    assert r["stage"] == "qwen7b_folded_remote_package_decode"
    assert r["boundary_mode"] == "lite"
    assert r["gpu_worker_remote"] is True
    assert r["folded_package_loaded"] is True
    assert r["folded_package_valid"] is True
    assert r["package_backed_prefill"] is True
    assert r["package_backed_decode"] is True
    assert r["max_new_tokens"] == N_NEW
    assert r["expected_token_ids"] == expected
    assert r["package_token_ids"] == expected
    assert r["tokens_exact_match"] is True
    assert r["token_match_rate"] == 1.0
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["audit_passed"] is True
