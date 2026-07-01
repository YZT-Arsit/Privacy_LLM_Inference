"""T2 correctness: batched folded decode == per-stream sequential decode.

The latency win of batching B independent prompt streams through the GPU worker in
ONE round trip per step is only admissible if it changes NO result: stream i in a
batched run must produce the EXACT tokens it would produce decoded alone. This
proves that, bit for bit, on a tiny real folded package + boundary artifact (same
fixtures as tests/test_folded_package_remote_lite.py).

Batching is exact when every stream in the batch is at the SAME position (equal
prompt length), because the folded forward is a plain tensor op over the batch
dim and RoPE/causal masking key off the single shared ``position``. This test
uses equal-length prompts; variable lengths (left-pad + per-stream positions) are
a separate extension and are NOT claimed here.

Run: python -m pytest tests/test_folded_batched_decode.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.folded_probe_common import LiteBoundary  # noqa: E402
from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker  # noqa: E402
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)

SEED, N_LAYERS, SEQ_LEN, N_NEW, BATCH = 20240722, 4, 8, 5, 3


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
    builder = _load("buildpkg_b", "scripts/build_qwen7b_folded_package.py")
    pkg = tmp_path / "pkg"
    assert _main(builder, ["prog", "--dry-run", "--output-dir", str(pkg),
                           "--num-layers", str(N_LAYERS), "--seed", str(SEED),
                           "--write-manifest", "true"]) == 0
    embuild = _load("buildemb_b", "scripts/build_qwen7b_embedding_artifact.py")
    art = tmp_path / "art"
    assert _main(embuild, ["prog", "--dry-run", "--output-dir", str(art),
                           "--seed", str(SEED)]) == 0
    return pkg, art


def _to_np(t):
    return np.asarray(t.detach().to("cpu").float().numpy())


def _greedy(rec):
    return rec.argmax(-1).reshape(-1).tolist()


def _run(pkg, art, ids_batch, batched):
    """Decode ``ids_batch`` [B, SEQ_LEN] either as one batched B-stream run
    (batched=True) or as B independent single-stream runs. Returns [[tokens]*B]."""
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port
    meta = boundary.exec_metadata(seq_len=SEQ_LEN, max_new_tokens=N_NEW)

    def _decode_group(ids):                     # ids: [G, SEQ_LEN]
        g = ids.shape[0]
        worker = RemoteGpuWorker(url, "qwen7b_folded_package")
        worker.init(BoundaryInitRequest(
            session_id="s", hidden_size=int(meta["hidden_size"]),
            vocab_size=int(meta["vocab_size"]), num_layers=N_LAYERS,
            dtype="float32", gpu_backend="qwen7b_folded_package",
            folded_lm_head=None, public_metadata=meta))
        h = boundary.mask_embeddings(ids)                       # [G, T, H]
        pre = worker.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=_to_np(h),
            positions=list(range(SEQ_LEN)), batch_size=g, seq_len=SEQ_LEN))
        toks = boundary.recover(torch.as_tensor(np.asarray(pre.masked_logits)).to(
            boundary.compute_device, boundary.fdtype))
        cur = _greedy(toks)
        cols = [[t] for t in cur]
        pos = SEQ_LEN
        for step in range(N_NEW - 1):
            x = boundary.mask_token_embedding(torch.tensor(cur))  # [G,1,H]
            dec = worker.decode(MaskedDecodeRequest(
                session_id="s", masked_embedding=_to_np(x), position=pos,
                step=step + 1))
            rec = boundary.recover(torch.as_tensor(np.asarray(
                dec.masked_logits)).to(boundary.compute_device, boundary.fdtype))
            cur = _greedy(rec)
            for i, t in enumerate(cur):
                cols[i].append(t)
            pos += 1
        worker.close()
        return cols

    try:
        if batched:
            return _decode_group(ids_batch)
        return [_decode_group(ids_batch[i:i + 1])[0]
                for i in range(ids_batch.shape[0])]
    finally:
        server.shutdown()


def test_batched_decode_matches_sequential_bit_identical(pkg_and_artifact):
    pkg, art = pkg_and_artifact
    torch.manual_seed(7)
    ids = torch.randint(0, 256, (BATCH, SEQ_LEN))
    sequential = _run(pkg, art, ids, batched=False)
    batched = _run(pkg, art, ids, batched=True)
    assert batched == sequential, (batched, sequential)
    assert len(batched) == BATCH and all(len(c) == N_NEW for c in batched)


def _server(pkg):
    s = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    s.start_background()
    return s, "http://127.0.0.1:%d" % s.port


def test_batched_driver_mixed_lengths_and_eos_match_sequential(pkg_and_artifact):
    from pllo.experiments.folded_batched_decode import batched_greedy_decode
    pkg, art = pkg_and_artifact
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    torch.manual_seed(9)
    # MIXED lengths -> exercises length-bucketing (6,8,6,8,7)
    lens = [6, 8, 6, 8, 7]
    id_lists = [torch.randint(0, 256, (L,)).tolist() for L in lens]
    MNT = 6

    srv, url = _server(pkg)
    try:
        w = RemoteGpuWorker(url, "qwen7b_folded_package")
        # full (no EOS) batched vs per-prompt sequential
        batched = batched_greedy_decode(
            boundary, w, id_lists, max_new_tokens=MNT, num_layers=N_LAYERS)
        seq = [batched_greedy_decode(
            boundary, w, [ids], max_new_tokens=MNT, num_layers=N_LAYERS)[0]
            for ids in id_lists]
        assert batched == seq, (batched, seq)
        assert [len(t) for t in batched] == [MNT] * len(id_lists)

        # EOS truncation: use a token that appears mid-sequence in stream 0
        eos_tok = batched[0][2]
        eos_batched = batched_greedy_decode(
            boundary, w, id_lists, max_new_tokens=MNT, eos_ids=[eos_tok],
            num_layers=N_LAYERS)
        eos_seq = [batched_greedy_decode(
            boundary, w, [ids], max_new_tokens=MNT, eos_ids=[eos_tok],
            num_layers=N_LAYERS)[0] for ids in id_lists]
        assert eos_batched == eos_seq, (eos_batched, eos_seq)
        # stream 0 stopped at its first eos occurrence
        assert eos_batched[0][-1] == eos_tok
        assert eos_tok not in eos_batched[0][:-1]
        w.close()
    finally:
        srv.shutdown()
