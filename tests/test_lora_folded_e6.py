"""E6: private folded-LoRA inference -- build/verify, local correctness, remote
HTTP decode, and no-LoRA backward compatibility. Tiny CPU only (no H800/TDX/CUDA).

Run: python -m pytest tests/test_lora_folded_e6.py -q
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment.lora_folded_package import (  # noqa: E402
    ALL_TARGET_MODULES,
    DEFAULT_TARGET_MODULES,
    apply_lora_to_model,
    fold_lora_for_layer,
    lora_scaling,
    merge_folded_lora,
    synthetic_lora_adapter,
    verify_lora_folded_package,
)
from pllo.experiments.folded_probe_common import LiteBoundary, tiny_model  # noqa: E402
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402
from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker  # noqa: E402
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)

SEED = 2035
N = 4
SL = 8
NN = 4
RANK = 8
ALPHA = 16.0


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


def _cfg():
    return MemoryOptimizedConfig(
        num_layers=N, batch_size=1, seq_len=SL, max_new_tokens=NN, device="cpu",
        dtype="float32", folding_dtype="float32", folded_weight_device="cpu",
        seed=SEED)


def test_folding_matches_reference_all_modules() -> None:
    """folded base + merged folded-LoRA == folded(base + raw LoRA), all modules."""
    torch.manual_seed(0)
    model, mc = tiny_model()
    model_lora = copy.deepcopy(model)
    tm = list(ALL_TARGET_MODULES)
    scaling = lora_scaling(ALPHA, RANK)
    lora = synthetic_lora_adapter(mc, N, tm, RANK, seed=SEED)
    sess = MaskedQwenSession(model, mc, _cfg())
    apply_lora_to_model(model_lora, lora, tm, scaling)
    sess_lora = MaskedQwenSession(model_lora, mc, _cfg())

    max_err = 0.0
    for ell in range(N):
        base = {k: v.clone()
                for k, v in sess.export_folded_layer_tensors(ell).items()}
        fl = fold_lora_for_layer(sess, ell, lora[ell], scaling=scaling,
                                 rank=RANK, rank_seed=SEED, target_modules=tm)
        merged = merge_folded_lora(base, fl, tm)
        ref = sess_lora.export_folded_layer_tensors(ell)
        for k in ref:
            max_err = max(max_err, (merged[k] - ref[k]).abs().max().item())
    assert max_err < 1e-4


@pytest.fixture()
def lora_pkgs(tmp_path):
    """Tiny base + folded-LoRA packages (same seed) + boundary artifact."""
    base = tmp_path / "base"
    bp = _load("bp", "scripts/build_qwen7b_folded_package.py")
    assert _main(bp, ["x", "--dry-run", "--output-dir", str(base),
                      "--num-layers", str(N), "--seed", str(SEED),
                      "--write-manifest", "true"]) == 0
    lpkg = tmp_path / "lora"
    lb = _load("lb", "scripts/build_qwen7b_lora_folded_package.py")
    assert _main(lb, ["x", "--dry-run", "--output-dir", str(lpkg),
                      "--base-folded-package-path", str(base),
                      "--target-modules", ",".join(DEFAULT_TARGET_MODULES),
                      "--rank", str(RANK), "--alpha", str(ALPHA),
                      "--seed", str(SEED)]) == 0
    art = tmp_path / "art"
    eb = _load("eb", "scripts/build_qwen7b_embedding_artifact.py")
    assert _main(eb, ["x", "--dry-run", "--output-dir", str(art),
                      "--seed", str(SEED)]) == 0
    return base, lpkg, art


def test_build_and_verify(lora_pkgs) -> None:
    base, lpkg, _ = lora_pkgs
    rep = verify_lora_folded_package(str(lpkg))
    assert rep["lora_package_valid"] is True
    assert rep["forbidden_fields_found"] == []
    assert rep["raw_lora_tensor_names_found"] == []
    assert rep["contains_raw_lora"] is False
    assert rep["contains_optimizer_state"] is False
    assert rep["contains_training_data"] is False
    assert rep["contains_mask_secrets"] is False
    assert rep["target_modules_missing_coverage"] == []
    assert rep["rank"] == RANK
    # base manifest compatibility check
    from pllo.deployment import compute_manifest_hash, load_manifest
    bh = compute_manifest_hash(load_manifest(str(base)))
    rep2 = verify_lora_folded_package(str(lpkg), base_manifest_hash=bh)
    assert rep2["lora_package_valid"] is True
    assert rep2["base_manifest_match"] is True
    rep3 = verify_lora_folded_package(str(lpkg), base_manifest_hash="WRONG")
    assert rep3["lora_package_valid"] is False


def test_local_probe_script(tmp_path) -> None:
    probe = _load("lp", "scripts/run_qwen7b_lora_folded_local_probe.py")
    js = tmp_path / "lp.json"
    rc = _main(probe, ["x", "--dry-run", "--target-modules",
                       ",".join(ALL_TARGET_MODULES), "--rank", str(RANK),
                       "--alpha", str(ALPHA), "--seq-len", str(SL),
                       "--max-new-tokens", str(NN), "--device", "cpu",
                       "--output-json", str(js)])
    assert rc == 0
    r = json.loads(js.read_text())
    assert r["stage"] == "qwen7b_lora_folded_local_probe"
    assert r["lora_enabled"] is True
    assert r["allclose"] is True
    assert r["top1_match"] is True
    assert r["next_token_match"] is True
    assert r["topk_overlap"] == 1.0
    assert r["tokens_exact_match"] is True
    assert r["token_match_rate"] == 1.0
    assert r["worker_has_raw_lora"] is False
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []


def _expected_lora_tokens(ids, tm):
    """Trusted base + raw synthetic LoRA reference tokens for fixed ids."""
    model, mc = tiny_model()
    model_l = copy.deepcopy(model)
    scaling = lora_scaling(ALPHA, RANK)
    lora = synthetic_lora_adapter(mc, N, tm, RANK, seed=SEED)
    apply_lora_to_model(model_l, lora, tm, scaling)
    sb = MaskedQwenSession(model, mc, _cfg())
    sl = MaskedQwenSession(model_l, mc, _cfg())
    h = sb.mask_embeddings(ids)
    out = sl.worker_prefill(h)
    tok = int(sb.recover(out["logits_tilde"][:, -1, :]).argmax(-1))
    toks, kv, pos = [tok], out["kv"], SL
    for _ in range(NN - 1):
        x = sb.mask_token_embedding(torch.tensor([tok]))
        out = sl.worker_decode(x, kv, pos)
        kv = out["kv"]
        tok = int(sb.recover(out["logits_tilde"][:, -1, :]).argmax(-1))
        toks.append(tok)
        pos += 1
    return toks


def test_remote_lora_decode_matches_reference(lora_pkgs) -> None:
    base, lpkg, art = lora_pkgs
    tm = list(DEFAULT_TARGET_MODULES)
    torch.manual_seed(123)
    ids = torch.randint(0, 256, (1, SL))
    expected = _expected_lora_tokens(ids, tm)

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(base),
                        "folded_lora_package_path": str(lpkg),
                        "device": "cpu", "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    meta = boundary.exec_metadata(seq_len=SL, max_new_tokens=NN)

    def _to_np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    def _tok(ml):
        return int(boundary.recover(torch.as_tensor(np.asarray(ml))).argmax(-1))

    try:
        worker = RemoteGpuWorker(url, "qwen7b_folded_package")
        init_resp = worker.init(BoundaryInitRequest(
            session_id="s", hidden_size=int(meta["hidden_size"]),
            vocab_size=int(meta["vocab_size"]), num_layers=N, dtype="float32",
            gpu_backend="qwen7b_folded_package", folded_lm_head=None,
            public_metadata=meta))
        notes = json.loads(init_resp.notes)
        assert notes["lora_enabled"] is True
        assert notes["folded_lora_loaded"] is True
        assert notes["folded_lora_valid"] is True
        assert notes["worker_has_raw_lora"] is False
        assert notes["lora_rank"] == RANK

        h = boundary.mask_embeddings(ids)
        pre = worker.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=_to_np(h),
            positions=list(range(SL)), batch_size=1, seq_len=SL))
        tok = _tok(pre.masked_logits)
        got, pos = [tok], SL
        for _ in range(NN - 1):
            x = boundary.mask_token_embedding(torch.tensor([tok]))
            dec = worker.decode(MaskedDecodeRequest(
                session_id="s", masked_embedding=_to_np(x), position=pos, step=1))
            tok = _tok(dec.masked_logits)
            got.append(tok)
            pos += 1
        worker.close()
    finally:
        server.shutdown()
    assert got == expected


def test_no_lora_path_unchanged(lora_pkgs) -> None:
    """A worker started WITHOUT a LoRA package reports no-LoRA and still decodes
    (backward compatibility of the base folded path)."""
    base, _lpkg, art = lora_pkgs
    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(base), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    meta = boundary.exec_metadata(seq_len=SL, max_new_tokens=1)
    try:
        worker = RemoteGpuWorker(url, "qwen7b_folded_package")
        resp = worker.init(BoundaryInitRequest(
            session_id="s", hidden_size=int(meta["hidden_size"]),
            vocab_size=int(meta["vocab_size"]), num_layers=N, dtype="float32",
            gpu_backend="qwen7b_folded_package", folded_lm_head=None,
            public_metadata=meta))
        notes = json.loads(resp.notes)
        assert notes["lora_enabled"] is False
        assert notes["folded_lora_loaded"] is False
        ids = torch.zeros((1, SL), dtype=torch.long)
        h = boundary.mask_embeddings(ids)
        pre = worker.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=np.asarray(
                h.detach().to("cpu").float().numpy()),
            positions=list(range(SL)), batch_size=1, seq_len=SL))
        assert np.asarray(pre.masked_logits).shape[-1] == int(meta["vocab_size"])
        worker.close()
    finally:
        server.shutdown()
