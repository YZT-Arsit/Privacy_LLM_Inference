"""Tiny folded-package proof (no GPU, no checkpoint).

Builds a folded package for a small synthetic linear "tiny model" from public
weights + an internally-generated signed-permutation mask schedule, then proves:

* the package contains folded operators + a manifest, and NO mask secrets;
* loading the package and applying it to a masked input on a worker that has NO
  masks reproduces the in-process protected masked output exactly;
* recovering the package output equals the plaintext reference;
* manifest hash validation passes;
* tampering with a single shard makes verification fail.

Run: python -m pytest tests/test_folded_package_tiny.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pllo.deployment import (  # noqa: E402
    FoldedPackageWriter,
    build_manifest,
    compute_manifest_hash,
    fold_linear,
    forbidden_tensor_names,
    list_package_shards,
    load_manifest,
    load_shard,
    validate_manifest,
    verify_package,
    write_manifest,
)


def _signed_perm(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """An invertible orthogonal mask N (signed permutation) and its inverse."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    signs = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1)
    N = torch.zeros(n, n)
    N[torch.arange(n), perm] = signs            # row i -> column perm[i]
    return N, torch.linalg.inv(N)


def _tiny_model(dims, seed=7):
    """A chain of plain linear weights W_i of shape [dims[i], dims[i+1]]."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(dims[i], dims[i + 1], generator=g)
            for i in range(len(dims) - 1)]


def _build_package(tmp_path, dims, seed=7):
    """Trusted setup: fold W_i = N_i^{-1} W_i N_{i+1}, write shards + manifest.
    Returns (package_dir, masks N_0..N_L, plain weights)."""
    weights = _tiny_model(dims, seed)
    masks, mask_inv = [], []
    for i, d in enumerate(dims):
        N, Ninv = _signed_perm(d, seed + i)
        masks.append(N)
        mask_inv.append(Ninv)
    writer = FoldedPackageWriter(tmp_path / "pkg")
    for i, W in enumerate(weights):
        w_tilde = fold_linear(W, mask_inv[i], masks[i + 1])   # N_in^{-1} W N_out
        writer.add_shard(f"layer_{i}", {f"w{i}_tilde": w_tilde})
    manifest = build_manifest(
        package_type="base_model", model_name="tiny-linear",
        model_path_or_id=None, num_layers=len(weights), dtype="float32",
        nonlinear_backend="current", created_by="test",
        shard_index=writer.shard_index, hidden_size=dims[0],
        mask_schedule_id="tiny-sched-0", created_at="2026-06-24T00:00:00Z")
    write_manifest(manifest, tmp_path / "pkg")
    return tmp_path / "pkg", masks, weights


def test_tiny_package_builds_with_no_mask_secrets(tmp_path) -> None:
    pkg, masks, weights = _build_package(tmp_path, [8, 8, 8])
    manifest = load_manifest(pkg)
    assert manifest.num_shards == 2
    assert manifest.contains_mask_secrets is False
    ok, problems = validate_manifest(manifest)
    assert ok, problems
    # every stored tensor is a folded operator; no mask/secret names anywhere
    for shard in list_package_shards(pkg):
        names = list(load_shard(shard).keys())
        assert forbidden_tensor_names(names) == []
        assert all(n.endswith("_tilde") for n in names)


def test_worker_without_masks_matches_in_process_and_recovers(tmp_path) -> None:
    dims = [8, 8, 8]
    pkg, masks, weights = _build_package(tmp_path, dims, seed=11)
    x = torch.randn(1, dims[0])

    # plaintext reference (trusted)
    h = x
    for W in weights:
        h = h @ W
    plain_ref = h

    # in-process protected path (trusted has masks): x_tilde = x @ N_0; per layer
    # fold + matmul; output masked by N_L.
    _, mask_inv0 = masks[0], torch.linalg.inv(masks[0])
    x_tilde = x @ masks[0]
    ht = x_tilde
    for i, W in enumerate(weights):
        w_tilde = fold_linear(W, torch.linalg.inv(masks[i]), masks[i + 1])
        ht = ht @ w_tilde
    in_process_masked = ht

    # WORKER path: load only the package shards (NO masks), apply to x_tilde.
    manifest = load_manifest(pkg)
    hw = x_tilde
    for i, sh in enumerate(manifest.shard_index):
        tensors = load_shard(pkg / sh["path"])
        hw = hw @ tensors[f"w{i}_tilde"]
    worker_masked = hw

    # worker output == in-process protected output (folded weights only)
    assert torch.allclose(worker_masked, in_process_masked, atol=1e-5)
    # trusted recovery: y = y_tilde @ N_L^{-1} == plaintext reference
    recovered = worker_masked @ torch.linalg.inv(masks[-1])
    assert torch.allclose(recovered, plain_ref, atol=1e-4)


def test_package_hash_validation_passes(tmp_path) -> None:
    pkg, *_ = _build_package(tmp_path, [6, 6, 6])
    rep = verify_package(pkg, check_manifest=True, check_hashes=True,
                         check_no_secret_fields=True)
    assert rep["package_valid"] is True
    assert rep["missing_shards"] == []
    assert rep["hash_mismatches"] == []
    assert rep["forbidden_fields_found"] == []
    # manifest hash is stable + matches the embedded value
    assert rep["manifest_hash"] == compute_manifest_hash(load_manifest(pkg))


def test_tampering_one_shard_fails_verification(tmp_path) -> None:
    pkg, *_ = _build_package(tmp_path, [6, 6, 6])
    shard = list_package_shards(pkg)[0]
    with open(shard, "r+b") as fh:          # flip a byte in the middle
        fh.seek(shard.stat().st_size // 2)
        b = fh.read(1)
        fh.seek(shard.stat().st_size // 2)
        fh.write(bytes([b[0] ^ 0xFF]))
    rep = verify_package(pkg, check_hashes=True)
    assert rep["package_valid"] is False
    assert len(rep["hash_mismatches"]) == 1
    assert rep["hash_mismatches"][0]["path"] == shard.name


def test_writer_rejects_forbidden_tensor_names(tmp_path) -> None:
    writer = FoldedPackageWriter(tmp_path / "pkg")
    with pytest.raises(ValueError):
        writer.add_shard("bad", {"residual_perm": torch.zeros(2, 2)})
    with pytest.raises(ValueError):
        writer.add_shard("bad2", {"vocab_mask": torch.zeros(2, 2)})


def test_folded_package_worker_loads_without_masks(tmp_path) -> None:
    """The untrusted worker loads + verifies a folded package on init, reports it
    holds no mask secrets, and refuses package-backed prefill until the PUBLIC
    exec metadata is supplied (it never asks for a mask secret). The wired
    HTTP prefill/decode path is covered in test_folded_package_remote_exec.py."""
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest,
        MaskedPrefillRequest,
    )

    pkg, *_ = _build_package(tmp_path, [8, 8, 8])
    backend = Qwen7BFoldedPackageGpuBackend(folded_package_path=str(pkg),
                                            device="cpu", dtype="float32")
    resp = backend.init(BoundaryInitRequest(
        session_id="s", hidden_size=8, vocab_size=8, num_layers=2,
        dtype="float32", gpu_backend="qwen7b_folded_package"))
    assert resp.tee_used_on_gpu is False
    d = backend.describe()
    assert d["folded_package_loaded"] is True
    assert d["worker_has_mask_secrets"] is False
    assert d["package_valid"] is True
    assert d["manifest_hash"] == compute_manifest_hash(load_manifest(pkg))
    # package-backed prefill needs the PUBLIC exec metadata in init; without it
    # the worker raises a clear RuntimeError (never asks for a mask secret).
    with pytest.raises(RuntimeError) as ei:
        backend.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=[[0.0] * 8],
            positions=[0], batch_size=1, seq_len=1))
    assert "mask secrets are ever required" in str(ei.value).lower()
