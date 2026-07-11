"""Tests for the private-base packager / plaintext-absence scanner / runtime guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.deployment.private_base_package import (  # noqa: E402
    PrivateBasePackageError,
    PrivateBasePackager,
    assert_private_base_or_abort,
    scan_package_for_plaintext,
)


def _pkg(tmp):
    p = PrivateBasePackager(out_dir=tmp, model_id="Qwen2.5-0.5B",
                            config={"hidden": 896, "layers": 24})
    p.add_transformed_tensor("model.layers.0.self_attn.wq_tilde", torch.randn(4, 4))
    p.add_transformed_tensor("model.embed_tokens_tilde", torch.randn(8, 4))
    p.add_transformed_tensor("lm_head_tilde", torch.randn(8, 4))
    p.record_mask_domain("feature", ["N_in", "N_out", "R", "S"])
    p.record_mask_domain("vocab", ["D_vocab", "Pi_vocab"])
    return p


def test_packager_writes_transformed_only_and_manifest(tmp_path):
    m = _pkg(tmp_path).finalize()
    assert m["contains_plaintext_base_weights"] is False
    assert m["contains_mask_secrets"] is False
    assert m["private_base_required"] is True
    assert m["artifact_count"] == 3
    assert "feature" in m["mask_domains_recorded"]


def test_packager_refuses_untransformed_name(tmp_path):
    p = PrivateBasePackager(out_dir=tmp_path, model_id="x")
    with pytest.raises(PrivateBasePackageError):
        p.add_transformed_tensor("model.layers.0.q_proj.weight", torch.randn(2, 2))


def test_packager_refuses_mask_secret(tmp_path):
    p = PrivateBasePackager(out_dir=tmp_path, model_id="x")
    with pytest.raises(PrivateBasePackageError):
        p.add_transformed_tensor("n_res_tilde", torch.randn(2, 2))     # _inv/n_res
    with pytest.raises(PrivateBasePackageError):
        p.add_transformed_tensor("pi_vocab_tilde", torch.randn(2, 2))


def test_packager_refuses_plaintext_tensor_hint(tmp_path):
    p = PrivateBasePackager(out_dir=tmp_path, model_id="x")
    with pytest.raises(PrivateBasePackageError):
        # even with a _tilde suffix, an embed_tokens.weight substring is refused
        p.add_transformed_tensor("model.embed_tokens.weight_tilde", torch.randn(2, 2))


def test_scan_clean_package(tmp_path):
    _pkg(tmp_path).finalize()
    rep = scan_package_for_plaintext(tmp_path)
    assert rep["clean"] is True, rep["findings"]
    assert rep["finding_count"] == 0


def test_scan_flags_planted_plaintext_shard(tmp_path):
    _pkg(tmp_path).finalize()
    # simulate a leaked plaintext checkpoint shard
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00\x01")
    rep = scan_package_for_plaintext(tmp_path)
    assert rep["clean"] is False
    kinds = {f["kind"] for f in rep["findings"]}
    assert "plaintext_checkpoint" in kinds or "untransformed_tensor" in kinds


def test_scan_flags_serialized_mask(tmp_path):
    _pkg(tmp_path).finalize()
    (tmp_path / "n_res_inv.pt").write_bytes(b"\x00")
    rep = scan_package_for_plaintext(tmp_path)
    assert rep["clean"] is False
    assert any(f["kind"] == "mask_secret" for f in rep["findings"])


def test_runtime_guard_aborts_on_plaintext_path(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    with pytest.raises(PrivateBasePackageError):
        assert_private_base_or_abort(gpu_process_paths=[str(tmp_path)])


def test_runtime_guard_passes_on_clean_package(tmp_path):
    _pkg(tmp_path).finalize()
    assert_private_base_or_abort(gpu_process_paths=[str(tmp_path)])   # no raise


def test_runtime_guard_can_be_disabled(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    assert_private_base_or_abort(gpu_process_paths=[str(tmp_path)], require=False)
