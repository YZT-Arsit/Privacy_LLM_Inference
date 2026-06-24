"""Tests for inspect_folded_package.py + run_qwen7b_folded_package_load_probe.py.

Builds a tiny folded package directly (no model needed) and checks the inspector
reports per-shard sizes/dtypes regardless of manifest schema, and the load probe
loads + verifies the package (no /prefill /decode) with the required fields.

Run: python -m pytest tests/test_folded_package_inspect_and_load.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment import (  # noqa: E402
    FoldedPackageWriter,
    build_manifest,
    write_manifest,
)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_pkg(pkg_dir, n_layers=2):
    w = FoldedPackageWriter(pkg_dir)
    for i in range(n_layers):
        w.add_shard(f"layer_{i:03d}", {
            "wq_tilde": torch.zeros(8, 8), "wo_tilde": torch.zeros(8, 8),
            "wgate_tilde": torch.zeros(8, 16), "wup_tilde": torch.zeros(8, 16),
            "wdown_tilde": torch.zeros(16, 8)})
    w.add_shard("head", {"w_lm_tilde": torch.zeros(8, 32)})
    manifest = build_manifest(
        package_type="base_model", model_name="tiny", model_path_or_id=None,
        num_layers=n_layers, dtype="bfloat16", nonlinear_backend="current",
        created_by="test", shard_index=w.shard_index, hidden_size=8,
        vocab_size=32, mask_schedule_id="t-seed2035-n%d" % n_layers,
        created_at="2026-06-24T00:00:00Z")
    write_manifest(manifest, pkg_dir)
    return pkg_dir


def _run(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


def test_inspector_scans_filesystem_regardless_of_schema(tmp_path) -> None:
    pkg = _build_pkg(tmp_path / "pkg", n_layers=2)
    mod = _load("inspect", "scripts/inspect_folded_package.py")
    js = tmp_path / "insp.json"
    rc = _run(mod, ["prog", "--package-dir", str(pkg), "--output-json", str(js)])
    assert rc == 0
    r = json.loads(js.read_text())
    assert r["num_shards"] == 3                       # 2 layers + head, from FS
    assert len(r["per_shard_size_gb"]) == 3
    assert r["total_size_gb"] > 0
    assert "F32" in r["store_dtypes"]                 # float32 storage exposed
    # schema-tolerant manifest cross-check: our schema uses shard_index
    assert r["manifest"]["present"] is True
    assert r["manifest"]["declared_shard_key"] == "shard_index"
    assert r["manifest"]["declared_shard_count"] == 3


def test_load_probe_reports_required_fields(tmp_path) -> None:
    pkg = _build_pkg(tmp_path / "pkg", n_layers=2)
    mod = _load("loadprobe", "scripts/run_qwen7b_folded_package_load_probe.py")
    js = tmp_path / "load.json"
    rc = _run(mod, ["prog", "--folded-package-path", str(pkg),
                    "--device", "cpu", "--dtype", "float32",
                    "--output-json", str(js)])
    assert rc == 0
    r = json.loads(js.read_text())
    for k in ("folded_package_loaded", "folded_package_valid", "load_time_s",
              "package_size_gb", "num_layers", "num_shards", "manifest_hash",
              "worker_has_mask_secrets", "tee_used_on_gpu",
              "gpu_visible_plaintext_fields", "leaked_secret_fields",
              "per_shard_size_gb", "total_size_gb"):
        assert k in r, f"missing {k}"
    assert r["folded_package_loaded"] is True
    assert r["folded_package_valid"] is True
    assert r["num_layers"] == 2
    assert r["num_shards"] == 3
    assert len(r["per_shard_size_gb"]) == 3
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["load_time_s"] >= 0.0
