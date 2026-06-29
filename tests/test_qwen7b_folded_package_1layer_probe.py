"""Test the Qwen 1-layer folded-package probe script (dry-run, tiny Qwen2, CPU).

Asserts the probe builds + verifies + loads a 1-layer package in a worker with no
masks, reproduces the in-process masked output (allclose), and emits every
required report field.

Run: python -m pytest tests/test_qwen7b_folded_package_1layer_probe.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REPO_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED = (
    "folded_package_loaded", "folded_package_valid", "package_size_gb",
    "manifest_hash", "num_layers", "worker_has_mask_secrets", "tee_used_on_gpu",
    "max_abs_error", "mean_abs_error", "relative_l2_error", "allclose",
    "gpu_visible_plaintext_fields", "leaked_secret_fields",
)


def _run(tmp_path, *extra):
    spec = importlib.util.spec_from_file_location(
        "probe", REPO_ROOT / "scripts" / "run_qwen7b_folded_package_1layer_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    js = tmp_path / "probe.json"
    argv = ["prog", "--dry-run", "--seq-len", "8",
            "--folded-package-path", str(tmp_path / "pkg"),
            "--output-json", str(js), "--output-md", str(tmp_path / "probe.md"),
            *extra]
    old = sys.argv
    try:
        sys.argv = argv
        rc = mod.main()
    finally:
        sys.argv = old
    return rc, json.loads(js.read_text())


def test_probe_passes_and_reports_required_fields(tmp_path) -> None:
    rc, r = _run(tmp_path)
    assert rc == 0
    for k in _REQUIRED:
        assert k in r, f"missing {k}"
    assert r["folded_package_loaded"] is True
    assert r["folded_package_valid"] is True
    assert r["num_layers"] == 1
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["allclose"] is True
    assert r["max_abs_error"] == 0.0      # build+reference share the fold exactly
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert (tmp_path / "probe.md").exists()


def test_probe_consumes_existing_package(tmp_path) -> None:
    rc1, _ = _run(tmp_path)               # builds the package
    assert rc1 == 0
    assert (tmp_path / "pkg" / "manifest.json").exists()
    rc2, r2 = _run(tmp_path)              # consumes the existing package
    assert rc2 == 0
    assert r2["allclose"] is True
    assert r2["folded_package_valid"] is True


def test_probe_consumes_package_built_by_build_script(tmp_path) -> None:
    """Cross-script flow (the real H800 flow): the package is built by
    build_qwen7b_folded_package.py and the probe CONSUMES it. The two scripts'
    dry-run tiny models must be identical, else layer-0 weights differ and the
    comparison fails -- this guards that parity."""
    import importlib.util
    import json
    import sys
    pkg = tmp_path / "built_pkg"

    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    builder = _load("buildpkg", "scripts/build_qwen7b_folded_package.py")
    old = sys.argv
    try:
        # mask-only build so the probe's mask-only reference matches bit-exactly
        # (Linear-boundary pad is the build default and perturbs the operand view)
        sys.argv = ["prog", "--dry-run", "--output-dir", str(pkg),
                    "--num-layers", "1", "--seed", "2035",
                    "--no-linear-boundary-pad",
                    "--write-manifest", "true"]
        assert builder.main() == 0
    finally:
        sys.argv = old
    assert (pkg / "manifest.json").exists()

    probe = _load("probe", "scripts/run_qwen7b_folded_package_1layer_probe.py")
    js = tmp_path / "probe.json"
    try:
        sys.argv = ["prog", "--dry-run", "--seq-len", "8",
                    "--folded-package-path", str(pkg),
                    "--output-json", str(js),
                    "--output-md", str(tmp_path / "p.md")]
        assert probe.main() == 0
    finally:
        sys.argv = old
    r = json.loads(js.read_text())
    assert r["allclose"] is True
    assert r["max_abs_error"] == 0.0      # identical model + masks across scripts
    assert r["worker_has_mask_secrets"] is False
    assert r["leaked_secret_fields"] == []
