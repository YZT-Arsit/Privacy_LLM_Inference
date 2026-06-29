"""Executable folded-package probes: multi-layer prefill, one-step logits, decode.

Builds a dry-run folded package with the build script, then drives the three
execution probes (prefill / one-step-logits / short-decode) end-to-end on a tiny
Qwen2 (CPU). Asserts the package-backed compute matches the in-process folded
reference and that no mask secrets reach the worker.

Run: python -m pytest tests/test_folded_package_prefill_exec.py -q
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
def pkg4(tmp_path):
    """A dry-run 4-layer folded package built by the build script."""
    builder = _load("buildpkg", "scripts/build_qwen7b_folded_package.py")
    pkg = tmp_path / "pkg4"
    # mask-only build so the probe's mask-only reference matches bit-exactly
    # (the Linear-boundary pad is the build default; it perturbs the matmul
    # operand view -> output is mathematically identical but not bit-exact)
    assert _main(builder, ["prog", "--dry-run", "--output-dir", str(pkg),
                           "--num-layers", "4", "--seed", "2035",
                           "--no-linear-boundary-pad",
                           "--write-manifest", "true"]) == 0
    assert (pkg / "manifest.json").exists()
    return pkg


@pytest.mark.parametrize("k", [1, 4])
def test_package_backed_prefill_matches_reference(pkg4, tmp_path, k) -> None:
    probe = _load("prefill", "scripts/run_qwen7b_folded_package_prefill_probe.py")
    js = tmp_path / ("prefill_%d.json" % k)
    rc = _main(probe, ["prog", "--dry-run", "--seq-len", "8",
                       "--num-exec-layers", str(k),
                       "--folded-package-path", str(pkg4),
                       "--output-json", str(js),
                       "--output-md", str(tmp_path / ("p%d.md" % k))])
    assert rc == 0
    r = json.loads(js.read_text())
    assert r["stage"] == "qwen7b_folded_package_prefill_probe"
    assert r["num_exec_layers"] == k
    assert r["package_backed_prefill"] is True
    assert r["folded_package_loaded"] is True
    assert r["folded_package_valid"] is True
    assert r["allclose"] is True
    assert r["max_abs_error"] == 0.0          # identical model + masks
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["num_package_layers"] == 4


def test_package_backed_onestep_logits(pkg4, tmp_path) -> None:
    probe = _load("onestep",
                  "scripts/run_qwen7b_folded_package_onestep_logits_probe.py")
    js = tmp_path / "onestep.json"
    rc = _main(probe, ["prog", "--dry-run", "--seq-len", "8", "--topk", "3",
                       "--folded-package-path", str(pkg4),
                       "--output-json", str(js),
                       "--output-md", str(tmp_path / "o.md")])
    assert rc == 0
    r = json.loads(js.read_text())
    for f in ("logits_max_abs_error", "logits_mean_abs_error",
              "logits_relative_l2_error", "top1_match", "topk_overlap",
              "next_token_match"):
        assert f in r
    assert r["top1_match"] is True
    assert r["next_token_match"] is True
    assert r["topk_overlap"] == 1.0
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["package_backed_head"] is True


def test_package_backed_short_decode(pkg4, tmp_path) -> None:
    probe = _load("decode", "scripts/run_qwen7b_folded_package_decode_probe.py")
    js = tmp_path / "decode.json"
    rc = _main(probe, ["prog", "--dry-run", "--seq-len", "8",
                       "--max-new-tokens", "4",
                       "--folded-package-path", str(pkg4),
                       "--output-json", str(js),
                       "--output-md", str(tmp_path / "d.md")])
    assert rc == 0
    r = json.loads(js.read_text())
    assert r["package_backed_decode"] is True
    assert len(r["package_token_ids"]) == 4
    assert r["tokens_exact_match"] is True
    assert r["token_match_rate"] == 1.0
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["leaked_secret_fields"] == []
