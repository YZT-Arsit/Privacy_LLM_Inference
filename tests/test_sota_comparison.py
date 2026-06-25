"""Fixture tests for the SOTA comparison table renderer.

Run: python -m pytest tests/test_sota_comparison.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

COMMITTED_YAML = REPO_ROOT / "baselines" / "privacy_inference_methods.yaml"

REQUIRED_FIELDS = [
    "method", "paper", "year", "protects_input", "protects_logits",
    "protects_kv", "protects_lora", "requires_gpu_tee", "requires_mpc_fhe",
    "tee_holds_full_model", "runs_real_7b", "real_attestation",
    "reported_latency", "source_type", "notes",
]


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


def test_render_committed_yaml(tmp_path) -> None:
    mod = _load("sota", "scripts/render_sota_comparison_tables.py")
    js = tmp_path / "sota.json"
    md = tmp_path / "sota.md"
    csv = tmp_path / "sota.csv"
    tex = tmp_path / "sota.tex"
    rc = _main(mod, ["x", "--methods-yaml", str(COMMITTED_YAML),
                     "--output-json", str(js), "--output-md", str(md),
                     "--output-csv", str(csv), "--output-tex", str(tex)])
    assert rc == 0

    report = json.loads(js.read_text())
    assert report["stage"] == "sota_comparison"
    assert report["row_count"] > 0
    assert len(report["rows"]) == report["row_count"]

    methods = [r["method"] for r in report["rows"]]
    assert any(r["source_type"] == "ours" for r in report["rows"])
    assert any("Ours" in m for m in methods)

    # every row has exactly the required fields
    for r in report["rows"]:
        assert set(r.keys()) == set(REQUIRED_FIELDS)

    # unknown numeric/bool fields stay null in JSON (reported_latency is all null)
    assert all(r["reported_latency"] is None for r in report["rows"])
    # at least some protects_kv values are null (unknown) in prior work
    assert any(r["protects_kv"] is None for r in report["rows"])

    # null renders as "?" in MD/CSV and "--" in LaTeX
    md_text = md.read_text()
    csv_text = csv.read_text()
    tex_text = tex.read_text()
    assert " ? " in md_text or "| ? " in md_text
    assert "?" in csv_text
    assert "--" in tex_text
    assert "\\begin{tabular}" in tex_text


def test_ours_row_claims(tmp_path) -> None:
    mod = _load("sota2", "scripts/render_sota_comparison_tables.py")
    js = tmp_path / "sota.json"
    rc = _main(mod, ["x", "--methods-yaml", str(COMMITTED_YAML),
                     "--output-json", str(js)])
    assert rc == 0
    rows = json.loads(js.read_text())["rows"]
    ours = [r for r in rows if r["source_type"] == "ours"]
    assert len(ours) == 1
    o = ours[0]
    assert o["protects_input"] is True
    assert o["protects_logits"] is True
    assert o["protects_kv"] is True
    assert o["protects_lora"] is True
    assert o["requires_gpu_tee"] is False
    assert o["requires_mpc_fhe"] is False
    assert o["tee_holds_full_model"] is False
    assert o["runs_real_7b"] is True
    assert o["real_attestation"] is True
    assert o["reported_latency"] is None


def test_nonlinear_backend_fields_included(tmp_path) -> None:
    mod = _load("sota3", "scripts/render_sota_comparison_tables.py")
    js = tmp_path / "sota.json"
    rc = _main(mod, ["x", "--methods-yaml", str(COMMITTED_YAML),
                     "--nonlinear-backend", "trusted_shortcut",
                     "--output-json", str(js)])
    assert rc == 0
    report = json.loads(js.read_text())
    assert report["nonlinear_backend"] == "trusted_shortcut"
    assert "nonlinear_design_metadata_hash" in report


def test_bad_yaml_path_raises() -> None:
    mod = _load("sota4", "scripts/render_sota_comparison_tables.py")
    with pytest.raises(Exception):
        _main(mod, ["x", "--methods-yaml", "/no/such/file.yaml",
                    "--output-json", "/tmp/should_not_matter.json"])
