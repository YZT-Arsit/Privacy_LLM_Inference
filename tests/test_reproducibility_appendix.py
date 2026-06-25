"""Fixture tests for the reproducibility appendix generator.

Run: python -m pytest tests/test_reproducibility_appendix.py -q
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


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


def _fixtures(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "model_name": "Qwen2.5-7B",
        "model_path": "/models/qwen7b",
        "runtime_hash": "abc123runtime",
        "mr_td": "MRTDVALUE",
        "manifest_hash": "manifesthash00",
        "evidence_report_data": "00" * 32,
    }))
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"name": "mmlu", "split": "test", "n": 100}))
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"hello reproducibility")
    return result, card, artifact


def test_appendix_basic(tmp_path) -> None:
    mod = _load("repro", "scripts/render_reproducibility_appendix.py")
    result, card, artifact = _fixtures(tmp_path)
    md = tmp_path / "appendix.md"
    js = tmp_path / "appendix.json"
    tex = tmp_path / "appendix.tex"
    rc = _main(mod, ["x", "--result-json", str(result),
                     "--dataset-card-json", str(card),
                     "--output-artifact", str(artifact),
                     "--hardware", "1x H800 80GB",
                     "--output-md", str(md), "--output-json", str(js),
                     "--output-tex", str(tex)])
    assert rc == 0
    assert md.exists()
    md_text = md.read_text()
    assert "Qwen2.5-7B" in md_text
    assert "abc123runtime" in md_text
    assert "1x H800 80GB" in md_text

    report = json.loads(js.read_text())
    assert report["stage"] == "reproducibility_appendix"
    assert report["model"]["name"] == "Qwen2.5-7B"
    assert report["runtime_hash"] == "abc123runtime"
    art = report["output_artifact_hashes"]
    assert len(art) == 1
    (only_hash,) = list(art.values())
    assert re.fullmatch(r"[0-9a-f]{64}", only_hash)

    # software versions auto-collected (python at least)
    assert "python" in report["software_versions"]

    assert tex.exists()
    assert "\\begin{tabular}" in tex.read_text()


def test_appendix_nonlinear_design_fields(tmp_path) -> None:
    mod = _load("repro2", "scripts/render_reproducibility_appendix.py")
    result, card, artifact = _fixtures(tmp_path)
    md = tmp_path / "appendix.md"
    js = tmp_path / "appendix.json"
    rc = _main(mod, ["x", "--result-json", str(result),
                     "--nonlinear-backend", "trusted_shortcut",
                     "--output-md", str(md), "--output-json", str(js)])
    assert rc == 0
    report = json.loads(js.read_text())
    assert report["nonlinear_backend"] == "trusted_shortcut"
    assert "nonlinear_design_metadata_hash" in report
    assert "nonlinear_design_label" in report
    assert "trusted_shortcut" in md.read_text()


def test_appendix_missing_inputs_robust(tmp_path) -> None:
    mod = _load("repro3", "scripts/render_reproducibility_appendix.py")
    md = tmp_path / "appendix.md"
    js = tmp_path / "appendix.json"
    rc = _main(mod, ["x", "--output-md", str(md), "--output-json", str(js)])
    assert rc == 0
    md_text = md.read_text()
    assert "(not provided)" in md_text
    report = json.loads(js.read_text())
    assert report["model"]["name"] is None
