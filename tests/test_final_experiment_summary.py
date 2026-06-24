"""Tests for the consolidated final experiment summary generator.

Asserts the four sections are present, the three claims are marked complete, full
28-layer package-backed decode is explicitly TODO (not claimed), the size
paragraph explains 26.34 GB vs 13.17 GB, and measured numbers can be refreshed
from a supplied artifact JSON.

Run: python -m pytest tests/test_final_experiment_summary.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(argv):
    spec = importlib.util.spec_from_file_location(
        "finalsum", REPO_ROOT / "scripts" / "build_final_experiment_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


def test_summary_structure_and_no_overclaim(tmp_path) -> None:
    js = tmp_path / "s.json"
    md = tmp_path / "s.md"
    assert _run(["prog", "--output-json", str(js), "--output-md", str(md)]) == 0
    s = json.loads(js.read_text())

    # four required sections present
    for sec in ("completed_claims", "supporting_artifacts", "measured_numbers",
                "limitations_todo"):
        assert sec in s and s[sec]

    ids = [c["id"] for c in s["completed_claims"]]
    assert ids == ["C1", "C2", "C3"]
    assert all(c["status"] == "complete" for c in s["completed_claims"])

    # full 28-layer package-backed decode is NOT claimed complete
    assert s["full_28layer_package_backed_decode_status"] == "TODO_not_implemented"
    assert any("28-layer package-backed prefill/decode is NOT implemented" in t
               for t in s["limitations_todo"])
    for c in s["completed_claims"]:
        txt = (c["claim"] + c["summary"]).lower()
        assert not ("28-layer" in txt and "decode" in txt), c["id"]

    # measured numbers groups + key values
    mn = s["measured_numbers"]
    assert mn["folded_package_build_verify"]["package_size_gb"] == 26.339369
    assert mn["folded_package_layer0_correctness"]["allclose"] is True
    assert mn["cross_machine_attested_mock"]["boundary_attested"] is True
    assert mn["standalone_e1_e2"][
        "teacher_forced_top1_match_rate_plain_masked"] == 1.0

    # size paragraph cites both figures
    assert "26.34" in s["size_explanation"] and "13.17" in s["size_explanation"]

    md_txt = md.read_text()
    assert "Why the folded package is 26.34 GB" in md_txt
    assert "No claim of full 28-layer package-backed decode" in md_txt


def test_measured_numbers_override_from_artifact(tmp_path) -> None:
    fb = tmp_path / "fb.json"
    fb.write_text(json.dumps({"manifest_hash": "deadbeef", "num_shards": 29,
                              "folded_weight_size_gb": 99.9}))
    js = tmp_path / "s.json"
    assert _run(["prog", "--folded-build-json", str(fb),
                 "--output-json", str(js),
                 "--output-md", str(tmp_path / "s.md")]) == 0
    f = json.loads(js.read_text())["measured_numbers"]["folded_package_build_verify"]
    assert f["manifest_hash"] == "deadbeef"
    assert f["package_size_gb"] == 99.9
