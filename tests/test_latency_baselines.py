"""Latency / overhead baselines -- pure parsing, stdlib + numpy only.

Run: python -m pytest tests/test_latency_baselines.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.latency_baselines import (  # noqa: E402
    build_latency_table,
    parse_backend_row,
    render_csv,
    render_latex,
)


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


def test_parse_backend_row_latency() -> None:
    rep = {"stage": "x", "dry_run": False, "latency_s": 60.0,
           "max_new_tokens": 8, "trusted_bytes": 100, "gpu_bytes": 2000,
           "boundary_calls": {"a": 2, "b": 3}, "peak_gpu_memory_mb": 2200.0}
    row = parse_backend_row("folded_h800_remote", rep)
    assert row["total_latency_s"] == pytest.approx(60.0)
    assert row["latency_per_token_s"] == pytest.approx(7.5)
    assert row["tokens_per_s"] == pytest.approx(8 / 60.0)
    assert row["boundary_calls"] == pytest.approx(5.0)
    assert row["paper_ready"] is True


def test_build_table_overhead() -> None:
    plain = {"stage": "p", "dry_run": False, "latency_s": 30.0,
             "max_new_tokens": 8}
    remote = {"stage": "r", "dry_run": False, "latency_s": 60.0,
              "max_new_tokens": 8}
    table = build_latency_table({"plaintext_h800": plain,
                                 "folded_h800_remote": remote})
    by = {r["backend"]: r for r in table["rows"]}
    assert by["folded_h800_remote"]["overhead_vs_plaintext_h800"] == \
        pytest.approx(2.0)
    assert by["plaintext_h800"]["overhead_vs_plaintext_h800"] == \
        pytest.approx(1.0)


def test_run_e12_script(tmp_path) -> None:
    plain = {"stage": "p", "dry_run": False, "latency_s": 30.0,
             "max_new_tokens": 8}
    remote = {"stage": "r", "dry_run": False, "latency_s": 60.0,
              "max_new_tokens": 8, "peak_gpu_memory_mb": 2200.0}
    pp = tmp_path / "plain.json"
    pr = tmp_path / "remote.json"
    pp.write_text(json.dumps(plain))
    pr.write_text(json.dumps(remote))
    mod = _load("e12", "scripts/run_e12_latency_baselines.py")
    oj = tmp_path / "out.json"
    oc = tmp_path / "out.csv"
    om = tmp_path / "out.md"
    ot = tmp_path / "out.tex"
    rc = _main(mod, ["x", "--plaintext-h800-json", str(pp),
                     "--folded-h800-remote-json", str(pr),
                     "--output-json", str(oj), "--output-csv", str(oc),
                     "--output-md", str(om), "--output-tex", str(ot)])
    assert rc == 0
    out = json.loads(oj.read_text())
    assert out["stage"] == "latency_baselines"
    assert len(out["rows"]) == 2
    csv = oc.read_text()
    assert csv.splitlines()[0].startswith("backend,")
    tex = ot.read_text()
    assert "tabular" in tex


def test_render_csv_and_latex_smoke() -> None:
    table = build_latency_table({"plaintext_h800": {
        "stage": "p", "dry_run": False, "latency_s": 30.0,
        "max_new_tokens": 8}})
    csv = render_csv(table)
    assert "backend" in csv.splitlines()[0]
    tex = render_latex(table)
    assert r"\begin{tabular}" in tex and r"\end{tabular}" in tex
