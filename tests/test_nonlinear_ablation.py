"""E16 nonlinear ablation tests (fixtures only, stdlib).

Run: python -m pytest tests/test_nonlinear_ablation.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_ablation import (  # noqa: E402
    build_nonlinear_ablation,
    render_csv,
    render_latex,
    render_md,
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


def _decode(backend, *, latency, trusted_bytes, boundary_calls):
    return {
        "stage": "tee_gpu_protocol_demo",
        "nonlinear_backend": backend,
        "tokens_exact_match": True,
        "latency_s": latency,
        "decode_latency": latency,
        "max_new_tokens": 4,
        "trusted_bytes": trusted_bytes,
        "gpu_bytes": 1000,
        "boundary_calls": boundary_calls,
        "peak_gpu_memory_mb": 2048,
        "worker_has_mask_secrets": False,
        "runtime_hash_bound": True,
        "boundary_attested": True,
        "lora_enabled": True,
        "dry_run": False,
        "paper_ready": True,
    }


def _build(backend, size_gb):
    return {"stage": "folded_package_build", "nonlinear_backend": backend,
            "folded_weight_size_gb": size_gb,
            "folded_weight_generation_time_s": 100.0}


def _negative(backend):
    return {"stage": "security_negative_tests", "nonlinear_backend": backend,
            "all_passed": True}


def _transcript(backend):
    return {"stage": "security_transcript_scan", "nonlinear_backend": backend,
            "fail": False}


def _two_backend_fixtures():
    return {
        "current": [
            _decode("current", latency=1.2, trusted_bytes=5000,
                    boundary_calls=12),
            _build("current", 26.3),
            _negative("current"), _transcript("current"),
        ],
        "trusted_shortcut": [
            _decode("trusted_shortcut", latency=0.9, trusted_bytes=3500,
                    boundary_calls=4),
            _build("trusted_shortcut", 27.1),
            _negative("trusted_shortcut"), _transcript("trusted_shortcut"),
        ],
    }


def test_build_ablation_rows_and_deltas():
    rep = build_nonlinear_ablation(_two_backend_fixtures())
    assert rep["stage"] == "e16_nonlinear_ablation"
    metrics = [r["metric"] for r in rep["rows"]]
    expected = [
        "design", "nonlinear_boundary_calls", "trusted_bytes_due_to_nonlinear",
        "latency_overhead_due_to_nonlinear", "security_difference",
        "package_size_difference", "lora_compatibility_difference",
    ]
    for m in expected:
        assert m in metrics
    assert len(rep["rows"]) == 7
    # at least one numeric delta computed
    assert rep["deltas_summary"]
    assert "trusted_bytes_due_to_nonlinear" in rep["deltas_summary"]
    # trusted_shortcut - current = 3500 - 5000 = -1500
    assert rep["deltas_summary"]["trusted_bytes_due_to_nonlinear"] == -1500
    # boundary_calls delta = 4 - 12 = -8
    assert rep["deltas_summary"]["nonlinear_boundary_calls"] == -8


def test_renderers_mention_both_backends():
    rep = build_nonlinear_ablation(_two_backend_fixtures())
    md = render_md(rep)
    csv = render_csv(rep)
    tex = render_latex(rep)
    for blob in (md, csv, tex):
        assert blob
        assert "current" in blob
        assert "trusted_shortcut" in blob


def test_script_writes_four_outputs(tmp_path):
    mod = _load("e16", "scripts/run_e16_nonlinear_ablation_report.py")
    fx = _two_backend_fixtures()

    def _w(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return str(p)

    cur = [_w("c0.json", fx["current"][0]), _w("c1.json", fx["current"][1]),
           _w("c2.json", fx["current"][2]), _w("c3.json", fx["current"][3])]
    ts = [_w("t0.json", fx["trusted_shortcut"][0]),
          _w("t1.json", fx["trusted_shortcut"][1]),
          _w("t2.json", fx["trusted_shortcut"][2]),
          _w("t3.json", fx["trusted_shortcut"][3])]

    oj = tmp_path / "e16.json"
    om = tmp_path / "e16.md"
    oc = tmp_path / "e16.csv"
    ot = tmp_path / "e16.tex"
    argv = ["x"]
    for p in cur:
        argv += ["--current-json", p]
    for p in ts:
        argv += ["--trusted-shortcut-json", p]
    argv += ["--output-json", str(oj), "--output-md", str(om),
             "--output-csv", str(oc), "--output-tex", str(ot)]
    rc = _main(mod, argv)
    assert rc == 0
    assert oj.is_file() and om.is_file() and oc.is_file() and ot.is_file()
    r = json.loads(oj.read_text())
    assert r["stage"] == "e16_nonlinear_ablation"
    assert len(r["rows"]) == 7
