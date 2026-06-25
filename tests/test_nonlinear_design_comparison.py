"""E15 dual-design comparison tests (fixtures only, stdlib).

Run: python -m pytest tests/test_nonlinear_design_comparison.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.nonlinear_design_comparison import (  # noqa: E402
    build_comparison,
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


# --- fixture report shapes -------------------------------------------------


def _decode(backend, *, latency, trusted_bytes, gpu_bytes=1000,
            boundary_calls=10):
    return {
        "stage": "tee_gpu_protocol_demo",
        "nonlinear_backend": backend,
        "tokens_exact_match": True,
        "operator_allclose": True,
        "logits_error": 1e-7,
        "latency_s": latency,
        "decode_latency": latency,
        "max_new_tokens": 4,
        "trusted_bytes": trusted_bytes,
        "gpu_bytes": gpu_bytes,
        "boundary_calls": boundary_calls,
        "peak_gpu_memory_mb": 2048,
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": [],
        "worker_has_mask_secrets": False,
        "worker_has_raw_lora": False,
        "audit_passed": True,
        "package_backed_decode": True,
        "boundary_attested": True,
        "runtime_hash_bound": True,
        "gpu_worker_remote": True,
        "lora_enabled": False,
        "dry_run": False,
        "paper_ready": True,
    }


def _build(backend, *, size_gb=26.3, gen_time=120.0):
    return {
        "stage": "folded_package_build",
        "nonlinear_backend": backend,
        "folded_weight_size_gb": size_gb,
        "folded_weight_generation_time_s": gen_time,
    }


def _negative(backend):
    return {"stage": "security_negative_tests", "nonlinear_backend": backend,
            "all_passed": True}


def _transcript(backend):
    return {"stage": "security_transcript_scan", "nonlinear_backend": backend,
            "fail": False}


def _pairwise(backend, delta=0.005):
    return {"stage": "e9_pairwise_utility_preservation",
            "nonlinear_backend": backend, "utility_preserved": True,
            "delta_abs": delta, "paper_ready": True, "dry_run": False}


def _complete(backend, *, latency, trusted_bytes):
    return [
        _decode(backend, latency=latency, trusted_bytes=trusted_bytes),
        _build(backend),
        _negative(backend),
        _transcript(backend),
        _pairwise(backend),
    ]


# --- complete + favorable -> recommendation ok -----------------------------


def test_complete_both_backends_ok_recommendation():
    rbb = {
        "current": _complete("current", latency=1.0, trusted_bytes=5000),
        "trusted_shortcut": _complete("trusted_shortcut", latency=0.8,
                                      trusted_bytes=4000),
    }
    rep = build_comparison(rbb)
    assert rep["stage"] == "e15_nonlinear_design_comparison"
    rec = rep["recommendation"]
    assert rec["recommendation_status"] == "ok"
    assert rec["final_recommendation"] is not None
    # security must favor the formally-claimed `current`, not trusted_shortcut
    assert rec["per_axis_winners"]["security"] == "current"
    assert rec["final_recommendation"] == "current"
    # 5 tables populated
    for tbl in ("correctness", "security", "performance", "deployment"):
        assert set(rep[tbl].keys()) == {"current", "trusted_shortcut"}
    assert rep["correctness"]["current"]["token_exact_match"] is True
    assert rep["performance"]["trusted_shortcut"]["decode_latency"] == 0.8
    assert rep["security"]["current"]["negative_tests"] is True


def test_security_never_recommends_not_formally_claimed():
    # trusted_shortcut faster but security must NOT name it as winner
    rbb = {
        "current": _complete("current", latency=2.0, trusted_bytes=9000),
        "trusted_shortcut": _complete("trusted_shortcut", latency=0.5,
                                      trusted_bytes=3000),
    }
    rep = build_comparison(rbb)
    rec = rep["recommendation"]
    assert rec["per_axis_winners"]["security"] == "current"
    # latency winner is trusted_shortcut
    assert rec["per_axis_winners"]["latency"] == "trusted_shortcut"
    # final recommendation defers to the secure formally-claimed design
    assert rec["final_recommendation"] == "current"


# --- one backend only / missing axis -> insufficient_evidence --------------


def test_only_one_backend_insufficient():
    rbb = {"current": _complete("current", latency=1.0, trusted_bytes=5000),
           "trusted_shortcut": []}
    rep = build_comparison(rbb)
    rec = rep["recommendation"]
    assert rec["recommendation_status"] == "insufficient_evidence"
    assert rec["missing_evidence"]
    assert rec.get("final_recommendation") is None
    assert any("trusted_shortcut" in m for m in rec["missing_evidence"])


def test_missing_security_axis_insufficient():
    # both have decode/build but trusted_shortcut lacks security evidence
    rbb = {
        "current": _complete("current", latency=1.0, trusted_bytes=5000),
        "trusted_shortcut": [
            _decode("trusted_shortcut", latency=0.8, trusted_bytes=4000),
            _build("trusted_shortcut"),
        ],
    }
    rep = build_comparison(rbb)
    rec = rep["recommendation"]
    assert rec["recommendation_status"] == "insufficient_evidence"
    assert rec.get("final_recommendation") is None
    assert any("security" in m for m in rec["missing_evidence"])


# --- script via importlib + argv swap --------------------------------------


def test_script_runs(tmp_path):
    mod = _load("e15", "scripts/run_e15_nonlinear_design_comparison.py")

    def _w(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return str(p)

    cur_decode = _w("cur_decode.json",
                    _decode("current", latency=1.0, trusted_bytes=5000))
    cur_build = _w("cur_build.json", _build("current"))
    cur_neg = _w("cur_neg.json", _negative("current"))
    cur_trans = _w("cur_trans.json", _transcript("current"))
    cur_pw = _w("cur_pw.json", _pairwise("current"))
    ts_decode = _w("ts_decode.json",
                   _decode("trusted_shortcut", latency=0.8, trusted_bytes=4000))
    ts_build = _w("ts_build.json", _build("trusted_shortcut"))
    ts_neg = _w("ts_neg.json", _negative("trusted_shortcut"))
    ts_trans = _w("ts_trans.json", _transcript("trusted_shortcut"))
    ts_pw = _w("ts_pw.json", _pairwise("trusted_shortcut"))

    oj = tmp_path / "e15.json"
    om = tmp_path / "e15.md"
    rc = _main(mod, [
        "x",
        "--current-json", cur_decode, "--current-json", cur_build,
        "--current-json", cur_neg, "--current-json", cur_trans,
        "--current-json", cur_pw,
        "--trusted-shortcut-json", ts_decode, "--trusted-shortcut-json", ts_build,
        "--trusted-shortcut-json", ts_neg, "--trusted-shortcut-json", ts_trans,
        "--trusted-shortcut-json", ts_pw,
        "--output-json", str(oj), "--output-md", str(om)])
    assert rc == 0
    assert oj.is_file() and om.is_file()
    r = json.loads(oj.read_text())
    assert r["stage"] == "e15_nonlinear_design_comparison"
    assert r["recommendation"]["recommendation_status"] == "ok"
    md = om.read_text()
    assert "E15" in md and "Recommendation" in md
