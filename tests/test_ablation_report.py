"""Fixture tests for the E14 ablation report (module + CLI). stdlib only.

Run: python -m pytest tests/test_ablation_report.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.ablation_report import build_ablation_report  # noqa: E402


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


def _decode(*, latency, trusted_bytes, gpu_bytes, boundary_calls=2,
            tokens=True, paper_ready=True, dry_run=False, **extra):
    r = {"stage": "package_backed_decode", "tokens_exact_match": tokens,
         "latency_s": latency, "trusted_bytes": trusted_bytes,
         "gpu_bytes": gpu_bytes, "boundary_calls": boundary_calls,
         "paper_ready": paper_ready, "dry_run": dry_run}
    r.update(extra)
    return r


def _write(p, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_module_boundary_and_attested_axes_available() -> None:
    inputs = {
        "full_reference_decode": _decode(latency=1.0, trusted_bytes=100,
                                         gpu_bytes=1000, boundary_calls=4),
        "lite_decode": _decode(latency=0.7, trusted_bytes=40, gpu_bytes=1000,
                               boundary_calls=2),
        "attested_decode": _decode(latency=1.2, trusted_bytes=110, gpu_bytes=1000,
                                   boundary_attested=True,
                                   runtime_hash_bound=True),
        "nonattested_decode": _decode(latency=1.0, trusted_bytes=100,
                                      gpu_bytes=1000, boundary_attested=False,
                                      runtime_hash_bound=False),
    }
    rep = build_ablation_report(inputs)
    assert rep["stage"] == "e14_ablation_report"
    assert "boundary_full_reference_vs_lite" in rep["axes_available"]
    assert "attested_vs_non_attested" in rep["axes_available"]

    b = rep["axes"]["boundary_full_reference_vs_lite"]
    assert abs(b["deltas"]["latency_s"]["delta"] - (-0.3)) < 1e-9
    assert b["deltas"]["trusted_bytes"]["delta"] == -60

    a = rep["axes"]["attested_vs_non_attested"]
    assert abs(a["attestation_overhead_latency_s"] - 0.2) < 1e-9
    assert a["tokens_equivalent"] is True
    # all inputs paper_ready, none dry_run
    assert rep["paper_ready"] is True


def test_missing_inputs_mark_axes_unavailable() -> None:
    rep = build_ablation_report({})
    for name in ("boundary_full_reference_vs_lite", "folded_storage_f32_vs_bf16",
                 "lora_rankmask_on_vs_off", "attested_vs_non_attested",
                 "max_new_tokens_scaling"):
        assert rep["axes"][name]["available"] is False
        assert rep["axes"][name].get("reason")
    assert rep["axes_available"] == []
    # no inputs -> not paper_ready
    assert rep["paper_ready"] is False


def test_lora_rankmask_requires_safe_fixture_mode() -> None:
    on = _decode(latency=1.0, trusted_bytes=100, gpu_bytes=1000, rank=8)
    off = _decode(latency=0.9, trusted_bytes=90, gpu_bytes=1000, rank=8)
    # not safe fixture -> unavailable
    rep = build_ablation_report({"lora_rankmask_on": on, "lora_rankmask_off": off})
    assert rep["axes"]["lora_rankmask_on_vs_off"]["available"] is False
    assert "safe fixture" in rep["axes"]["lora_rankmask_on_vs_off"]["reason"]
    # safe fixture -> available
    rep2 = build_ablation_report({"lora_rankmask_on": on,
                                  "lora_rankmask_off": off,
                                  "safe_fixture_mode": True})
    assert rep2["axes"]["lora_rankmask_on_vs_off"]["available"] is True


def test_dry_run_input_blocks_paper_ready() -> None:
    inputs = {
        "full_reference_decode": _decode(latency=1.0, trusted_bytes=100,
                                         gpu_bytes=1000, paper_ready=False,
                                         dry_run=True),
        "lite_decode": _decode(latency=0.7, trusted_bytes=40, gpu_bytes=1000),
    }
    rep = build_ablation_report(inputs)
    assert rep["any_input_dry_run"] is True
    assert rep["paper_ready"] is False


def test_cli_runs_and_writes(tmp_path) -> None:
    mod = _load("e14", "scripts/run_e14_ablation_report.py")
    full = _write(tmp_path / "full.json",
                  _decode(latency=1.0, trusted_bytes=100, gpu_bytes=1000,
                          boundary_calls=4))
    lite = _write(tmp_path / "lite.json",
                  _decode(latency=0.7, trusted_bytes=40, gpu_bytes=1000,
                          boundary_calls=2))
    mnt1 = _write(tmp_path / "mnt8.json",
                  _decode(latency=1.0, trusted_bytes=80, gpu_bytes=1000,
                          max_new_tokens=8))
    mnt2 = _write(tmp_path / "mnt32.json",
                  _decode(latency=3.2, trusted_bytes=320, gpu_bytes=1000,
                          max_new_tokens=32))
    oj = tmp_path / "e14.json"
    om = tmp_path / "e14.md"
    rc = _main(mod, ["x",
                     "--full-reference-decode-json", str(full),
                     "--lite-decode-json", str(lite),
                     "--max-new-tokens-decode-json", str(mnt1),
                     "--max-new-tokens-decode-json", str(mnt2),
                     "--nonlinear-backend", "current",
                     "--output-json", str(oj), "--output-md", str(om)])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["stage"] == "e14_ablation_report"
    assert "boundary_full_reference_vs_lite" in r["axes_available"]
    assert "max_new_tokens_scaling" in r["axes_available"]
    assert r.get("nonlinear_backend") == "current"
    # per-token trends present
    rows = r["axes"]["max_new_tokens_scaling"]["rows"]
    assert any(row["latency_per_token_s"] is not None for row in rows)
    assert om.is_file() and "E14 ablation report" in om.read_text()
