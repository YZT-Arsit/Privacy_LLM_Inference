"""Local (CPU, dry-run) tests for resident-vs-non-resident correctness validation.

Drives scripts/compare_resident_folded_correctness.py over a tiny dry-run folded
package on CPU -- no Qwen weights, no GPU, no server paths. Confirms the validator
(a) PASSES when resident == non-resident (the real, un-mutated case), (b) FAILS +
flags when a resident weight is mutated in-place, (c) flags a dtype mismatch, and
(d) flags a shape mismatch (via the pure comparison helper). Security fields stay
clean throughout.

Run: python -m pytest tests/test_resident_folded_correctness.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load("cmp_res", "scripts/compare_resident_folded_correctness.py")


def _run(extra, tmp_path, name):
    out = tmp_path / ("%s.json" % name)
    argv = ["x", "--dry-run", "--num-layers", "4", "--device", "cpu",
            "--dtype", "float32", "--seq-len", "8", "--decode-steps", "4",
            "--output-json", str(out)] + (extra or [])
    old = sys.argv
    try:
        sys.argv = argv
        rc = _MOD.main()
    finally:
        sys.argv = old
    return rc, json.loads(out.read_text())


# ---- pure comparison helper -----------------------------------------------

def test_compare_arrays_identical() -> None:
    a = [np.ones((1, 5), dtype="float32")]
    c = _MOD._compare_arrays(a, [a[0].copy()], atol=1e-6, rtol=1e-6)
    assert c["shape_match"] and c["dtype_match"] and c["allclose_atol_rtol"]
    assert c["max_abs_err"] == 0.0
    assert c["token_match_if_logits_available"] is True


def test_compare_arrays_shape_mismatch() -> None:
    a = [np.ones((1, 5), dtype="float32")]
    b = [np.ones((1, 6), dtype="float32")]
    c = _MOD._compare_arrays(a, b, atol=1e-6, rtol=1e-6)
    assert c["shape_match"] is False
    assert c["max_abs_err"] is None          # not computed on a shape mismatch


def test_compare_arrays_dtype_mismatch() -> None:
    a = [np.ones((1, 5), dtype="float32")]
    b = [np.ones((1, 5), dtype="float64")]
    c = _MOD._compare_arrays(a, b, atol=1e-6, rtol=1e-6)
    assert c["dtype_match"] is False


def test_compare_arrays_value_divergence() -> None:
    a = [np.zeros((1, 4), dtype="float32")]
    b = [np.array([[0, 0, 0, 9]], dtype="float32")]
    c = _MOD._compare_arrays(a, b, atol=1e-6, rtol=1e-6)
    assert c["allclose_atol_rtol"] is False
    assert c["max_abs_err"] == 9.0
    assert c["token_match_if_logits_available"] is False   # argmax differs


# ---- end-to-end validator (dry-run package) -------------------------------

def test_resident_correctness_passes_when_identical(tmp_path) -> None:
    rc, r = _run([], tmp_path, "clean")
    assert rc == 0
    assert r["resident_correctness_passed"] is True
    assert r["resident_vs_nonresident_max_abs_err"] == 0.0
    assert r["resident_vs_nonresident_mean_abs_err"] == 0.0
    assert r["resident_weight_mutated"] is False
    assert r["resident_dtype_mismatch"] is False
    assert r["first_divergent_layer"] is None
    assert r["token_match_if_logits_available"] is True
    assert r["allclose_atol_rtol"] is True
    # the dtype lead from the server: fold compute dtype == resident cache dtype
    assert r["fold_compute_dtype"] == r["resident_cache_dtype"]


def test_resident_correctness_fails_on_inplace_mutation(tmp_path) -> None:
    rc, r = _run(["--inject-test-mutation"], tmp_path, "mut")
    assert rc == 1
    assert r["resident_correctness_passed"] is False
    assert r["resident_weight_mutated"] is True
    assert r["first_divergent_layer"] == 0          # mutated layer 0
    assert r["resident_vs_nonresident_max_abs_err"] > 0.0


def test_resident_correctness_flags_dtype_mismatch(tmp_path) -> None:
    rc, r = _run(["--inject-dtype-mismatch"], tmp_path, "dt")
    assert rc == 1
    assert r["resident_dtype_mismatch"] is True
    assert r["resident_correctness_passed"] is False
    assert r["resident_cache_dtype"] != r["fold_compute_dtype"]


def test_resident_correctness_security_clean(tmp_path) -> None:
    for extra, name in (([], "s_clean"), (["--inject-test-mutation"], "s_mut")):
        _, r = _run(extra, tmp_path, name)
        # a correctness regression must NEVER relax the security posture
        assert r["audit_passed"] is True
        assert r["tee_used_on_gpu"] is False
        assert r["worker_has_mask_secrets"] is False
        assert r["worker_has_raw_lora"] is False
        assert r["leaked_secret_fields"] == []
        assert r["schedule_secret_leaked_to_gpu"] is False
        assert r["gpu_request_contains_schedule_secret"] is False
        assert r["optimization_is_performance_only"] is True
