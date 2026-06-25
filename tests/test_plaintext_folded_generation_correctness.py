"""Local (CPU, dry-run) tests for the plaintext-vs-folded generation diagnostic.

Drives scripts/compare_plaintext_folded_generation_correctness.py over the tiny
dry-run folded package on CPU (no Qwen weights, no GPU, no server). Confirms the
diagnostic (a) PASSES when plaintext == folded (the real, un-broken case), (b)
localises first_divergent_step when the folded logits are corrupted, (c) detects a
resident-vs-non-resident mismatch, (d) reports shape / dtype mismatch (pure
helper), and (e) never serialises a forbidden GPU-boundary secret field.

Run: python -m pytest tests/test_plaintext_folded_generation_correctness.py -q
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


_MOD = _load("cmp_pf", "scripts/compare_plaintext_folded_generation_correctness.py")


def _run(extra, tmp_path, name):
    out = tmp_path / ("%s.json" % name)
    argv = ["x", "--dry-run", "--num-layers", "4", "--device", "cpu",
            "--dtype", "float32", "--seq-len", "8", "--max-steps", "4",
            "--output-json", str(out)] + (extra or [])
    old = sys.argv
    try:
        sys.argv = argv
        rc = _MOD.main()
    finally:
        sys.argv = old
    return rc, json.loads(out.read_text())


# ---- pure comparison helpers ----------------------------------------------

def test_compare_steps_identical() -> None:
    a = [np.array([1.0, 2.0, 3.0], dtype="float32"),
         np.array([3.0, 1.0, 0.0], dtype="float32")]
    rows, summ = _MOD._compare_logit_steps([x.copy() for x in a], a, topk=2,
                                           atol=1e-6, rtol=1e-6)
    assert summ["first_divergent_step"] is None
    assert summ["top1_match_rate"] == 1.0
    assert summ["max_abs_logit_err_max"] == 0.0
    assert all(r["top1_match"] for r in rows)


def test_compare_steps_value_divergence_first_step() -> None:
    plain = [np.array([1.0, 0.0], dtype="float32"),
             np.array([0.0, 1.0], dtype="float32"),
             np.array([1.0, 0.0], dtype="float32")]
    folded = [np.array([1.0, 0.0], dtype="float32"),   # step 0 matches
              np.array([1.0, 0.0], dtype="float32"),   # step 1 argmax flips
              np.array([1.0, 0.0], dtype="float32")]
    rows, summ = _MOD._compare_logit_steps(plain, folded, topk=2, atol=1e-6,
                                           rtol=1e-6)
    assert summ["first_divergent_step"] == 1
    assert rows[1]["top1_match"] is False
    assert rows[0]["top1_match"] is True
    assert summ["max_abs_logit_err_max"] == 1.0


def test_compare_steps_shape_mismatch_is_reported() -> None:
    plain = [np.ones((5,), dtype="float32")]
    folded = [np.ones((6,), dtype="float32")]
    rows, summ = _MOD._compare_logit_steps(plain, folded, topk=2, atol=1e-6,
                                           rtol=1e-6)
    assert rows[0]["shape_match"] is False
    assert rows[0]["max_abs_logit_err"] is None        # not computed on mismatch
    assert summ["first_divergent_step"] == 0


def test_compare_steps_dtype_mismatch_is_reported() -> None:
    plain = [np.ones((4,), dtype="float32")]
    folded = [np.ones((4,), dtype="float64")]
    rows, _ = _MOD._compare_logit_steps(plain, folded, topk=2, atol=1e-6,
                                        rtol=1e-6)
    assert rows[0]["dtype_match"] is False


def test_np_err_and_rel() -> None:
    a = np.array([0.0, 0.0, 3.0])
    b = np.array([0.0, 0.0, 0.0])
    mx, mn = _MOD._np_err(a, b)
    assert mx == 3.0 and abs(mn - 1.0) < 1e-12
    assert _MOD._np_rel(b, a) == 1.0                     # ||b-a||/||a|| = 3/3


# ---- end-to-end diagnostic (dry-run tiny package) -------------------------

def test_clean_plaintext_folded_passes(tmp_path) -> None:
    rc, r = _run(["--resident-folded-weights", "--compare-nonresident"],
                 tmp_path, "clean")
    assert rc == 0
    assert r["correctness_passed"] is True
    assert r["plaintext_vs_folded_top1_match_rate"] == 1.0
    assert r["first_divergent_step"] is None
    assert r["first_divergent_stage"] is None
    assert r["suspected_root_cause"] is None
    # resident is a pure performance optimisation -> bit-exact vs non-resident
    assert r["resident_vs_nonresident_correctness_passed"] is True
    assert r["resident_vs_nonresident_max_abs_err"] == 0.0
    assert r["resident_weight_mutated"] is False
    assert r["resident_dtype_mismatch"] is False
    # the dtype lead: fold compute dtype == resident cache dtype
    assert r["fold_compute_dtype"] == r["resident_cache_dtype"]
    # plaintext-vs-folded numeric error is tiny (exact masking algebra)
    assert r["plaintext_vs_folded_max_abs_logit_err_max"] < 1e-2


def test_inject_folded_divergence_localises_step(tmp_path) -> None:
    rc, r = _run(["--inject-folded-divergence-step", "2"], tmp_path, "div")
    assert rc == 1
    assert r["correctness_passed"] is False
    assert r["first_divergent_step"] == 2
    assert r["first_divergent_stage"] == "decode"
    assert r["plaintext_vs_folded_top1_match_rate"] < 1.0
    assert "decode_path" in r["suspected_root_cause"]


def test_inject_folded_divergence_step0_is_prefill(tmp_path) -> None:
    _, r = _run(["--inject-folded-divergence-step", "0"], tmp_path, "div0")
    assert r["first_divergent_step"] == 0
    assert r["first_divergent_stage"] == "prefill"
    assert "prefill_path" in r["suspected_root_cause"]


def test_inject_resident_divergence_is_detected(tmp_path) -> None:
    rc, r = _run(["--resident-folded-weights", "--compare-nonresident",
                  "--inject-resident-divergence"], tmp_path, "rvn")
    assert rc == 1
    assert r["resident_vs_nonresident_correctness_passed"] is False
    assert r["resident_vs_nonresident_max_abs_err"] > 0.0
    assert r["resident_vs_nonresident_top1_match"] is False
    assert "resident_cache" in r["suspected_root_cause"]


def test_security_fields_clean_and_no_forbidden_keys(tmp_path) -> None:
    from pllo.protocol.remote import FORBIDDEN_WIRE_FIELDS

    def _walk_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _walk_keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _walk_keys(v)

    for extra, name in (([], "s_clean"),
                        (["--inject-folded-divergence-step", "1"], "s_div"),
                        (["--resident-folded-weights", "--compare-nonresident"],
                         "s_res")):
        _, r = _run(extra, tmp_path, name)
        # a correctness regression must NEVER relax the security posture
        assert r["audit_passed"] is True
        assert r["tee_used_on_gpu"] is False
        assert r["worker_has_mask_secrets"] is False
        assert r["worker_has_raw_lora"] is False
        assert r["leaked_secret_fields"] == []
        assert r["schedule_secret_leaked_to_gpu"] is False
        assert r["gpu_request_contains_schedule_secret"] is False
        assert r["pad_disabled_for_correctness"] is False
        # no GPU-boundary secret field name is ever serialised in the report
        keys = set(_walk_keys(r))
        assert keys.isdisjoint(FORBIDDEN_WIRE_FIELDS)
