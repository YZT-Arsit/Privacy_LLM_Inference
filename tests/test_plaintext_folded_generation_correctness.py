"""Local (CPU, dry-run) tests for the plaintext-vs-folded generation diagnostic.

Drives scripts/compare_plaintext_folded_generation_correctness.py over the tiny
dry-run folded package on CPU (no Qwen weights, no GPU, no server). Confirms:

* the **plaintext reproduction gate** -- the tool's step-by-step plaintext path
  reproduces the authoritative ``generate`` path; a forced mismatch BLOCKS every
  folded correctness conclusion;
* **teacher-forcing** vs **free-running** rollout: clean -> both match; a forced
  free-running fork is localised to ``first_free_running_divergent_step`` and (when
  teacher-forcing still matches) flagged as autoregressive-state divergence;
* a forced teacher-forced step divergence is localised;
* resident == non-resident folded (bit-exact) + injected resident divergence is
  detected;
* shape / dtype mismatch are reported by the pure helper;
* no forbidden GPU-boundary secret field is ever serialised.

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
            "--dtype", "float32", "--seq-len", "8", "--max-steps", "6",
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
    folded = [np.array([1.0, 0.0], dtype="float32"),
              np.array([1.0, 0.0], dtype="float32"),   # step 1 argmax flips
              np.array([1.0, 0.0], dtype="float32")]
    rows, summ = _MOD._compare_logit_steps(plain, folded, topk=2, atol=1e-6,
                                           rtol=1e-6)
    assert summ["first_divergent_step"] == 1
    assert rows[1]["token_match"] is False and rows[0]["token_match"] is True
    assert summ["max_abs_logit_err_max"] == 1.0


def test_compare_steps_shape_mismatch_is_reported() -> None:
    rows, summ = _MOD._compare_logit_steps([np.ones((5,), dtype="float32")],
                                           [np.ones((6,), dtype="float32")],
                                           topk=2, atol=1e-6, rtol=1e-6)
    assert rows[0]["shape_match"] is False
    assert rows[0]["max_abs_logit_err"] is None
    assert summ["first_divergent_step"] == 0


def test_compare_steps_dtype_mismatch_is_reported() -> None:
    rows, _ = _MOD._compare_logit_steps([np.ones((4,), dtype="float32")],
                                        [np.ones((4,), dtype="float64")],
                                        topk=2, atol=1e-6, rtol=1e-6)
    assert rows[0]["dtype_match"] is False


def test_first_divergent_tokens_and_match_rate() -> None:
    assert _MOD._first_divergent_tokens([1, 2, 3], [1, 2, 3]) is None
    assert _MOD._first_divergent_tokens([1, 2, 3], [1, 9, 3]) == 1
    assert _MOD._first_divergent_tokens([1, 2], [1, 2, 3]) == 2   # prefix
    assert _MOD._token_match_rate([1, 2, 3, 4], [1, 2, 9, 4]) == 0.75


# ---- end-to-end diagnostic (dry-run tiny package) -------------------------

def test_clean_both_passes(tmp_path) -> None:
    rc, r = _run(["--rollout-mode", "both", "--verify-plaintext-reproduction",
                  "--resident-folded-weights", "--compare-nonresident"],
                 tmp_path, "clean")
    assert rc == 0
    assert r["plaintext_reproduction_passed"] is True
    assert r["correctness_blocked_by_plaintext_reproduction_mismatch"] is False
    assert r["correctness_passed"] is True
    assert r["teacher_forcing_top1_match_rate"] == 1.0
    assert r["teacher_forcing_first_divergent_step"] is None
    assert r["first_free_running_divergent_step"] is None
    assert r["free_running_token_match_rate"] == 1.0
    assert r["suspected_root_cause"] is None
    assert r["resident_vs_nonresident_correctness_passed"] is True
    assert r["resident_vs_nonresident_max_abs_err"] == 0.0
    assert r["resident_weight_mutated"] is False
    assert r["fold_compute_dtype"] == r["resident_cache_dtype"]


def test_plaintext_reproduction_mismatch_blocks_correctness(tmp_path) -> None:
    rc, r = _run(["--rollout-mode", "both", "--inject-plaintext-repro-mismatch"],
                 tmp_path, "repro")
    assert rc == 1
    assert r["plaintext_reproduction_passed"] is False
    assert r["first_plaintext_reproduction_divergent_step"] == 0
    assert r["correctness_blocked_by_plaintext_reproduction_mismatch"] is True
    # a blocked run must NOT claim plaintext-vs-folded correctness passed
    assert r["correctness_passed"] is None
    assert "reproduction mismatch" in r["suspected_root_cause"]
    # folded comparisons are skipped while blocked
    assert r["teacher_forcing_per_step"] == []
    assert r["free_running_per_step"] == []


def test_authoritative_processor_divergence_is_attributed(tmp_path) -> None:
    # the ifeval baseline (model.generate) diverges from raw greedy, but the
    # tool's raw decode is faithful -> attribute to generation config, not a bug
    rc, r = _run(["--rollout-mode", "both",
                  "--inject-authoritative-processor-divergence"], tmp_path, "proc")
    assert rc == 1
    assert r["plaintext_reproduction_passed"] is False        # vs authoritative
    assert r["manual_reproduces_raw_greedy"] is True          # tool is correct
    assert r["authoritative_uses_generation_processors"] is True
    assert r["correctness_blocked_by_plaintext_reproduction_mismatch"] is True
    assert r["correctness_passed"] is None
    assert "generation-config mismatch" in r["suspected_root_cause"]
    # folded diagnostics still ran (against the faithful raw-greedy reference)
    assert r["teacher_forcing_per_step"] != []


def test_free_running_divergence_localised(tmp_path) -> None:
    rc, r = _run(["--rollout-mode", "both", "--inject-free-divergence-step", "3"],
                 tmp_path, "free")
    assert r["plaintext_reproduction_passed"] is True
    assert r["first_free_running_divergent_step"] == 3
    assert r["free_running_token_match_rate"] < 1.0
    # teacher-forcing per-step fidelity still holds -> autoregressive divergence
    assert r["teacher_forcing_first_divergent_step"] is None
    assert r["suspected_root_cause"] == \
        "autoregressive state divergence or generation-state mismatch"
    assert isinstance(r["folded_token_ids"], list)
    assert isinstance(r["plaintext_token_ids"], list)


def test_teacher_forcing_divergence_localised(tmp_path) -> None:
    rc, r = _run(["--rollout-mode", "teacher_forcing",
                  "--inject-folded-divergence-step", "2"], tmp_path, "tf")
    assert rc == 1
    assert r["correctness_passed"] is False
    assert r["teacher_forcing_first_divergent_step"] == 2
    assert r["first_divergent_stage"] == "decode"
    assert "teacher_forcing" in r["suspected_root_cause"]


def test_inject_resident_divergence_is_detected(tmp_path) -> None:
    rc, r = _run(["--rollout-mode", "teacher_forcing", "--resident-folded-weights",
                  "--compare-nonresident", "--inject-resident-divergence"],
                 tmp_path, "rvn")
    assert rc == 1
    assert r["resident_vs_nonresident_correctness_passed"] is False
    assert r["resident_vs_nonresident_max_abs_err"] > 0.0
    assert r["resident_vs_nonresident_top1_match"] is False


def test_reproduction_metadata_present(tmp_path) -> None:
    _, r = _run(["--rollout-mode", "teacher_forcing"], tmp_path, "meta")
    for key in ("run_ifeval_plaintext_token_ids", "correctness_plaintext_token_ids",
                "generation_config_summary", "eos_token_id", "pad_token_id",
                "do_sample", "num_beams", "use_cache", "attention_mask_policy",
                "position_ids_policy", "prompt_token_count", "add_generation_prompt"):
        assert key in r
    assert r["do_sample"] is False and r["num_beams"] == 1 and r["use_cache"] is True
    assert r["generation_config_summary"]["do_sample"] is False


def _run_diag(extra, tmp_path, name, *, prompts_jsonl=None):
    out = tmp_path / ("%s.json" % name)
    argv = ["x", "--dry-run", "--num-layers", "4", "--device", "cpu",
            "--dtype", "float32", "--seq-len", "8", "--max-steps", "6",
            "--diagnose-divergence", "--output-json", str(out)]
    if prompts_jsonl:
        argv += ["--input-jsonl", str(prompts_jsonl)]
    argv += (extra or [])
    old = sys.argv
    try:
        sys.argv = argv
        rc = _MOD.main()
    finally:
        sys.argv = old
    return rc, json.loads(out.read_text())


def test_classify_divergence() -> None:
    assert _MOD._classify_divergence(None)[0] == "no_token_divergence"
    assert _MOD._classify_divergence(0)[0] == "step0_divergence"
    assert _MOD._classify_divergence(3)[0] == "early_decode_divergence"


def test_diagnosis_clean_no_token_divergence(tmp_path) -> None:
    rc, r = _run_diag([], tmp_path, "dclean")
    assert rc == 0
    assert r["stage"] == "plaintext_vs_folded_divergence_diagnosis"
    d = r["per_example_diagnosis"][0]
    assert d["first_free_running_divergent_step"] is None
    assert d["divergence_class"] == "no_token_divergence"
    # required per-example fields present
    for k in ("prompt_token_count", "chat_template_sha256",
              "plaintext_first_token_ids", "folded_first_token_ids",
              "plaintext_total_tokens", "folded_total_tokens"):
        assert k in d
    assert r["plaintext_logits_or_sampling_on_gpu"] is False


def test_diagnosis_inject_early_decode(tmp_path) -> None:
    rc, r = _run_diag(["--diag-inject-divergence-step", "2"], tmp_path, "ddiv")
    d = r["per_example_diagnosis"][0]
    assert d["first_free_running_divergent_step"] == 2
    assert d["divergence_class"] == "early_decode_divergence"
    assert d["plaintext_token_at_divergence"] != d["folded_token_at_divergence"]


def test_diagnosis_inject_step0(tmp_path) -> None:
    _, r = _run_diag(["--diag-inject-divergence-step", "0"], tmp_path, "ddiv0")
    d = r["per_example_diagnosis"][0]
    assert d["first_free_running_divergent_step"] == 0
    assert d["divergence_class"] == "step0_divergence"


def test_diagnosis_batch_indices(tmp_path) -> None:
    pj = tmp_path / "p.jsonl"
    pj.write_text("\n".join(
        json.dumps({"key": "k%d" % i, "prompt": "prompt %d" % i})
        for i in range(3)) + "\n")
    rc, r = _run_diag(["--example-indices", "0,2"], tmp_path, "dbatch",
                      prompts_jsonl=pj)
    assert r["num_examples_diagnosed"] == 2
    assert r["example_indices"] == [0, 2]
    assert len(r["per_example_diagnosis"]) == 2
    assert "divergence_class_counts" in r


def test_diagnosis_batch_from_strict_gap(tmp_path) -> None:
    pj = tmp_path / "p.jsonl"
    pj.write_text("\n".join(
        json.dumps({"key": "k%d" % i, "prompt": "prompt %d" % i})
        for i in range(3)) + "\n")
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps(
        {"strict": {"plaintext_pass_folded_fail": ["k1", "k2"]}}))
    _, r = _run_diag(["--batch-from-strict-gap", str(gap)], tmp_path, "dgap",
                     prompts_jsonl=pj)
    assert r["example_indices"] == [1, 2]
    assert r["num_examples_diagnosed"] == 2


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

    for extra, name in (
            (["--rollout-mode", "both"], "s_clean"),
            (["--rollout-mode", "both", "--inject-free-divergence-step", "2"],
             "s_free"),
            (["--rollout-mode", "both", "--inject-plaintext-repro-mismatch"],
             "s_block"),
            (["--rollout-mode", "teacher_forcing", "--resident-folded-weights",
              "--compare-nonresident"], "s_res")):
        _, r = _run(extra, tmp_path, name)
        assert r["audit_passed"] is True
        assert r["tee_used_on_gpu"] is False
        assert r["worker_has_mask_secrets"] is False
        assert r["worker_has_raw_lora"] is False
        assert r["leaked_secret_fields"] == []
        assert r["schedule_secret_leaked_to_gpu"] is False
        assert r["gpu_request_contains_schedule_secret"] is False
        assert r["plaintext_logits_or_sampling_on_gpu"] is False
        assert r["pad_disabled_for_correctness"] is False
        keys = set(_walk_keys(r))
        assert keys.isdisjoint(FORBIDDEN_WIRE_FIELDS)
