"""Tests for the current-vs-trusted_shortcut generation-backend evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pllo.benchmarks import generation_backend_eval as gbe
from pllo.benchmarks import ifeval_scoring


TINY_CONFIG = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 8,
    "vocab_size": 256, "max_position_embeddings": 1024, "_name_or_path": "tiny",
}


# --------------------------------------------------------------------------- #
# 1. trusted_shortcut -> amulet_migrated
# --------------------------------------------------------------------------- #
def test_trusted_shortcut_maps_to_amulet_migrated():
    r = gbe.resolve_backend("trusted_shortcut")
    assert r["nonlinear_backend"] == "trusted_shortcut"
    assert r["op_backend"] == "amulet_migrated"
    # alias resolves the same
    assert gbe.resolve_backend("amulet_migrated")["op_backend"] == \
        "amulet_migrated"
    # current stays current
    assert gbe.resolve_backend("current")["op_backend"] == "current"


# --------------------------------------------------------------------------- #
# 2. no silent fallback
# --------------------------------------------------------------------------- #
def test_no_silent_fallback_detected_when_worker_is_current():
    health = {"nonlinear_backend": "current",
              "nonlinear_execution_evidence": {
                  "nonlinear_op_backend": "current"}}
    v = gbe.verify_backend_execution(health, "trusted_shortcut",
                                     "amulet_migrated")
    assert v["backend_verification_passed"] is False
    assert v["silent_fallback_detected"] is True
    assert v["reasons"]


def test_backend_verification_passes_on_match():
    health = {"nonlinear_backend": "trusted_shortcut",
              "nonlinear_execution_evidence": {
                  "nonlinear_op_backend": "amulet_migrated"}}
    v = gbe.verify_backend_execution(health, "trusted_shortcut",
                                     "amulet_migrated")
    assert v["backend_verification_passed"] is True
    assert v["silent_fallback_detected"] is False


# --------------------------------------------------------------------------- #
# 3. dimension audit schema
# --------------------------------------------------------------------------- #
def test_dimension_audit_schema():
    a = gbe.build_dimension_audit(
        TINY_CONFIG, prompt_len=10, max_new_tokens=8,
        nonlinear_backend="trusted_shortcut", lift_k=2)
    for sec in gbe.REQUIRED_AUDIT_SECTIONS:
        assert sec in a, f"missing section {sec}"
    m = a["model"]
    for k in ("num_layers", "hidden_size", "intermediate_size",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "dtype", "device"):
        assert k in m
    # attention template + KV + MLP + LM head fields
    at = a["attention"]["per_layer_template"]
    for k in ("q_proj_weight", "k_proj_weight", "v_proj_weight", "o_proj_weight",
              "attention_score", "attention_probability", "residual_output"):
        assert k in at and at[k]["shape"] is not None
    kv = a["kv_cache"]
    assert kv["key_cache_after_prefill"]["shape"] == [1, 2, 10, 8]
    assert kv["cache_num_key_value_heads"] == 2
    mlp = a["mlp"]["per_layer_template"]
    for k in ("gate_proj_weight", "up_proj_weight", "down_proj_weight",
              "nonlinear_input", "down_output"):
        assert k in mlp
    lm = a["lm_head"]
    assert lm["lm_head_weight"]["shape"] == [256, 64]
    # nonlinear evidence: trusted_shortcut -> silu is lifted with lift_k
    silu = [o for o in a["nonlinear"]["per_layer_ops"]
            if o["op_type"] == "silu"][0]
    assert silu["backend_used"] == "amulet_migrated"
    assert silu["lifted_representation_used"] is True
    assert silu["lift_k"] == 2
    assert silu["lifted_tensor_shape"] == [1, 10, 128, 2]
    # rows flatten cleanly
    rows = gbe.dimension_audit_to_rows(a)
    assert len(rows) > 0 and all("shape" in r for r in rows)


def test_dimension_audit_marks_measured_vs_derived():
    a = gbe.build_dimension_audit(
        TINY_CONFIG, prompt_len=10, max_new_tokens=8,
        nonlinear_backend="current",
        measured={"input_ids": [1, 10], "recovered_logits": [1, 256]})
    assert a["inputs"]["input_ids"]["source"] == "measured"
    assert a["inputs"]["prefill_hidden"]["source"] == "derived_from_config"
    assert a["lm_head"]["recovered_logits"]["source"] == "measured"


# --------------------------------------------------------------------------- #
# 4. nonlinear evidence required
# --------------------------------------------------------------------------- #
def test_nonlinear_evidence_required_flags_missing():
    s = gbe.summarize_nonlinear_evidence({}, "trusted_shortcut",
                                         "amulet_migrated")
    assert s["nonlinear_execution_evidence_missing"] is True
    assert s["diagnostics"]


def test_nonlinear_evidence_present_passes():
    ev = {
        "nonlinear_backend": "trusted_shortcut",
        "nonlinear_op_backend": "amulet_migrated",
        "amulet_lift_executed": True, "amulet_backend_used": True,
        "lifted_nonlinear_ops_count": 56, "lift_k": 2,
        "lifted_gpu_bytes": 123456,
        "trusted_nonlinear_ops_count": 0,
        "migrated_ops_by_type": {"softmax": 28, "rmsnorm": 57, "silu": 28},
        "nonlinear_execution_status": "lifted_on_accelerator",
    }
    s = gbe.summarize_nonlinear_evidence(ev, "trusted_shortcut",
                                         "amulet_migrated")
    assert s["nonlinear_execution_evidence_missing"] is False
    assert s["amulet_real_path_executed"] is True
    assert s["trusted_shortcut_tag_only"] is False
    assert s["nonlinear_ops_count"]["softmax"] == 28


# --------------------------------------------------------------------------- #
# 5. generation eval smoke on a stub predictor
# --------------------------------------------------------------------------- #
class _StubPredictor:
    def __init__(self):
        self._captured = [[0.1] * 256]

    def enable_decode_profiling(self, *a, **k):
        pass

    def enable_logits_capture(self, *a, **k):
        pass

    def captured_logits(self):
        return self._captured

    def generate(self, prompt):
        toks = [10, 11, 12, 13]
        return {"text": "hello world answer", "token_ids": toks,
                "finish_reason": "eos", "stopped_by_eos": True,
                "prompt_token_count": 7}

    def stats(self):
        return {"boundary_calls": 5, "gpu_calls": 4, "trusted_bytes": 100,
                "gpu_bytes": 200, "finish_reason": "eos",
                "nonlinear_backend": "current"}

    def close(self):
        pass


def _make_args(**over):
    base = dict(
        backend="current", model_path="tiny", tokenizer_path=None,
        folded_package_path=None, embedding_path=None,
        gpu_worker_url="http://x", prompt_file=None, max_new_tokens=8,
        temperature=0.0, top_p=1.0, do_sample=False, seq_len=1024,
        dtype="bfloat16", device="cpu", use_chat_template=False, limit=0,
        audit_dimensions=True, audit_nonlinear=True,
        expected_nonlinear_backend="current", expected_op_backend=None,
        compare_baseline_dir=None, allow_missing_evidence=False,
        no_worker_timing=True, profile=False, output_dir=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_generation_eval_smoke(tmp_path):
    import scripts.run_generation_backend_eval as script  # noqa

    pf = tmp_path / "prompts.jsonl"
    pf.write_text(
        '{"id": "p1", "category": "en_qa", "prompt": "What is 2+2?"}\n'
        '{"id": "p2", "category": "code", "prompt": "def f(): pass"}\n')
    out = tmp_path / "run"
    args = _make_args(prompt_file=str(pf), output_dir=str(out))

    health = {"nonlinear_backend": "current",
              "nonlinear_execution_evidence": {
                  "nonlinear_op_backend": "current"},
              "peak_gpu_memory_mb": 42}
    summary = script.run_eval(
        args,
        predictor_factory=lambda a, r: _StubPredictor(),
        config_loader=lambda mp: dict(TINY_CONFIG),
        manifest_loader=lambda pd: {"seq_len": 1024, "nonlinear_backend":
                                    "current", "num_shards": 3},
        verify_fn=lambda pd: {"package_valid": True},
        health_fn=lambda url: health)

    assert (out / "generations.jsonl").exists()
    assert (out / "report.md").exists()
    assert (out / "dimension_audit.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "nonlinear_execution_evidence.json").exists()
    recs = [json.loads(x) for x in open(out / "generations.jsonl") if x.strip()]
    assert len(recs) == 2
    assert recs[0]["generated_text"] == "hello world answer"
    assert recs[0]["output_length"] == 4
    assert summary["exit_code"] == 0
    # measured input_ids captured from the stub's prompt_token_count
    audit = json.loads((out / "dimension_audit.json").read_text())
    assert audit["inputs"]["input_ids"]["source"] == "measured"
    assert audit["inputs"]["input_ids"]["shape"] == [1, 7]


def test_generation_eval_smoke_trusted_shortcut_missing_evidence_fails(tmp_path):
    import scripts.run_generation_backend_eval as script  # noqa

    pf = tmp_path / "prompts.jsonl"
    pf.write_text('{"id": "p1", "prompt": "hi"}\n')
    out = tmp_path / "run_ts"
    args = _make_args(backend="trusted_shortcut",
                      expected_nonlinear_backend="trusted_shortcut",
                      expected_op_backend="amulet_migrated",
                      prompt_file=str(pf), output_dir=str(out))
    # worker claims trusted_shortcut but returns NO lift evidence -> must fail
    health = {"nonlinear_backend": "trusted_shortcut",
              "nonlinear_execution_evidence": {}}
    summary = script.run_eval(
        args,
        predictor_factory=lambda a, r: _StubPredictor(),
        config_loader=lambda mp: dict(TINY_CONFIG),
        manifest_loader=lambda pd: {"seq_len": 1024},
        verify_fn=lambda pd: {},
        health_fn=lambda url: health)
    assert summary["nonlinear_execution_evidence_missing"] is True
    assert summary["exit_code"] == 3


# --------------------------------------------------------------------------- #
# 6. seq-len compatibility
# --------------------------------------------------------------------------- #
def test_seq_len_compatibility():
    ok, _ = gbe.seq_len_compatible(1024, 100, 64)
    assert ok is True
    bad, msg = gbe.seq_len_compatible(128, 500, 64)
    assert bad is False and "exceeds package seq_len" in msg
    warn, msg2 = gbe.seq_len_compatible(128, 100, 64)
    assert warn is True and "horizon" not in msg2 or "may truncate" in msg2


# --------------------------------------------------------------------------- #
# extra: IFEval approximate scorer sanity
# --------------------------------------------------------------------------- #
def test_ifeval_scorer_basic():
    meta = {"instruction_id_list": ["punctuation:no_comma",
                                    "change_case:english_lowercase"],
            "kwargs": [{}, {}]}
    good = score = ifeval_scoring.score_response(meta, "hello there world")
    assert good["prompt_strict"] is True
    bad = ifeval_scoring.score_response(meta, "Hello, WORLD")
    assert bad["prompt_strict"] is False
    agg = ifeval_scoring.aggregate_ifeval([good, bad])
    assert agg["coverage_pct"] == 100.0
    assert 0.0 <= agg["strict_prompt_acc"] <= 1.0


def test_ifeval_number_words_and_forbidden():
    r = ifeval_scoring.score_instruction(
        "length_constraints:number_words", "one two three four five",
        {"relation": "at least", "num_words": 3})
    assert r["strict"] is True
    r2 = ifeval_scoring.score_instruction(
        "keywords:forbidden_words", "a clean sentence",
        {"forbidden_words": ["banana"]})
    assert r2["strict"] is True
    r3 = ifeval_scoring.score_instruction(
        "keywords:forbidden_words", "i ate a banana today",
        {"forbidden_words": ["banana"]})
    assert r3["strict"] is False


def test_ifeval_unsupported_instruction_excluded():
    r = ifeval_scoring.score_instruction("some:unknown_type", "text", {})
    assert r["supported"] is False
    agg = ifeval_scoring.aggregate_ifeval(
        [ifeval_scoring.score_response(
            {"instruction_id_list": ["some:unknown_type"], "kwargs": [{}]},
            "x")])
    # nothing supported -> coverage 0, accuracies None (not silently 1.0)
    assert agg["coverage_pct"] == 0.0
    assert agg["strict_prompt_acc"] is None
