"""Dry-run tests for the E1/E2 no-LoRA Qwen generation runners (tiny Qwen2, CPU).

Validates the runner end-to-end on a tiny random model (never a paper result):
all three decoding modes produce metrics, masking preserves correctness
(plain-vs-masked top-1 == 1.0), and the protocol fields hold
(``tee_used_on_gpu=False``, empty plaintext/secret field lists).

Run: python -m pytest tests/test_qwen_generation_experiments.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pllo.experiments.qwen_generation_experiments import (  # noqa: E402
    build_context_fields,
    protocol_accounting,
    run_e1_nolora,
    run_e2_token_scaling,
    teacher_forced_block,
)
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_e1_script():
    spec = importlib.util.spec_from_file_location(
        "e1cli", REPO_ROOT / "scripts" / "run_qwen7b_e1_nolora_generation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _cfg(mc, seq_len=6, n=4):
    return MemoryOptimizedConfig(
        num_layers=mc.num_hidden_layers, batch_size=1, seq_len=seq_len,
        max_new_tokens=n, device="cpu", dtype="float32", folding_dtype="float32",
        folded_weight_device="cpu", mlp_down_chunk_size=64, seed=2035)


def test_protocol_accounting_positive() -> None:
    a = protocol_accounting(128, 3584, 152064, 64, "bfloat16")
    assert a["gpu_bytes"] > 0 and a["trusted_bytes"] > 0
    assert a["boundary_calls"]["embed_and_mask"] == 64
    assert a["boundary_calls"]["recover_logits"] == 64


def test_e1_all_modes_run_and_masking_is_correct() -> None:
    model, mc = _tiny()
    cfg = _cfg(mc, seq_len=6, n=4)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, mc.vocab_size, (1, 6), generator=g)
    r = run_e1_nolora(model, mc, ids, cfg,
                      modes=("greedy", "teacher_forced", "sampling"), topk=3)
    assert r["tee_used_on_gpu"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["trusted_bytes"] > 0 and r["gpu_bytes"] > 0

    tf = r["modes"]["teacher_forced"]
    # masking is exact -> recovered == plain -> plain/masked top-1 agree
    assert tf["teacher_forced_top1_match_rate_plain_masked"] >= 0.99
    assert 0.0 <= tf["teacher_forced_top1_match_rate_hf_masked"] <= 1.0
    assert tf["teacher_forced_steps_evaluated"] == 4
    assert 0.0 <= tf["topk_overlap"] <= 1.0
    assert tf["logits_max_abs_error"] < 1e-1     # tiny fp masking error

    g_ = r["modes"]["greedy"]
    assert g_["plain_vs_masked_token_match_rate"] is not None
    assert g_["latency_s"] >= 0.0

    s = r["modes"]["sampling"]
    assert s["generation_completion_tokens"] == 4
    assert s["temperature"] == 0.7 and s["top_p"] == 0.9


def test_e2_token_scaling_rows() -> None:
    model, mc = _tiny()
    cfg = _cfg(mc, seq_len=6, n=1)
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, mc.vocab_size, (1, 6), generator=g)
    out = run_e2_token_scaling(model, mc, ids, cfg, token_grid=(1, 2, 4),
                               modes=("greedy", "teacher_forced"), topk=3)
    assert out["stage"] == "E2_token_scaling"
    assert [row["max_new_tokens"] for row in out["rows"]] == [1, 2, 4]
    for row in out["rows"]:
        assert row["tee_used_on_gpu"] is False
        assert row["gpu_bytes"] > 0 and row["trusted_bytes"] > 0
        assert "tf_top1_plain_masked" in row


# --- reporting-fix regression tests (seq_len / attention_mask / padding) ----

def test_hf_generate_passes_attention_mask_when_pad_eq_eos() -> None:
    """The HF generation path must ALWAYS receive an attention_mask, even when
    pad_token_id == eos_token_id (otherwise transformers warns + may misbehave)."""
    from pllo.experiments import qwen_generation_experiments as Q
    model, mc = _tiny()
    model.config.pad_token_id = model.config.eos_token_id = 0
    ids = torch.randint(0, mc.vocab_size, (1, 5))
    captured = {}
    orig = model.generate

    def spy(*a, **kw):
        captured.update(kw)
        return orig(*a, **kw)

    model.generate = spy
    Q._hf_generate(model, ids, 2, do_sample=False)
    assert "attention_mask" in captured
    am = captured["attention_mask"]
    assert tuple(am.shape) == (1, 5) and int(am.sum()) == 5    # all-ones over real


def test_report_distinguishes_requested_vs_effective_seq_len() -> None:
    model, mc = _tiny()
    ids = torch.randint(0, mc.vocab_size, (1, 6))             # effective_len=6
    cfg = _cfg(mc, seq_len=6, n=2)
    ctx = build_context_fields(
        seq_len_requested=128, effective_prompt_len=6, pad_to_seq_len=False,
        padded_seq_len=None, decode_start_index=6, attention_mask_used=True,
        dtype="float32", model_name="Qwen2.5-7B-Instruct", model_path=None,
        prompt_text="hello", dry_run=True)
    r = run_e1_nolora(model, mc, ids, cfg, modes=("greedy",), topk=3, context=ctx)
    assert r["seq_len_requested"] == 128          # the --seq-len budget
    assert r["effective_prompt_len"] == 6         # real tokens fed to masked path
    assert r["seq_len"] == 6                       # back-compat alias == effective
    assert r["padded_seq_len"] is None
    assert r["context_mode"] == "natural_prompt"
    assert r["attention_mask_used"] is True
    assert r["decode_start_index"] == 6
    assert r["prompt_sha256_16"] and len(r["prompt_sha256_16"]) == 16


def test_pad_to_seq_len_true_produces_padded_seq_len_equal_requested() -> None:
    demo = _load_e1_script()
    model, mc = _tiny()
    ids = torch.randint(0, mc.vocab_size, (1, 6))

    class _Args:
        seq_len = 32
        pad_to_seq_len = "true"
        model_name = "Qwen2.5-7B-Instruct"
        model_path = None

    attn, padded_ids, padded_mask, ctx = demo._build_context_inputs(
        _Args(), ids, None, mc, "hello", "cpu", "float32", True)
    assert ctx["padded_seq_len"] == ctx["seq_len_requested"] == 32
    assert ctx["effective_prompt_len"] == 6
    assert ctx["decode_start_index"] == 6        # still the real prompt length
    assert padded_ids.shape[1] == 32             # left-padded to the budget
    assert int(padded_mask[:, :32 - 6].sum()) == 0   # pad positions masked out
    assert int(padded_mask[:, 32 - 6:].sum()) == 6   # real positions attended


def test_pad_mode_padding_invariance_holds() -> None:
    """fixed-padded mode must not change results: HF padded+mask greedy == HF
    unpadded greedy, and the masked/plain comparisons stay on real tokens."""
    demo = _load_e1_script()
    model, mc = _tiny()
    ids = torch.randint(0, mc.vocab_size, (1, 6))

    class _Args:
        seq_len = 24
        pad_to_seq_len = "true"
        model_name = "m"
        model_path = None

    attn, padded_ids, padded_mask, ctx = demo._build_context_inputs(
        _Args(), ids, None, mc, "p", "cpu", "float32", True)
    cfg = _cfg(mc, seq_len=6, n=3)
    r = run_e1_nolora(model, mc, ids, cfg, modes=("greedy", "teacher_forced"),
                      topk=3, context=ctx, attention_mask=attn,
                      padded_input_ids=padded_ids, padded_attention_mask=padded_mask)
    assert r["padding_invariance_hf_token_match_rate"] == 1.0
    assert r["padding_invariance_ok"] is True


_E1_FLAT_METRICS = (
    "teacher_forced_top1_match_rate_hf_plain",
    "teacher_forced_top1_match_rate_hf_masked",
    "teacher_forced_top1_match_rate_plain_masked",
    "plain_vs_masked_token_match_rate", "topk_overlap",
    "logits_max_abs_error", "logits_mean_abs_error",
    "logits_relative_l2_error", "latency_s",
)
# every field each E2 row must expose (presence); a subset must be non-None when
# the contributing mode ran.
_E2_ROW_FIELDS = (
    "max_new_tokens", "seq_len_requested", "effective_prompt_len",
    "padded_seq_len", "decode_start_index", "attention_mask_used",
    "teacher_forced_top1_match_rate_hf_plain",
    "teacher_forced_top1_match_rate_hf_masked",
    "teacher_forced_top1_match_rate_plain_masked",
    "plain_vs_masked_token_match_rate", "topk_overlap", "logits_max_abs_error",
    "logits_mean_abs_error", "logits_relative_l2_error", "latency_s",
    "peak_gpu_memory_mb", "trusted_bytes", "gpu_bytes", "boundary_calls",
    "gpu_visible_plaintext_fields", "leaked_secret_fields", "tee_used_on_gpu",
)


def _json_roundtrip(obj):
    import json
    return json.loads(json.dumps(obj, default=str))


def test_e1_json_top_level_paper_metrics_not_none() -> None:
    """E1 JSON top-level paper-critical metrics are populated (not None) when the
    contributing modes ran -- this is the schema fix the paper reader needs."""
    model, mc = _tiny()
    cfg = _cfg(mc, seq_len=6, n=3)
    ids = torch.randint(0, mc.vocab_size, (1, 6))
    r = _json_roundtrip(run_e1_nolora(
        model, mc, ids, cfg, modes=("greedy", "teacher_forced"), topk=3))
    for k in _E1_FLAT_METRICS:
        assert r.get(k) is not None, f"E1 top-level {k} is None"
    # nested detail retained
    assert "teacher_forced" in r["modes"] and "greedy" in r["modes"]
    assert r["latency_s"] == r["greedy_latency_s"]   # latency == greedy gen time


def test_e1_metric_none_only_when_mode_absent() -> None:
    """Greedy-only: plain_vs_masked is populated; teacher-forced fields are None
    (None is meaningful = mode not run, not a dropped value)."""
    model, mc = _tiny()
    cfg = _cfg(mc, seq_len=6, n=3)
    ids = torch.randint(0, mc.vocab_size, (1, 6))
    r = _json_roundtrip(run_e1_nolora(model, mc, ids, cfg, modes=("greedy",),
                                      topk=3))
    assert r["plain_vs_masked_token_match_rate"] is not None
    assert r["latency_s"] is not None
    assert r["teacher_forced_top1_match_rate_hf_plain"] is None
    assert r["topk_overlap"] is None


def test_e2_rows_expose_flattened_fields() -> None:
    model, mc = _tiny()
    cfg = _cfg(mc, seq_len=6, n=1)
    ids = torch.randint(0, mc.vocab_size, (1, 6))
    out = _json_roundtrip(run_e2_token_scaling(
        model, mc, ids, cfg, token_grid=(1, 4),
        modes=("greedy", "teacher_forced"), topk=3))
    assert [row["max_new_tokens"] for row in out["rows"]] == [1, 4]
    for row in out["rows"]:
        for k in _E2_ROW_FIELDS:
            assert k in row, f"E2 row missing {k}"
        # metrics from modes that ran must be populated
        for k in _E1_FLAT_METRICS:
            assert row[k] is not None, f"E2 row {k} is None"
        assert row["tee_used_on_gpu"] is False
        assert row["gpu_visible_plaintext_fields"] == []


def test_teacher_forced_same_prefix_semantics_across_hf_plain_masked() -> None:
    """HF, plain, and masked are scored on the SAME real-token prefix; with exact
    masking and a tiny fp model all three agree at top-1."""
    model, mc = _tiny()
    cfg = _cfg(mc, seq_len=6, n=4)
    prompt = torch.randint(0, mc.vocab_size, (1, 6))
    ref = torch.randint(0, mc.vocab_size, (1, 4))
    tf = teacher_forced_block(model, mc, prompt, ref, cfg, topk=3)
    assert tf["teacher_forced_steps_evaluated"] == 4
    assert tf["teacher_forced_top1_match_rate_hf_plain"] == 1.0
    assert tf["teacher_forced_top1_match_rate_plain_masked"] >= 0.99
    assert tf["logits_max_abs_error"] < 1e-1
