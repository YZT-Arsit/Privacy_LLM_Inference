"""Stage 5.5b — tests for the real-token-prompted trace collector.

All tests run with ``attempt_tokenizer_load=False`` and
``attempt_real_model_load=False`` so the test suite never hits the
HuggingFace network.
"""

from __future__ import annotations

import json

import torch

from pllo.experiments.real_token_trace import (
    DEFAULT_PROMPTS,
    DEFAULT_TARGET_TENSORS,
    RealTokenTraceConfig,
    collect_real_token_traces,
)


def _small_config(**overrides):
    cfg = dict(
        seed=2026,
        attempt_real_model_load=False,
        attempt_tokenizer_load=False,
        allow_synthetic_fallback=True,
        max_layers=2,
        max_new_tokens=2,
        prompt_max_length=6,
        num_prompts=4,
        use_pad=True,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        synthetic_vocab_size=64,
        synthetic_hidden_size=16,
        synthetic_intermediate_size=32,
        synthetic_num_attention_heads=4,
        synthetic_num_key_value_heads=2,
        synthetic_head_dim=4,
    )
    cfg.update(overrides)
    return RealTokenTraceConfig(**cfg)


def test_synthetic_token_fallback_does_not_require_network() -> None:
    pkg = collect_real_token_traces(_small_config())
    assert pkg["tokenizer_loading"]["tokenizer_status"] == "not_requested"
    assert pkg["prompt_summary"]["token_source"] == "synthetic_token_ids"
    assert pkg["model_loading"]["load_status"] in (
        "synthetic_only", "skipped",
    )


def test_collect_returns_top_level_sections() -> None:
    pkg = collect_real_token_traces(_small_config())
    for key in (
        "config", "model_loading", "tokenizer_loading", "source",
        "prompt_summary", "trace_summary", "traces", "generation_summary",
        "metadata", "block_spec_summary", "per_prompt_metadata",
        "decode_step_log",
    ):
        assert key in pkg, f"missing top-level key {key!r}"


def test_trace_summary_includes_prefill_scope() -> None:
    pkg = collect_real_token_traces(_small_config())
    prefill = pkg["trace_summary"]["prefill"]
    assert prefill, "prefill trace summary is empty"
    # Must cover at least the SwiGLU island tensors.
    for name in ("gate", "up", "swiglu_intermediate", "post_island"):
        assert name in prefill, f"missing prefill tensor {name!r}"


def test_decode_traces_present_when_max_new_tokens_gt_0() -> None:
    pkg = collect_real_token_traces(_small_config(max_new_tokens=3))
    decode = pkg["trace_summary"]["decode"]
    assert decode, "decode trace summary is empty"
    for name in ("gate", "up", "swiglu_intermediate"):
        assert name in decode, f"missing decode tensor {name!r}"


def test_generation_token_match_summary_present() -> None:
    pkg = collect_real_token_traces(_small_config(max_new_tokens=3))
    g = pkg["generation_summary"]
    assert "mean_token_match_rate" in g
    # Synthetic prefill / decode are deterministic and Stage 6.4c verified
    # exact match — the obf / plain token streams must agree.
    assert g["all_sequences_exact_match"] is True
    assert g["mean_token_match_rate"] == 1.0


def test_trace_summary_excludes_raw_tensors() -> None:
    pkg = collect_real_token_traces(_small_config())
    text = json.dumps(
        {
            "trace_summary": pkg["trace_summary"],
            "prompt_summary": pkg["prompt_summary"],
            "metadata": pkg["metadata"],
            "generation_summary": pkg["generation_summary"],
        },
        default=str,
    )
    assert "tensor(" not in text
    # No giant numeric arrays — only scalar statistics.
    for ch in pkg["trace_summary"]["prefill"].values():
        assert isinstance(ch["plain_statistics"]["fingerprint_sha256_prefix"], str)


def test_use_pad_true_full_bundle_prefill_allclose() -> None:
    pkg = collect_real_token_traces(
        _small_config(use_pad=True, mitigation_bundle="fresh_perm_plus_sandwich_plus_pad")
    )
    assert pkg["metadata"]["all_prefill_allclose"] is True


def test_synthetic_prompt_input_ids_deterministic() -> None:
    pkg_a = collect_real_token_traces(_small_config())
    pkg_b = collect_real_token_traces(_small_config())
    # Same seed + same shape ⇒ same prompt fingerprints.
    fp_a = pkg_a["trace_summary"]["prefill"]["boundary_input"][
        "plain_statistics"
    ]["fingerprint_sha256_prefix"]
    fp_b = pkg_b["trace_summary"]["prefill"]["boundary_input"][
        "plain_statistics"
    ]["fingerprint_sha256_prefix"]
    assert fp_a == fp_b


def test_decode_step_log_records_position_and_cache_seq_len() -> None:
    pkg = collect_real_token_traces(_small_config(max_new_tokens=3))
    log = pkg["decode_step_log"]
    assert log, "decode_step_log is empty"
    for entry in log:
        for step in entry["steps"]:
            assert "position" in step
            assert "cache_seq_len_before" in step
            assert "cache_seq_len_after" in step
            assert step["cache_seq_len_after"] >= step["cache_seq_len_before"]


def test_prompt_set_uses_default_prompts() -> None:
    pkg = collect_real_token_traces(_small_config(num_prompts=4))
    assert pkg["prompt_summary"]["prompts_used"] == list(DEFAULT_PROMPTS[:4])


def test_target_tensor_inventory_matches_default() -> None:
    pkg = collect_real_token_traces(_small_config())
    prefill = pkg["trace_summary"]["prefill"]
    for name in DEFAULT_TARGET_TENSORS:
        assert name in prefill, f"missing default target {name!r}"
