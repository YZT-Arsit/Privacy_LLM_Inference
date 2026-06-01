"""Stage 5.5 — real activation trace collector tests."""

from __future__ import annotations

import json
import re

import pytest
import torch

from pllo.experiments.real_activation_trace import (
    DEFAULT_TARGET_TENSORS,
    RealActivationTraceConfig,
    collect_real_activation_traces,
)


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


@pytest.fixture(scope="module")
def trace_pkg() -> dict:
    cfg = RealActivationTraceConfig(
        num_samples=128, batch_size=2, seq_len=8,
        synthetic_hidden_size=32, synthetic_intermediate_size=64,
        synthetic_num_attention_heads=4, synthetic_num_key_value_heads=2,
        synthetic_head_dim=8,
    )
    return collect_real_activation_traces(cfg)


# ---------------------------------------------------------------------------
# Synthetic fallback / no network
# ---------------------------------------------------------------------------


def test_synthetic_fallback_does_not_hit_network(trace_pkg) -> None:
    assert trace_pkg["source"] == "synthetic_block"
    assert trace_pkg["model_loading"]["load_status"] == "synthetic_only"
    assert trace_pkg["model_loading"]["candidates_tried"] == []


def test_top_level_keys_present(trace_pkg) -> None:
    for k in (
        "config", "model_loading", "block_spec",
        "traces", "trace_summary", "metadata", "source",
    ):
        assert k in trace_pkg, f"missing top-level key {k!r}"


# ---------------------------------------------------------------------------
# Trace_summary covers required tensors
# ---------------------------------------------------------------------------


def test_trace_summary_covers_required_tensors(trace_pkg) -> None:
    required = {"gate", "up", "swiglu_intermediate", "post_island"}
    assert required <= set(trace_pkg["trace_summary"].keys()), (
        trace_pkg["trace_summary"].keys()
    )


def test_trace_summary_records_shape_and_fingerprint(trace_pkg) -> None:
    for name, s in trace_pkg["trace_summary"].items():
        assert "plain_shape" in s
        assert "visible_shape" in s
        assert "feature_dim" in s
        assert "num_samples" in s
        assert "plain_statistics" in s
        assert "visible_statistics" in s
        assert (
            "fingerprint_sha256_prefix" in s["plain_statistics"]
        )
        assert (
            "fingerprint_sha256_prefix" in s["visible_statistics"]
        )


# ---------------------------------------------------------------------------
# JSON-safe — no raw tensor, no overlong arrays
# ---------------------------------------------------------------------------


def test_json_summary_excludes_raw_tensors(trace_pkg) -> None:
    json_safe = {
        "config": trace_pkg["config"],
        "model_loading": trace_pkg["model_loading"],
        "block_spec": trace_pkg["block_spec"],
        "trace_summary": trace_pkg["trace_summary"],
        "metadata": trace_pkg["metadata"],
        "source": trace_pkg["source"],
    }
    blob = json.dumps(json_safe)
    assert "tensor(" not in blob
    assert "torch.Tensor" not in blob
    assert _LONG_NUMBER_ARRAY.search(blob) is None


def test_traces_dict_holds_tensors_in_memory(trace_pkg) -> None:
    for name, pair in trace_pkg["traces"].items():
        assert isinstance(pair["plain"], torch.Tensor)
        assert isinstance(pair["visible"], torch.Tensor)


# ---------------------------------------------------------------------------
# Full-bundle final allclose
# ---------------------------------------------------------------------------


def test_full_bundle_use_pad_true_allclose() -> None:
    cfg = RealActivationTraceConfig(
        num_samples=64, batch_size=2, seq_len=4,
        synthetic_hidden_size=32, synthetic_intermediate_size=64,
        synthetic_num_attention_heads=4, synthetic_num_key_value_heads=2,
        synthetic_head_dim=8,
        use_pad=True,
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    r = collect_real_activation_traces(cfg)
    assert r["metadata"]["all_sessions_allclose"] is True
    assert r["metadata"]["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"


# ---------------------------------------------------------------------------
# Flattened shape sanity
# ---------------------------------------------------------------------------


def test_target_tensor_flatten_shapes_reasonable(trace_pkg) -> None:
    for name, pair in trace_pkg["traces"].items():
        plain = pair["plain"]
        visible = pair["visible"]
        assert plain.ndim == 2
        assert visible.ndim == 2
        assert plain.shape == visible.shape


def test_per_tensor_feature_dims_match_block_spec(trace_pkg) -> None:
    spec = trace_pkg["block_spec"]
    H = spec["hidden_size"]
    I = spec["intermediate_size"]
    D = spec["head_dim"]
    summary = trace_pkg["trace_summary"]
    if "gate" in summary:
        assert summary["gate"]["feature_dim"] == I
    if "up" in summary:
        assert summary["up"]["feature_dim"] == I
    if "swiglu_intermediate" in summary:
        assert summary["swiglu_intermediate"]["feature_dim"] == I
    if "post_island" in summary:
        assert summary["post_island"]["feature_dim"] == H
    if "boundary_input" in summary:
        assert summary["boundary_input"]["feature_dim"] == H
    if "q" in summary:
        assert summary["q"]["feature_dim"] == D
    if "k" in summary:
        assert summary["k"]["feature_dim"] == D
    if "v" in summary:
        assert summary["v"]["feature_dim"] == D


def test_default_target_tensors_match_spec_inventory() -> None:
    for required in ("gate", "up", "swiglu_intermediate", "post_island"):
        assert required in DEFAULT_TARGET_TENSORS
