"""Stage 6.6 tests -- HF LLaMA/Qwen single-block adapter (CPU, float64).

Transformers is optional: tests that need it use ``importorskip`` and the
family's layer class, so nothing fails merely because transformers (or a
given model family) is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from pllo.experiments.hf_single_block_probe import (
    HFSingleBlockProbeConfig,
    run_hf_single_block_probe,
)
from pllo.hf_wrappers.llama_qwen_single_block import (
    extract_hf_single_block_weights,
    generate_hf_single_block_masks,
    has_transformers,
    hf_single_block_masked_decode,
    hf_single_block_masked_prefill,
    infer_config_from_hf_layer,
    make_random_hf_decoder_layer,
)

ATOL = 1e-8
RTOL = 1e-8
DTYPE = torch.float64


def _layer_or_skip(family: str):
    pytest.importorskip("transformers")
    try:
        return make_random_hf_decoder_layer(family, seed=2029)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{family} decoder layer unavailable: {exc!r}")


def _prep(family: str):
    layer, mc = _layer_or_skip(family)
    cfg = infer_config_from_hf_layer(layer, mc, dtype=DTYPE)
    weights = extract_hf_single_block_weights(layer, DTYPE)
    masks = generate_hf_single_block_masks(cfg, seed=2029)
    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(1, 8, cfg.hidden_size, generator=g, dtype=DTYPE)
    return layer, cfg, weights, masks, x, g


# 1.
def test_hf_wrapper_imports_are_optional() -> None:
    # Importing the module + calling has_transformers must never raise,
    # regardless of whether transformers is installed.
    assert isinstance(has_transformers(), bool)


# 2.
def test_extract_linear_weight_transposes_row_vector_convention() -> None:
    lin = torch.nn.Linear(4, 6, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
        lin.bias.copy_(torch.arange(6, dtype=torch.float32))

    class _FakeAttn:
        q_proj = k_proj = v_proj = o_proj = lin

    class _FakeMLP:
        gate_proj = up_proj = down_proj = lin

    class _FakeLayer:
        self_attn = _FakeAttn()
        mlp = _FakeMLP()

        class input_layernorm:  # noqa: N801
            weight = torch.ones(4)

        class post_attention_layernorm:  # noqa: N801
            weight = torch.ones(4)

    w = extract_hf_single_block_weights(_FakeLayer(), DTYPE)
    # Row-vector W must be the transpose of the HF [out, in] weight.
    torch.testing.assert_close(
        w.q_proj_weight, lin.weight.detach().to(DTYPE).t(), atol=ATOL,
        rtol=RTOL)
    assert w.q_proj_weight.shape == (4, 6)
    torch.testing.assert_close(
        w.q_proj_bias, lin.bias.detach().to(DTYPE), atol=ATOL, rtol=RTOL)
    # Original layer weights must be untouched (float32 preserved).
    assert lin.weight.dtype == torch.float32


# 3.
def test_random_llama_layer_extraction_shapes() -> None:
    _, cfg, w, _, _, _ = _prep("llama")
    assert cfg.model_type == "llama"
    assert cfg.head_dim == 8 and cfg.head_dim % 2 == 0
    assert w.q_proj_weight.shape == (cfg.hidden_size, cfg.num_heads * cfg.head_dim)
    assert w.k_proj_weight.shape == (
        cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
    assert w.gate_proj_weight.shape == (cfg.hidden_size, cfg.intermediate_size)
    assert w.down_proj_weight.shape == (cfg.intermediate_size, cfg.hidden_size)
    assert cfg.attention_bias is False  # LLaMA: no attention bias


# 4.
def test_random_qwen2_layer_extraction_shapes() -> None:
    _, cfg, w, _, _, _ = _prep("qwen2")
    assert cfg.model_type == "qwen2"
    assert cfg.attention_bias is True  # Qwen2: q/k/v bias present
    assert w.q_proj_bias is not None
    assert w.k_proj_bias is not None
    assert w.o_proj_bias is None      # Qwen2: o_proj has no bias


# 5.
def test_random_llama_single_block_masked_prefill_correctness() -> None:
    _, cfg, w, masks, x, _ = _prep("llama")
    res = hf_single_block_masked_prefill(x, w, cfg, masks, decode_steps=2)
    assert res["metrics"]["final_output_max_abs_error"] <= 1e-8
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL)
    assert res["metrics"]["allclose"] is True


# 6.
def test_random_qwen2_single_block_masked_prefill_correctness() -> None:
    _, cfg, w, masks, x, _ = _prep("qwen2")
    res = hf_single_block_masked_prefill(x, w, cfg, masks, decode_steps=2)
    assert res["metrics"]["final_output_max_abs_error"] <= 1e-8
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL)
    assert res["metrics"]["allclose"] is True


def _decode_check(family: str) -> None:
    _, cfg, w, masks, x, g = _prep(family)
    pre = hf_single_block_masked_prefill(x, w, cfg, masks, decode_steps=2)
    ct, cp = pre["cache_tilde"], pre["cache_plain"]
    for step in range(2):
        x_new = torch.randn(1, 1, cfg.hidden_size, generator=g, dtype=DTYPE)
        dec = hf_single_block_masked_decode(
            x_new, ct, cp, w, cfg, masks, position=8 + step)
        torch.testing.assert_close(
            dec["y_new_tilde"], dec["expected_y_new_tilde"], atol=ATOL,
            rtol=RTOL)
        torch.testing.assert_close(
            dec["appended_key_tilde"], dec["expected_appended_key_tilde"],
            atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            dec["appended_value_tilde"], dec["expected_appended_value_tilde"],
            atol=ATOL, rtol=RTOL)
        assert dec["metrics"]["allclose"] is True
        ct, cp = dec["cache_tilde"], dec["cache_plain"]
    assert ct["key_rope_tilde"].shape[2] == 8 + 2


# 7.
def test_random_llama_single_block_decode_correctness() -> None:
    _decode_check("llama")


# 8.
def test_random_qwen2_single_block_decode_correctness() -> None:
    _decode_check("qwen2")


# 9.
def test_probe_random_llama_reports_allclose() -> None:
    pytest.importorskip("transformers")
    report = run_hf_single_block_probe(HFSingleBlockProbeConfig(
        model_family="llama"))
    if report["status"] != "ok":
        pytest.skip(f"probe skipped: {report.get('reason')}")
    assert report["allclose"] is True
    assert report["source"] == "random_config"
    assert report["prefill_metrics"]["final_output_max_abs_error"] <= 1e-8


# 10.
def test_probe_metadata_no_security_claim() -> None:
    pytest.importorskip("transformers")
    report = run_hf_single_block_probe(HFSingleBlockProbeConfig(
        model_family="llama"))
    if report["status"] != "ok":
        pytest.skip(f"probe skipped: {report.get('reason')}")
    md = report["metadata"]
    assert md["security_status"] == (
        "operator_compatible_leakage_reduction_not_semantic_security")
    assert md["no_intermediate_tee"] is True
    assert md["no_network_download"] is True
    assert md["semantic_security_claimed"] is False
    assert md["formal_security_claimed"] is False
    assert md["cryptographic_security_claimed"] is False


# 11.
def test_local_model_path_missing_skips_or_reports_cleanly() -> None:
    report = run_hf_single_block_probe(HFSingleBlockProbeConfig(
        model_family="llama",
        local_model_path="/nonexistent/path/to/model/xyz"))
    # Must not crash; status is a clean skip and never "ok".
    assert report["status"].startswith("skipped")
    assert report["status"] != "ok"
    assert "reason" in report
