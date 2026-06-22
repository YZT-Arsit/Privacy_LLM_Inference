"""Stage 7.0 tests -- full-pipeline cost/leakage evaluation (CPU, no HF)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pllo.experiments.full_pipeline_cost_leakage import (
    FullPipelineCostLeakageConfig,
    compute_cost_breakdown,
    compute_leakage_surface,
    run_full_pipeline_cost_leakage,
)


def _cfg(**kw) -> FullPipelineCostLeakageConfig:
    base = dict(batch_size=2, prefill_seq_len=4, decode_steps=2, vocab_size=64,
                hidden_size=32, intermediate_size=64, num_layers=3, num_heads=4,
                num_key_value_heads=2, num_repeats=2)
    base.update(kw)
    return FullPipelineCostLeakageConfig(**base)


# 1.
def test_cost_config_validates_shapes() -> None:
    _cfg().validate()
    with pytest.raises(ValueError):
        FullPipelineCostLeakageConfig(hidden_size=30, num_heads=4).validate()
    with pytest.raises(ValueError):
        FullPipelineCostLeakageConfig(num_layers=0).validate()


# 2.
def test_cost_model_boundary_calls_formula() -> None:
    cfg = _cfg(decode_steps=2)
    masked = compute_cost_breakdown("masked_per_layer_with_vocab_scaling", cfg)
    assert masked.boundary_calls == 2 + 2 * cfg.decode_steps
    plain = compute_cost_breakdown("plain_synthetic", cfg)
    assert plain.boundary_calls == 0


# 3.
def test_cost_model_handoff_zero_for_shared_mask() -> None:
    cfg = _cfg()
    assert compute_cost_breakdown("masked_same_residual_mask",
                                  cfg).handoff_gemm_flops == 0.0
    assert compute_cost_breakdown("plain_synthetic",
                                  cfg).handoff_gemm_flops == 0.0


# 4.
def test_cost_model_handoff_positive_for_per_layer_mask() -> None:
    cfg = _cfg()
    for v in ("masked_per_layer_residual_mask",
              "masked_per_layer_with_vocab_scaling", "gpu_masked_lm_head"):
        assert compute_cost_breakdown(v, cfg).handoff_gemm_flops > 0.0


# 5.
def test_cost_model_lm_head_gpu_vs_tee_difference() -> None:
    cfg = _cfg()
    gpu = compute_cost_breakdown("gpu_masked_lm_head", cfg)
    tee = compute_cost_breakdown("output_hidden_to_tee", cfg)
    assert gpu.lm_head_gpu_flops > 0.0 and gpu.lm_head_tee_flops == 0.0
    assert tee.lm_head_tee_flops > 0.0 and tee.lm_head_gpu_flops == 0.0
    assert gpu.logits_recovery_flops > 0.0      # TEE recovers masked logits
    assert tee.logits_recovery_flops == 0.0     # TEE computes logits directly


# 6.
def test_transfer_bytes_masked_logits_vs_hidden() -> None:
    cfg = _cfg()  # vocab_size (64) > hidden_size (32): logits transfer larger
    gpu = compute_cost_breakdown("gpu_masked_lm_head", cfg)
    tee = compute_cost_breakdown("output_hidden_to_tee", cfg)
    assert gpu.transfer_bytes_prefill > tee.transfer_bytes_prefill
    assert gpu.transfer_bytes_decode > tee.transfer_bytes_decode


# 7.
def test_leakage_surface_plain_baseline_flags() -> None:
    s = compute_leakage_surface("plain_synthetic")
    assert s.input_ids_visible_to_gpu is True
    assert s.plaintext_logits_visible_to_gpu is True
    assert s.masked_logits_visible_to_gpu is False
    assert s.final_output_text_semantics_protected is False


# 8.
def test_leakage_surface_masked_logits_flags() -> None:
    s = compute_leakage_surface("masked_per_layer_with_vocab_scaling")
    assert s.input_ids_visible_to_gpu is False
    assert s.plaintext_embedding_visible_to_gpu is False
    assert s.plaintext_logits_visible_to_gpu is False
    assert s.masked_logits_visible_to_gpu is True
    assert s.masked_hidden_visible_to_gpu is True
    assert s.attention_scores_visible_to_gpu is True  # honest caveat


# 9.
def test_leakage_surface_output_hidden_to_tee_flags() -> None:
    s = compute_leakage_surface("output_hidden_to_tee")
    assert s.plaintext_logits_visible_to_gpu is False
    assert s.masked_logits_visible_to_gpu is False   # GPU never sees logits
    assert s.masked_hidden_visible_to_gpu is True


# 10.
def test_vocab_mask_leakage_proxy_runs() -> None:
    report = run_full_pipeline_cost_leakage(_cfg())
    vl = report["leakage_proxy"]["vocab_mask"]
    # TEE recovers exactly in all modes.
    for k in ("no_mask", "permutation_only", "permutation_plus_scaling"):
        assert vl[k]["tee_recovered_top1_matches_plain"] == 1.0
    # GPU-side argmax alignment is exact only without a mask.
    assert vl["no_mask"]["gpu_argmax_token_index_matches_plain"] == 1.0
    assert (vl["permutation_only"]["gpu_argmax_token_index_matches_plain"]
            < vl["no_mask"]["gpu_argmax_token_index_matches_plain"])


# 11.
def test_safe_and_unsafe_claims_present() -> None:
    pc = run_full_pipeline_cost_leakage(_cfg())["paper_claims"]
    assert pc["safe_claims"] and pc["unsafe_claims"] and pc["required_caveats"]
    joined = " ".join(pc["unsafe_claims"]).lower()
    assert "semantic security" in joined
    assert "cryptographic security" in joined


# 12.
def test_experiment_runs_small_config() -> None:
    report = run_full_pipeline_cost_leakage(_cfg())
    assert report["stage"] == "7.0_full_pipeline_cost_leakage"
    assert len(report["cost_breakdown"]) == 7
    assert len(report["leakage_surfaces"]) == 7
    assert report["summary"]["recommended_default"] == (
        "masked_per_layer_with_vocab_scaling")


# 13.
def test_experiment_reports_handoff_caveat() -> None:
    report = run_full_pipeline_cost_leakage(_cfg())
    caveats = " ".join(report["paper_claims"]["required_caveats"]).lower()
    assert "handoff" in caveats
    assert report["summary"]["handoff_gemm_required_for_per_layer_masks"] is True
    # the per-layer cost notes mention the skip path cannot be folded offline
    per_layer = next(c for c in report["cost_breakdown"]
                     if c["variant"] == "masked_per_layer_with_vocab_scaling")
    assert any("skip" in n.lower() for n in per_layer["notes"])


# 14.
def test_timing_breakdown_schema_if_wallclock_enabled() -> None:
    report = run_full_pipeline_cost_leakage(_cfg(run_wallclock=True))
    tb = report["timing_breakdown"]
    assert len(tb) >= 1
    names = {t["variant"] for t in tb}
    assert "plain_synthetic" in names
    assert "masked_per_layer_residual_mask" in names
    for t in tb:
        assert t["total_ms_mean"] >= 0.0
        assert t["num_repeats"] == report["config"]["num_repeats"]
    # disabling wall-clock yields an empty timing list
    assert run_full_pipeline_cost_leakage(
        _cfg(run_wallclock=False))["timing_breakdown"] == []


# 15.
def test_markdown_required_statement_if_helper_exposed() -> None:
    report = run_full_pipeline_cost_leakage(_cfg())
    assert "does not claim semantic" in report["statement"].lower()
    script = (Path(__file__).resolve().parents[1] / "scripts"
              / "run_full_pipeline_cost_leakage.py")
    if script.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_run_full_pipeline_cost_leakage", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rendered = mod._render_markdown(report)
        assert report["statement"] in rendered
        assert "semantic" in rendered.lower()
