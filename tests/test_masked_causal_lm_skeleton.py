"""Stage 6.8 tests -- full masked CausalLM skeleton (CPU, float64, no HF)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pllo.experiments.masked_causal_lm_skeleton_probe import (
    MaskedCausalLMSkeletonProbeConfig,
    run_masked_causal_lm_skeleton_probe,
)
from pllo.ops.masked_causal_lm_skeleton import (
    MaskedCausalLMSkeletonConfig,
    causal_lm_masked_greedy_decode,
    causal_lm_masked_prefill,
    causal_lm_plain_prefill,
    generate_skeleton_masks,
    init_masked_causal_lm_skeleton_weights,
)
from pllo.ops.rope import build_rope_cache

ATOL = 1e-8
RTOL = 1e-8
DTYPE = torch.float64


def _cfg(**kw) -> MaskedCausalLMSkeletonConfig:
    base = dict(batch_size=2, prefill_seq_len=4, decode_steps=2, vocab_size=64,
                hidden_size=32, intermediate_size=64, num_layers=3, num_heads=4,
                num_key_value_heads=2, dtype=DTYPE, device="cpu", seed=2031)
    base.update(kw)
    return MaskedCausalLMSkeletonConfig(**base)


def _g(seed: int = 2031) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _setup(**kw):
    cfg = _cfg(**kw)
    g = _g(cfg.seed)
    weights = init_masked_causal_lm_skeleton_weights(cfg, g)
    masks = generate_skeleton_masks(cfg, g)
    ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.prefill_seq_len),
                        generator=g)
    return cfg, weights, masks, ids


# 1.
def test_skeleton_config_validates_shapes() -> None:
    cfg = _cfg()
    cfg.validate()
    assert cfg.head_dim == 8 and cfg.head_dim % 2 == 0
    with pytest.raises(ValueError):
        MaskedCausalLMSkeletonConfig(hidden_size=30, num_heads=4).validate()
    with pytest.raises(ValueError):
        MaskedCausalLMSkeletonConfig(num_layers=0).validate()


# 2.
def test_generate_skeleton_masks_shapes() -> None:
    cfg, _, masks, _ = _setup()
    assert len(masks.residual_masks) == cfg.num_layers + 1
    assert len(masks.residual_mask_inverses) == cfg.num_layers + 1
    assert len(masks.layer_masks) == cfg.num_layers
    for ell, lm in enumerate(masks.layer_masks):
        assert lm.n_in.shape == (cfg.hidden_size, cfg.hidden_size)
        assert torch.equal(lm.n_in, masks.residual_masks[ell])
        assert torch.equal(lm.n_out, masks.residual_masks[ell + 1])
    assert masks.input_pad is not None  # use_input_pad default True


# 3.
def test_residual_masks_are_orthogonal() -> None:
    cfg, _, masks, _ = _setup()
    eye = torch.eye(cfg.hidden_size, dtype=DTYPE)
    for m in masks.residual_masks:
        torch.testing.assert_close(m @ m.T, eye, atol=ATOL, rtol=RTOL)


# 4.
def test_plain_prefill_shapes() -> None:
    cfg, w, masks, ids = _setup()
    cos, sin = build_rope_cache(cfg.prefill_seq_len + cfg.decode_steps + 1,
                                cfg.head_dim, cfg.rope_base, DTYPE, "cpu")
    plain = causal_lm_plain_prefill(ids, w, masks, cfg, cos, sin)
    assert plain["logits_plain"].shape == (
        cfg.batch_size, cfg.prefill_seq_len, cfg.vocab_size)
    assert len(plain["hidden_by_layer_plain"]) == cfg.num_layers + 1
    assert plain["next_token_plain"].shape == (cfg.batch_size,)


# 5.
def test_masked_prefill_embedding_boundary() -> None:
    cfg, w, masks, ids = _setup()
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    assert pre["metrics"]["embedding_mask_max_abs_error"] <= 1e-8


# 6.
def test_masked_prefill_layer_handoff_single_layer() -> None:
    cfg, w, masks, ids = _setup(num_layers=1)
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    errs = pre["metrics"]["per_layer_handoff_max_abs_error"]
    assert len(errs) == 2  # N_0 (input) and N_1 (after the single layer)
    assert all(e <= 1e-8 for e in errs)


# 7.
def test_masked_prefill_layer_handoff_multi_layer() -> None:
    cfg, w, masks, ids = _setup(num_layers=3)
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    errs = pre["metrics"]["per_layer_handoff_max_abs_error"]
    assert len(errs) == 4
    assert all(e <= 1e-8 for e in errs)


# 8.
def test_masked_prefill_final_hidden_invariant() -> None:
    cfg, w, masks, ids = _setup()
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    assert pre["metrics"]["final_hidden_max_abs_error"] <= 1e-8


# 9.
def test_masked_prefill_masked_logits_recover() -> None:
    cfg, w, masks, ids = _setup()
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    assert pre["metrics"]["masked_logits_max_abs_error"] <= 1e-8
    assert pre["metrics"]["recovered_logits_max_abs_error"] <= 1e-8


# 10.
def test_masked_prefill_greedy_token_match() -> None:
    cfg, w, masks, ids = _setup()
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    assert pre["metrics"]["greedy_token_match_rate"] == 1.0
    torch.testing.assert_close(pre["next_token_plain"],
                               pre["next_token_from_masked"])


# 11.
def test_masked_prefill_cache_shapes() -> None:
    cfg, w, masks, ids = _setup()
    pre = causal_lm_masked_prefill(ids, w, masks, cfg)
    assert len(pre["caches_tilde"]) == cfg.num_layers
    for c in pre["caches_tilde"]:
        assert c["key_rope_tilde"].shape == (
            cfg.batch_size, cfg.num_key_value_heads, cfg.prefill_seq_len,
            cfg.head_dim)


# 12.
def test_masked_decode_one_step_token_match() -> None:
    cfg, w, masks, ids = _setup(decode_steps=1)
    dec = causal_lm_masked_greedy_decode(ids, w, masks, cfg)
    assert dec["decode_step_metrics"][0]["sampled_token_match"] == 1.0


# 13.
def test_masked_decode_multi_step_token_match() -> None:
    cfg, w, masks, ids = _setup(decode_steps=2)
    dec = causal_lm_masked_greedy_decode(ids, w, masks, cfg)
    assert dec["token_match_rate"] == 1.0
    for s in dec["decode_step_metrics"]:
        assert s["sampled_token_match"] == 1.0


# 14.
def test_masked_decode_per_layer_cache_append_invariant() -> None:
    cfg, w, masks, ids = _setup()
    dec = causal_lm_masked_greedy_decode(ids, w, masks, cfg)
    for s in dec["decode_step_metrics"]:
        assert s["per_layer_cache_append_key_error"] <= 1e-8
        assert s["per_layer_cache_append_value_error"] <= 1e-8
        assert s["per_layer_output_error"] <= 1e-8


# 15.
def test_probe_runs_and_allclose() -> None:
    report = run_masked_causal_lm_skeleton_probe(
        MaskedCausalLMSkeletonProbeConfig())
    assert report["status"] == "ok"
    assert report["all_allclose"] is True
    assert report["token_match_rate"] == 1.0


# 16.
def test_probe_metadata_boundaries() -> None:
    md = run_masked_causal_lm_skeleton_probe(
        MaskedCausalLMSkeletonProbeConfig())["metadata"]
    assert md["input_ids_visible_to_gpu"] is False
    assert md["plaintext_embedding_visible_to_gpu"] is False
    assert md["plaintext_logits_visible_to_gpu"] is False
    assert md["masked_logits_visible_to_gpu"] is True
    assert md["logits_recovered_in_tee"] is True
    assert md["sampling_boundary"] == "trusted_side"
    assert md["decoder_runs_on_gpu_assumption"] == "masked_tensors_only"


# 17.
def test_probe_metadata_no_security_claim() -> None:
    md = run_masked_causal_lm_skeleton_probe(
        MaskedCausalLMSkeletonProbeConfig())["metadata"]
    assert md["security_status"] == (
        "operator_compatible_leakage_reduction_not_semantic_security")
    assert md["semantic_security_claimed"] is False
    assert md["formal_security_claimed"] is False
    assert md["cryptographic_security_claimed"] is False


# 18.
def test_probe_markdown_required_statement_if_helper_exposed() -> None:
    report = run_masked_causal_lm_skeleton_probe(
        MaskedCausalLMSkeletonProbeConfig())
    assert "does not validate production generation" in report["statement"]
    script = (Path(__file__).resolve().parents[1] / "scripts"
              / "run_masked_causal_lm_skeleton_probe.py")
    if script.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_run_masked_causal_lm_skeleton_probe", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rendered = mod._render_markdown(report)
        assert report["statement"] in rendered
        assert "semantic security" in rendered.lower()


# 19.
def test_num_layers_one_edge_case() -> None:
    cfg, w, masks, ids = _setup(num_layers=1)
    dec = causal_lm_masked_greedy_decode(ids, w, masks, cfg)
    assert dec["prefill_metrics"]["allclose"] is True
    assert dec["token_match_rate"] == 1.0


# 20.
def test_mha_variant_num_heads_equals_num_key_value_heads() -> None:
    cfg, w, masks, ids = _setup(num_key_value_heads=4)  # MHA
    dec = causal_lm_masked_greedy_decode(ids, w, masks, cfg)
    assert dec["prefill_metrics"]["allclose"] is True
    assert dec["token_match_rate"] == 1.0
