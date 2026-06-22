"""Stage 6.7 tests -- masked CausalLM boundaries (CPU, float64, no HF)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pllo.experiments.causal_lm_boundary_probe import (
    CausalLMBoundaryProbeConfig,
    run_causal_lm_boundary_probe,
)
from pllo.ops.causal_lm_boundaries import (
    CausalLMBoundaryConfig,
    apply_temperature,
    apply_vocab_logit_mask,
    embedding_boundary_forward,
    final_norm_lm_head_masked,
    final_norm_lm_head_plain,
    fold_final_norm_lm_head_with_vocab_mask,
    greedy_sample,
    init_causal_lm_boundary_weights,
    make_vocab_logit_mask,
    recover_vocab_logits,
    sample_from_logits,
    top_k_filter,
    top_p_filter,
    trusted_embedding_lookup,
    trusted_next_token_to_masked_embedding,
    trusted_sample_from_masked_logits,
)

ATOL = 1e-8
RTOL = 1e-8
DTYPE = torch.float64


def _g(seed: int = 2030) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _orthogonal(dim: int, g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=DTYPE))
    return q


def _cfg(**kw) -> CausalLMBoundaryConfig:
    base = dict(batch_size=2, seq_len=8, vocab_size=32, hidden_size=16,
                dtype=DTYPE, device="cpu", seed=2030)
    base.update(kw)
    return CausalLMBoundaryConfig(**base)


def _setup(**kw):
    cfg = _cfg(**kw)
    g = _g(cfg.seed)
    weights = init_causal_lm_boundary_weights(cfg, g)
    n_res = _orthogonal(cfg.hidden_size, g)
    n_res_inv = n_res.transpose(-2, -1).contiguous()
    input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len),
                              generator=g)
    return cfg, weights, n_res, n_res_inv, input_ids, g


# 1.
def test_trusted_embedding_lookup_shapes() -> None:
    cfg, w, _, _, ids, _ = _setup()
    x = trusted_embedding_lookup(ids, w.embed_tokens_weight)
    assert x.shape == (cfg.batch_size, cfg.seq_len, cfg.hidden_size)


# 2.
def test_embedding_boundary_masks_output_without_pad() -> None:
    cfg, w, n_res, _, ids, _ = _setup()
    res = embedding_boundary_forward(ids, w, n_res, pad_in=None)
    torch.testing.assert_close(
        res["x_tilde"], res["x_plain"] @ n_res, atol=ATOL, rtol=RTOL)
    assert res["metadata"]["used_input_pad"] is False
    assert res["metadata"]["input_ids_visible_to_gpu"] is False


# 3.
def test_embedding_boundary_masks_output_with_pad() -> None:
    cfg, w, n_res, _, ids, g = _setup()
    pad = torch.randn(cfg.hidden_size, generator=g, dtype=DTYPE)
    res = embedding_boundary_forward(ids, w, n_res, pad_in=pad)
    torch.testing.assert_close(
        res["x_tilde"], (res["x_plain"] - pad) @ n_res, atol=ATOL, rtol=RTOL)
    assert res["metadata"]["used_input_pad"] is True


# 4.
def test_vocab_logit_mask_roundtrip() -> None:
    g = _g(5)
    mask = make_vocab_logit_mask(32, DTYPE, "cpu", g)
    logits = torch.randn(2, 8, 32, generator=g, dtype=DTYPE)
    masked = apply_vocab_logit_mask(logits, mask)
    recovered = recover_vocab_logits(masked, mask)
    torch.testing.assert_close(recovered, logits, atol=ATOL, rtol=RTOL)


# 5.
def test_vocab_logit_mask_inverse_permutation() -> None:
    g = _g(6)
    mask = make_vocab_logit_mask(32, DTYPE, "cpu", g)
    arange = torch.arange(32)
    torch.testing.assert_close(mask.inverse_permutation[mask.permutation],
                               arange)
    assert torch.all(mask.scale > 0)


# 6.
def test_final_norm_lm_head_plain_shapes() -> None:
    cfg, w, _, _, _, g = _setup()
    h = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, generator=g,
                    dtype=DTYPE)
    out = final_norm_lm_head_plain(h, w.final_norm_weight, w.lm_head_weight,
                                   cfg.rms_norm_eps)
    assert out["core"].shape == h.shape
    assert out["logits"].shape == (cfg.batch_size, cfg.seq_len, cfg.vocab_size)


# 7.
def test_fold_final_norm_lm_head_with_vocab_mask_correctness() -> None:
    cfg, w, n_res, n_res_inv, _, g = _setup()
    mask = make_vocab_logit_mask(cfg.vocab_size, DTYPE, "cpu", g)
    h_plain = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size,
                          generator=g, dtype=DTYPE)
    h_tilde = h_plain @ n_res
    from pllo.ops.nonlinear_islands import rmsnorm_core
    core_tilde = rmsnorm_core(h_tilde, cfg.rms_norm_eps)
    w_lm_tilde = fold_final_norm_lm_head_with_vocab_mask(
        w.final_norm_weight, w.lm_head_weight, n_res_inv, mask)
    logits_tilde = core_tilde @ w_lm_tilde
    plain = final_norm_lm_head_plain(h_plain, w.final_norm_weight,
                                     w.lm_head_weight, cfg.rms_norm_eps)
    torch.testing.assert_close(
        logits_tilde, apply_vocab_logit_mask(plain["logits"], mask),
        atol=ATOL, rtol=RTOL)


# 8.
def test_final_norm_lm_head_masked_logits_match_expected_masked_logits() -> None:
    cfg, w, n_res, n_res_inv, _, g = _setup()
    mask = make_vocab_logit_mask(cfg.vocab_size, DTYPE, "cpu", g)
    h_plain = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size,
                          generator=g, dtype=DTYPE)
    h_tilde = h_plain @ n_res
    out = final_norm_lm_head_masked(h_tilde, h_plain, w, n_res_inv, mask,
                                    cfg.rms_norm_eps)
    torch.testing.assert_close(
        out["core_tilde"], out["expected_core_tilde"], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        out["logits_tilde"], out["expected_logits_tilde"], atol=ATOL, rtol=RTOL)


# 9.
def test_recovered_logits_match_plain_logits() -> None:
    cfg, w, n_res, n_res_inv, _, g = _setup()
    mask = make_vocab_logit_mask(cfg.vocab_size, DTYPE, "cpu", g)
    h_plain = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size,
                          generator=g, dtype=DTYPE)
    out = final_norm_lm_head_masked(h_plain @ n_res, h_plain, w, n_res_inv,
                                    mask, cfg.rms_norm_eps)
    torch.testing.assert_close(
        out["logits_recovered"], out["logits_plain"], atol=ATOL, rtol=RTOL)


# 10.
def test_greedy_sample_matches_argmax() -> None:
    logits = torch.randn(2, 8, 32, generator=_g(9), dtype=DTYPE)
    torch.testing.assert_close(greedy_sample(logits), logits.argmax(dim=-1))


# 11.
def test_trusted_greedy_from_masked_logits_matches_plain_greedy() -> None:
    cfg, w, n_res, n_res_inv, _, g = _setup()
    mask = make_vocab_logit_mask(cfg.vocab_size, DTYPE, "cpu", g)
    h_plain = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size,
                          generator=g, dtype=DTYPE)
    out = final_norm_lm_head_masked(h_plain @ n_res, h_plain, w, n_res_inv,
                                    mask, cfg.rms_norm_eps)
    trusted = trusted_sample_from_masked_logits(
        out["logits_tilde"], mask, mode="greedy")
    torch.testing.assert_close(trusted["tokens"],
                               greedy_sample(out["logits_plain"]))


# 12.
def test_temperature_validation() -> None:
    logits = torch.randn(2, 32, generator=_g(10), dtype=DTYPE)
    torch.testing.assert_close(apply_temperature(logits, 1.0), logits)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, -0.5)


# 13.
def test_top_k_filter_keeps_k_entries() -> None:
    logits = torch.randn(4, 32, generator=_g(11), dtype=DTYPE)
    filtered = top_k_filter(logits, 5)
    finite = torch.isfinite(filtered).sum(dim=-1)
    assert torch.all(finite == 5)
    with pytest.raises(ValueError):
        top_k_filter(logits, 0)
    with pytest.raises(ValueError):
        top_k_filter(logits, 999)
    torch.testing.assert_close(top_k_filter(logits, None), logits)


# 14.
def test_top_p_filter_keeps_nonempty_prefix() -> None:
    logits = torch.randn(4, 32, generator=_g(12), dtype=DTYPE)
    filtered = top_p_filter(logits, 0.9)
    finite = torch.isfinite(filtered).sum(dim=-1)
    assert torch.all(finite >= 1)
    assert torch.all(finite <= 32)
    with pytest.raises(ValueError):
        top_p_filter(logits, 0.0)
    with pytest.raises(ValueError):
        top_p_filter(logits, 1.5)
    torch.testing.assert_close(top_p_filter(logits, None), logits)


# 15.
def test_seeded_sampling_is_deterministic() -> None:
    logits = torch.randn(2, 8, 32, generator=_g(13), dtype=DTYPE)
    s1 = sample_from_logits(logits, temperature=0.7, top_k=10, top_p=0.9,
                            generator=_g(99))
    s2 = sample_from_logits(logits, temperature=0.7, top_k=10, top_p=0.9,
                            generator=_g(99))
    assert torch.equal(s1, s2)
    assert s1.shape == (2, 8)


# 16.
def test_next_token_to_masked_embedding_boundary() -> None:
    cfg, w, n_res, _, _, g = _setup()
    next_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size,), generator=g)
    pad = torch.randn(cfg.hidden_size, generator=g, dtype=DTYPE)
    res = trusted_next_token_to_masked_embedding(next_ids, w, n_res, pad)
    expected = (w.embed_tokens_weight[next_ids] - pad) @ n_res
    torch.testing.assert_close(res["x_next_tilde"], expected, atol=ATOL,
                               rtol=RTOL)
    assert res["metadata"]["next_token_ids_visible_to_gpu"] is False


# 17.
def test_boundary_probe_runs_and_allclose() -> None:
    report = run_causal_lm_boundary_probe(CausalLMBoundaryProbeConfig())
    assert report["status"] == "ok"
    assert report["all_allclose"] is True
    m = report["metrics"]
    assert m["logits_recovered_allclose"] is True
    assert m["greedy_token_match_rate"] == 1.0
    assert m["trusted_greedy_from_masked_match_rate"] == 1.0
    assert m["seeded_sampling_deterministic"] is True


# 18.
def test_probe_metadata_input_ids_not_visible_to_gpu() -> None:
    md = run_causal_lm_boundary_probe(CausalLMBoundaryProbeConfig())["metadata"]
    assert md["input_ids_visible_to_gpu"] is False
    assert md["plaintext_embedding_visible_to_gpu"] is False
    assert md["next_token_ids_visible_to_gpu"] is False
    assert md["released_to_gpu_at_input"] == "masked_embeddings_only"


# 19.
def test_probe_metadata_plaintext_logits_not_visible_to_gpu() -> None:
    md = run_causal_lm_boundary_probe(CausalLMBoundaryProbeConfig())["metadata"]
    assert md["plaintext_logits_visible_to_gpu"] is False
    assert md["masked_logits_visible_to_gpu"] is True
    assert md["logits_recovered_in_tee"] is True
    assert md["sampling_boundary"] == "trusted_side"


# 20.
def test_probe_metadata_no_security_claim() -> None:
    md = run_causal_lm_boundary_probe(CausalLMBoundaryProbeConfig())["metadata"]
    assert md["security_status"] == (
        "operator_compatible_leakage_reduction_not_semantic_security")
    assert md["semantic_security_claimed"] is False
    assert md["formal_security_claimed"] is False
    assert md["cryptographic_security_claimed"] is False
    assert md["dense_vocab_mask_used"] is False


# 21.
def test_tied_embedding_lm_head_shapes() -> None:
    cfg = _cfg(tie_word_embeddings=True)
    w = init_causal_lm_boundary_weights(cfg, _g(cfg.seed))
    assert w.lm_head_weight.shape == (cfg.hidden_size, cfg.vocab_size)
    torch.testing.assert_close(
        w.lm_head_weight, w.embed_tokens_weight.t(), atol=ATOL, rtol=RTOL)


# 22.
def test_boundary_markdown_required_statement_if_script_helper_exposed() -> None:
    # The required statement must be present in the probe report and, if the
    # script's markdown renderer is importable, in its rendered output.
    report = run_causal_lm_boundary_probe(CausalLMBoundaryProbeConfig())
    assert "does not validate full-model generation" in report["statement"]
    script = (Path(__file__).resolve().parents[1] / "scripts"
              / "run_causal_lm_boundary_probe.py")
    if script.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_run_causal_lm_boundary_probe", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rendered = mod._render_markdown(report)
        assert report["statement"] in rendered
        assert "semantic security" in rendered.lower()
