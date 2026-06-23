"""Stage 6.9 tests -- HF full-model masked CausalLM skeleton.

Skips cleanly if transformers is unavailable. CPU-only, float64 internally,
atol=rtol=1e-8. Tiny dimensions only; no HF generate, no network.
"""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pllo.hf_wrappers.hf_causal_lm_skeleton import (  # noqa: E402
    HFCausalLMSkeletonConfig,
    extract_hf_causal_lm_skeleton_weights,
    generate_hf_causal_lm_masks,
    has_transformers,
    hf_causal_lm_masked_greedy_decode,
    hf_causal_lm_masked_only_decode,
    hf_causal_lm_masked_prefill,
    hf_causal_lm_plain_prefill,
    make_random_tiny_hf_causal_lm,
)
from pllo.experiments.hf_causal_lm_skeleton_probe import (  # noqa: E402
    HFCausalLMSkeletonProbeConfig,
    run_hf_causal_lm_skeleton_probe,
)

TOL = 1e-8


def _config(family: str = "llama", *, max_layers: int = 2,
            prefill_seq_len: int = 4, decode_steps: int = 2,
            max_vocab_size: int = 256) -> HFCausalLMSkeletonConfig:
    return HFCausalLMSkeletonConfig(
        model_family=family, batch_size=1, prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps, max_layers=max_layers,
        max_vocab_size=max_vocab_size, dtype=torch.float64, device="cpu",
        seed=2033)


def _build(family: str, cfg: HFCausalLMSkeletonConfig):
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    weights, layer_configs, meta = extract_hf_causal_lm_skeleton_weights(
        model, mc, max_layers=cfg.max_layers, dtype=torch.float64,
        device="cpu", mask_family=cfg.mask_family)
    masks = generate_hf_causal_lm_masks(weights, layer_configs, cfg)
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, meta["vocab_size"],
                              (cfg.batch_size, cfg.prefill_seq_len),
                              generator=g)
    return model, mc, weights, layer_configs, masks, meta, input_ids


# 1.
def test_hf_causal_lm_imports_optional() -> None:
    assert has_transformers() is True  # importorskip guards this module


# 2.
def test_random_tiny_llama_causal_lm_constructs() -> None:
    cfg = _config("llama")
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    assert mc.model_type == "llama"
    assert len(model.model.layers) == cfg.max_layers
    assert model.model.embed_tokens.weight.shape[1] == 32


# 3.
def test_random_tiny_qwen2_causal_lm_constructs() -> None:
    cfg = _config("qwen2")
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    assert mc.model_type == "qwen2"
    assert len(model.model.layers) == cfg.max_layers


# 4.
def test_extract_hf_causal_lm_weights_shapes_llama() -> None:
    cfg = _config("llama", max_vocab_size=128)
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    weights, layer_configs, meta = extract_hf_causal_lm_skeleton_weights(
        model, mc, max_layers=cfg.max_layers, dtype=torch.float64)
    assert weights.embed_tokens_weight.shape == (meta["vocab_size"], 32)
    assert weights.lm_head_weight.shape == (32, meta["vocab_size"])
    assert weights.final_norm_weight.shape == (32,)
    assert len(weights.layer_weights) == cfg.max_layers
    assert len(layer_configs) == cfg.max_layers
    assert meta["model_type"] == "llama"


# 5.
def test_extract_hf_causal_lm_weights_shapes_qwen2() -> None:
    cfg = _config("qwen2", max_vocab_size=128)
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    weights, layer_configs, meta = extract_hf_causal_lm_skeleton_weights(
        model, mc, max_layers=cfg.max_layers, dtype=torch.float64)
    assert weights.lm_head_weight.shape == (32, meta["vocab_size"])
    assert len(weights.layer_weights) == cfg.max_layers
    # Qwen2 carries q/k/v projection biases.
    lw = weights.layer_weights[0]
    assert lw.q_proj_bias is not None
    assert lw.k_proj_bias is not None
    assert lw.v_proj_bias is not None
    assert meta["model_type"] == "qwen2"


# 6.
def test_plain_prefill_shapes_llama() -> None:
    cfg = _config("llama")
    _, _, weights, layer_configs, masks, meta, input_ids = _build("llama", cfg)
    plain = hf_causal_lm_plain_prefill(input_ids, weights, layer_configs, masks,
                                       cfg)
    assert plain["logits_plain"].shape == (
        cfg.batch_size, cfg.prefill_seq_len, meta["vocab_size"])
    assert plain["next_token_plain"].shape == (cfg.batch_size,)
    assert len(plain["hidden_by_layer_plain"]) == cfg.max_layers + 1
    assert len(plain["caches_plain"]) == cfg.max_layers


# 7.
def test_masked_prefill_correctness_llama() -> None:
    cfg = _config("llama")
    _, _, weights, layer_configs, masks, _, input_ids = _build("llama", cfg)
    pre = hf_causal_lm_masked_prefill(input_ids, weights, layer_configs, masks,
                                      cfg)
    m = pre["metrics"]
    assert m["embedding_mask_max_abs_error"] <= TOL
    assert max(m["per_layer_handoff_max_abs_error"]) <= TOL
    assert m["final_hidden_max_abs_error"] <= TOL
    assert m["masked_logits_max_abs_error"] <= TOL
    assert m["recovered_logits_max_abs_error"] <= TOL
    assert m["greedy_token_match_rate"] == 1.0
    assert m["allclose"] is True


# 8.
def test_masked_prefill_correctness_qwen2() -> None:
    cfg = _config("qwen2")
    _, _, weights, layer_configs, masks, _, input_ids = _build("qwen2", cfg)
    pre = hf_causal_lm_masked_prefill(input_ids, weights, layer_configs, masks,
                                      cfg)
    assert pre["metrics"]["allclose"] is True
    for pl in pre["metrics"]["per_layer"]:
        assert pl["final_output_max_abs_error"] <= TOL
        assert pl["mlp_output_max_abs_error"] <= TOL


# 9.
def test_masked_greedy_decode_token_match_llama() -> None:
    cfg = _config("llama")
    _, _, weights, layer_configs, masks, _, input_ids = _build("llama", cfg)
    res = hf_causal_lm_masked_greedy_decode(input_ids, weights, layer_configs,
                                            masks, cfg)
    assert res["token_match_rate"] == 1.0
    assert res["generated_plain_tokens"].shape == (1, 1 + cfg.decode_steps)
    for s in res["decode_step_metrics"]:
        assert s["per_layer_output_error"] <= TOL
        assert s["recovered_logits_error"] <= TOL
        assert s["sampled_token_match"] == 1.0


# 10.
def test_masked_greedy_decode_token_match_qwen2() -> None:
    cfg = _config("qwen2")
    _, _, weights, layer_configs, masks, _, input_ids = _build("qwen2", cfg)
    res = hf_causal_lm_masked_greedy_decode(input_ids, weights, layer_configs,
                                            masks, cfg)
    assert res["token_match_rate"] == 1.0


# 11.
def test_probe_random_llama_reports_allclose() -> None:
    pc = HFCausalLMSkeletonProbeConfig(
        model_family="llama", max_layers=2, prefill_seq_len=4, decode_steps=2,
        max_vocab_size=128)
    report = run_hf_causal_lm_skeleton_probe(pc)
    assert report["status"] == "ok"
    assert report["metadata"]["source"] == "random_tiny_hf_model"
    assert report["prefill_metrics"]["allclose"] is True
    assert report["decode_metrics"]["token_match_rate"] == 1.0


# 12.
def test_probe_random_qwen2_reports_allclose() -> None:
    pc = HFCausalLMSkeletonProbeConfig(
        model_family="qwen2", max_layers=2, prefill_seq_len=4, decode_steps=2,
        max_vocab_size=128)
    report = run_hf_causal_lm_skeleton_probe(pc)
    assert report["status"] == "ok"
    assert report["metadata"]["model_type"] == "qwen2"
    assert report["prefill_metrics"]["allclose"] is True
    assert report["decode_metrics"]["token_match_rate"] == 1.0


# 13.
def test_missing_local_path_returns_clean_skipped_status() -> None:
    pc = HFCausalLMSkeletonProbeConfig(
        local_model_path="/nonexistent/path/to/model",
        use_random_tiny_if_no_path=False)
    report = run_hf_causal_lm_skeleton_probe(pc)
    assert report["status"].startswith("skipped")
    assert "metadata" in report  # no crash, structured result


# 14.
def test_metadata_boundaries_and_no_security_claim() -> None:
    pc = HFCausalLMSkeletonProbeConfig(max_layers=1, prefill_seq_len=3,
                                       decode_steps=1, max_vocab_size=128)
    md = run_hf_causal_lm_skeleton_probe(pc)["metadata"]
    assert md["input_ids_visible_to_gpu"] is False
    assert md["plaintext_embedding_visible_to_gpu"] is False
    assert md["plaintext_logits_visible_to_gpu"] is False
    assert md["masked_logits_visible_to_gpu"] is True
    assert md["logits_recovered_in_tee"] is True
    assert md["sampling_boundary"] == "trusted_side"
    assert md["semantic_security_claimed"] is False
    assert md["formal_security_claimed"] is False
    assert md["cryptographic_security_claimed"] is False
    assert md["no_network_download"] is True
    assert md["no_gpu_required"] is True


# 15b.
def test_masked_only_decode_matches_masked_greedy() -> None:
    """The masked-only (no-plain-reference) runtime must produce exactly the
    same masked tokens as the verification decode -- it is the same masked
    path, just without the diagnostic plaintext recompute."""
    cfg = _config("qwen2", max_layers=2, prefill_seq_len=6, decode_steps=3)
    _, _, weights, lc, masks, _, input_ids = _build("qwen2", cfg)
    full = hf_causal_lm_masked_greedy_decode(input_ids, weights, lc, masks, cfg)
    only = hf_causal_lm_masked_only_decode(input_ids, weights, lc, masks, cfg)
    assert torch.equal(full["generated_from_masked_tokens"],
                       only["generated_from_masked_tokens"])
    # and the masked path reproduces the plaintext tokens here (fp64 tiny model)
    assert full["token_match_rate"] == 1.0


# 15.
def test_reports_do_not_include_full_tensor_dumps() -> None:
    pc = HFCausalLMSkeletonProbeConfig(max_layers=1, prefill_seq_len=3,
                                       decode_steps=1, max_vocab_size=128)
    report = run_hf_causal_lm_skeleton_probe(pc)
    text = json.dumps(report, default=str)
    assert "tensor(" not in text
    # no long flat numeric arrays (a logits/hidden dump would trip this)
    import re
    assert re.search(r"(-?\d+\.\d+\s*,\s*){50,}", text) is None
    assert len(text) < 200_000
