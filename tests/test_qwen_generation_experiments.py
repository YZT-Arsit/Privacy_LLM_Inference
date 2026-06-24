"""Dry-run tests for the E1/E2 no-LoRA Qwen generation runners (tiny Qwen2, CPU).

Validates the runner end-to-end on a tiny random model (never a paper result):
all three decoding modes produce metrics, masking preserves correctness
(plain-vs-masked top-1 == 1.0), and the protocol fields hold
(``tee_used_on_gpu=False``, empty plaintext/secret field lists).

Run: python -m pytest tests/test_qwen_generation_experiments.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pllo.experiments.qwen_generation_experiments import (  # noqa: E402
    protocol_accounting,
    run_e1_nolora,
    run_e2_token_scaling,
)
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402


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
