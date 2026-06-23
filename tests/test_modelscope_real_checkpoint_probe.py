"""Stage 8.2 tests -- real ModelScope checkpoint probe.

No ModelScope network, no real checkpoint, no CUDA required. Heavy paths are
exercised only via clean-skip statuses and pure mask/dtype helpers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pllo.experiments import modelscope_real_checkpoint_probe as M
from pllo.experiments.modelscope_real_checkpoint_probe import (
    ModelScopeRealCheckpointProbeConfig,
    resolve_dtype,
    run_modelscope_real_checkpoint_probe,
)
from pllo.hf_wrappers.hf_causal_lm_skeleton import (
    HFCausalLMSkeletonConfig,
    generate_hf_causal_lm_masks,
    make_residual_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "run_modelscope_real_checkpoint_probe.py"


# 1.
def test_config_defaults() -> None:
    c = ModelScopeRealCheckpointProbeConfig()
    assert c.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert c.dtype == "bfloat16"
    assert c.mask_mode == "signed_permutation"
    assert c.residual_mask_strategy == "shared"
    assert c.prefill_seq_len == 16 and c.decode_steps == 8
    assert c.max_report_mb == 10


# 2.
def test_signed_permutation_mask_roundtrip() -> None:
    g = torch.Generator().manual_seed(0)
    h = 128
    m, m_inv = make_residual_mask(h, "signed_permutation", torch.float32,
                                  torch.device("cpu"), g)
    # exact orthogonality + roundtrip (no float error for ±1/0 entries)
    assert torch.allclose(m @ m_inv, torch.eye(h), atol=1e-12)
    x = torch.randn(2, 3, h)
    assert torch.allclose((x @ m) @ m_inv, x, atol=1e-12)
    # exactly one nonzero per row/col, values in {-1, +1}
    nz = (m != 0)
    assert nz.sum().item() == h
    assert set(m[nz].abs().unique().tolist()) == {1.0}


# 3.
def test_block_orthogonal_mask_roundtrip_small() -> None:
    g = torch.Generator().manual_seed(1)
    h = 96
    m, m_inv = make_residual_mask(h, "block_orthogonal", torch.float64,
                                  torch.device("cpu"), g, block_size=32)
    assert torch.allclose(m @ m_inv, torch.eye(h, dtype=torch.float64),
                          atol=1e-10)
    # block-diagonal structure: zero outside 32-blocks
    assert m[:32, 32:].abs().max().item() < 1e-12


# 4.
def test_dense_mask_rejected_for_large_hidden_without_override() -> None:
    g = torch.Generator().manual_seed(2)
    with pytest.raises(ValueError):
        make_residual_mask(2048, "dense_orthogonal", torch.float32,
                           torch.device("cpu"), g)
    # override allows it (small enough to actually build here -> use 1100)
    m, _ = make_residual_mask(1100, "dense_orthogonal", torch.float32,
                              torch.device("cpu"), g, allow_dense_large=True)
    assert m.shape == (1100, 1100)


# 5.
def test_probe_skips_cleanly_without_modelscope(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    monkeypatch.setattr(M, "has_transformers", lambda: True)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    assert r["status"] == "skipped_modelscope_unavailable"
    assert "caveats" in r and r["caveats"]


# 6.
def test_probe_skips_cleanly_without_transformers(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_transformers", lambda: False)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    assert r["status"] == "skipped_transformers_unavailable"


# 7.
def test_report_schema_compact(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    for key in ("stage", "config", "status", "resolved_dtype", "environment",
                "required_statement", "caveats"):
        assert key in r, key
    assert r["stage"] == "8.2_modelscope_real_checkpoint"
    assert r["status"].startswith("skipped")


# 8.
def test_report_has_no_tensor_dumps(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu")
    r = run_modelscope_real_checkpoint_probe(cfg)
    text = json.dumps(r, default=str)
    assert "tensor(" not in text
    import re
    assert re.search(r"(-?\d+\.\d+\s*,\s*){50,}", text) is None
    assert len(text) < 200_000


# 9.
def test_real_checkpoint_cli_help() -> None:
    proc = subprocess.run([sys.executable, str(CLI), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "model-id" in proc.stdout


# 10.
def test_dtype_selection_bfloat16_or_float16() -> None:
    allowed = {torch.bfloat16, torch.float16, torch.float32}
    for name in ("bfloat16", "float16", "float32"):
        for dev in ("cuda", "cpu"):
            dt = resolve_dtype(name, dev)
            assert dt in allowed
            assert dt is not torch.float64
    # explicit fp16 is always honored
    assert resolve_dtype("float16", "cuda") is torch.float16
    assert resolve_dtype("float16", "cpu") is torch.float16
    # explicit fp32 honored; bf16 never degrades to float64
    assert resolve_dtype("float32", "cpu") is torch.float32


# ---------------------------------------------------------------------------
# QR-safety regression tests (Stage 8.2 bf16 CUDA fix)
# ---------------------------------------------------------------------------


def _tiny_skeleton(mask_mode: str, strategy: str, dtype=torch.float32):
    """Build a tiny random Qwen2 + extract + return (weights, configs, cfg)."""
    pytest.importorskip("transformers")
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        extract_hf_causal_lm_skeleton_weights, make_random_tiny_hf_causal_lm)
    cfg = HFCausalLMSkeletonConfig(
        model_family="qwen2", max_layers=2, prefill_seq_len=8, decode_steps=4,
        max_vocab_size=256, dtype=dtype, device="cpu", seed=5,
        mask_mode=mask_mode, residual_mask_strategy=strategy)
    model, mc = make_random_tiny_hf_causal_lm(cfg)
    w, lc, _ = extract_hf_causal_lm_skeleton_weights(
        model, mc, max_layers=2, dtype=dtype, device="cpu")
    return w, lc, cfg


# 11.
def test_signed_permutation_real_mask_does_not_call_qr(monkeypatch) -> None:
    w, lc, cfg = _tiny_skeleton("signed_permutation", "shared")

    def _boom(*a, **k):
        raise AssertionError("torch.linalg.qr must not be called for "
                             "signed_permutation")

    monkeypatch.setattr(torch.linalg, "qr", _boom)
    masks = generate_hf_causal_lm_masks(w, lc, cfg)  # must not raise
    assert masks.metadata["qr_free_residual_mask"] is True
    assert masks.metadata["materialized_dense_from_signed_perm"] is True


# 12.
def test_block_orthogonal_qr_runs_cpu_float32(monkeypatch) -> None:
    seen: list[tuple] = []
    real_qr = torch.linalg.qr

    def _spy(inp, *a, **k):
        seen.append((inp.dtype, inp.device.type))
        return real_qr(inp, *a, **k)

    monkeypatch.setattr(torch.linalg, "qr", _spy)
    g = torch.Generator().manual_seed(3)
    # bf16 target => QR must be CPU float32, output cast to bf16.
    m, _inv = make_residual_mask(96, "block_orthogonal", torch.bfloat16,
                                 torch.device("cpu"), g, block_size=32)
    assert seen, "QR should have run for block_orthogonal"
    assert all(dt == torch.float32 and dev == "cpu" for dt, dev in seen)
    assert m.dtype == torch.bfloat16


# 13.
def test_bfloat16_cuda_signed_permutation_mask_generation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    g = torch.Generator(device="cuda").manual_seed(0)
    m, inv = make_residual_mask(128, "signed_permutation", torch.bfloat16,
                                torch.device("cuda"), g)
    assert m.dtype == torch.bfloat16 and m.device.type == "cuda"
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16, device="cuda")
    assert torch.allclose((x @ m) @ inv, x, atol=1e-2)


# 14.
def test_dense_orthogonal_large_hidden_rejected_without_override() -> None:
    g = torch.Generator().manual_seed(4)
    with pytest.raises(ValueError):
        make_residual_mask(2048, "dense_orthogonal", torch.bfloat16,
                           torch.device("cpu"), g)
    m, _ = make_residual_mask(1100, "dense_orthogonal", torch.bfloat16,
                              torch.device("cpu"), g, allow_dense_large=True)
    assert m.shape == (1100, 1100) and m.dtype == torch.bfloat16


# 15.
def test_stage8_2_config_signed_permutation_is_default() -> None:
    c = ModelScopeRealCheckpointProbeConfig()
    assert c.mask_mode == "signed_permutation"
    assert c.residual_mask_strategy == "shared"


# 16.
def test_build_attention_mask_all_ones_long() -> None:
    from pllo.experiments.modelscope_real_checkpoint_probe import (
        build_attention_mask)
    ids = torch.tensor([[5, 9, 2, 2]], dtype=torch.long)
    am = build_attention_mask(ids)
    assert am.shape == ids.shape
    assert am.dtype == torch.long
    assert am.device == ids.device
    assert torch.equal(am, torch.ones_like(ids))


# 17.
def test_probe_passes_explicit_attention_mask_when_pad_eq_eos() -> None:
    """When pad_token_id == eos_token_id, the HF baseline must still receive an
    explicit all-ones attention_mask (no padding) -- silencing the warning."""
    from pllo.experiments.modelscope_real_checkpoint_probe import _hf_baseline

    captured: dict = {}

    class _FakeTok:
        pad_token_id = 7
        eos_token_id = 7  # pad == eos: HF cannot infer the mask

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3, 7]], dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            return "hi"

    class _FakeModel:
        def generate(self, input_ids, attention_mask=None, **kw):
            captured["attention_mask"] = attention_mask
            captured["input_ids"] = input_ids
            return torch.cat([input_ids, input_ids[:, :1]], dim=1)

    cfg = ModelScopeRealCheckpointProbeConfig(device="cpu", decode_steps=1)
    out = _hf_baseline(_FakeModel(), _FakeTok(), cfg, "cpu")
    am = captured["attention_mask"]
    assert am is not None, "attention_mask must be passed explicitly"
    assert am.dtype == torch.long
    assert torch.equal(am, torch.ones_like(captured["input_ids"]))
    assert out["new_token_ids"]  # baseline still produced tokens


# 18.
def test_report_has_attention_mask_explicit_field(monkeypatch) -> None:
    monkeypatch.setattr(M, "has_modelscope", lambda: False)
    r = run_modelscope_real_checkpoint_probe(
        ModelScopeRealCheckpointProbeConfig(device="cpu"))
    assert r["attention_mask_explicit"] is True
    assert r["no_padding_assumption"] is True


# ---------------------------------------------------------------------------
# Audit / prompt-file / negative-control tests (Stage 8.2 audit)
#
# Run the *full* probe path (extraction -> masked decode -> audit) against a
# tiny random Qwen2 + a deterministic fake tokenizer by monkeypatching the
# ModelScope loader. CPU, float32; no network, no CUDA, no real checkpoint.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic char-level tokenizer over a tiny vocab (no network)."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, vocab: int) -> None:
        self._vocab = vocab

    def _ids(self, text: str) -> list[int]:
        ids = [(ord(c) % (self._vocab - 1)) + 1 for c in text] or [1]
        return ids

    def __call__(self, text, return_tensors=None):
        ids = self._ids(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "x"


def _patch_tiny_loaded(monkeypatch, *, total_layers: int = 2,
                       vocab: int = 64):
    """Monkeypatch the loader to return a tiny random Qwen2 + fake tokenizer."""
    pytest.importorskip("transformers")
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        HFCausalLMSkeletonConfig, make_random_tiny_hf_causal_lm)
    skel = HFCausalLMSkeletonConfig(
        model_family="qwen2", max_layers=total_layers, max_vocab_size=vocab,
        dtype=torch.float32, device="cpu", seed=7)
    model, _mc = make_random_tiny_hf_causal_lm(skel)
    tok = _FakeTokenizer(int(model.config.vocab_size))

    def _fake_load(model_id, cache_dir, dtype, device):
        return {"status": "ok", "model": model, "tokenizer": tok,
                "local_path": "/fake/modelscope_cache/" + model_id,
                "moved_to_cuda": False}

    monkeypatch.setattr(M, "load_modelscope_checkpoint", _fake_load)
    monkeypatch.setattr(M, "has_modelscope", lambda: True)
    monkeypatch.setattr(M, "has_transformers", lambda: True)
    return model, tok


def _tiny_cfg(**kw):
    base = dict(device="cpu", dtype="float32", folding_dtype="float32",
                folded_weight_runtime_dtype="float32", recovery_dtype="float32",
                compare_dtype="float32", prefill_seq_len=8, decode_steps=2,
                max_layers="1", mask_mode="signed_permutation",
                residual_mask_strategy="shared")
    base.update(kw)
    return ModelScopeRealCheckpointProbeConfig(**base)


def _write_prompt_file(tmp_path):
    p = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "en_privacy", "prompt": "How is my private data protected?"},
        {"id": "zh_privacy", "prompt": "我的隐私数据如何被保护?"},
        {"id": "code", "prompt": "def add(a, b):\n    return a + b"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    return p, rows


AUDIT_REQUIRED_FIELDS = (
    "model_id", "local_checkpoint_path", "checkpoint_source",
    "hf_remote_download_used", "tokenizer_used", "input_source",
    "prompt_count", "prefill_seq_len", "decode_steps", "input_ids_shape",
    "attention_mask_explicit", "num_hidden_layers_total", "max_layers_executed",
    "hidden_size", "intermediate_size", "num_attention_heads",
    "num_key_value_heads", "vocab_size", "dtype", "folding_dtype",
    "folded_weight_runtime_dtype", "recovery_dtype", "compare_dtype",
    "mask_mode", "residual_mask_strategy", "used_extracted_weights",
    "used_masked_embedding_boundary", "used_masked_decoder_blocks",
    "used_folded_qkv", "used_folded_mlp", "used_masked_kv_cache",
    "used_masked_lm_head", "used_vocab_mask_recovery", "hf_baseline_only",
    "simulated_tee_only",
)


# 19.
def test_audit_fields_present(monkeypatch) -> None:
    _patch_tiny_loaded(monkeypatch)
    r = run_modelscope_real_checkpoint_probe(_tiny_cfg())
    assert r["status"] == "ok"
    audit = r["audit"]
    for f in AUDIT_REQUIRED_FIELDS:
        assert f in audit, f"missing audit field {f!r}"
    # provenance / posture
    assert audit["checkpoint_source"] == "ModelScope"
    assert audit["hf_remote_download_used"] is False
    assert audit["simulated_tee_only"] is True
    assert audit["hf_baseline_only"] is False
    # partial-layer probe: executed < total
    assert audit["max_layers_executed"] == 1
    assert audit["num_hidden_layers_total"] == 2
    # the masked operator path actually ran
    for k in ("used_extracted_weights", "used_masked_embedding_boundary",
              "used_masked_decoder_blocks", "used_folded_qkv",
              "used_folded_mlp", "used_masked_kv_cache", "used_masked_lm_head",
              "used_vocab_mask_recovery"):
        assert audit[k] is True, k
    # normal run -> expected to match, and it does
    assert r["negative_control"] == "none"
    assert r["expected_to_match"] is True
    assert r["negative_control_passed"] is True


# 20.
def test_prompt_file_tokenization_real_tokenizer_or_mock(monkeypatch,
                                                         tmp_path) -> None:
    _patch_tiny_loaded(monkeypatch)
    pf, rows = _write_prompt_file(tmp_path)
    r = run_modelscope_real_checkpoint_probe(
        _tiny_cfg(prompt_file=str(pf), prefill_seq_len=16))
    assert r["status"] == "ok"
    audit = r["audit"]
    assert audit["input_source"] == "prompt_file"
    assert audit["prompt_count"] == len(rows)
    assert audit["tokenizer_used"] is True
    # input_ids: [N prompts, prefill_seq_len]
    assert audit["input_ids_shape"] == [len(rows), 16]
    # length stats recorded; full prompt text NOT stored by default
    stats = audit["tokenized_length_stats"]
    assert stats["count"] == len(rows) and stats["max"] >= stats["min"]
    text = json.dumps(r, default=str)
    assert "How is my private data protected?" not in text
    # prompt ids ARE stored
    assert "prompt_ids" in r["input"] and len(r["input"]["prompt_ids"]) == 3
    # include flag stores prompts
    r2 = run_modelscope_real_checkpoint_probe(
        _tiny_cfg(prompt_file=str(pf), prefill_seq_len=16,
                  include_prompts_in_report=True))
    assert "How is my private data protected?" in json.dumps(r2, default=str)


# 21.
def test_negative_control_wrong_vocab_recovery_fails(monkeypatch,
                                                     tmp_path) -> None:
    _patch_tiny_loaded(monkeypatch)
    pf, _ = _write_prompt_file(tmp_path)
    r = run_modelscope_real_checkpoint_probe(
        _tiny_cfg(prompt_file=str(pf), prefill_seq_len=16,
                  negative_control="wrong_vocab_recovery"))
    assert r["status"] == "ok"
    assert r["negative_control"] == "wrong_vocab_recovery"
    assert r["expected_to_match"] is False
    mr = r["masked_runtime"]
    # mismatch is the expected (passing) outcome
    assert mr["token_match_rate_vs_extracted"] < 1.0
    assert mr["recovered_logits_max_abs_error"] > 1e-3
    assert r["negative_control_passed"] is True
    assert r["audit"]["used_vocab_mask_recovery"] is False


# 22.
def test_negative_control_plaintext_weights_on_masked_hidden_fails(
        monkeypatch, tmp_path) -> None:
    _patch_tiny_loaded(monkeypatch)
    pf, _ = _write_prompt_file(tmp_path)
    r = run_modelscope_real_checkpoint_probe(
        _tiny_cfg(prompt_file=str(pf), prefill_seq_len=16,
                  negative_control="plaintext_weights_on_masked_hidden"))
    assert r["status"] == "ok"
    assert r["negative_control"] == "plaintext_weights_on_masked_hidden"
    assert r["expected_to_match"] is False
    mr = r["masked_runtime"]
    assert mr["token_match_rate_vs_extracted"] < 1.0
    assert mr["recovered_logits_max_abs_error"] > 1e-3
    assert r["negative_control_passed"] is True


# 23.
def test_attention_mask_explicit_for_prompt_file(monkeypatch,
                                                 tmp_path) -> None:
    _patch_tiny_loaded(monkeypatch)
    pf, rows = _write_prompt_file(tmp_path)
    r = run_modelscope_real_checkpoint_probe(
        _tiny_cfg(prompt_file=str(pf), prefill_seq_len=16))
    audit = r["audit"]
    assert audit["attention_mask_explicit"] is True
    assert audit["attention_mask_shape"] == [len(rows), 16]
    assert r["input"]["attention_mask_all_ones"] is True


# 24.
def test_unknown_negative_control_rejected(monkeypatch) -> None:
    _patch_tiny_loaded(monkeypatch)
    with pytest.raises(ValueError):
        run_modelscope_real_checkpoint_probe(_tiny_cfg(negative_control="bogus"))
