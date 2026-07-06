"""ObfuscaTune-Qwen KV cache correctness (prefill/decode) + honest cache labels."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.qwen_config import (  # noqa: E402
    extract_arch, load_qwen, make_tiny_qwen_config)
from pllo.baselines.obfuscatune.qwen_cache import (  # noqa: E402
    QwenObfCache, cache_metrics, hf_cache_shapes)
from pllo.baselines.obfuscatune.qwen_obfuscatune import (  # noqa: E402
    run_decode_step, run_prefill)
from pllo.baselines.obfuscatune.qwen_metrics import qwen_security_proxy  # noqa: E402


def _model(dtype=torch.float64):
    cfg = make_tiny_qwen_config()
    return load_qwen(tiny_random_qwen=cfg, seed=0, dtype=dtype)


def test_prefill_cache_shape_matches_hf():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 9))
    r = run_prefill(model, ids, "orthogonal", dtype=torch.float64)
    ref = model(ids, use_cache=True)
    obf_shapes = r["cache"].shapes()
    ref_shapes = hf_cache_shapes(ref.past_key_values)
    assert obf_shapes == ref_shapes
    assert r["cache"].length() == 9


def test_decode_appends_one_and_matches_recompute():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 9))
    nxt = torch.randint(0, arch.vocab_size, (1, 1))
    r = run_decode_step(model, ids, nxt, "orthogonal", dtype=torch.float64)
    assert r["prefill_cache_len"] == 9
    assert r["decode_cache_len"] == 10
    assert r["correctness"]["max_abs_error"] < 1e-8
    assert r["argmax_match_rate"] == 1.0


def test_gqa_cache_head_count():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 5))
    r = run_prefill(model, ids, "orthogonal", dtype=torch.float64)
    # cache keys have n_kv heads, not n_attention heads (GQA)
    for shape in r["cache"].shapes():
        assert shape[1] == arch.num_key_value_heads


def test_security_proxy_does_not_overclaim_kv_protection():
    sp = qwen_security_proxy()
    assert sp["protected_kv_cache"] == "false"      # plaintext K/V exposed
    assert sp["public_qkv"] is True
    assert QwenObfCache.cache_obfuscation == "plain_intermediate_exposed"


def test_cache_metrics_shape_mismatch_detected():
    m = cache_metrics(enabled=True, prefill_cache_len=5, decode_cache_len=6,
                      obf_shapes=[(1, 2, 5, 8)], ref_shapes=[(1, 2, 6, 8)])
    assert m["cache_shape_match"] is False
    assert m["cache_obfuscation"] == "plain_intermediate_exposed"
