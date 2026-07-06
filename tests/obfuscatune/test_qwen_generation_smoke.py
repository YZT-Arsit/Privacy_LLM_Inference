"""ObfuscaTune-Qwen greedy generation smoke (tiny random Qwen, no download)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.qwen_config import (  # noqa: E402
    extract_arch, load_qwen, make_tiny_qwen_config)
from pllo.baselines.obfuscatune.qwen_obfuscatune import generate_greedy  # noqa: E402


def _model(dtype=torch.float64):
    cfg = make_tiny_qwen_config()
    return load_qwen(tiny_random_qwen=cfg, seed=0, dtype=dtype)


def test_generation_runs_and_shapes_ok():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 6))
    r = generate_greedy(model, ids, max_new_tokens=4, mode="orthogonal", dtype=torch.float64)
    assert r["tokens"].shape == (1, 4)
    assert r["full"].shape == (1, 10)


def test_generation_orthogonal_matches_unprotected():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 6))
    ref = generate_greedy(model, ids, 4, "unprotected", dtype=torch.float64)
    obf = generate_greedy(model, ids, 4, "orthogonal", dtype=torch.float64)
    ref_t, obf_t = ref["tokens"][0].tolist(), obf["tokens"][0].tolist()
    match = sum(a == b for a, b in zip(ref_t, obf_t)) / len(ref_t)
    assert match == 1.0


def test_generation_cache_equals_no_cache():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 6))
    a = generate_greedy(model, ids, 4, "orthogonal", dtype=torch.float64, use_cache=True)
    b = generate_greedy(model, ids, 4, "orthogonal", dtype=torch.float64, use_cache=False)
    assert a["tokens"][0].tolist() == b["tokens"][0].tolist()


def test_generation_random_mode_finite():
    model = _model()
    arch = extract_arch(model)
    ids = torch.randint(0, arch.vocab_size, (1, 6))
    r = generate_greedy(model, ids, 4, "random", dtype=torch.float64)
    assert r["tokens"].shape == (1, 4)
    assert torch.isfinite(r["tokens"].float()).all()
