"""Baseline audits: STIP Theorem 1 holds; ObfuscaTune correct; gaps flagged."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config
from pllo.baselines.obfuscatune.audit import audit_obfuscatune_qwen
from pllo.baselines.stip.audit import audit_stip_qwen


def _model():
    return load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=0, dtype=torch.float64)


def test_stip_theorem1_holds_and_correct():
    r = audit_stip_qwen(_model())
    assert r["status"] == "correct"
    assert r["correctness"]["theorem1_holds"] is True
    assert r["correctness"]["theorem1_recovery_max_abs_error"] < 1e-5
    assert r["correctness"]["per_linear_max_abs_error"] < 1e-8


def test_stip_flags_paper_gaps_and_leakage():
    r = audit_stip_qwen(_model())
    assert len(r["paper_spec_gaps_flagged"]) >= 5
    assert r["leakage"]["attention_scores_unpermuted"] is True
    assert r["leakage"]["kv_cache_protected"] is False
    assert r["qwen_adaptation"]["uses_whole_head_rope_safe_permutation"] is True
    assert r["qwen_adaptation"]["swiglu_gate_up_share_intermediate_perm"] is True


def test_obfuscatune_correct_and_boundaries_honest():
    r = audit_obfuscatune_qwen(_model())
    assert r["status"] == "correct"
    assert r["correctness"]["logits_match_plaintext"] is True
    assert r["boundary_labels"]["rmsnorm_not_run_in_obfuscated_domain"] is True
    assert r["boundary_labels"]["silu_not_run_in_obfuscated_domain"] is True
    assert r["boundary_labels"]["kv_cache_protected"] is False
    assert r["boundary_labels"]["qkv_exposed_plaintext"] is True
