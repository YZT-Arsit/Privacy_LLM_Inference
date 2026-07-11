"""Contract tests for the Gate 3.5 unified masked-training CPU prototype.

Asserts the masked A_rightmul forward + rank-masked LoRA backward/SGD close
together exactly (fp64), that every residual state is masked (never plaintext),
zero trusted nonlinear crossings, and that the observed facts satisfy the
paper_safe execution profile. CPU/synthetic contract; NOT a real model result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.execution_profile import (  # noqa: E402
    security_claim_allowed,
    validate_run,
)
from pllo.experiments.unified_masked_training import (  # noqa: E402
    BlockConfig,
    observed_facts_for_profile,
    run_contract,
)


def test_contract_all_checks_pass():
    res = run_contract()
    for name, ok in res.checks.items():
        assert ok, f"contract check failed: {name} (metrics={res.metrics})"
    assert res.all_passed


def test_forward_recovers_to_machine_epsilon():
    res = run_contract()
    assert res.metrics["forward_logits_max_abs_err"] < 1e-10
    assert res.metrics["next_step_logits_max_err"] < 1e-10


def test_inter_block_residual_is_masked_not_plaintext():
    res = run_contract()
    # masked == plaintext @ N exactly ...
    assert res.metrics["residual_h2_mask_err"] < 1e-9
    assert res.metrics["residual_h3_mask_err"] < 1e-9
    # ... and genuinely different from the plaintext hidden (masked, not identity)
    assert res.metrics["residual_h2_plain_gap"] > 1e-2
    assert res.metrics["residual_h3_plain_gap"] > 1e-2
    assert res.counters["plaintext_hidden_materializations"] == 0


def test_masked_lora_backward_and_sgd_exact():
    res = run_contract()
    assert res.metrics["gradA_recovered_max_err"] < 1e-8
    assert res.metrics["gradB_recovered_max_err"] < 1e-8
    assert res.metrics["masked_sgd_update_max_err"] < 1e-8


def test_private_ce_matches_plaintext():
    res = run_contract()
    assert res.metrics["loss_abs_diff_vs_plaintext"] < 1e-9


def test_boundary_discipline_counters():
    res = run_contract()
    assert res.counters["nonlinear_trusted_calls"] == 0
    assert res.counters["trusted_nonlinear_ops_count"] == 0
    assert res.counters["packed_update"] == 0
    assert res.counters["trusted_optimizer_calls"] == 0
    # exactly one trusted crossing: the private cross-entropy boundary
    assert res.counters["trusted_boundary_calls"] == 1
    # the masked nonlinear islands (3 rmsnorm + softmax + silu) ran on-accelerator
    assert res.counters["accelerator_nonlinear_ops"] == 5


def test_observed_facts_satisfy_paper_safe():
    res = run_contract()
    obs = observed_facts_for_profile(res)
    ok, violations = validate_run("paper_safe", obs)
    assert ok and not violations
    assert security_claim_allowed("paper_safe", obs) is True


@pytest.mark.parametrize("seed", [1, 7, 99])
def test_contract_stable_across_seeds(seed):
    cfg = BlockConfig(seed=seed)
    res = run_contract(cfg)
    assert res.all_passed, res.checks


def test_gqa_shape_variation():
    # n_kv_heads=1 (full GQA collapse) and =n_heads (MHA) both must close
    for nkv in (1, 2, 4):
        cfg = BlockConfig(n_kv_heads=nkv)
        res = run_contract(cfg)
        assert res.all_passed, (nkv, res.checks)
