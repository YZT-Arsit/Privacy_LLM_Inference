"""Correctness + attack tests for protected (masked) LoRA training.

Verifies protected LoRA training is mathematically equivalent to plaintext LoRA
training (synthetic linear + tiny transformer, ranks 4/8/16) and that the attack
baselines behave as expected. numpy only.

Run: python -m pytest tests/test_lora_training_protection.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pllo.experiments.lora_training_protection import (
    adapter_recovery_attack,
    gradient_inversion_attack,
    membership_attack,
    run_all,
    run_gpt2_probe,
    run_qwen_probe,
    run_synthetic_linear,
    run_tiny_transformer,
)

RANKS = [4, 8, 16]


@pytest.mark.parametrize("rank", RANKS)
def test_synthetic_linear_exact_equivalence(rank) -> None:
    r = run_synthetic_linear(rank)
    m = r["metrics"]
    assert m["final_logits_allclose"] is True
    assert m["top1_match_rate"] == 1.0
    assert m["max_lora_a_error"] < 1e-9
    assert m["max_lora_b_error"] < 1e-9
    assert m["max_delta_w_error"] < 1e-9
    assert m["max_grad_a_error"] < 1e-9
    assert m["max_grad_b_error"] < 1e-9
    assert m["max_optimizer_state_error"] < 1e-9
    assert m["loss_curve_distance"] < 1e-7
    assert m["final_eval_delta"] < 1e-7
    assert m["tee_used_on_gpu"] is False
    # training actually learned the low-rank signal
    assert r["history"]["train_loss_protected"][-1] < \
        0.5 * r["history"]["train_loss_protected"][0]


@pytest.mark.parametrize("rank", RANKS)
def test_tiny_transformer_exact_equivalence(rank) -> None:
    r = run_tiny_transformer(rank)
    m = r["metrics"]
    assert m["final_logits_allclose"] is True
    assert m["top1_match_rate"] == 1.0
    assert m["max_param_error"] < 1e-9
    assert m["max_grad_error"] < 1e-9
    assert m["final_task_metric_plain"] == m["final_task_metric_protected"]
    assert m["tee_used_on_gpu"] is False
    # attention-V + MLP LoRA backward is correct -> loss drops substantially
    assert r["history"]["train_loss_protected"][-1] < \
        0.6 * r["history"]["train_loss_protected"][0]


@pytest.mark.parametrize("rank", RANKS)
def test_adapter_recovery_attack_fails(rank) -> None:
    r = run_synthetic_linear(rank)
    a = adapter_recovery_attack(r)
    # the GPU trace has zero information about A/B -> recovery no better than zero
    assert a["adapter_recovery_relative_error"] >= 0.99
    assert a["delta_w_recovery_relative_error"] >= 0.99
    assert a["baseline_exposed_relative_error"] == 0.0


@pytest.mark.parametrize("rank", RANKS)
def test_gradient_inversion_baseline_leaks_but_protected_does_not(rank) -> None:
    g = gradient_inversion_attack(rank)
    assert g["gradient_inversion_baseline_error"] < 1e-6      # plaintext leaks X
    assert g["gradient_inversion_reconstruction_error"] > 0.5  # protected does not


@pytest.mark.parametrize("rank", RANKS)
def test_membership_attack_near_random(rank) -> None:
    r = run_synthetic_linear(rank)
    m = membership_attack(r)
    assert 0.30 <= m["membership_attack_auc"] <= 0.70       # ~ chance
    assert m["membership_baseline_auc"] >= 0.9              # plaintext distinguishes


def test_gpt2_probe_returns_status() -> None:
    p = run_gpt2_probe()
    assert p["task"] == "gpt2"
    assert p["status"] in {"skipped", "available_not_run"}
    assert p["tee_used_on_gpu"] is False


def test_qwen_probe_is_feasibility_probe() -> None:
    p = run_qwen_probe()
    assert p["task"] == "qwen2.5-7b"
    assert p["tee_used_on_gpu"] is False
    assert p.get("probe_only") is True
    # without CUDA + checkpoint it must be skipped, never claimed complete
    assert p["status"] in {"skipped", "available_not_run"}


def test_run_all_summary_passes() -> None:
    s = run_all((4, 8), seed=0)
    assert s["all_allclose"] is True
    assert s["all_audits_passed"] is True
    assert s["tee_used_on_gpu"] is False
    assert len(s["correctness"]) == 4                      # 2 tasks x 2 ranks
