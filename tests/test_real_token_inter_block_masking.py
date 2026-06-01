"""Stage 5.6 extension — tests for real-token attacker under masked boundary."""

from __future__ import annotations

import pytest

from pllo.experiments.real_token_activation_attacker import (
    RealTokenActivationAttackConfig,
    run_real_token_activation_attacks,
)


def _cfg(**overrides):
    cfg = dict(
        seed=2026,
        num_prompts=4,
        prompt_max_length=6,
        max_new_tokens=2,
        attacker_steps=10,
        mlp_hidden_size=32,
        mlp_batch_size=16,
        use_pad=True,
        nonlinear_mode="compatible_islands",
        synthetic_vocab_size=32,
        synthetic_hidden_size=16,
        synthetic_intermediate_size=32,
        synthetic_num_attention_heads=4,
        synthetic_num_key_value_heads=2,
        synthetic_head_dim=4,
        max_layers=2,
    )
    cfg.update(overrides)
    return RealTokenActivationAttackConfig(**cfg)


def test_attacker_accepts_inter_block_mask_mode() -> None:
    cfg = _cfg(inter_block_mask_mode="masked_boundary_experimental")
    r = run_real_token_activation_attacks(cfg)
    assert r["metadata"]["inter_block_mask_mode"] == "masked_boundary_experimental"
    assert r["metadata"]["boundary_mask_status"] == "masked"
    assert r["recommendation"]["inter_block_mask_mode"] == "masked_boundary_experimental"


def test_plain_boundary_marks_boundary_input_inter_block_plain() -> None:
    cfg = _cfg(inter_block_mask_mode="plain_boundary")
    r = run_real_token_activation_attacks(cfg)
    full = r["target_tensor_results"]["fresh_perm_plus_sandwich_plus_pad"]["prefill"]
    assert full["boundary_input"]["inter_block_plain"] is True
    assert full["final"]["inter_block_plain"] is True


def test_masked_boundary_unmarks_inter_block_plain() -> None:
    cfg = _cfg(inter_block_mask_mode="masked_boundary_experimental")
    r = run_real_token_activation_attacks(cfg)
    full = r["target_tensor_results"]["fresh_perm_plus_sandwich_plus_pad"]["prefill"]
    assert full["boundary_input"]["inter_block_plain"] is False
    assert full["final"]["inter_block_plain"] is False
    # And those tensors should no longer trivially recover (linear rel_l2 > 0.5).
    assert full["boundary_input"]["linear_inverter"]["relative_l2_error"] > 0.5
    assert full["final"]["linear_inverter"]["relative_l2_error"] > 0.5
