"""Stage 6.4 — GQA / MQA probe tests."""

from __future__ import annotations

import pytest
import torch

from pllo.experiments.gqa_probe import GqaProbeConfig, repeat_kv, run_gqa_probe


# ---------------------------------------------------------------------------
# repeat_kv
# ---------------------------------------------------------------------------


def test_repeat_kv_shape() -> None:
    x = torch.randn(2, 2, 8, 16)
    y = repeat_kv(x, 4)
    assert y.shape == (2, 8, 8, 16)
    # Each kv head should appear ``n_rep`` times in adjacent slots.
    for k in range(2):
        for r in range(4):
            assert torch.equal(y[:, k * 4 + r, :, :], x[:, k, :, :])


def test_repeat_kv_identity_for_nrep_one() -> None:
    x = torch.randn(2, 4, 8, 16)
    assert torch.equal(repeat_kv(x, 1), x)


# ---------------------------------------------------------------------------
# Probe invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("hq", "hk"), [(4, 1), (4, 2), (4, 4), (8, 2)])
def test_gqa_score_and_value_invariants_allclose(hq: int, hk: int) -> None:
    report = run_gqa_probe(
        GqaProbeConfig(
            batch_size=2,
            num_query_heads=hq,
            num_kv_heads=hk,
            seq_len=8,
            head_dim=16,
        )
    )
    assert report["status"] == "ok"
    assert report["score_path"]["allclose"] is True, report["score_path"]
    assert report["value_path"]["allclose"] is True, report["value_path"]
    assert report["qk_constraint_max_error_per_q_head"] < 1e-3


def test_gqa_attention_variant_label() -> None:
    assert (
        run_gqa_probe(GqaProbeConfig(num_query_heads=4, num_kv_heads=1))[
            "attention_variant"
        ]
        == "mqa"
    )
    assert (
        run_gqa_probe(GqaProbeConfig(num_query_heads=4, num_kv_heads=2))[
            "attention_variant"
        ]
        == "gqa"
    )
    assert (
        run_gqa_probe(GqaProbeConfig(num_query_heads=4, num_kv_heads=4))[
            "attention_variant"
        ]
        == "mha"
    )


def test_gqa_mask_dimension_is_head_dim() -> None:
    """Mask dimension must equal head_dim, NOT hidden_size and NOT num_heads."""
    report = run_gqa_probe(
        GqaProbeConfig(num_query_heads=4, num_kv_heads=2, head_dim=16)
    )
    assert report["mask_dimension"] == 16
    assert report["mask_is_per_head_not_hidden_size"] is True
    assert report["mask_is_per_head_not_num_heads"] is True


def test_gqa_rejects_non_divisible_heads() -> None:
    report = run_gqa_probe(GqaProbeConfig(num_query_heads=5, num_kv_heads=2))
    assert report["status"] == "skipped"
    assert "divisible" in report["reason"]
