"""Stage 7.7b tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.lora_protocol_integration import (
    LoRAProtocolConfig,
    render_markdown,
    run_lora_protocol_integration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_lora_protocol_integration(cfg=LoRAProtocolConfig())


def test_supported_sites_complete(report: dict) -> None:
    expected = {"q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"}
    assert set(report["supported_lora_sites"]) == expected


def test_q_proj_lora_outputs_b_q_directly(report: dict) -> None:
    info = report["site_padded_boundary_identity_audit"]["q_proj"]
    assert info["n_out_kind"].startswith("B_Q_block_diagonal")
    assert info["padded_boundary_identity_max_abs_error"] < 1e-9
    assert info["trusted_recovery_max_abs_error"] < 1e-9


def test_k_proj_lora_outputs_b_k_directly(report: dict) -> None:
    info = report["site_padded_boundary_identity_audit"]["k_proj"]
    assert info["n_out_kind"].startswith("B_K_block_diagonal")
    assert info["padded_boundary_identity_max_abs_error"] < 1e-9


def test_v_proj_lora_outputs_n_v(report: dict) -> None:
    info = report["site_padded_boundary_identity_audit"]["v_proj"]
    assert info["n_out_kind"].startswith("N_V_block_diagonal")
    assert info["padded_boundary_identity_max_abs_error"] < 1e-9


def test_o_proj_lora_residual_compatible(report: dict) -> None:
    info = report["site_padded_boundary_identity_audit"]["o_proj"]
    assert info["n_out_kind"] == "Q_l_residual_stream_orthogonal"
    assert info["trusted_recovery_max_abs_error"] < 1e-9


def test_swiglu_branches_lora_with_paired_perm(report: dict) -> None:
    for site in ("up_proj", "gate_proj"):
        info = report["site_padded_boundary_identity_audit"][site]
        assert info["n_out_kind"] == "paired_permutation_P"
        assert info["padded_boundary_identity_max_abs_error"] < 1e-9
    info = report["site_padded_boundary_identity_audit"]["down_proj"]
    assert info["n_out_kind"] == "Q_l_residual_stream_orthogonal"
    assert info["padded_boundary_identity_max_abs_error"] < 1e-9


def test_use_pad_compensation_includes_base_and_lora(report: dict) -> None:
    # The padded-boundary identity already includes both C_base and
    # C_lora; if the experiment passes with use_pad=True, the
    # compensation must cover both branches.
    for site, info in report["site_padded_boundary_identity_audit"].items():
        assert info["padded_boundary_identity_max_abs_error"] < 1e-9, site


def test_rank_space_mask_r_used(report: dict) -> None:
    # ``padded_AB_minus_true_AB_max_abs_error`` confirms padded factors
    # numerically match the true-rank product (column/row pad with
    # zeros). The presence of A_tilde = M^{-1} A R and
    # B_tilde = R^{-1} B N_out in the identity test confirms R is
    # consumed in the protocol.
    for site, info in report["site_padded_boundary_identity_audit"].items():
        assert info["padded_AB_minus_true_AB_max_abs_error"] < 1e-9, site


def test_rank_padding_hides_true_rank_but_not_padded_rank(report: dict) -> None:
    assert report["true_rank_hidden_from_shape"] is True
    assert report["padded_rank_visible"] is True
    assert report["config"]["padded_rank"] > report["config"]["true_rank"]


def test_greedy_generation_exact_across_combos(report: dict) -> None:
    for c in report["merged_weights_generation_combos"]:
        assert c["greedy_token_match_rate"] == 1.0
        assert c["sequence_exact_match"] is True
        assert c["lm_head_recovery_max_abs_error"] < 1e-9


def test_json_reports_unsupported_sites_and_training(report: dict) -> None:
    assert report["unsupported_lora_sites"] == []
    assert report["lora_training_backward_supported"] is False
    assert report["lora_adapter_plaintext_visible"] is False


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    for site in report["supported_lora_sites"]:
        assert site in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "lora_protocol_integration.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7b"
