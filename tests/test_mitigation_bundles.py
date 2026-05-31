"""Stage 5.3e — mitigation bundle enum + helpers tests."""

from __future__ import annotations

import pytest

from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    RECOMMENDED_DEFAULT_ON_BUNDLE,
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
    describe_mitigation_bundle,
    normalize_mitigation_bundle,
)


def test_default_bundle_is_fresh_perm_only() -> None:
    assert DEFAULT_MITIGATION_BUNDLE == "fresh_perm_only"


def test_recommended_default_on_bundle_is_full() -> None:
    assert RECOMMENDED_DEFAULT_ON_BUNDLE == "fresh_perm_plus_sandwich_plus_pad"


def test_valid_bundles_contains_both() -> None:
    assert set(VALID_MITIGATION_BUNDLES) == {
        "fresh_perm_only",
        "fresh_perm_plus_sandwich_plus_pad",
    }


def test_normalize_none_returns_default() -> None:
    assert normalize_mitigation_bundle(None) == DEFAULT_MITIGATION_BUNDLE


def test_normalize_passthrough_valid() -> None:
    assert (
        normalize_mitigation_bundle("fresh_perm_only") == "fresh_perm_only"
    )
    assert (
        normalize_mitigation_bundle("fresh_perm_plus_sandwich_plus_pad")
        == "fresh_perm_plus_sandwich_plus_pad"
    )


def test_normalize_invalid_raises() -> None:
    with pytest.raises(ValueError):
        normalize_mitigation_bundle("not_a_bundle")


def test_describe_fresh_perm_only_flags() -> None:
    d = describe_mitigation_bundle("fresh_perm_only")
    assert d.fresh_permutation_enabled is True
    assert d.dense_sandwich_enabled is False
    assert d.boundary_pad_required is False
    assert d.default_on_candidate_under_stage_5_4 is False
    assert d.risk_level_from_stage_5_4 == "medium"
    assert d.default_on_recommendation == "needs_more_evaluation"
    assert d.activation_input_form == "ZP"
    assert d.activation_pad_forbidden is True


def test_describe_full_bundle_flags() -> None:
    d = describe_mitigation_bundle("fresh_perm_plus_sandwich_plus_pad")
    assert d.fresh_permutation_enabled is True
    assert d.dense_sandwich_enabled is True
    assert d.boundary_pad_required is True
    assert d.default_on_candidate_under_stage_5_4 is True
    assert d.risk_level_from_stage_5_4 == "low"
    assert d.default_on_recommendation == "acceptable_with_mitigation"
    assert d.activation_input_form == "ZP"
    assert d.activation_pad_forbidden is True
    assert d.post_island_dense_mask is True


def test_bundle_metadata_default_on_candidate_requires_pad() -> None:
    full_no_pad = bundle_metadata(
        "fresh_perm_plus_sandwich_plus_pad", use_pad=False
    )
    full_with_pad = bundle_metadata(
        "fresh_perm_plus_sandwich_plus_pad", use_pad=True
    )
    assert full_no_pad["default_on_candidate_under_stage_5_4"] is False
    assert full_with_pad["default_on_candidate_under_stage_5_4"] is True
    assert full_no_pad["boundary_pad_enabled"] is False
    assert full_with_pad["boundary_pad_enabled"] is True
    assert full_no_pad["boundary_pad_required"] is True
    assert full_with_pad["boundary_pad_required"] is True


def test_bundle_metadata_records_no_extra_matmul() -> None:
    for bundle in VALID_MITIGATION_BUNDLES:
        for use_pad in (False, True):
            m = bundle_metadata(bundle, use_pad=use_pad)
            assert m["online_extra_matmul_count"] == 0
            assert m["activation_input_form"] == "ZP"
            assert m["activation_pad_forbidden"] is True
            assert m["pad_placement"] in {"linear_boundary_only", "n/a"}


def test_bundle_metadata_preprocessing_only_transformations() -> None:
    m_only = bundle_metadata("fresh_perm_only", use_pad=True)
    m_full = bundle_metadata("fresh_perm_plus_sandwich_plus_pad", use_pad=True)
    assert "permutation_absorbed_into_weights" in m_only["preprocessing_only_transformations"]
    assert (
        "dense_input_mask"
        in m_full["preprocessing_only_transformations"]
    )
    assert (
        "dense_output_mask"
        in m_full["preprocessing_only_transformations"]
    )
