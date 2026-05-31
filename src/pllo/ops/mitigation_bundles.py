"""Stage 5.3e — mitigation bundle enum for the compatible nonlinear-island
feature flag.

Stage 5.4's adaptive proxy attacker identified

    fresh_permutation + dense_sandwich + pad at Linear boundaries

as the only mask-strategy combination that lands at ``risk_level="low"``
and ``default_on_recommendation="acceptable_with_mitigation"`` under the
tested adaptive attacker. This module names that bundle as the explicit
opt-in for wrappers / probes / scripts.

Two bundles are defined:

* ``"fresh_perm_only"`` (DEFAULT) — preserves the Stage 5.2a / 5.3a /
  5.3b / 5.3c / 6.4 behaviour exactly. Pads may or may not be applied;
  N_in / N_out are still freshly sampled per call by
  ``SimulatedTEE.create_linear_mask_state``, but the bundle does not
  *require* either pad or dense sandwich semantics.
* ``"fresh_perm_plus_sandwich_plus_pad"`` — opt-in. Treats the per-call
  fresh ``N_in`` and ``N_out`` as a dense sandwich around the
  permutation island, forces ``use_pad`` at the Linear boundary
  whenever the caller supplies a pad slot, and surfaces structured
  metadata that downstream reports use to label the bundle as the
  Stage 5.4 default-on candidate.

Mathematics unchanged: the bundle does not introduce new online
matmuls. The existing per-call ``N_in`` / ``perm`` / ``N_out`` sampling
in the Stage 5.2a island APIs already satisfies the dense-sandwich +
fresh-permutation contract; this module is the explicit metadata
contract over that behaviour.

Security disclaimer: the labels in this module describe behaviour
*under the Stage 5.4 adaptive proxy attacker only*. They are not formal
security claims, not real TEE measurements, and the wider system
default mode (``nonlinear_mode``) remains ``"trusted"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


VALID_MITIGATION_BUNDLES: Final[tuple[str, ...]] = (
    "fresh_perm_only",
    "fresh_perm_plus_sandwich_plus_pad",
)
DEFAULT_MITIGATION_BUNDLE: Final[str] = "fresh_perm_only"
RECOMMENDED_DEFAULT_ON_BUNDLE: Final[str] = "fresh_perm_plus_sandwich_plus_pad"


def normalize_mitigation_bundle(bundle: str | None) -> str:
    """Validate and canonicalize a user-supplied ``mitigation_bundle``.

    * ``None`` → :data:`DEFAULT_MITIGATION_BUNDLE`.
    * Any other value must be in :data:`VALID_MITIGATION_BUNDLES`;
      otherwise raises ``ValueError`` with a clear message.
    """
    if bundle is None:
        return DEFAULT_MITIGATION_BUNDLE
    if bundle not in VALID_MITIGATION_BUNDLES:
        raise ValueError(
            f"mitigation_bundle must be one of {VALID_MITIGATION_BUNDLES},"
            f" got {bundle!r}"
        )
    return bundle


@dataclass(frozen=True)
class MitigationBundleDescriptor:
    """Static description of a mitigation bundle.

    Used by reports / wrappers / scripts to surface a structured opinion
    on what the bundle protects against without leaking secret tensors.
    """

    name: str
    fresh_permutation_enabled: bool
    dense_sandwich_enabled: bool
    boundary_pad_required: bool
    activation_pad_forbidden: bool
    island_view_lifetime: str
    activation_input_form: str
    post_island_dense_mask: bool
    default_on_candidate_under_stage_5_4: bool
    risk_level_from_stage_5_4: str
    default_on_recommendation: str
    security_profile_detail: str
    notes: str


_FRESH_PERM_ONLY = MitigationBundleDescriptor(
    name="fresh_perm_only",
    fresh_permutation_enabled=True,
    dense_sandwich_enabled=False,
    boundary_pad_required=False,
    activation_pad_forbidden=True,
    island_view_lifetime="short_lived",
    activation_input_form="ZP",
    post_island_dense_mask=False,
    default_on_candidate_under_stage_5_4=False,
    risk_level_from_stage_5_4="medium",
    default_on_recommendation="needs_more_evaluation",
    security_profile_detail="adaptive-proxy-evaluated, not formal",
    notes=(
        "Stage 5.2a / 5.3a baseline. Per-call fresh permutation only. The"
        " Stage 5.4 adaptive cross-session signature-aggregation attacker"
        " still recovers permutation top1 ≈ 0.25 at hidden=64."
    ),
)


_FRESH_PERM_PLUS_SANDWICH_PLUS_PAD = MitigationBundleDescriptor(
    name="fresh_perm_plus_sandwich_plus_pad",
    fresh_permutation_enabled=True,
    dense_sandwich_enabled=True,
    boundary_pad_required=True,
    activation_pad_forbidden=True,
    island_view_lifetime="short_lived",
    activation_input_form="ZP",
    post_island_dense_mask=True,
    default_on_candidate_under_stage_5_4=True,
    risk_level_from_stage_5_4="low",
    default_on_recommendation="acceptable_with_mitigation",
    security_profile_detail="adaptive-proxy-mitigated, not formal",
    notes=(
        "Stage 5.3e bundle. Per-call fresh permutation, dense N_in / N_out"
        " sandwiching the activation island, and pad applied at every"
        " Linear boundary (pad is never pushed through the activation)."
        " Stage 5.4's tested adaptive proxy attackers (ridge linear, small"
        " MLP, Sinkhorn-style permutation recovery) place this bundle at"
        " low risk with default_on_recommendation='acceptable_with_mitigation'."
    ),
)


_DESCRIPTORS: Final[dict[str, MitigationBundleDescriptor]] = {
    _FRESH_PERM_ONLY.name: _FRESH_PERM_ONLY,
    _FRESH_PERM_PLUS_SANDWICH_PLUS_PAD.name: _FRESH_PERM_PLUS_SANDWICH_PLUS_PAD,
}


def describe_mitigation_bundle(bundle: str | None) -> MitigationBundleDescriptor:
    """Return the static descriptor for a mitigation bundle."""
    return _DESCRIPTORS[normalize_mitigation_bundle(bundle)]


def bundle_metadata(
    bundle: str | None, *, use_pad: bool, online_extra_matmul_count: int = 0
) -> dict:
    """Return a JSON-safe metadata dict describing the bundle in context.

    ``use_pad`` is the caller's actual pad selection at the Linear
    boundary. For ``fresh_perm_plus_sandwich_plus_pad`` the
    ``boundary_pad_enabled`` field is ``False`` (and not
    ``not_applicable_without_pad``) when the caller explicitly opts out
    of pad — the descriptor still records the bundle's *required*
    mitigations via ``boundary_pad_required`` so downstream readers can
    tell the difference between "bundle does not need pad" and "bundle
    needs pad but caller disabled it".
    """
    desc = describe_mitigation_bundle(bundle)
    return {
        "mitigation_bundle": desc.name,
        "fresh_permutation_enabled": desc.fresh_permutation_enabled,
        "dense_sandwich_enabled": desc.dense_sandwich_enabled,
        "boundary_pad_required": desc.boundary_pad_required,
        "boundary_pad_enabled": bool(use_pad),
        "activation_pad_forbidden": desc.activation_pad_forbidden,
        "pad_placement": "linear_boundary_only" if use_pad else "n/a",
        "activation_input_form": desc.activation_input_form,
        "post_island_dense_mask": desc.post_island_dense_mask,
        "island_view_lifetime": desc.island_view_lifetime,
        "default_on_candidate_under_stage_5_4": (
            desc.default_on_candidate_under_stage_5_4
            and (use_pad or not desc.boundary_pad_required)
        ),
        "risk_level_from_stage_5_4": desc.risk_level_from_stage_5_4,
        "default_on_recommendation": desc.default_on_recommendation,
        "security_profile_detail": desc.security_profile_detail,
        "online_extra_matmul_count": int(online_extra_matmul_count),
        "preprocessing_only_transformations": (
            ["dense_input_mask", "permutation_absorbed_into_weights",
             "dense_output_mask"]
            if desc.dense_sandwich_enabled
            else ["permutation_absorbed_into_weights"]
        ),
    }


__all__ = [
    "DEFAULT_MITIGATION_BUNDLE",
    "MitigationBundleDescriptor",
    "RECOMMENDED_DEFAULT_ON_BUNDLE",
    "VALID_MITIGATION_BUNDLES",
    "bundle_metadata",
    "describe_mitigation_bundle",
    "normalize_mitigation_bundle",
]
