"""Stage 5.3a — feature-flag enum for the GPT-2 single-block wrapper.

Two modes are exposed:

* ``"trusted"`` — default; LayerNorm and GELU run as trusted-side shortcuts
  exactly as in Stage 4.6 / 4.7 / 4.9. All existing tests assume this mode
  and continue to pass without modification.
* ``"compatible_islands"`` — Stage 5.2a GELU MLP permutation island is
  spliced into the block's MLP path. LayerNorm remains a trusted shortcut,
  the LM head is not modified, and the KV cache / generation paths are not
  modified. Pad compensation, when ``use_pad=True``, is applied only at
  the Linear boundary (``c_fc``); the pad is never pushed through GELU.

Security caveat: ``"compatible_islands"`` is gated on the Stage 5.2b
naive-observer security proxy. It is *not* formally secure. Production
deployments must follow the Stage 5.2b mitigations (fresh permutation per
session, dense-mask sandwiching at Linear boundaries, pad at Linear
boundaries only) and remain behind a feature flag — this mode must not
be enabled by default.
"""

from __future__ import annotations

from typing import Final

VALID_NONLINEAR_MODES: Final[tuple[str, ...]] = ("trusted", "compatible_islands")
DEFAULT_NONLINEAR_MODE: Final[str] = "trusted"


def normalize_nonlinear_mode(mode: str | None) -> str:
    """Validate and canonicalize a user-supplied ``nonlinear_mode`` argument."""
    if mode is None:
        return DEFAULT_NONLINEAR_MODE
    if mode not in VALID_NONLINEAR_MODES:
        raise ValueError(
            f"nonlinear_mode must be one of {VALID_NONLINEAR_MODES}, got {mode!r}"
        )
    return mode


__all__ = [
    "DEFAULT_NONLINEAR_MODE",
    "VALID_NONLINEAR_MODES",
    "normalize_nonlinear_mode",
]
