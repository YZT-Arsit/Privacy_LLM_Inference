"""Registry for selectable nonlinear-handling backends.

``current`` (Line A) and ``amulet_migrated`` (Line B). Use
:func:`make_nonlinear_backend` to construct one by name (the value of the
``--nonlinear-backend`` CLI flag)."""

from __future__ import annotations

from typing import Any

from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
from pllo.nonlinear.amulet_secure_r_backend import AmuletSecureRNonlinearBackend
from pllo.nonlinear.backends import NonlinearBackend
from pllo.nonlinear.current_backend import CurrentNonlinearBackend
from pllo.nonlinear.right_multiply_backend import RightMultiplyNonlinearBackend

__all__ = [
    "NONLINEAR_BACKENDS",
    "make_nonlinear_backend",
    "available_backends",
    "backend_security_status",
]

NONLINEAR_BACKENDS: dict[str, type[NonlinearBackend]] = {
    "current": CurrentNonlinearBackend,
    "amulet_migrated": AmuletMigratedNonlinearBackend,
    # A_rightmul: compatible right-multiply nonlinear islands (paper design A_rightmul)
    "compatible_right_multiply": RightMultiplyNonlinearBackend,
    # amulet_secure_R: hardened Amulet (dense single-one R, no trusted shortcut)
    "amulet_secure_R": AmuletSecureRNonlinearBackend,
}


def available_backends() -> list[str]:
    return list(NONLINEAR_BACKENDS)


def make_nonlinear_backend(name: str, **kwargs: Any) -> NonlinearBackend:
    """Construct a nonlinear backend by registry name.

    ``current`` ignores kwargs; ``amulet_migrated`` accepts ``lift_k`` / ``seed``.
    """
    cls = NONLINEAR_BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown nonlinear backend {name!r}; expected one of "
            f"{available_backends()}")
    if name == "current":
        return cls()
    return cls(**kwargs)


def backend_security_status() -> dict[str, str]:
    """Map backend name -> declared security status (for reports)."""
    return {name: cls.security_status for name, cls in NONLINEAR_BACKENDS.items()}


def backend_security_claim_status() -> dict[str, str]:
    """Map backend name -> paper-facing security_claim_status (E3).

    ``amulet_migrated`` is ``under_discussion`` until the advisor confirms the
    formal security boundary; no security is proven here."""
    return {name: cls.security_claim_status
            for name, cls in NONLINEAR_BACKENDS.items()}
