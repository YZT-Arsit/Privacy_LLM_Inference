"""Selectable nonlinear-handling backends (Line A: current; Line B: Amulet).

A thin, comparable layer over the repo's two nonlinear-island strategies so a
single ``--nonlinear-backend current|amulet_migrated`` flag selects how GELU/SiLU,
Softmax, and LayerNorm/RMSNorm are handled. See :mod:`pllo.nonlinear.backends`.

No security claim is made here. The Amulet migration's security is
``not_formally_claimed`` (under discussion); this layer measures correctness +
efficiency only.
"""

from __future__ import annotations

from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
from pllo.nonlinear.backends import (
    NonlinearBackend,
    NonlinearOpResult,
    OP_NAMES,
    tensor_bytes,
)
from pllo.nonlinear.current_backend import CurrentNonlinearBackend
from pllo.nonlinear.registry import (
    NONLINEAR_BACKENDS,
    available_backends,
    backend_security_status,
    make_nonlinear_backend,
)

__all__ = [
    "NonlinearBackend",
    "NonlinearOpResult",
    "OP_NAMES",
    "tensor_bytes",
    "CurrentNonlinearBackend",
    "AmuletMigratedNonlinearBackend",
    "NONLINEAR_BACKENDS",
    "available_backends",
    "backend_security_status",
    "make_nonlinear_backend",
]
