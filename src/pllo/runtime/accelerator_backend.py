"""Stage 7.5c - re-export of the canonical accelerator backend interface.

This module exists so external code can write ``from pllo.runtime
import accelerator_backend`` to reach the protocol type, and so a future
TEE / GPU backend can re-use the same name without grepping for it.
"""

from __future__ import annotations

from pllo.runtime.interfaces import (
    AcceleratorBackend,
    UnsupportedBackendOp,
)

__all__ = ["AcceleratorBackend", "UnsupportedBackendOp"]
