"""Stage 7.5c - deployable runtime boundary between trusted controller
and accelerator backend.

The protocol logic lives in :mod:`pllo.runtime.trusted_controller`. The
only concrete backend in this artifact is :class:`LocalCPUBackend`; a
future TEE / GPU backend implements the same
:class:`AcceleratorBackend` protocol and is registered through
:func:`pllo.runtime.backend_registry.register_backend`.

**Stage 7.5c is NOT a real TEE and NOT a GPU runtime.** The interface
is backend-ready; the hardware wire-up is future work.
"""

from pllo.runtime.interfaces import (
    AcceleratorBackend,
    UnsupportedBackendOp,
)
from pllo.runtime.local_cpu_backend import LocalCPUBackend
from pllo.runtime.trusted_controller import (
    TrustedController,
    TrustedControllerConfig,
)
from pllo.runtime.transcript import RuntimeTranscript
from pllo.runtime.backend_registry import (
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "AcceleratorBackend",
    "LocalCPUBackend",
    "RuntimeTranscript",
    "TrustedController",
    "TrustedControllerConfig",
    "UnsupportedBackendOp",
    "get_backend",
    "list_backends",
    "register_backend",
]
