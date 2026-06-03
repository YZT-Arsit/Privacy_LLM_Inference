"""Stage 7.5c - backend registry.

Tiny indirection layer so the controller can ask for a named backend
("local_cpu", and in the future "tee", "gpu", ...) without importing the
concrete class.

Future TEE / GPU backends register themselves here; the controller does
not need to be modified.
"""

from __future__ import annotations

from typing import Any, Callable

from pllo.runtime.local_cpu_backend import LocalCPUBackend


_REGISTRY: dict[str, Callable[..., Any]] = {
    "local_cpu": LocalCPUBackend,
}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    """Register a new backend factory under ``name``.

    Calling ``register_backend("tee", TEEBackend)`` is the only step a
    future TEE deployment needs.
    """
    _REGISTRY[name] = factory


def get_backend(name: str, **kwargs: Any) -> Any:
    """Look up a backend by name and instantiate it.

    Raises :class:`KeyError` if no backend is registered for ``name``.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"no backend registered for {name!r}; available: "
            f"{sorted(_REGISTRY)}",
        )
    return _REGISTRY[name](**kwargs)


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["register_backend", "get_backend", "list_backends"]
