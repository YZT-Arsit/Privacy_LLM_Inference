"""Execution backends for trusted and untrusted domains."""

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor

__all__ = ["SimulatedTEE", "UntrustedGPUExecutor"]
