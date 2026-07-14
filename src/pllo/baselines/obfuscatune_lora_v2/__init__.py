"""Isolated ObfuscaTune-style adapted LoRA baseline.

This package implements the repository adaptation frozen by contract
``OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1``.  It deliberately has
no dependency on the protected G2 training path.
"""

from .runtime import ExternalLoRALinear, RuntimeCounters, TrustedRuntime

__all__ = ["ExternalLoRALinear", "RuntimeCounters", "TrustedRuntime"]
