"""Configuration for the ObfuscaTune baseline (arXiv:2407.02960).

Kept backward-compatible with the original single-file
``pllo.baselines.obfuscatune`` module: the fields ``dtype``, ``device``,
``condition_number`` and ``seed`` and the :meth:`torch_dtype` accessor are
preserved, so the legacy protocol class and its tests keep working. New,
optional fields carry defaults and are only used by the extended package.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Canonical string identifiers used across the comparison schema so that
# ObfuscaTune rows can be dropped into the same table as our amulet-style /
# trusted-shortcut rows. These are *baseline method names*, never our scheme.
METHOD_UNPROTECTED = "unprotected"
METHOD_ORTHOGONAL = "obfuscatune_orthogonal"
METHOD_RANDOM = "obfuscatune_random"


def method_cond(cond: float) -> str:
    """Method label for a fixed condition-number ObfuscaTune run."""
    c = int(cond) if float(cond).is_integer() else cond
    return f"obfuscatune_cond_{c}"


def torch_dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float64", "double", "fp64"):
        return torch.float64
    if name in ("float32", "float", "fp32"):
        return torch.float32
    if name in ("float16", "half", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype string: {name!r}")


@dataclass
class ObfuscaTuneConfig:
    """Run configuration.

    ``condition_number`` == 1.0 selects the paper's orthogonal obfuscation
    (Protected(ours)); a larger value selects a fixed-condition-number matrix
    (App. B). ``random_matrix_type`` == "random" selects a naive Gaussian
    invertible matrix (Protected(random)) whose condition number is left
    uncontrolled.
    """

    dtype: str = "float64"
    device: str = "cpu"
    condition_number: float = 1.0          # 1.0 => orthogonal (paper's choice)
    seed: int = 0
    random_matrix_type: str = "orthogonal"  # "orthogonal" | "random" | "cond"
    gaussian_jitter: float = 1e-3           # ridge added to naive random matrices

    def torch_dtype(self) -> torch.dtype:
        return torch_dtype_from_str(self.dtype)

    def method_name(self) -> str:
        if self.random_matrix_type == "random":
            return METHOD_RANDOM
        if self.random_matrix_type == "cond" and self.condition_number > 1.0:
            return method_cond(self.condition_number)
        return METHOD_ORTHOGONAL


__all__ = [
    "ObfuscaTuneConfig",
    "torch_dtype_from_str",
    "method_cond",
    "METHOD_UNPROTECTED",
    "METHOD_ORTHOGONAL",
    "METHOD_RANDOM",
]
