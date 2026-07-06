"""ObfuscaTune baseline package (arXiv:2407.02960).

A self-contained, simulator-only baseline for the paper's obfuscate-outside /
non-linear-in-TEE scheme, kept strictly separate from our amulet-style /
trusted-shortcut mainline. See ``docs/obfuscatune_baseline.md``.

Backward compatibility: the original single-file module exported
``ObfuscaTune``, ``ObfuscaTuneConfig`` and ``matrix_with_condition_number`` from
this path; those names are re-exported unchanged (the latter is the legacy
generator-based API in :mod:`.protocol`). New code should import the seed-based
generators from :mod:`.random_matrices`.
"""

from __future__ import annotations

from .config import (
    METHOD_ORTHOGONAL,
    METHOD_RANDOM,
    METHOD_UNPROTECTED,
    ObfuscaTuneConfig,
    method_cond,
)
from .protocol import ObfuscaTune, matrix_with_condition_number
from . import (
    hf_gpt2_obfuscatune,
    linear_obfuscation,
    metrics,
    modules,
    random_matrices,
    qwen_cache,
    qwen_config,
    qwen_metrics,
    qwen_modules,
    qwen_obfuscatune,
)

__all__ = [
    # legacy surface
    "ObfuscaTune",
    "ObfuscaTuneConfig",
    "matrix_with_condition_number",
    # config helpers
    "METHOD_ORTHOGONAL",
    "METHOD_RANDOM",
    "METHOD_UNPROTECTED",
    "method_cond",
    # submodules
    "random_matrices",
    "linear_obfuscation",
    "modules",
    "metrics",
    "hf_gpt2_obfuscatune",
    # qwen adaptation
    "qwen_config",
    "qwen_modules",
    "qwen_cache",
    "qwen_obfuscatune",
    "qwen_metrics",
]
