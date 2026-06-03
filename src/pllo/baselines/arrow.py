"""Stage 7.5c - Arrow baseline placeholder.

The user explicitly forbids substituting a generic Arrow-style proxy. The
specific Arrow nonlinear primitive formula is NOT present in the
repository's reference materials at the time of this commit, so we
record this as ``exact_primitive_implemented = False`` and
``missing_paper_formula = True``. No measured baseline is produced.

To replace this stub with a real Arrow primitive in a later stage, drop
the closed-form formula into ``ArrowDirectPrimitive.forward`` and flip
``_DECLARE.exact_primitive_implemented`` to ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pllo.baselines.baseline_protocol import (
    BaselineProtocol,
    BaselineSelfDeclaration,
    UnsupportedResult,
)


_DECLARE = BaselineSelfDeclaration(
    name="arrow_direct_primitive_or_unavailable",
    paper="Arrow (nonlinear-primitive obfuscation; specific formula not available in repository materials)",
    exact_primitive_implemented=False,
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=False,
    supports_decoder_generation=False,
    supports_kv_cache_append=False,
    supports_lora_training=False,
    notes=(
        "No proxy / generic Arrow-like primitive is substituted. The"
        " closed-form Arrow nonlinear formula is missing from this"
        " repository's references; this baseline is recorded as"
        " unavailable rather than fabricated."
    ),
)


@dataclass
class ArrowConfig:
    dtype: str = "float64"
    device: str = "cpu"


class ArrowDirectPrimitive(BaselineProtocol):
    declare = _DECLARE

    missing_paper_formula: bool = True

    def __init__(self, config: ArrowConfig | None = None) -> None:
        self.config = config or ArrowConfig()

    def forward(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="Arrow primitive is unavailable in this repository",
            paper_scope_reason=(
                "Arrow paper specifies a nonlinear obfuscation primitive"
                " whose closed-form formula is not present in the repo's"
                " reference materials at the time of this commit."
            ),
            implementation_scope_reason=(
                "Stage 7.5c forbids substituting a generic Arrow-like"
                " proxy; this baseline is recorded as missing-formula"
                " rather than fabricated."
            ),
        )


__all__ = ["ArrowConfig", "ArrowDirectPrimitive"]
