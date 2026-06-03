"""Stage 7.5c - Amulet-style static left+right masking primitive.

Implements the static ``P H Q`` left+right masking primitive over a
linear layer and demonstrates the **decoder KV-append counterexample**:
a static left mask ``P`` does NOT generally commute with token-axis
concatenation in autoregressive decoding unless ``P`` is block-compatible
across decoding steps. The counterexample is itself the experimental
result; we do not paper over the gap by introducing a generic proxy.

References:
- The static left+right masking formula ``H_tilde = P H Q`` and the
  linear identity ``H_tilde W_tilde = P (HW) Q_out`` are textbook matrix
  identities; we implement them directly.
- The nonlinear primitive(s) from the Amulet paper are not reproduced in
  full here: the available repository artifacts do not pin a closed-form
  nonlinear formula. ``exact_nonlinear_primitive_implemented`` is
  recorded as ``False`` and ``missing_formula`` is ``True`` rather than
  silently substituting a generic stand-in.

This module ONLY implements primitives. Threat-model statements and any
claim of full-system reproduction are intentionally NOT made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.baselines.baseline_protocol import (
    BaselineProtocol,
    BaselineSelfDeclaration,
    UnsupportedResult,
)


_DECLARE = BaselineSelfDeclaration(
    name="amulet_static_left_right_mask",
    paper="Amulet (matrix-obfuscation family); referenced as the static PHQ primitive",
    exact_primitive_implemented=True,  # the linear PHQ identity
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=True,
    supports_decoder_generation=False,  # see KV append counterexample below
    supports_kv_cache_append=False,
    supports_lora_training=False,
    notes=(
        "Linear PHQ identity implemented from textbook matrix algebra;"
        " nonlinear primitive(s) not reproduced; KV append counterexample"
        " demonstrates the static-left-mask gap for autoregressive decoding."
    ),
)


@dataclass
class AmuletConfig:
    dtype: str = "float64"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    def torch_device(self) -> torch.device:
        return torch.device(self.device)


def _gen_invertible(
    dim: int, dtype: torch.dtype, device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if generator is None:
        m = torch.randn(dim, dim, dtype=dtype, device=device)
    else:
        m = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(m)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(0, 1)


class AmuletStaticPHQ(BaselineProtocol):
    """Implements the static left+right masking primitive ``H_tilde = P H Q``.

    For a linear layer ``Y = H W``, choose ``W_tilde = Q^{-1} W Q_out``.
    Then ``H_tilde W_tilde = P (HW) Q_out``. This identity is purely
    linear algebra and we implement it as-is.
    """

    declare = _DECLARE

    def __init__(self, config: AmuletConfig | None = None) -> None:
        self.config = config or AmuletConfig()

    # ------------------------------------------------------------------
    # Static linear primitive
    # ------------------------------------------------------------------

    def static_linear_forward(
        self,
        h: torch.Tensor,
        w: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """``H_tilde = P H Q`` and ``W_tilde = Q^{-1} W Q_out``."""
        dtype = self.config.torch_dtype()
        device = self.config.torch_device()
        m, d = h.shape
        _, d_out = w.shape
        p, _ = _gen_invertible(m, dtype, device, generator)
        q, q_inv = _gen_invertible(d, dtype, device, generator)
        q_out, q_out_inv = _gen_invertible(d_out, dtype, device, generator)
        h_tilde = p @ h @ q
        w_tilde = q_inv @ w @ q_out
        y_masked_left_right = h_tilde @ w_tilde  # equals P (HW) Q_out
        y_plain = h @ w
        # Trusted-side recovery: Y = P^{-1} Y_masked Q_out^{-1}.
        p_inv = torch.linalg.inv(p)
        y_recovered = p_inv @ y_masked_left_right @ q_out_inv
        err = float((y_recovered - y_plain).abs().max().item())
        return {
            "y_plain": y_plain,
            "y_recovered": y_recovered,
            "y_masked_left_right": y_masked_left_right,
            "max_abs_error": err,
            "allclose": bool(torch.allclose(y_recovered, y_plain, atol=1e-9, rtol=1e-9)),
            "primitive": "static_left_right_mask",
        }

    def forward(
        self, h: torch.Tensor, w: torch.Tensor,
        *, generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        return self.static_linear_forward(h, w, generator=generator)

    # ------------------------------------------------------------------
    # Decoder KV append counterexample
    # ------------------------------------------------------------------

    def kv_append_counterexample(
        self,
        seq_len_old: int = 2,
        seq_len_new: int = 1,
        d: int = 8,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Construct a numerical witness that a generic static left mask
        ``P_3`` cannot agree with ``[P_2 K_{1:2}; ...]`` for arbitrary
        ``P_2 != P_3`` unless ``P_3`` is block-compatible with ``P_2``.

        Block-compatible means
            P_3 = [[P_2, 0], [0, p_new]]
        for some ``p_new``. We sample fresh per-step masks (the natural
        thing to do in an autoregressive setting), then check that the
        prefix block of ``P_3 [K_{1:t-1}; K_t]`` does NOT equal
        ``P_2 K_{1:t-1}`` unless re-computation is performed.
        """
        dtype = self.config.torch_dtype()
        device = self.config.torch_device()
        # K rows for steps 1..t-1 and one new row at step t.
        gen = generator
        K_prefix = (
            torch.randn(seq_len_old, d, dtype=dtype, device=device)
            if gen is None
            else torch.randn(seq_len_old, d, dtype=dtype, device=device, generator=gen)
        )
        K_new = (
            torch.randn(seq_len_new, d, dtype=dtype, device=device)
            if gen is None
            else torch.randn(seq_len_new, d, dtype=dtype, device=device, generator=gen)
        )
        # Fresh per-step left masks; size grows with the step index.
        P_old, _ = _gen_invertible(seq_len_old, dtype, device, gen)
        P_new, _ = _gen_invertible(seq_len_old + seq_len_new, dtype, device, gen)
        Q, _ = _gen_invertible(d, dtype, device, gen)

        cache_old_masked = P_old @ K_prefix @ Q
        K_concat = torch.cat([K_prefix, K_new], dim=0)
        cache_new_masked = P_new @ K_concat @ Q

        # Try to reuse the old cache: take its rows verbatim and prepend
        # them to a freshly-masked new row.
        appended = torch.cat(
            [cache_old_masked, P_new[seq_len_old:, seq_len_old:] @ K_new @ Q],
            dim=0,
        )
        gap = float((appended - cache_new_masked).abs().max().item())

        # Block-compatible construction: build a block-diagonal P3 from P2.
        block_p = torch.eye(seq_len_old + seq_len_new, dtype=dtype, device=device)
        block_p[:seq_len_old, :seq_len_old] = P_old
        # Choose p_new = identity for the new row.
        block_cache_new = block_p @ K_concat @ Q
        block_appended = torch.cat(
            [cache_old_masked, K_new @ Q], dim=0,
        )
        block_gap = float((block_appended - block_cache_new).abs().max().item())

        return {
            "counterexample_present": bool(gap > 1e-9),
            "max_gap": gap,
            "block_compatible_max_gap": block_gap,
            "block_compatible_condition": (
                "P_t = block-diag(P_{t-1}, p_new) -- a strict structural"
                " constraint not present under fresh per-step masks"
            ),
            "recompute_required": bool(gap > 1e-9),
            "kv_append_supported": False,
            "mathematical_reason": (
                "A generic invertible left mask P_t does not commute with"
                " token-axis concatenation unless P_t is block-diagonal"
                " aligned with the prefix masks. Refreshing P per step"
                " therefore invalidates the existing masked cache and"
                " forces re-computation."
            ),
        }

    def decode_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="decoder generation under fresh static left masks is unsupported",
            mathematical_reason=(
                "Static left mask does not commute with token-axis append"
                " unless block-compatible; see kv_append_counterexample()."
            ),
            paper_scope_reason="Amulet paper targets static feed-forward inference, not autoregressive generation.",
            implementation_scope_reason="no per-step block-diagonal mask refresh implemented",
        )

    def train_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="LoRA training is out of scope for the Amulet static primitive",
            paper_scope_reason="Amulet paper does not address LoRA personalization.",
        )


# ----------------------------------------------------------------------
# Our (right-mask) counterpart for the same KV-append scenario.
# ----------------------------------------------------------------------


def ours_right_mask_kv_append(
    seq_len_old: int = 2,
    seq_len_new: int = 1,
    d: int = 8,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Show that the right-mask construction does commute with append:

        K_tilde_{1:t} = K_{1:t} N_K  ==  cat([K_tilde_{1:t-1}, K_t N_K])
    """
    device = device or torch.device("cpu")
    K_prefix = (
        torch.randn(seq_len_old, d, dtype=dtype, device=device)
        if generator is None
        else torch.randn(seq_len_old, d, dtype=dtype, device=device, generator=generator)
    )
    K_new = (
        torch.randn(seq_len_new, d, dtype=dtype, device=device)
        if generator is None
        else torch.randn(seq_len_new, d, dtype=dtype, device=device, generator=generator)
    )
    Nk, _ = _gen_invertible(d, dtype, device, generator)
    cache_full = torch.cat([K_prefix, K_new], dim=0) @ Nk
    cache_prefix = K_prefix @ Nk
    appended = torch.cat([cache_prefix, K_new @ Nk], dim=0)
    err = float((cache_full - appended).abs().max().item())
    return {
        "ours_append_supported": True,
        "max_abs_error": err,
        "mathematical_reason": (
            "Right mask N distributes over token-axis concatenation:"
            " ([K_prefix; K_new]) @ N == [K_prefix @ N; K_new @ N]."
        ),
    }


__all__ = [
    "AmuletConfig",
    "AmuletStaticPHQ",
    "ours_right_mask_kv_append",
]
