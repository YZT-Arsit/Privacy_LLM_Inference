"""Trusted backend interface for future TEE implementations."""

from __future__ import annotations

from typing import Protocol

import torch

from pllo.masks.mask_state import MaskState


class TrustedBackend(Protocol):
    """Protocol for trusted-side mask, pad, transform, and recovery logic.

    Implementations may be a Python simulator or a later SGX, Gramine,
    Occlum, or RemoteTEEBackend. The trusted side is the only side expected to
    access plaintext inputs, pads, private LoRA adapters, mask inverses,
    compensation construction, and plaintext output recovery.
    """

    def create_linear_mask_state(
        self,
        x: torch.Tensor,
        d_out: int,
        use_pad: bool = False,
        pad_scale: float = 1.0,
    ) -> MaskState:
        """Create input/output masks and optional pad for a linear operation."""

    def transform_linear_weight(
        self,
        w: torch.Tensor,
        bias: torch.Tensor | None,
        state: MaskState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transform plaintext linear weights for untrusted execution."""

    def make_linear_pad_compensation(
        self,
        w: torch.Tensor,
        state: MaskState,
    ) -> torch.Tensor | None:
        """Create pad compensation for a standard linear operation."""

    def recover_output(self, y_tilde: torch.Tensor, state: MaskState) -> torch.Tensor:
        """Recover plaintext-domain output from an obfuscated output."""

    def transform_lora_adapters(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        state: MaskState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform LoRA adapters while preserving their low-rank branch."""

    def make_lora_pad_compensation(
        self,
        w: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: MaskState,
    ) -> torch.Tensor | None:
        """Create combined base-weight and LoRA pad compensation."""
