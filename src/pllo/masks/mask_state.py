"""Container for masks and optional pad state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class MaskState:
    """Trusted-side state for a single obfuscated operation."""

    n_in: torch.Tensor
    n_in_inv: torch.Tensor
    n_out: torch.Tensor
    n_out_inv: torch.Tensor
    rank_mask: torch.Tensor | None = None
    rank_mask_inv: torch.Tensor | None = None
    pad: torch.Tensor | None = None
