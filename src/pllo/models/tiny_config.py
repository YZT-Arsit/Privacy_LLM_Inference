"""Configuration for the tiny decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TinyTransformerConfig:
    """Small decoder-only Transformer configuration."""

    vocab_size: int = 128
    max_seq_len: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {self.ffn_dim}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got "
                f"{self.hidden_size} and {self.num_heads}"
            )
        if self.dropout != 0.0:
            raise ValueError("dropout must be 0.0 for deterministic correctness checks")
        if self.dtype not in (torch.float64, torch.float32):
            raise ValueError(f"dtype must be torch.float64 or torch.float32, got {self.dtype}")
