"""Base types for external model loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(slots=True)
class ExternalModelConfig:
    """Configuration for loading an external model."""

    source: str
    model_id: str
    local_dir: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    trust_remote_code: bool = False


class ModelLoader(Protocol):
    """Protocol implemented by external model loaders."""

    def load_tokenizer(self, config: ExternalModelConfig):
        """Load a tokenizer."""

    def load_model(self, config: ExternalModelConfig):
        """Load a causal language model."""

    def load(self, config: ExternalModelConfig):
        """Load tokenizer and model."""


def torch_dtype_from_string(dtype: str, device: str = "cpu") -> torch.dtype:
    """Map a dtype string to a torch dtype with device-aware validation."""
    normalized = dtype.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float64":
        return torch.float64
    if normalized == "float16":
        if torch.device(device).type != "cuda":
            raise ValueError("float16 loading is only allowed on CUDA for this prototype")
        return torch.float16
    raise ValueError(f"unsupported dtype {dtype!r}; expected float32, float64, or float16")
