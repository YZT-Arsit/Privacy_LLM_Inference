"""Model loader registry."""

from __future__ import annotations

from pllo.model_zoo.base import ModelLoader
from pllo.model_zoo.hf_loader import HuggingFaceModelLoader


def get_model_loader(source: str) -> ModelLoader:
    """Return a model loader for a source name."""
    normalized = source.lower()
    if normalized in {"huggingface", "hf"}:
        return HuggingFaceModelLoader()
    if normalized == "modelscope":
        raise NotImplementedError("ModelScope loading is planned for Stage 5 and is not supported yet")
    raise ValueError(f"unsupported model source {source!r}; expected huggingface/hf")
