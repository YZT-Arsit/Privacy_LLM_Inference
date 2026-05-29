"""External model loading and inspection utilities."""

from pllo.model_zoo.base import ExternalModelConfig, ModelLoader, torch_dtype_from_string
from pllo.model_zoo.gpt2_conv1d_adapter import (
    compare_c_attn_split_equivalence,
    compare_conv1d_equivalence,
    conv1d_forward_as_linear,
    extract_conv1d_as_linear,
    is_hf_gpt2_conv1d,
    split_gpt2_c_attn_weights,
)
from pllo.model_zoo.gpt2_mapping import build_gpt2_linear_mapping_report
from pllo.model_zoo.gpt2_spec import get_gpt2_module_spec, is_gpt2_like
from pllo.model_zoo.hf_loader import HuggingFaceModelLoader
from pllo.model_zoo.model_inspector import inspect_model_modules
from pllo.model_zoo.registry import get_model_loader

__all__ = [
    "ExternalModelConfig",
    "HuggingFaceModelLoader",
    "ModelLoader",
    "build_gpt2_linear_mapping_report",
    "compare_c_attn_split_equivalence",
    "compare_conv1d_equivalence",
    "conv1d_forward_as_linear",
    "extract_conv1d_as_linear",
    "get_gpt2_module_spec",
    "get_model_loader",
    "inspect_model_modules",
    "is_hf_gpt2_conv1d",
    "is_gpt2_like",
    "split_gpt2_c_attn_weights",
    "torch_dtype_from_string",
]
