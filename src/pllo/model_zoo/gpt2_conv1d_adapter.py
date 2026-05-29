"""Adapters for HuggingFace GPT-2 Conv1D projections."""

from __future__ import annotations

import torch
from torch import nn


def is_hf_gpt2_conv1d(module: object) -> bool:
    """Return True for HuggingFace GPT-2 Conv1D, not torch.nn.Conv1d."""
    if isinstance(module, nn.Conv1d):
        return False
    return (
        module.__class__.__name__ == "Conv1D"
        and hasattr(module, "weight")
        and hasattr(module, "bias")
        and isinstance(getattr(module, "weight"), torch.Tensor)
    )


def extract_conv1d_as_linear(module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract HF GPT-2 Conv1D as internal row-vector linear weights.

    HuggingFace GPT-2 ``Conv1D`` computes ``Y = X @ W + b`` for the last
    dimension. Its weight already has shape ``[d_in, d_out]``, which matches
    this project's internal row-vector convention.
    """
    if not is_hf_gpt2_conv1d(module):
        raise ValueError(f"expected HuggingFace GPT-2 Conv1D, got {module.__class__.__name__}")
    weight = module.weight.detach().clone()
    bias = None if module.bias is None else module.bias.detach().clone()
    if weight.ndim != 2:
        raise ValueError(f"Conv1D weight must be rank-2, got shape {tuple(weight.shape)}")
    if bias is not None and tuple(bias.shape) != (weight.shape[1],):
        raise ValueError(f"Conv1D bias must have shape {(weight.shape[1],)}, got {tuple(bias.shape)}")
    return weight, bias


def conv1d_forward_as_linear(x: torch.Tensor, module) -> torch.Tensor:
    """Run an HF GPT-2 Conv1D via internal row-vector linear convention."""
    weight, bias = extract_conv1d_as_linear(module)
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"x last dimension must be {weight.shape[0]}, got {x.shape[-1]}")
    out = x @ weight.to(dtype=x.dtype, device=x.device)
    if bias is not None:
        out = out + bias.to(dtype=x.dtype, device=x.device)
    return out


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    diff = candidate - reference
    denom = torch.linalg.vector_norm(reference.reshape(-1)).clamp_min(
        torch.as_tensor(1e-30, dtype=reference.dtype, device=reference.device)
    )
    return float((torch.linalg.vector_norm(diff.reshape(-1)) / denom).item())


def compare_conv1d_equivalence(
    module,
    x: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Compare HF Conv1D forward with internal linear-equivalent forward."""
    with torch.no_grad():
        hf_output = module(x)
        linear_output = conv1d_forward_as_linear(x, module)
    diff = (hf_output - linear_output).abs()
    return {
        "module_class": module.__class__.__name__,
        "input_shape": list(x.shape),
        "hf_output_shape": list(hf_output.shape),
        "linear_output_shape": list(linear_output.shape),
        "max_abs_error": float(diff.max().item()),
        "relative_l2_error": _relative_l2(hf_output, linear_output),
        "allclose": bool(torch.allclose(hf_output, linear_output, atol=atol, rtol=rtol)),
    }


def split_gpt2_c_attn_weights(module, hidden_size: int) -> dict[str, dict[str, torch.Tensor | None]]:
    """Split fused GPT-2 c_attn Conv1D weights into Q/K/V linear branches."""
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    weight, bias = extract_conv1d_as_linear(module)
    expected = (hidden_size, 3 * hidden_size)
    if tuple(weight.shape) != expected:
        raise ValueError(f"c_attn weight must have shape {expected}, got {tuple(weight.shape)}")
    if bias is not None and tuple(bias.shape) != (3 * hidden_size,):
        raise ValueError(f"c_attn bias must have shape {(3 * hidden_size,)}, got {tuple(bias.shape)}")

    result = {}
    for name, start in (("q", 0), ("k", hidden_size), ("v", 2 * hidden_size)):
        end = start + hidden_size
        result[name] = {
            "weight": weight[:, start:end].detach().clone(),
            "bias": None if bias is None else bias[start:end].detach().clone(),
        }
    return result


def compare_c_attn_split_equivalence(
    module,
    x: torch.Tensor,
    hidden_size: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Compare fused c_attn output with concat of split Q/K/V linears."""
    pieces = split_gpt2_c_attn_weights(module, hidden_size)
    outputs = []
    for name in ("q", "k", "v"):
        weight = pieces[name]["weight"].to(dtype=x.dtype, device=x.device)
        bias = pieces[name]["bias"]
        out = x @ weight
        if bias is not None:
            out = out + bias.to(dtype=x.dtype, device=x.device)
        outputs.append(out)
    with torch.no_grad():
        hf_output = module(x)
    split_output = torch.cat(outputs, dim=-1)
    diff = (hf_output - split_output).abs()
    return {
        "module_class": module.__class__.__name__,
        "input_shape": list(x.shape),
        "hf_output_shape": list(hf_output.shape),
        "split_output_shape": list(split_output.shape),
        "max_abs_error": float(diff.max().item()),
        "relative_l2_error": _relative_l2(hf_output, split_output),
        "allclose": bool(torch.allclose(hf_output, split_output, atol=atol, rtol=rtol)),
    }
