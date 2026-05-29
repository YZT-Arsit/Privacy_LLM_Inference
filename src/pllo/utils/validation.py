"""Validation helpers for tensor shape, dtype, and device checks."""

from __future__ import annotations

import torch

from pllo.masks.mask_state import MaskState


def require_rank2(name: str, tensor: torch.Tensor) -> None:
    """Require a tensor to have rank 2."""
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")


def require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    """Require an exact tensor shape."""
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


def require_same_dtype_device(reference_name: str, reference: torch.Tensor, **tensors: torch.Tensor | None) -> None:
    """Require tensors to share dtype and device with a reference tensor."""
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if tensor.dtype != reference.dtype:
            raise ValueError(
                f"{name} dtype must match {reference_name}: "
                f"{tensor.dtype} != {reference.dtype}"
            )
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} device must match {reference_name}: "
                f"{tensor.device} != {reference.device}"
            )


def validate_linear_inputs(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> None:
    """Validate plaintext row-vector linear inputs."""
    require_rank2("x", x)
    require_rank2("w", w)
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"w must have shape (d_in, d_out) with d_in={x.shape[1]}, "
            f"got {tuple(w.shape)}"
        )
    if bias is not None:
        require_shape("bias", bias, (w.shape[1],))
    require_same_dtype_device("x", x, w=w, bias=bias)


def validate_lora_inputs(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    """Validate plaintext row-vector LoRA linear inputs."""
    validate_linear_inputs(x, w, bias)
    require_rank2("a", a)
    require_rank2("b", b)
    d_in = x.shape[1]
    d_out = w.shape[1]
    if a.shape[0] != d_in:
        raise ValueError(f"a must have shape (d_in, rank) with d_in={d_in}, got {tuple(a.shape)}")
    if b.shape[0] != a.shape[1]:
        raise ValueError(
            f"b rank dimension must match a rank={a.shape[1]}, got {tuple(b.shape)}"
        )
    if b.shape[1] != d_out:
        raise ValueError(f"b must have shape (rank, d_out) with d_out={d_out}, got {tuple(b.shape)}")
    require_same_dtype_device("x", x, a=a, b=b)


def validate_mask_state_for_linear(state: MaskState, d_in: int, d_out: int) -> None:
    """Validate mask state shapes and tensor placement for a linear op."""
    require_shape("state.n_in", state.n_in, (d_in, d_in))
    require_shape("state.n_in_inv", state.n_in_inv, (d_in, d_in))
    require_shape("state.n_out", state.n_out, (d_out, d_out))
    require_shape("state.n_out_inv", state.n_out_inv, (d_out, d_out))
    require_same_dtype_device(
        "state.n_in",
        state.n_in,
        n_in_inv=state.n_in_inv,
        n_out=state.n_out,
        n_out_inv=state.n_out_inv,
        pad=state.pad,
    )


def validate_rank_mask(state: MaskState, rank: int) -> None:
    """Validate LoRA rank-space mask shape and placement."""
    if state.rank_mask is None or state.rank_mask_inv is None:
        raise ValueError("state.rank_mask and state.rank_mask_inv must both be present")
    require_shape("state.rank_mask", state.rank_mask, (rank, rank))
    require_shape("state.rank_mask_inv", state.rank_mask_inv, (rank, rank))
    require_same_dtype_device(
        "state.n_in",
        state.n_in,
        rank_mask=state.rank_mask,
        rank_mask_inv=state.rank_mask_inv,
    )
