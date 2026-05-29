"""Python simulation of the trusted execution environment."""

from __future__ import annotations

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.mask_state import MaskState
from pllo.masks.pad_generator import generate_pad
from pllo.utils.validation import (
    require_same_dtype_device,
    require_shape,
    validate_linear_inputs,
    validate_lora_inputs,
    validate_mask_state_for_linear,
    validate_rank_mask,
)


class SimulatedTEE:
    """Python simulation of the trusted side for obfuscated execution.

    This class is not a real TEE. It deliberately has access to plaintext
    inputs, pads, masks, inverse masks, LoRA adapters, and recovery state
    because it models the code that would run inside a trusted boundary.
    Later implementations can replace it with an SGX, Gramine, Occlum, or
    RemoteTEEBackend implementation of the same trusted-side interface.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        """Create a simulated trusted backend."""
        self.dtype = dtype
        self.device = torch.device(device)

    def create_linear_mask_state(
        self,
        x: torch.Tensor,
        d_out: int,
        use_pad: bool = False,
        pad_scale: float = 1.0,
    ) -> MaskState:
        """Create input/output masks and optional pad for a linear operation."""
        if x.ndim != 2:
            raise ValueError(f"x must be rank-2 with shape (s, d_in), got {tuple(x.shape)}")
        if d_out <= 0:
            raise ValueError(f"d_out must be positive, got {d_out}")

        d_in = x.shape[1]
        n_in, n_in_inv = generate_invertible_matrix(d_in, x.dtype, x.device)
        n_out, n_out_inv = generate_invertible_matrix(d_out, x.dtype, x.device)
        pad = generate_pad(tuple(x.shape), x.dtype, x.device, pad_scale) if use_pad else None
        return MaskState(n_in=n_in, n_in_inv=n_in_inv, n_out=n_out, n_out_inv=n_out_inv, pad=pad)

    def add_lora_rank_mask(self, state: MaskState, rank: int) -> MaskState:
        """Attach a rank-space mask to an existing state for LoRA execution."""
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        rank_mask, rank_mask_inv = generate_invertible_matrix(
            rank,
            state.n_in.dtype,
            state.n_in.device,
        )
        state.rank_mask = rank_mask
        state.rank_mask_inv = rank_mask_inv
        return state

    def obfuscate_input(self, x: torch.Tensor, state: MaskState) -> torch.Tensor:
        """Apply optional pad and right input mask to plaintext input."""
        if x.ndim != 2:
            raise ValueError(f"x must be rank-2 with shape (s, d_in), got {tuple(x.shape)}")
        validate_mask_state_for_linear(state, x.shape[1], state.n_out.shape[0])
        require_same_dtype_device("x", x, n_in=state.n_in, pad=state.pad)
        if state.pad is not None:
            require_shape("state.pad", state.pad, tuple(x.shape))
        if state.pad is not None:
            return (x - state.pad) @ state.n_in
        return x @ state.n_in

    def transform_linear_weight(
        self,
        w: torch.Tensor,
        bias: torch.Tensor | None,
        state: MaskState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transform plaintext linear weights for untrusted execution."""
        d_in = state.n_in.shape[0]
        d_out = state.n_out.shape[0]
        x_proxy = torch.empty(1, d_in, dtype=w.dtype, device=w.device)
        validate_linear_inputs(x_proxy, w, bias)
        validate_mask_state_for_linear(state, d_in, d_out)
        if w.shape[1] != d_out:
            raise ValueError(f"w output dimension must match state.n_out size {d_out}, got {w.shape[1]}")
        require_same_dtype_device("w", w, n_in_inv=state.n_in_inv, n_out=state.n_out)
        w_tilde = state.n_in_inv @ w @ state.n_out
        bias_tilde = None if bias is None else bias @ state.n_out
        return w_tilde, bias_tilde

    def make_linear_pad_compensation(
        self,
        w: torch.Tensor,
        state: MaskState,
    ) -> torch.Tensor | None:
        """Create pad compensation C_T = T W N_out for standard linear."""
        if state.pad is None:
            return None
        d_in = state.n_in.shape[0]
        d_out = state.n_out.shape[0]
        require_shape("w", w, (d_in, d_out))
        require_shape("state.pad", state.pad, (state.pad.shape[0], d_in))
        validate_mask_state_for_linear(state, d_in, d_out)
        require_same_dtype_device("w", w, pad=state.pad, n_out=state.n_out)
        return state.pad @ w @ state.n_out

    def recover_output(self, y_tilde: torch.Tensor, state: MaskState) -> torch.Tensor:
        """Recover plaintext-domain output from obfuscated output."""
        if y_tilde.ndim != 2:
            raise ValueError(f"y_tilde must be rank-2 with shape (s, d_out), got {tuple(y_tilde.shape)}")
        d_out = state.n_out.shape[0]
        require_shape("y_tilde", y_tilde, (y_tilde.shape[0], d_out))
        validate_mask_state_for_linear(state, state.n_in.shape[0], d_out)
        require_same_dtype_device("y_tilde", y_tilde, n_out_inv=state.n_out_inv)
        return y_tilde @ state.n_out_inv

    def transform_lora_adapters(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        state: MaskState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform LoRA adapters without merging them into the base weight."""
        if state.rank_mask is None or state.rank_mask_inv is None:
            self.add_lora_rank_mask(state, a.shape[1])
        d_in = state.n_in.shape[0]
        d_out = state.n_out.shape[0]
        w_proxy = torch.empty(d_in, d_out, dtype=a.dtype, device=a.device)
        validate_lora_inputs(torch.empty(1, d_in, dtype=a.dtype, device=a.device), w_proxy, a, b)
        validate_mask_state_for_linear(state, d_in, d_out)
        validate_rank_mask(state, a.shape[1])
        require_same_dtype_device(
            "a",
            a,
            b=b,
            n_in_inv=state.n_in_inv,
            n_out=state.n_out,
            rank_mask=state.rank_mask,
            rank_mask_inv=state.rank_mask_inv,
        )

        a_tilde = state.n_in_inv @ a @ state.rank_mask
        b_tilde = state.rank_mask_inv @ b @ state.n_out
        return a_tilde, b_tilde

    def make_lora_pad_compensation(
        self,
        w: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: MaskState,
    ) -> torch.Tensor | None:
        """Create combined compensation C = T W N_out + T A B N_out."""
        if state.pad is None:
            return None
        validate_lora_inputs(state.pad, w, a, b)
        validate_mask_state_for_linear(state, w.shape[0], w.shape[1])
        validate_rank_mask(state, a.shape[1])
        require_same_dtype_device("w", w, pad=state.pad, a=a, b=b, n_out=state.n_out)
        return state.pad @ w @ state.n_out + state.pad @ a @ b @ state.n_out
