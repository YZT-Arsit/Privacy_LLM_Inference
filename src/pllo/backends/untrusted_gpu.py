"""Untrusted GPU-side executor."""

from __future__ import annotations

import torch

from pllo.utils.validation import require_rank2, require_same_dtype_device, require_shape


class UntrustedGPUExecutor:
    """Simulated untrusted GPU executor.

    This class only processes obfuscated tensors, transformed base weights,
    transformed LoRA adapters, ``bias_tilde``, and trusted-side compensation
    tensors. It must not receive plaintext X, plaintext pad T, plaintext LoRA
    A/B, mask inverses, or recovered plaintext outputs.

    The executor does not generate masks, generate pads, create compensation,
    or recover outputs. Those responsibilities stay on the trusted side.
    """

    def linear_forward(
        self,
        x_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
        b_tilde: torch.Tensor | None = None,
        compensation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run an obfuscated standard linear layer."""
        require_rank2("x_tilde", x_tilde)
        require_rank2("w_tilde", w_tilde)
        if x_tilde.shape[1] != w_tilde.shape[0]:
            raise ValueError(
                f"w_tilde must have input dimension {x_tilde.shape[1]}, got {tuple(w_tilde.shape)}"
            )
        output_shape = (x_tilde.shape[0], w_tilde.shape[1])
        if b_tilde is not None:
            require_shape("b_tilde", b_tilde, (w_tilde.shape[1],))
        if compensation is not None:
            require_shape("compensation", compensation, output_shape)
        require_same_dtype_device("x_tilde", x_tilde, w_tilde=w_tilde, b_tilde=b_tilde, compensation=compensation)

        y_tilde = x_tilde @ w_tilde
        if b_tilde is not None:
            y_tilde = y_tilde + b_tilde
        if compensation is not None:
            y_tilde = y_tilde + compensation
        return y_tilde

    def lora_linear_forward(
        self,
        x_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
        a_tilde: torch.Tensor,
        b_tilde_lora: torch.Tensor,
        bias_tilde: torch.Tensor | None = None,
        compensation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run an obfuscated LoRA linear layer with a separate low-rank branch."""
        require_rank2("x_tilde", x_tilde)
        require_rank2("w_tilde", w_tilde)
        require_rank2("a_tilde", a_tilde)
        require_rank2("b_tilde_lora", b_tilde_lora)
        if x_tilde.shape[1] != w_tilde.shape[0]:
            raise ValueError(
                f"w_tilde must have input dimension {x_tilde.shape[1]}, got {tuple(w_tilde.shape)}"
            )
        if a_tilde.shape[0] != x_tilde.shape[1]:
            raise ValueError(
                f"a_tilde must have input dimension {x_tilde.shape[1]}, got {tuple(a_tilde.shape)}"
            )
        if b_tilde_lora.shape[0] != a_tilde.shape[1]:
            raise ValueError(
                f"b_tilde_lora rank must match a_tilde rank {a_tilde.shape[1]}, "
                f"got {tuple(b_tilde_lora.shape)}"
            )
        if b_tilde_lora.shape[1] != w_tilde.shape[1]:
            raise ValueError(
                f"b_tilde_lora output dimension must match w_tilde output {w_tilde.shape[1]}, "
                f"got {tuple(b_tilde_lora.shape)}"
            )
        output_shape = (x_tilde.shape[0], w_tilde.shape[1])
        if bias_tilde is not None:
            require_shape("bias_tilde", bias_tilde, (w_tilde.shape[1],))
        if compensation is not None:
            require_shape("compensation", compensation, output_shape)
        require_same_dtype_device(
            "x_tilde",
            x_tilde,
            w_tilde=w_tilde,
            a_tilde=a_tilde,
            b_tilde_lora=b_tilde_lora,
            bias_tilde=bias_tilde,
            compensation=compensation,
        )

        y_tilde = x_tilde @ w_tilde + (x_tilde @ a_tilde) @ b_tilde_lora
        if bias_tilde is not None:
            y_tilde = y_tilde + bias_tilde
        if compensation is not None:
            y_tilde = y_tilde + compensation
        return y_tilde
