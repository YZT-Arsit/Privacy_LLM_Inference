"""Stage 7.2 — Rank-padded LoRA primitive (hidden true rank from shape).

The Stage 7.0 / 7.1 LoRA forward + backward path exposes
``A_tilde ∈ R^{d_in × r}`` and ``B_tilde ∈ R^{r × d_out}`` (and gradients
of the same shape) to the GPU. The tensor dimensions therefore leak the
*true* LoRA rank ``r``.

Stage 7.2 hides ``r`` behind a *padded* rank ``r_pad ≥ r``. The trusted
side constructs

    A_pad ∈ R^{d_in × r_pad},   A_pad[:, :r] = A_real
    B_pad ∈ R^{r_pad × d_out},  B_pad[:r, :] = B_real

such that ``A_pad B_pad = A B`` exactly. All Stage 7.0 / 7.1 helpers are
reused with the rank dimension replaced by ``r_pad``; the LoRA scaling
factor stays ``α / r`` (NOT ``α / r_pad``) so the function value is
unchanged. The trusted side discards the dummy rank slice on the way
back; the optimizer state is sized only for the real rank.

Two dummy strategies are supported (`dummy_strategy`):

* ``zero_dummy`` — ``A_dummy`` random, ``B_dummy = 0``. Trivially
  satisfies ``A_pad B_pad = A B`` but ``B_pad`` then has *true rank*
  exactly ``r`` (the dummy rows are zero), so a spectral attacker can
  read off ``r`` directly. Baseline.
* ``paired_cancellation_dummy`` — pair dummies as
  ``[R, R], [S, -S]`` so each pair has rank 1 in ``B_pad`` and
  contributes ``R S + R(-S) = 0`` to ``A_pad B_pad``. Lifts the
  spectral rank of ``B_pad`` to ``r + ⌊(r_pad - r) / 2⌋`` (and
  ``r + ⌊(r_pad - r) / 2⌋ + 1`` if the trailing odd dummy is zero) so
  the attacker only learns an upper bound on ``r``. **NOT formal
  security** — the proxy explicitly measures residual leakage.

This module deliberately does NOT merge the adapter into the public
base weight ``W`` (constraint 7). Reports must emit fingerprints /
metrics — raw padded adapter tensors, dummy masks, and optimizer state
stay trusted-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    create_masked_lora_state,
    make_lora_pad_compensation,
    masked_lora_linear_forward,
    obfuscate_lora_input,
    recover_masked_output,
    transform_linear_weight_lora,
    transform_lora_adapter,
)
from pllo.ops.lora_backward import (
    make_lora_grad_pad_compensation,
    masked_lora_backward,
    recover_lora_gradients,
    transform_upstream_gradient,
)
from pllo.utils.validation import require_rank2, require_same_dtype_device


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


VALID_DUMMY_STRATEGIES: tuple[str, ...] = (
    "zero_dummy",
    "paired_cancellation_dummy",
)


@dataclass
class RankPaddingConfig:
    """Stage 7.2 rank-padding configuration.

    ``true_rank`` is the real LoRA rank (private). ``padded_rank`` is the
    rank dimension exposed to the GPU; the dummy slice has size
    ``padded_rank - true_rank``.
    """

    true_rank: int
    padded_rank: int
    dummy_scale: float = 1.0
    dummy_strategy: str = "paired_cancellation_dummy"
    fresh_dummy_per_step: bool = True
    dtype: str = "float64"
    device: str = "cpu"

    def dummy_size(self) -> int:
        return max(0, self.padded_rank - self.true_rank)

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype {self.dtype!r}")

    def torch_device(self) -> torch.device:
        return torch.device(self.device)


def validate_rank_padding_config(config: RankPaddingConfig) -> None:
    if config.true_rank <= 0:
        raise ValueError(
            f"true_rank must be > 0, got {config.true_rank}"
        )
    if config.padded_rank < config.true_rank:
        raise ValueError(
            f"padded_rank ({config.padded_rank}) must be >= "
            f"true_rank ({config.true_rank})"
        )
    if config.dummy_strategy not in VALID_DUMMY_STRATEGIES:
        raise ValueError(
            f"unknown dummy_strategy {config.dummy_strategy!r};"
            f" expected one of {VALID_DUMMY_STRATEGIES}"
        )
    if config.dummy_scale <= 0:
        raise ValueError(f"dummy_scale must be > 0, got {config.dummy_scale}")


# ---------------------------------------------------------------------------
# Adapter padding
# ---------------------------------------------------------------------------


def create_rank_padded_lora_adapters(
    a: torch.Tensor,
    b: torch.Tensor,
    config: RankPaddingConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Build ``(A_pad, B_pad)`` with ``A_pad B_pad == A B``.

    Returns a dict with keys ``a_pad`` / ``b_pad`` / ``true_rank`` /
    ``padded_rank`` / ``real_slice`` / ``dummy_slice`` / ``metadata``.
    """
    validate_rank_padding_config(config)
    require_rank2("a", a)
    require_rank2("b", b)
    if a.shape[1] != config.true_rank:
        raise ValueError(
            f"a rank dimension must match true_rank={config.true_rank},"
            f" got {tuple(a.shape)}"
        )
    if b.shape[0] != config.true_rank:
        raise ValueError(
            f"b rank dimension must match true_rank={config.true_rank},"
            f" got {tuple(b.shape)}"
        )
    require_same_dtype_device("a", a, b=b)

    r = config.true_rank
    r_pad = config.padded_rank
    d_in = a.shape[0]
    d_out = b.shape[1]
    dummy_size = r_pad - r

    a_pad = torch.empty(d_in, r_pad, dtype=a.dtype, device=a.device)
    b_pad = torch.empty(r_pad, d_out, dtype=a.dtype, device=a.device)
    a_pad[:, :r] = a
    b_pad[:r, :] = b

    dummy_strategy_effective = config.dummy_strategy
    if dummy_size == 0:
        dummy_strategy_effective = "no_padding"
    elif config.dummy_strategy == "zero_dummy":
        a_pad[:, r:] = _randn(
            (d_in, dummy_size), a.dtype, a.device, generator
        ) * config.dummy_scale
        b_pad[r:, :] = 0.0
    elif config.dummy_strategy == "paired_cancellation_dummy":
        num_pairs = dummy_size // 2
        leftover = dummy_size % 2
        for i in range(num_pairs):
            r_vec = _randn((d_in,), a.dtype, a.device, generator) * config.dummy_scale
            s_vec = _randn((d_out,), a.dtype, a.device, generator) * config.dummy_scale
            a_pad[:, r + 2 * i] = r_vec
            a_pad[:, r + 2 * i + 1] = r_vec
            b_pad[r + 2 * i, :] = s_vec
            b_pad[r + 2 * i + 1, :] = -s_vec
        if leftover == 1:
            a_pad[:, -1] = (
                _randn((d_in,), a.dtype, a.device, generator) * config.dummy_scale
            )
            b_pad[-1, :] = 0.0
            dummy_strategy_effective = (
                "paired_cancellation_dummy_with_zero_tail"
            )

    return {
        "a_pad": a_pad,
        "b_pad": b_pad,
        "true_rank": r,
        "padded_rank": r_pad,
        "real_slice": slice(0, r),
        "dummy_slice": slice(r, r_pad),
        "metadata": {
            "dummy_strategy_requested": config.dummy_strategy,
            "dummy_strategy_effective": dummy_strategy_effective,
            "dummy_size": int(dummy_size),
            "dummy_scale": float(config.dummy_scale),
            "fresh_dummy_per_step": bool(config.fresh_dummy_per_step),
        },
    }


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randn(shape, generator=generator, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Plain rank-padded forward + backward references
# ---------------------------------------------------------------------------


def plain_rank_padded_lora_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    true_rank: int,
    bias: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """``Y = X W + (alpha / true_rank) X A_pad B_pad + bias``.

    Scale uses ``alpha / true_rank``, NOT ``alpha / padded_rank``.
    """
    if true_rank <= 0:
        raise ValueError(f"true_rank must be > 0, got {true_rank}")
    require_rank2("x", x)
    require_rank2("w", w)
    require_rank2("a_pad", a_pad)
    require_rank2("b_pad", b_pad)
    if a_pad.shape[1] < true_rank:
        raise ValueError(
            f"a_pad rank dim ({a_pad.shape[1]}) must be >= true_rank ({true_rank})"
        )
    if b_pad.shape[0] != a_pad.shape[1]:
        raise ValueError(
            f"b_pad first dim must match a_pad rank dim ({a_pad.shape[1]}),"
            f" got {tuple(b_pad.shape)}"
        )
    require_same_dtype_device("x", x, w=w, a_pad=a_pad, b_pad=b_pad, bias=bias)

    scale = float(alpha) / max(true_rank, 1)
    y = x @ w + scale * (x @ a_pad) @ b_pad
    if bias is not None:
        y = y + bias
    return y


def plain_rank_padded_lora_backward_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    true_rank: int,
    upstream_grad: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Analytic plain-space rank-padded LoRA backward.

    Returns ``{"grad_x", "grad_a_pad", "grad_b_pad"}``. ``grad_a_pad`` has
    shape ``(d_in, padded_rank)`` and ``grad_b_pad`` has shape
    ``(padded_rank, d_out)``. The trusted side extracts the
    ``real_slice`` columns / rows to drive the optimizer update.
    """
    require_rank2("upstream_grad", upstream_grad)
    scale = float(alpha) / max(true_rank, 1)
    grad_b_pad = scale * (x @ a_pad).transpose(0, 1) @ upstream_grad
    grad_a_pad = scale * x.transpose(0, 1) @ upstream_grad @ b_pad.transpose(0, 1)
    grad_x = upstream_grad @ w.transpose(0, 1) + scale * (
        upstream_grad @ b_pad.transpose(0, 1)
    ) @ a_pad.transpose(0, 1)
    return {"grad_x": grad_x, "grad_a_pad": grad_a_pad, "grad_b_pad": grad_b_pad}


# ---------------------------------------------------------------------------
# Masked rank-padded forward (orchestrator)
# ---------------------------------------------------------------------------


def _effective_alpha(alpha: float, true_rank: int, padded_rank: int) -> float:
    """Return α' such that the Stage 7.0/7.1 helpers' internal
    ``scale = α' / padded_rank`` equals ``α / true_rank``.
    """
    if padded_rank <= 0:
        raise ValueError(f"padded_rank must be > 0, got {padded_rank}")
    return float(alpha) * padded_rank / max(true_rank, 1)


def run_masked_rank_padded_lora_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    true_rank: int,
    padded_rank: int,
    alpha: float,
    state: LoRAState | None,
    forward_config: MaskedLoRAForwardConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, LoRAState]:
    """End-to-end masked rank-padded LoRA forward.

    Mirrors :func:`pllo.ops.lora.run_masked_lora_linear` but with the rank
    dimension replaced by ``padded_rank`` and ``alpha`` adjusted so the
    effective scaling factor remains ``alpha / true_rank``.
    """
    if a_pad.shape[1] != padded_rank or b_pad.shape[0] != padded_rank:
        raise ValueError(
            f"a_pad / b_pad must have padded rank dim {padded_rank},"
            f" got {tuple(a_pad.shape)}, {tuple(b_pad.shape)}"
        )

    inner_cfg = LoRAConfig(
        d_in=a_pad.shape[0], d_out=b_pad.shape[1], rank=padded_rank,
        alpha=_effective_alpha(alpha, true_rank, padded_rank),
        use_bias=(bias is not None),
        dtype=forward_config.dtype, device=forward_config.device,
    )
    seq_len = x.shape[0]
    new_state = create_masked_lora_state(
        inner_cfg, forward_config, seq_len=seq_len, state=state, generator=generator,
    )

    x_tilde = obfuscate_lora_input(x, new_state.n_in, new_state.pad)
    w_tilde, bias_tilde = transform_linear_weight_lora(
        w, bias, new_state.n_in_inv, new_state.n_out,
    )
    a_pad_tilde, b_pad_tilde = transform_lora_adapter(
        a_pad, b_pad, new_state.n_in_inv, new_state.n_out,
        new_state.u, new_state.u_inv, alpha=inner_cfg.alpha,
    )
    compensation: torch.Tensor | None = None
    if new_state.pad is not None:
        compensation = make_lora_pad_compensation(
            w, a_pad, b_pad, new_state.pad, new_state.n_out,
            alpha=inner_cfg.alpha,
        )
    y_tilde = masked_lora_linear_forward(
        x_tilde, w_tilde, a_pad_tilde, b_pad_tilde,
        bias_tilde=bias_tilde, alpha=inner_cfg.alpha,
        pad_compensation=compensation,
    )
    y = recover_masked_output(y_tilde, new_state.n_out_inv)
    return y, new_state


# ---------------------------------------------------------------------------
# Masked rank-padded backward (orchestrator)
# ---------------------------------------------------------------------------


def run_masked_rank_padded_lora_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    true_rank: int,
    padded_rank: int,
    alpha: float,
    state: LoRAState,
    recover_grad_x: bool = False,
) -> dict[str, torch.Tensor | None]:
    """End-to-end masked rank-padded LoRA backward.

    Returns a dict with keys ``"grad_a_pad"`` (shape ``d_in × padded_rank``),
    ``"grad_b_pad"`` (shape ``padded_rank × d_out``), and optional
    ``"grad_x"``. The trusted side then slices ``[:, :true_rank]`` and
    ``[:true_rank, :]`` to recover the real-rank gradients.
    """
    if a_pad.shape[1] != padded_rank or b_pad.shape[0] != padded_rank:
        raise ValueError(
            f"a_pad / b_pad must have padded rank dim {padded_rank},"
            f" got {tuple(a_pad.shape)}, {tuple(b_pad.shape)}"
        )
    alpha_eff = _effective_alpha(alpha, true_rank, padded_rank)

    if state.pad is None:
        x_tilde = x @ state.n_in
    else:
        x_tilde = (x - state.pad) @ state.n_in
    w_tilde = state.n_in_inv @ w @ state.n_out
    a_pad_tilde = state.n_in_inv @ a_pad @ state.u
    b_pad_tilde = state.u_inv @ b_pad @ state.n_out
    grad_y_tilde = transform_upstream_gradient(upstream_grad, state.n_out)
    masked = masked_lora_backward(
        x_tilde, a_pad_tilde, b_pad_tilde, grad_y_tilde,
        alpha=alpha_eff,
        w_tilde=w_tilde if recover_grad_x else None,
        recover_grad_x=recover_grad_x,
    )
    pad_compensation = None
    if state.pad is not None:
        pad_compensation = make_lora_grad_pad_compensation(
            a_pad, b_pad, state.pad, upstream_grad, alpha=alpha_eff,
        )
    recovered = recover_lora_gradients(
        masked["grad_a_tilde"], masked["grad_b_tilde"],
        state.n_in, state.n_out, state.u,
        grad_x_tilde=masked["grad_x_tilde"],
        grad_a_pad_compensation=(
            pad_compensation["grad_a_pad_compensation"]
            if pad_compensation is not None else None
        ),
        grad_b_pad_compensation=(
            pad_compensation["grad_b_pad_compensation"]
            if pad_compensation is not None else None
        ),
    )
    return {
        "grad_a_pad": recovered["grad_a"],
        "grad_b_pad": recovered["grad_b"],
        "grad_x": recovered.get("grad_x"),
    }


# ---------------------------------------------------------------------------
# Helpers — real-slice extraction + shape fingerprint
# ---------------------------------------------------------------------------


def extract_real_gradients(
    grad_a_pad: torch.Tensor,
    grad_b_pad: torch.Tensor,
    true_rank: int,
) -> dict[str, torch.Tensor]:
    """Extract real-rank gradient slices for the optimizer."""
    return {
        "grad_a_real": grad_a_pad[:, :true_rank].contiguous(),
        "grad_b_real": grad_b_pad[:true_rank, :].contiguous(),
    }


def dummy_contribution_norm(
    a_pad: torch.Tensor, b_pad: torch.Tensor, true_rank: int,
) -> float:
    """Compute ‖A_pad[:, r:] B_pad[r:, :]‖_F (should be ≈ 0 for valid
    dummy strategies).
    """
    if a_pad.shape[1] == true_rank:
        return 0.0
    return float(
        (a_pad[:, true_rank:] @ b_pad[true_rank:, :]).norm().item()
    )


def visible_shape_fingerprint(
    a_pad_tilde: torch.Tensor,
    b_pad_tilde: torch.Tensor,
) -> dict[str, Any]:
    """Return the GPU-visible shape info (no raw tensor values)."""
    return {
        "a_tilde_pad_shape": list(a_pad_tilde.shape),
        "b_tilde_pad_shape": list(b_pad_tilde.shape),
        "visible_rank_from_a_shape": int(a_pad_tilde.shape[1]),
        "visible_rank_from_b_shape": int(b_pad_tilde.shape[0]),
    }


__all__ = [
    "RankPaddingConfig",
    "VALID_DUMMY_STRATEGIES",
    "create_rank_padded_lora_adapters",
    "dummy_contribution_norm",
    "extract_real_gradients",
    "plain_rank_padded_lora_backward_reference",
    "plain_rank_padded_lora_forward",
    "run_masked_rank_padded_lora_backward",
    "run_masked_rank_padded_lora_linear",
    "validate_rank_padding_config",
    "visible_shape_fingerprint",
]
