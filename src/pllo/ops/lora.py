"""Stage 7.0 — LoRA primitive (private adapter, public base weight).

Functional API for Linear + LoRA forward, where the GPU side only ever sees
masked / transformed adapter tensors and never sees the plaintext A / B or
the merged ΔW = A B. The base weight W is public (e.g. the frozen
pre-trained linear of a public base model), while A / B are private LoRA
adapter parameters that stay trusted-only.

Plain reference:
    Y = X W + (alpha / r) X A B + bias

Masked path (no merge into W):
    X_tilde      = (X - T_in) N_in    (or X N_in when use_pad=False)
    W_tilde      = N_in^{-1} W N_out
    A_tilde      = N_in^{-1} A U
    B_tilde      = U^{-1}   B N_out
    bias_tilde   = bias N_out
    C_W          = T_in W N_out
    C_LoRA       = (alpha / r) T_in A B N_out
    Y_tilde      = X_tilde W_tilde
                 + (alpha / r) (X_tilde A_tilde) B_tilde
                 + bias_tilde + C_W + C_LoRA
    Y_recovered  = Y_tilde N_out^{-1}

Stage 7.0 deliberately does NOT merge the adapter into the base weight
(constraint 7). ``U`` is a rank-space invertible mask; sampling fresh ``U``
per call obfuscates the adapter without altering the recovered output.

The module is **not a real TEE**. Anything written here that exposes
plaintext is by construction trusted-side state; reports must therefore
emit fingerprints/metrics rather than raw mask or adapter tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.utils.validation import (
    require_rank2,
    require_same_dtype_device,
    require_shape,
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


_DTYPE_TABLE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_TABLE[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype {name!r}; expected one of {tuple(_DTYPE_TABLE)}"
        ) from exc


def _torch_device(name: str) -> torch.device:
    return torch.device(name)


@dataclass
class LoRAConfig:
    """Public + private shape configuration for a single LoRA-augmented linear."""

    d_in: int
    d_out: int
    rank: int
    alpha: float = 1.0
    use_bias: bool = True
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.d_in <= 0:
            raise ValueError(f"d_in must be positive, got {self.d_in}")
        if self.d_out <= 0:
            raise ValueError(f"d_out must be positive, got {self.d_out}")
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.rank >= min(self.d_in, self.d_out):
            # We deliberately allow it for tests, but warn against the
            # "low-rank" claim — the rank-space mask still applies, but the
            # bandwidth-saving narrative goes away.
            pass
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    def torch_dtype(self) -> torch.dtype:
        return _torch_dtype(self.dtype)

    def torch_device(self) -> torch.device:
        return _torch_device(self.device)


@dataclass
class MaskedLoRAForwardConfig:
    """Runtime knobs for one masked LoRA forward call."""

    use_pad: bool = True
    fresh_u_per_call: bool = True
    fresh_masks_per_call: bool = True
    pad_scale: float = 1.0
    dtype: str = "float32"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        return _torch_dtype(self.dtype)

    def torch_device(self) -> torch.device:
        return _torch_device(self.device)


@dataclass
class LoRAState:
    """Trusted-side mask + pad state for one masked LoRA forward call.

    All tensor fields are trusted-only. Reports must NOT export this state
    directly; use :func:`lora_state_fingerprint` instead.
    """

    n_in: torch.Tensor
    n_in_inv: torch.Tensor
    n_out: torch.Tensor
    n_out_inv: torch.Tensor
    u: torch.Tensor
    u_inv: torch.Tensor
    pad: torch.Tensor | None
    rank: int
    alpha: float


# ---------------------------------------------------------------------------
# Adapter initialisation + state sampling
# ---------------------------------------------------------------------------


def init_lora_adapters(
    config: LoRAConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise ``(A, B)`` adapter pair.

    ``A`` is Kaiming-style normal-scaled; ``B`` is zero. This matches the
    standard LoRA "Y starts at the base model" convention.
    """
    dtype = config.torch_dtype()
    device = config.torch_device()
    scale = (1.0 / max(config.d_in, 1)) ** 0.5
    if generator is None:
        a = torch.randn(config.d_in, config.rank, dtype=dtype, device=device) * scale
    else:
        a = (
            torch.randn(
                config.d_in, config.rank,
                generator=generator, dtype=dtype, device=device,
            )
            * scale
        )
    b = torch.zeros(config.rank, config.d_out, dtype=dtype, device=device)
    return a, b


def create_masked_lora_state(
    config: LoRAConfig,
    forward_config: MaskedLoRAForwardConfig,
    *,
    seq_len: int,
    state: LoRAState | None = None,
    generator: torch.Generator | None = None,
) -> LoRAState:
    """Sample / refresh masks for one masked LoRA forward call.

    ``state`` lets the caller reuse a previously-sampled mask state. The
    ``fresh_masks_per_call`` / ``fresh_u_per_call`` toggles on
    ``forward_config`` decide which sub-tensors are re-sampled.

    ``generator`` controls the randomness; if ``None``, the global RNG is
    used. ``seq_len`` is only used when ``use_pad`` is True (the pad has
    shape ``(seq_len, d_in)``).
    """
    dtype = config.torch_dtype()
    device = config.torch_device()

    def _gen_matrix(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if generator is None:
            return generate_invertible_matrix(dim, dtype, device)
        # Use the explicit generator path so tests are reproducible.
        m = torch.randn(dim, dim, generator=generator, dtype=dtype, device=device)
        q, r = torch.linalg.qr(m)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        return q, q.transpose(-2, -1)

    if state is None or forward_config.fresh_masks_per_call:
        n_in, n_in_inv = _gen_matrix(config.d_in)
        n_out, n_out_inv = _gen_matrix(config.d_out)
    else:
        n_in, n_in_inv = state.n_in, state.n_in_inv
        n_out, n_out_inv = state.n_out, state.n_out_inv

    if state is None or forward_config.fresh_u_per_call:
        u, u_inv = _gen_matrix(config.rank)
    else:
        u, u_inv = state.u, state.u_inv

    pad: torch.Tensor | None = None
    if forward_config.use_pad:
        if generator is None:
            pad = generate_pad(
                (seq_len, config.d_in), dtype, device, forward_config.pad_scale,
            )
        else:
            pad = (
                torch.randn(
                    seq_len, config.d_in,
                    generator=generator, dtype=dtype, device=device,
                )
                * forward_config.pad_scale
            )

    return LoRAState(
        n_in=n_in, n_in_inv=n_in_inv,
        n_out=n_out, n_out_inv=n_out_inv,
        u=u, u_inv=u_inv,
        pad=pad,
        rank=config.rank,
        alpha=float(config.alpha),
    )


def lora_state_fingerprint(state: LoRAState) -> dict[str, Any]:
    """Return a JSON-safe fingerprint of a trusted-side LoRA state.

    The fingerprint exposes only shape + an opaque norm-based digest; the
    raw mask / pad tensors are NEVER published.
    """
    def _digest(t: torch.Tensor | None) -> dict[str, Any] | None:
        if t is None:
            return None
        flat = t.detach().to(torch.float64).reshape(-1)
        return {
            "shape": list(t.shape),
            "frobenius_norm_digest": float((flat * flat).sum().sqrt().item()),
        }

    return {
        "n_in": _digest(state.n_in),
        "n_out": _digest(state.n_out),
        "u": _digest(state.u),
        "pad_present": state.pad is not None,
        "pad": _digest(state.pad),
        "rank": int(state.rank),
        "alpha": float(state.alpha),
    }


# ---------------------------------------------------------------------------
# Plain LoRA forward
# ---------------------------------------------------------------------------


def plain_lora_linear_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Plaintext reference: ``Y = X W + (alpha / r) X A B + bias``."""
    require_rank2("x", x)
    require_rank2("w", w)
    require_rank2("a", a)
    require_rank2("b", b)
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"w must have d_in={x.shape[1]}, got {tuple(w.shape)}"
        )
    if a.shape[0] != x.shape[1]:
        raise ValueError(
            f"a must have d_in={x.shape[1]}, got {tuple(a.shape)}"
        )
    if b.shape[0] != a.shape[1]:
        raise ValueError(
            f"b rank must match a rank={a.shape[1]}, got {tuple(b.shape)}"
        )
    if b.shape[1] != w.shape[1]:
        raise ValueError(
            f"b must have d_out={w.shape[1]}, got {tuple(b.shape)}"
        )
    if bias is not None:
        require_shape("bias", bias, (w.shape[1],))
    require_same_dtype_device("x", x, w=w, a=a, b=b, bias=bias)

    rank = a.shape[1]
    scale = float(alpha) / max(rank, 1)
    y = x @ w + scale * (x @ a) @ b
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# Adapter / weight / input transforms (trusted-side)
# ---------------------------------------------------------------------------


def transform_lora_adapter(
    a: torch.Tensor,
    b: torch.Tensor,
    n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    u: torch.Tensor,
    u_inv: torch.Tensor,
    *,
    alpha: float = 1.0,  # kept on API for documentation; not folded into A/B
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform the LoRA adapters for masked GPU consumption.

    ``A_tilde = N_in^{-1} A U`` and ``B_tilde = U^{-1} B N_out``. Alpha is
    intentionally NOT folded into the adapter tensors so the forward keeps
    a single scaling site.
    """
    require_rank2("a", a)
    require_rank2("b", b)
    require_rank2("n_in_inv", n_in_inv)
    require_rank2("n_out", n_out)
    require_rank2("u", u)
    require_rank2("u_inv", u_inv)
    if b.shape[0] != a.shape[1]:
        raise ValueError(
            f"b rank must match a rank={a.shape[1]}, got {tuple(b.shape)}"
        )
    if u.shape != (a.shape[1], a.shape[1]):
        raise ValueError(
            f"u must be ({a.shape[1]}, {a.shape[1]}), got {tuple(u.shape)}"
        )
    if u_inv.shape != u.shape:
        raise ValueError(
            f"u_inv must match u shape {tuple(u.shape)}, got {tuple(u_inv.shape)}"
        )
    if n_in_inv.shape != (a.shape[0], a.shape[0]):
        raise ValueError(
            f"n_in_inv must be ({a.shape[0]}, {a.shape[0]}), got {tuple(n_in_inv.shape)}"
        )
    if n_out.shape != (b.shape[1], b.shape[1]):
        raise ValueError(
            f"n_out must be ({b.shape[1]}, {b.shape[1]}), got {tuple(n_out.shape)}"
        )
    require_same_dtype_device(
        "a", a, b=b, n_in_inv=n_in_inv, n_out=n_out, u=u, u_inv=u_inv,
    )
    _ = float(alpha)  # alpha intentionally unused here, only referenced.

    a_tilde = n_in_inv @ a @ u
    b_tilde = u_inv @ b @ n_out
    return a_tilde, b_tilde


def transform_linear_weight_lora(
    w: torch.Tensor,
    bias: torch.Tensor | None,
    n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transform the public base weight + bias into the masked domain."""
    require_rank2("w", w)
    require_rank2("n_in_inv", n_in_inv)
    require_rank2("n_out", n_out)
    if n_in_inv.shape != (w.shape[0], w.shape[0]):
        raise ValueError(
            f"n_in_inv must be ({w.shape[0]}, {w.shape[0]}), got {tuple(n_in_inv.shape)}"
        )
    if n_out.shape != (w.shape[1], w.shape[1]):
        raise ValueError(
            f"n_out must be ({w.shape[1]}, {w.shape[1]}), got {tuple(n_out.shape)}"
        )
    if bias is not None:
        require_shape("bias", bias, (w.shape[1],))
    require_same_dtype_device("w", w, n_in_inv=n_in_inv, n_out=n_out, bias=bias)

    w_tilde = n_in_inv @ w @ n_out
    bias_tilde = None if bias is None else bias @ n_out
    return w_tilde, bias_tilde


def obfuscate_lora_input(
    x: torch.Tensor,
    n_in: torch.Tensor,
    pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply optional input pad and right-mask ``X_tilde = (X - T) N_in``."""
    require_rank2("x", x)
    require_rank2("n_in", n_in)
    if n_in.shape != (x.shape[1], x.shape[1]):
        raise ValueError(
            f"n_in must be ({x.shape[1]}, {x.shape[1]}), got {tuple(n_in.shape)}"
        )
    if pad is not None:
        require_shape("pad", pad, tuple(x.shape))
    require_same_dtype_device("x", x, n_in=n_in, pad=pad)
    if pad is None:
        return x @ n_in
    return (x - pad) @ n_in


def make_lora_pad_compensation(
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    pad: torch.Tensor,
    n_out: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """``C = T W N_out + (alpha / r) T A B N_out``."""
    require_rank2("w", w)
    require_rank2("a", a)
    require_rank2("b", b)
    require_rank2("pad", pad)
    require_rank2("n_out", n_out)
    if pad.shape[1] != w.shape[0]:
        raise ValueError(
            f"pad d_in must match w d_in={w.shape[0]}, got {tuple(pad.shape)}"
        )
    if n_out.shape != (w.shape[1], w.shape[1]):
        raise ValueError(
            f"n_out must be ({w.shape[1]}, {w.shape[1]}), got {tuple(n_out.shape)}"
        )
    require_same_dtype_device("w", w, a=a, b=b, pad=pad, n_out=n_out)
    scale = float(alpha) / max(a.shape[1], 1)
    return pad @ w @ n_out + scale * (pad @ a) @ b @ n_out


# ---------------------------------------------------------------------------
# Masked LoRA forward (GPU domain)
# ---------------------------------------------------------------------------


def masked_lora_linear_forward(
    x_tilde: torch.Tensor,
    w_tilde: torch.Tensor,
    a_tilde: torch.Tensor,
    b_tilde: torch.Tensor,
    bias_tilde: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    pad_compensation: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPU-side masked LoRA forward.

    The GPU only sees masked / transformed tensors; this function
    intentionally takes no plaintext A / B / N / U.
    """
    require_rank2("x_tilde", x_tilde)
    require_rank2("w_tilde", w_tilde)
    require_rank2("a_tilde", a_tilde)
    require_rank2("b_tilde", b_tilde)
    if x_tilde.shape[1] != w_tilde.shape[0]:
        raise ValueError(
            f"w_tilde d_in must be {x_tilde.shape[1]}, got {tuple(w_tilde.shape)}"
        )
    if a_tilde.shape[0] != x_tilde.shape[1]:
        raise ValueError(
            f"a_tilde d_in must be {x_tilde.shape[1]}, got {tuple(a_tilde.shape)}"
        )
    if b_tilde.shape[0] != a_tilde.shape[1]:
        raise ValueError(
            f"b_tilde rank must match a_tilde rank={a_tilde.shape[1]},"
            f" got {tuple(b_tilde.shape)}"
        )
    if b_tilde.shape[1] != w_tilde.shape[1]:
        raise ValueError(
            f"b_tilde d_out must match w_tilde d_out={w_tilde.shape[1]},"
            f" got {tuple(b_tilde.shape)}"
        )
    output_shape = (x_tilde.shape[0], w_tilde.shape[1])
    if bias_tilde is not None:
        require_shape("bias_tilde", bias_tilde, (w_tilde.shape[1],))
    if pad_compensation is not None:
        require_shape("pad_compensation", pad_compensation, output_shape)
    require_same_dtype_device(
        "x_tilde", x_tilde,
        w_tilde=w_tilde, a_tilde=a_tilde, b_tilde=b_tilde,
        bias_tilde=bias_tilde, pad_compensation=pad_compensation,
    )

    rank = a_tilde.shape[1]
    scale = float(alpha) / max(rank, 1)
    y_tilde = x_tilde @ w_tilde + scale * (x_tilde @ a_tilde) @ b_tilde
    if bias_tilde is not None:
        y_tilde = y_tilde + bias_tilde
    if pad_compensation is not None:
        y_tilde = y_tilde + pad_compensation
    return y_tilde


def recover_masked_output(
    y_tilde: torch.Tensor,
    n_out_inv: torch.Tensor,
    pad_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover ``Y = Y_tilde @ N_out^{-1}`` (+ optional output pad)."""
    require_rank2("y_tilde", y_tilde)
    require_rank2("n_out_inv", n_out_inv)
    if n_out_inv.shape != (y_tilde.shape[1], y_tilde.shape[1]):
        raise ValueError(
            f"n_out_inv must be ({y_tilde.shape[1]}, {y_tilde.shape[1]}),"
            f" got {tuple(n_out_inv.shape)}"
        )
    if pad_out is not None:
        require_shape("pad_out", pad_out, tuple(y_tilde.shape))
    require_same_dtype_device(
        "y_tilde", y_tilde, n_out_inv=n_out_inv, pad_out=pad_out,
    )
    y = y_tilde @ n_out_inv
    if pad_out is not None:
        y = y + pad_out
    return y


# ---------------------------------------------------------------------------
# Convenience: one-shot trusted-orchestrated masked LoRA forward
# ---------------------------------------------------------------------------


def run_masked_lora_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    config: LoRAConfig,
    forward_config: MaskedLoRAForwardConfig,
    *,
    state: LoRAState | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, LoRAState]:
    """Run one trusted-orchestrated masked LoRA forward end-to-end.

    Returns ``(Y_recovered, lora_state)``. The GPU-domain tensors are
    intermediate and discarded — only the recovered plaintext-space output
    and the trusted state survive (the state is needed for downstream
    gradient computation by the caller in the trusted side).
    """
    seq_len = x.shape[0]
    new_state = create_masked_lora_state(
        config, forward_config, seq_len=seq_len, state=state, generator=generator,
    )
    x_tilde = obfuscate_lora_input(x, new_state.n_in, new_state.pad)
    w_tilde, bias_tilde = transform_linear_weight_lora(
        w, bias, new_state.n_in_inv, new_state.n_out,
    )
    a_tilde, b_tilde = transform_lora_adapter(
        a, b, new_state.n_in_inv, new_state.n_out, new_state.u, new_state.u_inv,
        alpha=config.alpha,
    )
    compensation: torch.Tensor | None = None
    if new_state.pad is not None:
        compensation = make_lora_pad_compensation(
            w, a, b, new_state.pad, new_state.n_out, alpha=config.alpha,
        )
    y_tilde = masked_lora_linear_forward(
        x_tilde, w_tilde, a_tilde, b_tilde,
        bias_tilde=bias_tilde,
        alpha=config.alpha,
        pad_compensation=compensation,
    )
    y = recover_masked_output(y_tilde, new_state.n_out_inv)
    return y, new_state


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "LoRAConfig",
    "MaskedLoRAForwardConfig",
    "LoRAState",
    "create_masked_lora_state",
    "init_lora_adapters",
    "lora_state_fingerprint",
    "make_lora_pad_compensation",
    "masked_lora_linear_forward",
    "obfuscate_lora_input",
    "plain_lora_linear_forward",
    "recover_masked_output",
    "run_masked_lora_linear",
    "transform_linear_weight_lora",
    "transform_lora_adapter",
]
