"""Stage 5.1 — Norm primitive experiments.

Two probes share this module:

1. ``run_trusted_norm_probe`` — drives the Stage-5.1 ``trusted_norm_forward``
   primitive over a (batch_size, seq_len, hidden_size, use_pad, norm_type)
   cell. Validates ``Y_tilde = Y N_out`` (or the pad variant).

2. ``run_rmsnorm_orthogonal_probe`` — explores RMSNorm commutation under an
   *orthogonal* right mask ``N``. The key facts the probe verifies:

       * ``N^T N ≈ I`` (orthogonal by QR construction)
       * ``rms(X N) ≈ rms(X)`` (column-wise norm preservation)
       * ``normalize(X N) ≈ normalize(X) N``
       * scalar gamma commutes with N (broadcast across hidden dim)
       * vector gamma generally does NOT commute with N

   The probe deliberately reports ``gamma_commutation_error`` for the
   vector-gamma case as a *non-zero* number — that's the point of the
   probe, and the report calls it out under
   "Vector gamma breaks simple right-mask commutation".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.experiments.report_utils import compare
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.model_zoo.base import torch_dtype_from_string
from pllo.ops.norm import (
    layer_norm_reference,
    rms_norm_reference,
    trusted_norm_forward,
)


# ---------------------------------------------------------------------------
# Trusted norm probe (drives trusted_norm_forward over one cell)
# ---------------------------------------------------------------------------


@dataclass
class TrustedNormProbeConfig:
    norm_type: str = "layernorm"          # 'layernorm' or 'rmsnorm'
    batch_size: int = 2
    seq_len: int = 8
    hidden_size: int = 64
    eps: float = 1e-5
    use_pad: bool = True
    use_weight: bool = True
    use_bias: bool = True                  # only meaningful for layernorm
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42


def run_trusted_norm_probe(config: TrustedNormProbeConfig) -> dict[str, Any]:
    """Run one cell of the trusted-norm primitive and report metrics."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)

    norm_type = config.norm_type.lower()
    if norm_type not in {"layernorm", "rmsnorm"}:
        raise ValueError(f"unknown norm_type {config.norm_type!r}")

    # Plaintext hidden, masks, optional pads.
    H = torch.randn(
        config.batch_size, config.seq_len, config.hidden_size,
        dtype=dtype, device=device,
    )
    flat = H.reshape(-1, config.hidden_size)

    n_in, n_in_inv = generate_invertible_matrix(
        config.hidden_size, dtype, device
    )
    n_out, n_out_inv = generate_invertible_matrix(
        config.hidden_size, dtype, device
    )

    if config.use_pad:
        pad_in_flat = generate_pad(tuple(flat.shape), dtype, device, 1.0)
        pad_out_flat = generate_pad(tuple(flat.shape), dtype, device, 1.0)
        x_tilde = (flat - pad_in_flat) @ n_in
    else:
        pad_in_flat = None
        pad_out_flat = None
        x_tilde = flat @ n_in

    weight = (
        torch.randn(config.hidden_size, dtype=dtype, device=device)
        if config.use_weight
        else None
    )
    bias = (
        torch.randn(config.hidden_size, dtype=dtype, device=device)
        if (config.use_bias and norm_type == "layernorm")
        else None
    )

    result = trusted_norm_forward(
        x_tilde=x_tilde,
        n_in_inv=n_in_inv,
        norm_weight=weight,
        norm_bias=bias,
        n_out=n_out,
        norm_type=norm_type,
        eps=config.eps,
        pad_in=pad_in_flat,
        pad_out=pad_out_flat,
    )

    # Independent verification: y_plain must match a fresh reference applied
    # directly to ``H``.
    if norm_type == "layernorm":
        y_reference = layer_norm_reference(flat, weight, bias, config.eps)
    else:
        y_reference = rms_norm_reference(flat, weight, config.eps)
    reference_metrics = compare(y_reference, result["y_plain"], atol=atol, rtol=rtol)

    # y_tilde shape invariant: Y_tilde = Y N_out (or (Y - T_out) N_out).
    expected_y_tilde = (
        result["y_plain"] @ n_out
        if pad_out_flat is None
        else (result["y_plain"] - pad_out_flat) @ n_out
    )
    y_tilde_invariant = compare(
        expected_y_tilde, result["y_tilde"], atol=atol, rtol=rtol
    )

    return {
        "config": asdict(config),
        "metrics": {
            "max_abs_error": result["max_abs_error"],
            "mean_abs_error": result["mean_abs_error"],
            "relative_l2_error": result["relative_l2_error"],
            "cosine_similarity": result["cosine_similarity"],
            "allclose": result["allclose"],
        },
        "reference_metrics": reference_metrics,
        "y_tilde_invariant_metrics": y_tilde_invariant,
        "pad_present": {
            "pad_in": pad_in_flat is not None,
            "pad_out": pad_out_flat is not None,
        },
        "weight_present": weight is not None,
        "bias_present": bias is not None,
    }


# ---------------------------------------------------------------------------
# Restricted RMSNorm orthogonal-mask probe
# ---------------------------------------------------------------------------


@dataclass
class RMSNormOrthogonalProbeConfig:
    batch_size: int = 2
    seq_len: int = 8
    hidden_size: int = 64
    num_trials: int = 16
    eps: float = 1e-6
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 7


def _generate_orthogonal(
    hidden: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Sample an orthogonal ``hidden × hidden`` matrix via QR decomposition."""
    g = torch.randn(hidden, hidden, dtype=torch.float64, device=device)
    q, _ = torch.linalg.qr(g)
    # Multiply by sign of diagonal of R to make the QR unique & Haar-ish.
    return q.to(dtype=dtype)


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def run_rmsnorm_orthogonal_probe(
    config: RMSNormOrthogonalProbeConfig,
) -> dict[str, Any]:
    """Verify the RMSNorm right-mask commutation under orthogonal masks."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)

    orthogonality_errors: list[float] = []
    rms_pres_errors: list[float] = []
    normalized_state_errors: list[float] = []
    scalar_gamma_errors: list[float] = []
    vector_gamma_errors: list[float] = []

    for trial in range(config.num_trials):
        torch.manual_seed(config.seed + trial)
        X = torch.randn(
            config.batch_size, config.seq_len, config.hidden_size,
            dtype=dtype, device=device,
        )
        N = _generate_orthogonal(config.hidden_size, dtype, device)

        # 1. Orthogonality: N^T N ≈ I
        eye = torch.eye(config.hidden_size, dtype=dtype, device=device)
        ortho_err = float((N.T @ N - eye).abs().max().item())
        orthogonality_errors.append(ortho_err)

        # 2. RMS preservation: rms(X N) ≈ rms(X)
        rms_X = _rms(X, config.eps)
        rms_XN = _rms(X @ N, config.eps)
        rms_pres_errors.append(float((rms_X - rms_XN).abs().max().item()))

        # 3. Normalized state: normalize(X N) ≈ normalize(X) N
        nxn = _normalize(X @ N, config.eps)
        nx_n = _normalize(X, config.eps) @ N
        normalized_state_errors.append(
            float((nxn - nx_n).abs().max().item())
        )

        # 4. Scalar gamma commutes: gamma * (normalize(X) N) = (gamma * normalize(X)) N
        gamma_scalar = torch.tensor(0.5 + 1.5 * trial / max(config.num_trials, 1),
                                    dtype=dtype, device=device)
        lhs_scalar = gamma_scalar * (_normalize(X, config.eps) @ N)
        rhs_scalar = (gamma_scalar * _normalize(X, config.eps)) @ N
        scalar_gamma_errors.append(
            float((lhs_scalar - rhs_scalar).abs().max().item())
        )

        # 5. Vector gamma generally does NOT commute:
        #    gamma ⊙ (normalize(X) N) ≠ (gamma ⊙ normalize(X)) N
        gamma_vector = torch.randn(
            config.hidden_size, dtype=dtype, device=device
        )
        lhs_vec = gamma_vector * (_normalize(X, config.eps) @ N)
        rhs_vec = (gamma_vector * _normalize(X, config.eps)) @ N
        vector_gamma_errors.append(
            float((lhs_vec - rhs_vec).abs().max().item())
        )

    def _stat(name: str, vals: list[float]) -> dict[str, float]:
        t = torch.tensor(vals, dtype=torch.float64)
        return {
            f"{name}_max": float(t.max().item()),
            f"{name}_mean": float(t.mean().item()),
        }

    orthogonality_max = max(orthogonality_errors)
    rms_pres_max = max(rms_pres_errors)
    normalized_state_max = max(normalized_state_errors)
    scalar_gamma_max = max(scalar_gamma_errors)
    vector_gamma_max = max(vector_gamma_errors)

    allclose_without_gamma = normalized_state_max < atol
    allclose_with_scalar_gamma = scalar_gamma_max < atol
    # Vector gamma is expected to break commutation; allclose is False unless
    # the random draw collapses to a degenerate case (e.g. zero gamma).
    allclose_with_vector_gamma = vector_gamma_max < atol

    return {
        "config": asdict(config),
        "num_trials": config.num_trials,
        "orthogonality_error": orthogonality_max,
        "rms_preservation_error": rms_pres_max,
        "normalized_state_error": normalized_state_max,
        "gamma_commutation_error": {
            "scalar_gamma_max": scalar_gamma_max,
            "vector_gamma_max": vector_gamma_max,
        },
        "allclose_without_gamma": allclose_without_gamma,
        "allclose_with_scalar_gamma": allclose_with_scalar_gamma,
        "allclose_with_vector_gamma": allclose_with_vector_gamma,
        "stats": {
            **_stat("orthogonality", orthogonality_errors),
            **_stat("rms_preservation", rms_pres_errors),
            **_stat("normalized_state", normalized_state_errors),
            **_stat("scalar_gamma", scalar_gamma_errors),
            **_stat("vector_gamma", vector_gamma_errors),
        },
        "note": (
            "Vector gamma generally breaks simple right-mask commutation"
            " because gamma is a per-channel scaling on the post-normalisation"
            " axis. Only scalar gamma (a single broadcast scalar) commutes"
            " with the right multiply N."
        ),
    }


__all__ = [
    "RMSNormOrthogonalProbeConfig",
    "TrustedNormProbeConfig",
    "run_rmsnorm_orthogonal_probe",
    "run_trusted_norm_probe",
]
