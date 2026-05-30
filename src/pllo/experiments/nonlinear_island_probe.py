"""Stage 5.2 — Nonlinear island correctness probe runner.

Sweeps four island categories and collects max-abs / cosine / allclose
metrics per cell. Returns a structured dict consumable by
``scripts/run_nonlinear_island_experiments.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.experiments.report_utils import compare
from pllo.model_zoo.base import torch_dtype_from_string
from pllo.ops.compatible_masks import (
    generate_dense_invertible,
    generate_mean_preserving_orthogonal,
    generate_orthogonal,
    generate_permutation,
    mean_preservation_error,
    orthogonal_error,
)
from pllo.ops.nonlinear_islands import (
    run_activation_permutation_island,
    run_gelu_mlp_island,
    run_layernorm_mean_preserving_island,
    run_rmsnorm_orthogonal_island,
    run_swiglu_mlp_island,
    run_swiglu_paired_permutation_island,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NonlinearIslandProbeConfig:
    batch_size: int = 2
    seq_len: int = 8
    hidden_size: int = 64
    intermediate_size: int = 256
    num_trials: int = 8
    use_pad: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    output_dir: str = "outputs"


# Sweeps actually executed by run_nonlinear_island_experiments.
NORM_HIDDEN_SWEEP = (64, 128)
NORM_USE_PAD_SWEEP = (True, False)
ACTIVATION_HIDDEN_SWEEP = (64, 128)
ACTIVATION_TYPES = ("gelu", "relu", "silu")
MLP_HIDDEN_SWEEP = (64, 128)
MLP_USE_PAD_SWEEP = (True, False)
MLP_TYPES = ("gelu_mlp", "relu_mlp", "silu_mlp", "swiglu_mlp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hidden_state(
    config: NonlinearIslandProbeConfig,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        config.batch_size, config.seq_len, hidden,
        dtype=dtype, device=device, generator=g,
    )


def _make_pad(
    config: NonlinearIslandProbeConfig,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    use_pad: bool,
) -> torch.Tensor | None:
    if not use_pad:
        return None
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    return torch.randn(
        config.batch_size, config.seq_len, hidden,
        dtype=dtype, device=device, generator=g,
    )


# ---------------------------------------------------------------------------
# Section A — Norm-compatible islands
# ---------------------------------------------------------------------------


def _norm_island_cells(
    config: NonlinearIslandProbeConfig,
    dtype: torch.dtype,
    device: torch.device,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for hidden in NORM_HIDDEN_SWEEP:
        n_out, _ = generate_dense_invertible(hidden, dtype, device)

        # RMSNorm with orthogonal mask
        N_o = generate_orthogonal(hidden, dtype, device)
        gamma_rms = torch.randn(hidden, dtype=dtype, device=device)
        W = torch.randn(hidden, hidden, dtype=dtype, device=device)
        b = torch.randn(hidden, dtype=dtype, device=device)
        x = _hidden_state(config, hidden, dtype, device, seed=100 + hidden)
        r = run_rmsnorm_orthogonal_island(
            x, N_o, gamma_rms, W, b, n_out
        )
        m = compare(r["expected_y_tilde"], r["y_tilde"], atol=atol, rtol=rtol)
        cells.append(
            {
                "island": "rmsnorm_orthogonal_affine_fold",
                "mask_family": "orthogonal",
                "activation_type": None,
                "hidden_size": hidden,
                "intermediate_size": None,
                "use_pad": False,
                "used_pad_at_linear_boundary": False,
                "online_extra_matmul_count": 0,
                "preprocessing_only_transformations": [
                    "N_in (orthogonal) folded into W via diag(gamma) and N.T pre-multiply",
                    "N_out fused into W and bias",
                ],
                "orthogonality_error": orthogonal_error(N_o),
                "mean_preservation_error": None,
                "metrics": m,
            }
        )

        # LayerNorm with mean-preserving orthogonal mask
        N_mp = generate_mean_preserving_orthogonal(hidden, dtype, device)
        gamma_ln = torch.randn(hidden, dtype=dtype, device=device)
        beta_ln = torch.randn(hidden, dtype=dtype, device=device)
        x = _hidden_state(config, hidden, dtype, device, seed=200 + hidden)
        r = run_layernorm_mean_preserving_island(
            x, N_mp, gamma_ln, beta_ln, W, b, n_out
        )
        m = compare(r["expected_y_tilde"], r["y_tilde"], atol=atol, rtol=rtol)
        cells.append(
            {
                "island": "layernorm_mean_preserving_affine_fold",
                "mask_family": "mean_preserving_orthogonal",
                "activation_type": None,
                "hidden_size": hidden,
                "intermediate_size": None,
                "use_pad": False,
                "used_pad_at_linear_boundary": False,
                "online_extra_matmul_count": 0,
                "preprocessing_only_transformations": [
                    "N_in (mean-preserving orthogonal) folded into W via diag(gamma) and N.T pre-multiply",
                    "beta @ W folded into bias",
                    "N_out fused into W and bias",
                ],
                "orthogonality_error": orthogonal_error(N_mp),
                "mean_preservation_error": mean_preservation_error(N_mp),
                "metrics": m,
            }
        )
    return cells


# ---------------------------------------------------------------------------
# Section B — Activation permutation islands
# ---------------------------------------------------------------------------


def _activation_island_cells(
    config: NonlinearIslandProbeConfig,
    dtype: torch.dtype,
    device: torch.device,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for hidden in ACTIVATION_HIDDEN_SWEEP:
        perm = generate_permutation(hidden, dtype, device)["perm"]
        for activation_type in ACTIVATION_TYPES:
            z = _hidden_state(config, hidden, dtype, device, seed=300 + hidden)
            r = run_activation_permutation_island(z, perm, activation_type)
            m = compare(r["rhs"], r["lhs"], atol=atol, rtol=rtol)
            cells.append(
                {
                    "island": "activation_permutation",
                    "mask_family": "permutation",
                    "activation_type": activation_type,
                    "hidden_size": hidden,
                    "intermediate_size": None,
                    "use_pad": False,
                    "used_pad_at_linear_boundary": False,
                    "online_extra_matmul_count": 0,
                    "preprocessing_only_transformations": [
                        "Permutation applied via index_select; commutes exactly with element-wise activation.",
                    ],
                    "orthogonality_error": None,
                    "mean_preservation_error": None,
                    "metrics": m,
                }
            )

        # SwiGLU paired permutation
        a = _hidden_state(config, hidden, dtype, device, seed=400 + hidden)
        b = _hidden_state(config, hidden, dtype, device, seed=500 + hidden)
        r = run_swiglu_paired_permutation_island(a, b, perm)
        m = compare(r["rhs"], r["lhs"], atol=atol, rtol=rtol)
        cells.append(
            {
                "island": "swiglu_paired_permutation",
                "mask_family": "paired_permutation",
                "activation_type": "swiglu",
                "hidden_size": hidden,
                "intermediate_size": None,
                "use_pad": False,
                "used_pad_at_linear_boundary": False,
                "online_extra_matmul_count": 0,
                "preprocessing_only_transformations": [
                    "Shared permutation P for up and gate branches; SwiGLU commutes exactly.",
                ],
                "orthogonality_error": None,
                "mean_preservation_error": None,
                "metrics": m,
            }
        )
    return cells


# ---------------------------------------------------------------------------
# Section C — Full MLP islands
# ---------------------------------------------------------------------------


def _mlp_island_cells(
    config: NonlinearIslandProbeConfig,
    dtype: torch.dtype,
    device: torch.device,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for hidden in MLP_HIDDEN_SWEEP:
        intermediate = hidden * 4
        n_in, n_in_inv = generate_dense_invertible(hidden, dtype, device)
        n_out, _ = generate_dense_invertible(hidden, dtype, device)
        perm = generate_permutation(intermediate, dtype, device)["perm"]

        # Scale weights for numerical stability at fp32. ``torch.randn`` for
        # large hidden produces O(sqrt(d)) outputs; we keep the test scale
        # modest so allclose at atol=1e-3 holds despite repeated matmuls.
        scale = 1.0 / max(1.0, (intermediate ** 0.5))

        for use_pad in MLP_USE_PAD_SWEEP:
            x = _hidden_state(config, hidden, dtype, device, seed=600 + hidden)
            pad_in = _make_pad(config, hidden, dtype, device, seed=601 + hidden, use_pad=use_pad)

            # ---- GELU / ReLU / SiLU MLP ----
            for mlp_type in ("gelu_mlp", "relu_mlp", "silu_mlp"):
                activation_type = mlp_type.split("_")[0]
                W1 = torch.randn(hidden, intermediate, dtype=dtype, device=device) * scale
                b1 = torch.randn(intermediate, dtype=dtype, device=device) * scale
                W2 = torch.randn(intermediate, hidden, dtype=dtype, device=device) * scale
                b2 = torch.randn(hidden, dtype=dtype, device=device) * scale

                r = run_gelu_mlp_island(
                    x, W1, b1, W2, b2,
                    n_in, n_in_inv, perm, n_out,
                    activation_type, pad_in=pad_in,
                )
                m = compare(r["expected_y_tilde"], r["y_tilde"], atol=atol, rtol=rtol)

                cells.append(
                    {
                        "island": "mlp_island",
                        "mlp_type": mlp_type,
                        "mask_family": "dense_invertible+permutation+dense_invertible",
                        "activation_type": activation_type,
                        "hidden_size": hidden,
                        "intermediate_size": intermediate,
                        "use_pad": use_pad,
                        "used_pad_at_linear_boundary": use_pad,
                        "online_extra_matmul_count": 0,
                        "preprocessing_only_transformations": [
                            "W1_tilde = N_in_inv @ W1[:, perm]",
                            "b1_tilde = b1[perm]",
                            "W2_tilde = W2[perm, :] @ N_out",
                            "b2_tilde = b2 @ N_out",
                        ],
                        "orthogonality_error": None,
                        "mean_preservation_error": None,
                        "metrics": m,
                    }
                )

            # ---- SwiGLU MLP ----
            W_up = torch.randn(hidden, intermediate, dtype=dtype, device=device) * scale
            b_up = torch.randn(intermediate, dtype=dtype, device=device) * scale
            W_gate = torch.randn(hidden, intermediate, dtype=dtype, device=device) * scale
            b_gate = torch.randn(intermediate, dtype=dtype, device=device) * scale
            W_down = torch.randn(intermediate, hidden, dtype=dtype, device=device) * scale
            b_down = torch.randn(hidden, dtype=dtype, device=device) * scale

            r = run_swiglu_mlp_island(
                x, W_up, b_up, W_gate, b_gate, W_down, b_down,
                n_in, n_in_inv, perm, n_out, pad_in=pad_in,
            )
            m = compare(r["expected_y_tilde"], r["y_tilde"], atol=atol, rtol=rtol)
            cells.append(
                {
                    "island": "mlp_island",
                    "mlp_type": "swiglu_mlp",
                    "mask_family": "dense_invertible+paired_permutation+dense_invertible",
                    "activation_type": "swiglu",
                    "hidden_size": hidden,
                    "intermediate_size": intermediate,
                    "use_pad": use_pad,
                    "used_pad_at_linear_boundary": use_pad,
                    "online_extra_matmul_count": 0,
                    "preprocessing_only_transformations": [
                        "W_up_tilde   = N_in_inv @ W_up[:, perm]",
                        "W_gate_tilde = N_in_inv @ W_gate[:, perm] (shared P)",
                        "W_down_tilde = W_down[perm, :] @ N_out",
                        "biases right-multiplied by N_out / permuted offline",
                    ],
                    "orthogonality_error": None,
                    "mean_preservation_error": None,
                    "metrics": m,
                }
            )
    return cells


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_nonlinear_island_experiments(
    config: NonlinearIslandProbeConfig,
) -> dict[str, Any]:
    """Run all four island categories and return a structured report dict."""
    torch.manual_seed(0)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = (1e-3, 1e-3) if dtype is torch.float32 else (1e-8, 1e-6)

    norm_cells = _norm_island_cells(config, dtype, device, atol, rtol)
    activation_cells = _activation_island_cells(config, dtype, device, atol, rtol)
    mlp_cells = _mlp_island_cells(config, dtype, device, atol, rtol)

    all_cells = norm_cells + activation_cells + mlp_cells

    return {
        "config": asdict(config),
        "norm_island_cells": norm_cells,
        "activation_island_cells": activation_cells,
        "mlp_island_cells": mlp_cells,
        "global_summary": {
            "num_cells": len(all_cells),
            "all_allclose": all(c["metrics"].get("allclose", False) for c in all_cells),
            "max_online_extra_matmul_count": max(
                c["online_extra_matmul_count"] for c in all_cells
            ),
        },
        "mask_family_assignments": {
            "linear_attention_kv_cache": "dense_invertible",
            "rmsnorm_core": "orthogonal",
            "layernorm_core": "mean_preserving_orthogonal",
            "activation_gelu_relu_silu": "permutation",
            "swiglu": "paired_permutation",
        },
        "pad_placement_rule": (
            "Pad is allowed at Linear boundaries only and compensated through"
            " the linear compensation term C = T W N_out. Pad is never pushed"
            " through an activation; the activation input is Z P (no pad)."
        ),
    }


__all__ = [
    "ACTIVATION_TYPES",
    "MLP_TYPES",
    "NonlinearIslandProbeConfig",
    "run_nonlinear_island_experiments",
]
