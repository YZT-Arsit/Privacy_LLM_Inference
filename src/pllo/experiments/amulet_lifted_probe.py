"""Amulet-style lifted nonlinear island probe (CPU correctness only).

Runs each lifted island once on small synthetic float64 tensors and
returns JSON-safe correctness metrics. No HuggingFace models, no GPU,
no integration into the GPT-2 wrapper. This is a correctness prototype;
the GELU / SiLU / SwiGLU selector islands may reveal selector positions
if the lifted projection has zero decoy rows.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.ops.amulet_lifted_islands import (
    run_layernorm_gadget_island,
    run_relu_lifted_mlp_island,
    run_selector_lifted_mlp_island,
    run_swiglu_selector_lifted_mlp_island,
)

_DTYPE = torch.float64


def _metrics(got: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = (got - expected).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "allclose": bool(torch.allclose(got, expected, atol=1e-8, rtol=1e-8)),
    }


def run_amulet_lifted_probe(
    *, m: int = 6, d: int = 8, h: int = 10, out: int = 8, k: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=_DTYPE)

    torch.manual_seed(seed)
    n_in, n_in_inv = generate_invertible_matrix(d, dtype=_DTYPE)
    n_out, _ = generate_invertible_matrix(out, dtype=_DTYPE)

    x = rn(m, d)
    w1, b1 = rn(d, h), rn(h)
    w2, b2 = rn(h, out), rn(out)

    results: dict[str, Any] = {}

    relu = run_relu_lifted_mlp_island(
        x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=k, seed=seed + 1,
    )
    results["relu"] = {
        **_metrics(relu["y_tilde"], relu["expected_y_tilde"]),
        "lift_factor": k,
        "activation_type": "relu",
        "lift_mode": relu["metadata"]["lift_mode"],
    }

    for act in ("gelu", "silu"):
        res = run_selector_lifted_mlp_island(
            act, x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=k, seed=seed + 2,
        )
        results[act] = {
            **_metrics(res["y_tilde"], res["expected_y_tilde"]),
            "lift_factor": k,
            "activation_type": act,
            "lift_mode": res["metadata"]["lift_mode"],
            "selector_warning": res["metadata"]["selector_leakage_warning"],
        }

    w_up, b_up = rn(d, h), rn(h)
    w_gate, b_gate = rn(d, h), rn(h)
    w_down, b_down = rn(h, out), rn(out)
    sw = run_swiglu_selector_lifted_mlp_island(
        x, w_up, b_up, w_gate, b_gate, w_down, b_down,
        n_in, n_in_inv, n_out, k=k, seed=seed + 3,
    )
    results["swiglu"] = {
        **_metrics(sw["y_tilde"], sw["expected_y_tilde"]),
        "lift_factor": k,
        "activation_type": "swiglu",
        "lift_mode": sw["metadata"]["lift_mode"],
        "selector_warning": sw["metadata"]["selector_leakage_warning"],
    }

    norm_weight, norm_bias = rn(d), rn(d)
    lin_w, lin_b = rn(d, out), rn(out)
    ln = run_layernorm_gadget_island(
        x, n_in, n_in_inv, norm_weight, norm_bias, lin_w, lin_b, n_out,
        eps=1e-5, seed=seed + 4,
    )
    results["layernorm_gadget"] = {
        **_metrics(ln["y_tilde"], ln["expected_y_tilde"]),
        "lift_factor": 1,
        "activation_type": "layernorm_gadget",
        "gadget_mode": ln["metadata"]["gadget_mode"],
    }

    all_close = all(v["allclose"] for v in results.values())
    return {
        "stage": "amulet_lifted_islands",
        "status": "ok",
        "gpu_used": False,
        "all_allclose": all_close,
        "islands": results,
        "limitations": [
            "CPU correctness prototype only; not integrated into the "
            "GPT-2 wrapper.",
            "Selector mode (GELU/SiLU/SwiGLU) may reveal valid selector "
            "positions if the lifted projection has zero decoy rows.",
            "No formal, cryptographic, or semantic security is claimed.",
        ],
    }


__all__ = ["run_amulet_lifted_probe"]
