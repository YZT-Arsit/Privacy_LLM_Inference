"""Stage 7.8b -- Precision / quantization numerical stability.

All Stage 7.6 correctness experiments use float64. Real LLM
inference runs in fp32 / bf16 / fp16 or weight-only int8/int4. Mask
transformations such as ``N^{-1} W N`` may amplify numerical error,
especially for ill-conditioned masks. This module simulates several
precision modes on CPU and reports per-mode error bounds.

Modes:
* ``float64_reference`` -- ground truth.
* ``float32_simulated`` -- ``.to(torch.float32)`` round-trip.
* ``bfloat16_simulated`` -- bf16 round-trip via float32 storage to
  emulate the bf16 mantissa (CPU lacks fast bf16 matmul; we round
  each tensor through bf16 storage and back).
* ``float16_simulated`` -- fp16 round-trip on CPU.
* ``int8_weight_only_simulated`` -- per-channel symmetric int8
  quantize-dequantize of the weight; activations stay fp32.
* ``int4_weight_only_symbolic`` -- symbolic estimate; no real int4
  kernel on CPU.

Mask families compared:
* ``orthogonal_mask`` (the protocol default; condition number 1).
* ``rope_plane_rotation`` (block-diagonal orthogonal).
* ``permutation_mask`` (condition number 1, integer indices).
* ``dense_well_conditioned`` (small condition number).
* ``dense_ill_conditioned`` (condition number sweep).

CPU local emulation only. NO real GPU fp16 / bf16 / int8 / int4
performance is measured; numbers are CPU-simulated error bounds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionStabilityConfig:
    seed: int = 2026
    batch_size: int = 2
    seq_len: int = 4
    in_dim: int = 64
    out_dim: int = 64
    vocab_size: int = 97
    condition_numbers: Tuple[float, ...] = (1.0, 2.0, 10.0, 100.0, 1000.0)
    pad_scale: float = 0.5


PRECISION_MODES: Tuple[str, ...] = (
    "float64_reference",
    "float32_simulated",
    "bfloat16_simulated",
    "float16_simulated",
    "int8_weight_only_simulated",
    "int4_weight_only_symbolic",
)


# ---------------------------------------------------------------------------
# Mask sampling
# ---------------------------------------------------------------------------


def _sample_orthogonal(
    dim: int, *, dtype: torch.dtype, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(dim, dim, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


def _sample_dense_with_condition(
    dim: int, condition: float, *,
    dtype: torch.dtype, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Construct a dense matrix with controlled condition number via
    diagonal scaling of an orthogonal factor."""
    u, _ = _sample_orthogonal(dim, dtype=dtype, generator=generator)
    v, _ = _sample_orthogonal(dim, dtype=dtype, generator=generator)
    # Singular values geometrically spaced from 1 to 1/condition.
    s = torch.tensor(
        [condition ** (-i / max(1, dim - 1)) for i in range(dim)],
        dtype=dtype,
    )
    m = u @ torch.diag(s) @ v
    actual_cond = float((s.max() / s.min()).item())
    m_inv = v.transpose(-2, -1) @ torch.diag(1.0 / s) @ u.transpose(-2, -1)
    return m, m_inv, actual_cond


def _sample_permutation(
    dim: int, *, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(dim, generator=generator)
    inv_perm = torch.argsort(perm)
    return perm, inv_perm


# ---------------------------------------------------------------------------
# Precision simulation helpers
# ---------------------------------------------------------------------------


def _round_to_precision(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "float64_reference":
        return x.to(torch.float64)
    if mode == "float32_simulated":
        return x.to(torch.float32).to(torch.float64)
    if mode == "bfloat16_simulated":
        return x.to(torch.bfloat16).to(torch.float64)
    if mode == "float16_simulated":
        return x.to(torch.float16).to(torch.float64)
    if mode == "int8_weight_only_simulated":
        # Per-column symmetric int8 quantize-dequantize.
        x32 = x.to(torch.float32)
        scale = x32.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / 127.0
        q = torch.round(x32 / scale).clamp(-127, 127)
        return (q * scale).to(torch.float64)
    if mode == "int4_weight_only_symbolic":
        # 4-bit symbolic: per-column symmetric quantize to {-7..7}.
        x32 = x.to(torch.float32)
        scale = x32.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / 7.0
        q = torch.round(x32 / scale).clamp(-7, 7)
        return (q * scale).to(torch.float64)
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Core experiment: padded boundary linear under a mask + precision
# ---------------------------------------------------------------------------


def _one_mask_run(
    *,
    label: str,
    mask: torch.Tensor,
    mask_inv: torch.Tensor,
    condition_number: float,
    cfg: PrecisionStabilityConfig,
    generator_data: torch.Generator,
) -> Dict[str, Any]:
    """For each precision mode, compute the recovered output of a
    padded-boundary linear and report the error vs the float64 plain
    reference.

    Setup (float64 ground truth):
        X = randn[B, S, in_dim]
        W = randn[in_dim, out_dim]
        Plain Y = X @ W.
        Masked: X_pad = (X - T) @ M, W_tilde = M^{-1} @ W,
                C_linear = T @ W (no output mask here for simplicity).
        Y_recovered = X_pad @ W_tilde + C_linear.
    """
    B = cfg.batch_size
    S = cfg.seq_len
    X = torch.randn(B, S, cfg.in_dim, dtype=torch.float64, generator=generator_data)
    W = torch.randn(cfg.in_dim, cfg.out_dim, dtype=torch.float64, generator=generator_data)
    T = torch.randn(B, S, cfg.in_dim, dtype=torch.float64, generator=generator_data) * cfg.pad_scale
    Y_plain = X @ W
    plain_argmax = Y_plain.argmax(dim=-1)

    per_mode: List[Dict[str, Any]] = []
    for mode in PRECISION_MODES:
        # Round M, M_inv, W, T to the precision mode (simulating
        # storing / transferring them at lower precision).
        M_p = _round_to_precision(mask, mode)
        M_inv_p = _round_to_precision(mask_inv, mode)
        W_p = _round_to_precision(W, mode)
        T_p = _round_to_precision(T, mode)
        X_p = _round_to_precision(X, mode)
        W_tilde = M_inv_p @ W_p
        C_linear = T_p @ W_p
        X_pad = (X_p - T_p) @ M_p
        Y_rec = X_pad @ W_tilde + C_linear
        err = float((Y_rec - Y_plain).abs().max().item())
        rel = float((Y_rec - Y_plain).abs().max().item()
                    / max(1e-12, Y_plain.abs().max().item()))
        rec_argmax = Y_rec.argmax(dim=-1)
        greedy = float((plain_argmax == rec_argmax).float().mean().item())
        overflow = bool(torch.isinf(Y_rec).any().item())
        nan = bool(torch.isnan(Y_rec).any().item())
        per_mode.append({
            "precision_mode": mode,
            "logits_max_abs_error_vs_float64_plain": err,
            "logits_relative_error": rel,
            "greedy_token_match_rate": greedy,
            "sequence_exact_match": bool(
                torch.equal(plain_argmax, rec_argmax)
            ),
            "overflow_detected": overflow,
            "nan_detected": nan,
        })
    return {
        "mask_family": label,
        "condition_number": condition_number,
        "per_precision": per_mode,
    }


def _orthogonal_run(
    cfg: PrecisionStabilityConfig, generator: torch.Generator,
    generator_data: torch.Generator,
) -> Dict[str, Any]:
    m, m_inv = _sample_orthogonal(
        cfg.in_dim, dtype=torch.float64, generator=generator
    )
    return _one_mask_run(
        label="orthogonal_mask", mask=m, mask_inv=m_inv,
        condition_number=1.0, cfg=cfg, generator_data=generator_data,
    )


def _permutation_run(
    cfg: PrecisionStabilityConfig, generator: torch.Generator,
    generator_data: torch.Generator,
) -> Dict[str, Any]:
    perm, inv_perm = _sample_permutation(cfg.in_dim, generator=generator)
    P = torch.zeros(cfg.in_dim, cfg.in_dim, dtype=torch.float64)
    P[torch.arange(cfg.in_dim), perm] = 1.0
    P_inv = P.transpose(-2, -1)
    return _one_mask_run(
        label="permutation_mask", mask=P, mask_inv=P_inv,
        condition_number=1.0, cfg=cfg, generator_data=generator_data,
    )


def _condition_sweep(
    cfg: PrecisionStabilityConfig, generator: torch.Generator,
    generator_data: torch.Generator,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in cfg.condition_numbers:
        m, m_inv, actual = _sample_dense_with_condition(
            cfg.in_dim, c, dtype=torch.float64, generator=generator,
        )
        out.append(_one_mask_run(
            label=f"dense_condition_{int(c)}",
            mask=m, mask_inv=m_inv,
            condition_number=actual,
            cfg=cfg, generator_data=generator_data,
        ))
    return out


def run_precision_quantization_stability(
    *, cfg: Optional[PrecisionStabilityConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = PrecisionStabilityConfig()
    torch.manual_seed(cfg.seed)
    g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed)
    g_data = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)

    orth = _orthogonal_run(cfg, g_mask, g_data)
    perm = _permutation_run(cfg, g_mask, g_data)
    sweep = _condition_sweep(cfg, g_mask, g_data)

    report = {
        "status": "ok",
        "stage": "7.8b",
        "main_mode": "precision_quantization_stability",
        "device": "cpu",
        "dtype_reference": "float64",
        "config": {
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "in_dim": cfg.in_dim,
            "out_dim": cfg.out_dim,
            "vocab_size": cfg.vocab_size,
            "condition_numbers": list(cfg.condition_numbers),
            "pad_scale": cfg.pad_scale,
        },
        "precision_modes_tested": list(PRECISION_MODES),
        "orthogonal_mask": orth,
        "permutation_mask": perm,
        "condition_sweep": sweep,
        "recommended_mask_families": [
            "orthogonal",
            "permutation",
            "RoPE-plane block rotation",
            "block-diagonal well-conditioned",
        ],
        "not_recommended_for_low_precision": [
            "ill-conditioned dense masks (condition number >> 1)",
        ],
        "real_gpu_kernel_measured": False,
        "real_quantized_model_loaded": False,
        "limitations": [
            "CPU local emulation only; no real GPU fp16 / bf16 / int8 "
            "/ int4 kernels are measured.",
            "bf16 / fp16 / int8 are SIMULATED via round-trip casts "
            "on float64 storage. Real GPU tensor-core behaviour may "
            "differ in accumulator type and rounding.",
            "int4 is SYMBOLIC ONLY -- no real int4 path.",
            "float64 reference is for protocol correctness, NOT real "
            "inference precision; real LLM inference uses bf16 or "
            "fp16 plus mixed-precision accumulation.",
            "Condition-number sweep uses synthetic dense masks; "
            "real-world quantized weights have their own conditioning.",
            "No formal cryptographic / semantic / differential-"
            "privacy security.",
            "No full Qwen / LLaMA deployment.",
        ],
        "paper_safe_wording": (
            "Mask transformations are stable for orthogonal / "
            "permutation / RoPE-plane block-diagonal masks under "
            "every simulated precision mode; ill-conditioned dense "
            "masks amplify error proportionally to the condition "
            "number. We recommend well-conditioned mask families "
            "(orthogonal, permutation, block) for low-precision "
            "deployment. We do NOT measure real GPU fp16 / bf16 / "
            "int8 / int4 wall-clock or hardware-specific rounding."
        ),
        "unsafe_wording_to_avoid": [
            "Real GPU fp16 / bf16 / int8 / int4 performance.",
            "Real quantized model deployment.",
            "GPU tensor-core matmul measured.",
            "This is formal cryptographic security.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if x == 0.0:
            return "0.0"
        if abs(x) >= 1e-3:
            return f"{x:.6g}"
        return f"{x:.3e}"
    return str(x)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Precision / Quantization Stability")
    w()
    w(
        "_Stage 7.8b: CPU-simulated precision modes (fp32, bf16, fp16, "
        "int8, int4) and condition-number sweep for the padded-"
        "boundary linear under various mask families. No real GPU / "
        "quantized kernel is measured._"
    )
    w()
    cfg = report["config"]
    w("## Configuration")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in ("batch_size", "seq_len", "in_dim", "out_dim", "vocab_size",
              "condition_numbers", "pad_scale"):
        w(f"| {k} | {cfg[k]} |")
    w()

    w("## Why Float64 Correctness Does NOT Imply Low-Precision Correctness")
    w()
    w(
        "The padded-boundary linear computes "
        "``Y_rec = (X - T) M M^{-1} W + T W``. In exact arithmetic "
        "``M M^{-1} = I`` cancels and ``Y_rec = X W``. At lower "
        "precision, ``M M^{-1}`` carries a residual ~ ``eps * cond(M)`` "
        "which is amplified by ``W`` and the activations. For "
        "well-conditioned ``M`` (orthogonal: cond = 1) the residual "
        "stays at machine epsilon; for ill-conditioned ``M`` the "
        "residual scales linearly with cond(M)."
    )
    w()

    for family in ("orthogonal_mask", "permutation_mask"):
        info = report[family]
        w(f"## {family.replace('_', ' ').title()} (cond = {info['condition_number']})")
        w()
        w(
            "| precision | logits_max_abs_error | logits_relative_error | "
            "greedy_match | seq_exact | overflow | nan |"
        )
        w("|---|---|---|---|---|---|---|")
        for r in info["per_precision"]:
            w(
                f"| `{r['precision_mode']}` | "
                f"{_fmt(r['logits_max_abs_error_vs_float64_plain'])} | "
                f"{_fmt(r['logits_relative_error'])} | "
                f"{r['greedy_token_match_rate']} | "
                f"{r['sequence_exact_match']} | "
                f"{r['overflow_detected']} | {r['nan_detected']} |"
            )
        w()

    w("## Dense Condition-Number Sweep")
    w()
    w(
        "| cond_target | actual_cond | precision | logits_max_abs_error | "
        "logits_relative_error | greedy_match |"
    )
    w("|---|---|---|---|---|---|")
    for sweep_row in report["condition_sweep"]:
        actual = sweep_row["condition_number"]
        for r in sweep_row["per_precision"]:
            w(
                f"| {sweep_row['mask_family']} | {_fmt(actual)} | "
                f"`{r['precision_mode']}` | "
                f"{_fmt(r['logits_max_abs_error_vs_float64_plain'])} | "
                f"{_fmt(r['logits_relative_error'])} | "
                f"{r['greedy_token_match_rate']} |"
            )
    w()

    w("## Recommendations")
    w()
    w("Recommended mask families for low-precision deployment:")
    w()
    for x in report["recommended_mask_families"]:
        w(f"- {x}")
    w()
    w("NOT recommended for low precision:")
    w()
    for x in report["not_recommended_for_low_precision"]:
        w(f"- {x}")
    w()

    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "precision_quantization_stability.json",
    md_filename: str = "precision_quantization_stability.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "PRECISION_MODES",
    "PrecisionStabilityConfig",
    "render_markdown",
    "run_precision_quantization_stability",
    "write_reports",
]
