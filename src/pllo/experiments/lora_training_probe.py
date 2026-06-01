"""Stage 7.0 — LoRA private training probe.

Runs a tiny LoRA training loop in two paths:

* ``plain``  — vanilla LoRA: ``Y = X W + (alpha / r) X A B``, then standard
  MSE loss + analytic backward + SGD / AdamW update over ``(A, B)``.
* ``masked`` — masked LoRA forward: trusted side hands the GPU only
  ``X_tilde / W_tilde / A_tilde / B_tilde / bias_tilde / pad_compensation``;
  GPU computes ``Y_tilde``; trusted side recovers ``Y`` and computes loss +
  gradients + update over the *plaintext* ``(A, B)`` in trusted memory.

This is a deliberately limited prototype:

  - The base weight ``W`` is **public**.
  - The training data ``(X_private, Y_target)`` is **private** to the
    trusted side; the GPU only ever sees masked activations.
  - The LoRA adapter ``(A, B)`` is **trusted-only**; GPU sees
    ``(A_tilde, B_tilde)`` whose factors depend on fresh per-step masks.
  - The optimizer state (SGD momentum / AdamW moments) is **trusted-only**.
  - Backward / gradient computation remains trusted — the report
    explicitly documents this as ``"training backward remains trusted in
    Stage 7.0 prototype"``. A masked-gradient GPU path is Stage 7.1+.

Correctness criterion: the masked path must reproduce the plain path
token-for-token up to numerical precision. The probe reports per-step
diffs and the final adapter update error so divergence is visible.

Constraints honoured:
  - No PEFT / DeepSpeed / vLLM / FlashAttention.
  - No real Qwen / TinyLlama / LLaMA fine-tuning.
  - No distributed training.
  - No real TEE; ``security_profile`` stays ``"proxy-evaluated, not formal"``.
  - Adapter is NEVER merged into ``W`` (constraint 7).
  - Outputs publish summary metrics + fingerprints; private data, raw
    adapter tensors, optimizer state, and dense masks are never exported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    create_masked_lora_state,
    init_lora_adapters,
    lora_state_fingerprint,
    make_lora_pad_compensation,
    masked_lora_linear_forward,
    obfuscate_lora_input,
    plain_lora_linear_forward,
    recover_masked_output,
    transform_linear_weight_lora,
    transform_lora_adapter,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


VALID_OPTIMIZERS: tuple[str, ...] = ("sgd", "adamw")


def normalize_optimizer(name: str | None) -> str:
    if name is None:
        return "sgd"
    if name not in VALID_OPTIMIZERS:
        raise ValueError(
            f"invalid optimizer {name!r}; expected one of {VALID_OPTIMIZERS}"
        )
    return name


@dataclass
class LoRATrainingProbeConfig:
    """All knobs for the Stage 7.0 LoRA training probe."""

    output_dir: str = "outputs"
    seed: int = 2026

    # Synthetic data shape
    batch_size: int = 4
    d_in: int = 32
    d_out: int = 16
    rank: int = 4
    alpha: float = 1.0
    use_bias: bool = True

    # Training schedule
    num_steps: int = 5
    lr: float = 1e-2
    optimizer: str = "sgd"
    weight_decay: float = 0.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8

    # Masking
    use_pad: bool = True
    fresh_u_per_step: bool = True
    fresh_masks_per_step: bool = True
    pad_scale: float = 1.0

    # Numerical
    dtype: str = "float64"  # default float64 so MSE-grad path is allclose
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Optimizer state (trusted-only)
# ---------------------------------------------------------------------------


@dataclass
class _OptimizerState:
    """Per-parameter optimizer scratch space; never exported to reports."""

    momentum: torch.Tensor | None = None  # used for SGD with momentum (unused in default)
    m: torch.Tensor | None = None  # AdamW first moment
    v: torch.Tensor | None = None  # AdamW second moment
    step: int = 0


# ---------------------------------------------------------------------------
# Plain LoRA training (reference) — analytic gradients, no autograd
# ---------------------------------------------------------------------------


def _compute_lora_gradients(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    y_target: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(grad_A, grad_B, mean_mse_loss)``.

    MSE per-sample-per-channel: ``loss = mean((Y - Y_target) ** 2)``.

    ``dL/dY     = 2/N * (Y - Y_target)`` (N = numel).
    ``dL/dB     = (alpha/r) * (X A)^T @ dL/dY``.
    ``dL/dA     = (alpha/r) * X^T @ (dL/dY @ B^T)``.
    """
    rank = a.shape[1]
    scale = float(alpha) / max(rank, 1)
    n = float(y.numel())
    diff = y - y_target
    loss = float((diff * diff).mean().item())
    dy = (2.0 / n) * diff
    grad_b = scale * (x @ a).transpose(0, 1) @ dy
    grad_a = scale * x.transpose(0, 1) @ (dy @ b.transpose(0, 1))
    return grad_a, grad_b, loss


def _apply_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: _OptimizerState,
    *,
    optimizer: str,
    lr: float,
    weight_decay: float,
    beta1: float, beta2: float, eps: float,
) -> torch.Tensor:
    """Return the post-update parameter."""
    state.step += 1
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    if optimizer == "sgd":
        return param - lr * grad
    # AdamW
    if state.m is None:
        state.m = torch.zeros_like(param)
        state.v = torch.zeros_like(param)
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * grad * grad
    m_hat = state.m / (1.0 - beta1 ** state.step)
    v_hat = state.v / (1.0 - beta2 ** state.step)
    return param - lr * m_hat / (v_hat.sqrt() + eps)


# ---------------------------------------------------------------------------
# Masked LoRA forward — used inside training probe
# ---------------------------------------------------------------------------


def _masked_lora_forward_recover(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    lora_cfg: LoRAConfig,
    fcfg: MaskedLoRAForwardConfig,
    *,
    state: LoRAState | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, LoRAState]:
    new_state = create_masked_lora_state(
        lora_cfg, fcfg, seq_len=x.shape[0], state=state, generator=generator,
    )
    x_tilde = obfuscate_lora_input(x, new_state.n_in, new_state.pad)
    w_tilde, bias_tilde = transform_linear_weight_lora(
        w, bias, new_state.n_in_inv, new_state.n_out,
    )
    a_tilde, b_tilde = transform_lora_adapter(
        a, b, new_state.n_in_inv, new_state.n_out, new_state.u, new_state.u_inv,
        alpha=lora_cfg.alpha,
    )
    compensation = None
    if new_state.pad is not None:
        compensation = make_lora_pad_compensation(
            w, a, b, new_state.pad, new_state.n_out, alpha=lora_cfg.alpha,
        )
    y_tilde = masked_lora_linear_forward(
        x_tilde, w_tilde, a_tilde, b_tilde,
        bias_tilde=bias_tilde,
        alpha=lora_cfg.alpha,
        pad_compensation=compensation,
    )
    y_recovered = recover_masked_output(y_tilde, new_state.n_out_inv)
    return y_recovered, new_state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "Stage 7.0 is a prototype LoRA private training path, not full Qwen/TinyLlama LoRA fine-tuning.",
    "Backward / optimizer update remains trusted in Stage 7.0 — only the forward is masked and offloaded.",
    "Optimizer state (SGD momentum / AdamW moments) is trusted-only and never exported to JSON/CSV/Markdown.",
    "PEFT / DeepSpeed / vLLM / FlashAttention are NOT integrated.",
    "Real TEE isolation is NOT evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "LoRA adapter is NEVER merged into the public base weight W.",
    "Distributed training is NOT implemented.",
    "Reported metrics are summary statistics + fingerprints. Private data, raw adapter tensors, masks, and pads are never emitted in outputs.",
    "No formal / cryptographic / semantic security is claimed.",
]


def run_lora_training_probe(
    config: LoRATrainingProbeConfig,
) -> dict[str, Any]:
    """Run the Stage 7.0 LoRA training-step correctness + leakage probe."""
    optimizer = normalize_optimizer(config.optimizer)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)

    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    torch.manual_seed(config.seed)

    # ----- 1. Build synthetic private training data + public base weight -----
    lora_cfg = LoRAConfig(
        d_in=config.d_in, d_out=config.d_out, rank=config.rank,
        alpha=config.alpha, use_bias=config.use_bias,
        dtype=config.dtype, device=config.device,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=config.use_pad,
        fresh_u_per_call=config.fresh_u_per_step,
        fresh_masks_per_call=config.fresh_masks_per_step,
        pad_scale=config.pad_scale,
        dtype=config.dtype, device=config.device,
    )

    # Synthetic private data D = (X, Y_target). Y_target is generated so the
    # task has signal in the LoRA subspace: Y_target = X W + small ΔY.
    x_private = torch.randn(
        config.batch_size, config.d_in,
        generator=gen, dtype=dtype, device=device,
    )
    y_target = torch.randn(
        config.batch_size, config.d_out,
        generator=gen, dtype=dtype, device=device,
    )
    w_public = torch.randn(
        config.d_in, config.d_out, generator=gen, dtype=dtype, device=device,
    )
    bias_public = (
        torch.randn(config.d_out, generator=gen, dtype=dtype, device=device)
        if config.use_bias else None
    )

    # Initial adapters; the masked + plain paths share the SAME init so
    # divergence is purely from masking, not initialisation.
    a0, b0 = init_lora_adapters(lora_cfg, generator=gen)
    # B starts at zero by convention — perturb slightly so step 0 gradient
    # for A is not exactly zero (which would mask numerical drift).
    b0 = b0 + 1e-3 * torch.randn(
        config.rank, config.d_out, generator=gen, dtype=dtype, device=device,
    )

    a_plain = a0.clone()
    b_plain = b0.clone()
    a_masked = a0.clone()
    b_masked = b0.clone()

    state_a_plain = _OptimizerState()
    state_b_plain = _OptimizerState()
    state_a_masked = _OptimizerState()
    state_b_masked = _OptimizerState()

    per_step: list[dict[str, Any]] = []
    lora_state: LoRAState | None = None

    for step in range(config.num_steps):
        # ----- Plain LoRA -----
        y_plain = plain_lora_linear_forward(
            x_private, w_public, a_plain, b_plain, bias_public,
            alpha=lora_cfg.alpha,
        )
        grad_a_plain, grad_b_plain, loss_plain = _compute_lora_gradients(
            x_private, a_plain, b_plain, y_plain, y_target,
            alpha=lora_cfg.alpha,
        )

        # ----- Masked LoRA (forward only) -----
        y_masked, lora_state = _masked_lora_forward_recover(
            x_private, w_public, a_masked, b_masked, bias_public,
            lora_cfg, fcfg, state=lora_state, generator=gen,
        )
        grad_a_masked, grad_b_masked, loss_masked = _compute_lora_gradients(
            x_private, a_masked, b_masked, y_masked, y_target,
            alpha=lora_cfg.alpha,
        )

        # Per-step diffs BEFORE the update.
        forward_err = float((y_plain - y_masked).abs().max().item())
        loss_diff = abs(loss_plain - loss_masked)
        grad_a_err = float((grad_a_plain - grad_a_masked).abs().max().item())
        grad_b_err = float((grad_b_plain - grad_b_masked).abs().max().item())

        # Apply update on both sides.
        a_plain_new = _apply_step(
            a_plain, grad_a_plain, state_a_plain,
            optimizer=optimizer, lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2,
            eps=config.adamw_eps,
        )
        b_plain_new = _apply_step(
            b_plain, grad_b_plain, state_b_plain,
            optimizer=optimizer, lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2,
            eps=config.adamw_eps,
        )
        a_masked_new = _apply_step(
            a_masked, grad_a_masked, state_a_masked,
            optimizer=optimizer, lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2,
            eps=config.adamw_eps,
        )
        b_masked_new = _apply_step(
            b_masked, grad_b_masked, state_b_masked,
            optimizer=optimizer, lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2,
            eps=config.adamw_eps,
        )

        update_a_err = float((a_plain_new - a_masked_new).abs().max().item())
        update_b_err = float((b_plain_new - b_masked_new).abs().max().item())

        per_step.append({
            "step": step,
            "loss_plain": loss_plain,
            "loss_masked": loss_masked,
            "loss_diff_abs": loss_diff,
            "forward_max_abs_err": forward_err,
            "grad_a_max_abs_err": grad_a_err,
            "grad_b_max_abs_err": grad_b_err,
            "adapter_a_update_max_abs_err": update_a_err,
            "adapter_b_update_max_abs_err": update_b_err,
            "lora_state_fingerprint": (
                lora_state_fingerprint(lora_state)
                if lora_state is not None else None
            ),
        })

        a_plain, b_plain = a_plain_new, b_plain_new
        a_masked, b_masked = a_masked_new, b_masked_new

    # ----- Final summary -----
    final_y_plain = plain_lora_linear_forward(
        x_private, w_public, a_plain, b_plain, bias_public, alpha=lora_cfg.alpha,
    )
    final_y_masked, _ = _masked_lora_forward_recover(
        x_private, w_public, a_masked, b_masked, bias_public,
        lora_cfg, fcfg, state=lora_state, generator=gen,
    )
    final_output_err = float((final_y_plain - final_y_masked).abs().max().item())
    final_a_err = float((a_plain - a_masked).abs().max().item())
    final_b_err = float((b_plain - b_masked).abs().max().item())

    max_loss_diff = max((row["loss_diff_abs"] for row in per_step), default=0.0)
    max_grad_a_err = max(
        (row["grad_a_max_abs_err"] for row in per_step), default=0.0,
    )
    max_grad_b_err = max(
        (row["grad_b_max_abs_err"] for row in per_step), default=0.0,
    )

    # Numerical tolerance: float32 forward-then-backward roundtrip is much
    # noisier than float64. We pick generous tolerances so float32 still
    # passes; the per-step report shows the actual diff.
    if config.dtype == "float64":
        tol_forward = 1e-9
        tol_grad = 1e-9
        tol_update = 1e-9
    else:
        tol_forward = 5e-4
        tol_grad = 5e-3
        tol_update = 5e-3

    allclose = (
        max((r["forward_max_abs_err"] for r in per_step), default=0.0) <= tol_forward
        and max_grad_a_err <= tol_grad
        and max_grad_b_err <= tol_grad
        and final_a_err <= tol_update
        and final_b_err <= tol_update
    )

    return {
        "config": {**asdict(config), "optimizer": optimizer},
        "lora_config_fingerprint": {
            "d_in": lora_cfg.d_in,
            "d_out": lora_cfg.d_out,
            "rank": lora_cfg.rank,
            "alpha": lora_cfg.alpha,
            "use_bias": lora_cfg.use_bias,
        },
        "training_step_correctness": {
            "num_steps": config.num_steps,
            "per_step": per_step,
            "max_loss_diff": max_loss_diff,
            "max_grad_a_err": max_grad_a_err,
            "max_grad_b_err": max_grad_b_err,
            "final_adapter_a_update_err": final_a_err,
            "final_adapter_b_update_err": final_b_err,
            "final_output_err": final_output_err,
            "tolerance_forward": tol_forward,
            "tolerance_grad": tol_grad,
            "tolerance_update": tol_update,
            "allclose": bool(allclose),
        },
        "gradient_and_optimizer_handling": {
            "backward_location": "trusted",
            "optimizer_state_location": "trusted",
            "adapter_location": "trusted",
            "private_data_location": "trusted",
            "gpu_visibility": {
                "x_tilde": True,
                "w_tilde": True,
                "a_tilde": True,
                "b_tilde": True,
                "bias_tilde": True,
                "pad_compensation": True,
                "raw_x": False,
                "raw_a": False,
                "raw_b": False,
                "grad_a": False,
                "grad_b": False,
                "optimizer_state": False,
                "private_target_y": False,
            },
            "trusted_backward_status": (
                "training backward remains trusted in Stage 7.0 prototype"
            ),
            "masked_backward_status": (
                "not_implemented; deferred to Stage 7.1 (masked backward / gradient-side"
                " obfuscation)"
            ),
            "merge_adapter_into_w": False,
        },
        "pad_compensation": {
            "use_pad": config.use_pad,
            "pad_scale": config.pad_scale,
            "compensation_formula": (
                "C = T_in W N_out + (alpha / r) T_in A B N_out"
            ),
            "is_trusted_only": True,
            "forward_err_under_pad": float(
                max((r["forward_max_abs_err"] for r in per_step), default=0.0)
                if config.use_pad else 0.0
            ),
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora": (
            "private-adapter-trusted-backward, not formal"
        ),
        "lora_private_training_status": "prototype",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.1 — masked backward path: send masked gradients to GPU"
            " (e.g. fold N into the gradient pipeline) so the trusted side"
            " only generates and applies the update.",
            "Stage 7.2 — multi-layer LoRA in a tiny transformer block end-to-end.",
            "Stage 7.3 — calibrated LoRA workload + LoRA timing-side proxy.",
        ],
    }


# ---------------------------------------------------------------------------
# Helper for the runner script (CSV row list)
# ---------------------------------------------------------------------------


def training_probe_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        rows.append({
            "section": "config", "step": "n/a", "metric": k, "value": v,
        })
    tc = report["training_step_correctness"]
    for r in tc["per_step"]:
        for k, v in r.items():
            if k == "lora_state_fingerprint":
                continue
            rows.append({
                "section": "per_step", "step": str(r["step"]),
                "metric": k, "value": v,
            })
    for k in (
        "max_loss_diff", "max_grad_a_err", "max_grad_b_err",
        "final_adapter_a_update_err", "final_adapter_b_update_err",
        "final_output_err", "allclose",
    ):
        rows.append({
            "section": "summary", "step": "final", "metric": k, "value": tc[k],
        })
    gr = report["gradient_and_optimizer_handling"]
    for k, v in gr.items():
        if k == "gpu_visibility":
            continue
        rows.append({
            "section": "gradient_handling", "step": "n/a", "metric": k,
            "value": v,
        })
    for k, v in gr["gpu_visibility"].items():
        rows.append({
            "section": "gpu_visibility", "step": "n/a", "metric": k, "value": v,
        })
    pc = report["pad_compensation"]
    for k, v in pc.items():
        rows.append({
            "section": "pad_compensation", "step": "n/a", "metric": k, "value": v,
        })
    return rows


__all__ = [
    "LoRATrainingProbeConfig",
    "VALID_OPTIMIZERS",
    "normalize_optimizer",
    "run_lora_training_probe",
    "training_probe_csv_rows",
]
