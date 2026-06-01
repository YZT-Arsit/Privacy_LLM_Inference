"""Stage 7.1 — LoRA masked-backward correctness probe.

Runs a tiny LoRA training loop where:

  * the trusted side holds private ``(X, Y_target)``, plaintext ``A`` / ``B``,
    and the optimizer state (SGD momentum / AdamW m, v);
  * the forward path is masked (Stage 7.0 contract — GPU sees only
    ``X_tilde / W_tilde / A_tilde / B_tilde / Y_tilde``);
  * the **backward** path is also masked: trusted side computes
    ``G = ∂L/∂Y`` and maps to ``G_tilde``; GPU runs
    :func:`pllo.ops.lora_backward.masked_lora_backward` on
    ``(X_tilde, A_tilde, B_tilde, G_tilde)`` and returns
    ``grad_A_tilde / grad_B_tilde`` (and optionally ``grad_X_tilde``);
    trusted side recovers and applies the SGD / AdamW update on plaintext
    ``A`` / ``B``.

The probe compares this masked-backward path against:

  * a fully plain LoRA training loop (analytic gradients), and
  * ``torch.autograd``-computed gradients for one step, to confirm the
    analytic plain reference itself is correct.

Reports per-step loss diffs, gradient errors, adapter-update errors, and a
final output error. Backward / optimizer remain trusted (the "trusted
optimizer" / "trusted loss" labels stay); Stage 7.1 only moves the matmul
arithmetic of backward through the GPU domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from pllo.ops.lora_backward import (
    invert_upstream_gradient_mask,
    make_lora_grad_pad_compensation,
    masked_lora_backward,
    plain_lora_backward_reference,
    recover_lora_gradients,
    transform_upstream_gradient,
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
class LoRABackwardProbeConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    batch_size: int = 4
    d_in: int = 32
    d_out: int = 16
    rank: int = 4
    alpha: float = 1.0
    use_bias: bool = True
    num_steps: int = 5
    lr: float = 1e-2
    optimizer: str = "sgd"
    weight_decay: float = 0.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    use_pad: bool = True
    fresh_u_per_step: bool = True
    fresh_masks_per_step: bool = True
    pad_scale: float = 1.0
    recover_grad_x: bool = False
    dtype: str = "float64"
    device: str = "cpu"


@dataclass
class _OptimizerState:
    m: torch.Tensor | None = None
    v: torch.Tensor | None = None
    step: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    state.step += 1
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    if optimizer == "sgd":
        return param - lr * grad
    if state.m is None:
        state.m = torch.zeros_like(param)
        state.v = torch.zeros_like(param)
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * grad * grad
    m_hat = state.m / (1.0 - beta1 ** state.step)
    v_hat = state.v / (1.0 - beta2 ** state.step)
    return param - lr * m_hat / (v_hat.sqrt() + eps)


def _masked_forward_recover(
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
        bias_tilde=bias_tilde, alpha=lora_cfg.alpha,
        pad_compensation=compensation,
    )
    y_recovered = recover_masked_output(y_tilde, new_state.n_out_inv)
    return y_recovered, new_state


def _masked_backward_recover(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    upstream_grad: torch.Tensor,
    state: LoRAState,
    *,
    alpha: float,
    recover_grad_x: bool,
) -> dict[str, torch.Tensor | None]:
    """Run one masked LoRA backward call given a forward state."""
    # Mirror Stage 7.0 forward operands so the masked tensors fed to the
    # GPU backward match those produced by the matching forward step.
    if state.pad is None:
        x_tilde = x @ state.n_in
    else:
        x_tilde = (x - state.pad) @ state.n_in
    w_tilde = state.n_in_inv @ w @ state.n_out
    a_tilde = state.n_in_inv @ a @ state.u
    b_tilde = state.u_inv @ b @ state.n_out
    grad_y_tilde = transform_upstream_gradient(upstream_grad, state.n_out)
    masked = masked_lora_backward(
        x_tilde, a_tilde, b_tilde, grad_y_tilde,
        alpha=alpha,
        w_tilde=w_tilde if recover_grad_x else None,
        recover_grad_x=recover_grad_x,
    )
    pad_compensation = None
    if state.pad is not None:
        pad_compensation = make_lora_grad_pad_compensation(
            a, b, state.pad, upstream_grad, alpha=alpha,
        )
    return recover_lora_gradients(
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


def _verify_chain_rule_invariance(
    upstream_grad: torch.Tensor,
    y_plain: torch.Tensor,
    n_out: torch.Tensor,
) -> float:
    """Return |tr(G^T Y) - tr(G_tilde^T Y_tilde)|."""
    g_tilde = transform_upstream_gradient(upstream_grad, n_out)
    y_tilde = y_plain @ n_out
    return float(abs((upstream_grad * y_plain).sum().item()
                     - (g_tilde * y_tilde).sum().item()))


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "Stage 7.1 implements masked-gradient prototype, not full private fine-tuning.",
    "Loss computation remains trusted (G = dL/dY is computed on the trusted side).",
    "Optimizer update remains trusted (SGD momentum / AdamW m, v never cross the boundary).",
    "PEFT / DeepSpeed / vLLM / FlashAttention are NOT integrated.",
    "This is not real TEE training; security_profile stays 'proxy-evaluated, not formal'.",
    "Rank padding is NOT implemented; LoRA rank r is still visible from A_tilde / B_tilde / grad_A_tilde / grad_B_tilde shapes (deferred to Stage 7.2).",
    "LoRA adapter is NEVER merged into the public base weight W.",
    "Distributed training is NOT implemented.",
    "Reports publish summary metrics + fingerprints only. Private data, raw adapter tensors, raw gradients, optimizer state, and masks are never emitted in outputs.",
    "No formal / cryptographic / semantic security is claimed.",
]


def run_lora_backward_probe(
    config: LoRABackwardProbeConfig,
) -> dict[str, Any]:
    """Run the Stage 7.1 masked-backward correctness probe."""
    optimizer = normalize_optimizer(config.optimizer)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    torch.manual_seed(config.seed)

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

    # ---- private data + public W ----
    x_private = torch.randn(
        config.batch_size, config.d_in, generator=gen, dtype=dtype, device=device,
    )
    y_target = torch.randn(
        config.batch_size, config.d_out, generator=gen, dtype=dtype, device=device,
    )
    w_public = torch.randn(
        config.d_in, config.d_out, generator=gen, dtype=dtype, device=device,
    )
    bias_public = (
        torch.randn(config.d_out, generator=gen, dtype=dtype, device=device)
        if config.use_bias else None
    )

    a0, b0 = init_lora_adapters(lora_cfg, generator=gen)
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

    # ---- One-time autograd cross-check at step 0 ----
    x_ag = x_private.detach().clone().requires_grad_(True)
    a_ag = a_plain.detach().clone().requires_grad_(True)
    b_ag = b_plain.detach().clone().requires_grad_(True)
    s = lora_cfg.alpha / max(lora_cfg.rank, 1)
    y_ag = x_ag @ w_public + s * (x_ag @ a_ag) @ b_ag
    if bias_public is not None:
        y_ag = y_ag + bias_public
    n = float(y_ag.numel())
    loss_ag = ((y_ag - y_target) ** 2).mean()
    grad_a_ag, grad_b_ag, grad_x_ag = torch.autograd.grad(
        loss_ag, [a_ag, b_ag, x_ag], retain_graph=False,
    )
    g_ag = (2.0 / n) * (y_ag.detach() - y_target)
    plain_ref = plain_lora_backward_reference(
        x_private, w_public, a_plain, b_plain, g_ag, alpha=lora_cfg.alpha,
    )
    autograd_vs_analytic = {
        "grad_a": float((grad_a_ag - plain_ref["grad_a"]).abs().max().item()),
        "grad_b": float((grad_b_ag - plain_ref["grad_b"]).abs().max().item()),
        "grad_x": float((grad_x_ag - plain_ref["grad_x"]).abs().max().item()),
    }

    per_step: list[dict[str, Any]] = []
    lora_state: LoRAState | None = None
    for step in range(config.num_steps):
        # --- Plain LoRA training step ---
        y_plain = plain_lora_linear_forward(
            x_private, w_public, a_plain, b_plain, bias_public, alpha=lora_cfg.alpha,
        )
        diff_plain = y_plain - y_target
        loss_plain = float((diff_plain * diff_plain).mean().item())
        g_plain = (2.0 / float(y_plain.numel())) * diff_plain
        ref = plain_lora_backward_reference(
            x_private, w_public, a_plain, b_plain, g_plain, alpha=lora_cfg.alpha,
        )
        grad_a_plain = ref["grad_a"]
        grad_b_plain = ref["grad_b"]
        grad_x_plain = ref["grad_x"]

        # --- Masked LoRA training step (forward + backward) ---
        y_masked, lora_state = _masked_forward_recover(
            x_private, w_public, a_masked, b_masked, bias_public,
            lora_cfg, fcfg, state=lora_state, generator=gen,
        )
        diff_masked = y_masked - y_target
        loss_masked = float((diff_masked * diff_masked).mean().item())
        g_masked = (2.0 / float(y_masked.numel())) * diff_masked
        upstream_invariance_err = _verify_chain_rule_invariance(
            g_masked, y_masked, lora_state.n_out,
        )
        recovered = _masked_backward_recover(
            x_private, w_public, a_masked, b_masked, g_masked, lora_state,
            alpha=lora_cfg.alpha, recover_grad_x=config.recover_grad_x,
        )
        grad_a_masked = recovered["grad_a"]
        grad_b_masked = recovered["grad_b"]
        grad_x_masked = recovered["grad_x"]

        # --- Diffs before update ---
        forward_err = float((y_plain - y_masked).abs().max().item())
        loss_diff = abs(loss_plain - loss_masked)
        grad_a_err = float((grad_a_plain - grad_a_masked).abs().max().item())
        grad_b_err = float((grad_b_plain - grad_b_masked).abs().max().item())
        grad_x_err = (
            float((grad_x_plain - grad_x_masked).abs().max().item())
            if grad_x_masked is not None else None
        )

        # --- Trusted optimizer update on both sides ---
        a_plain_new = _apply_step(
            a_plain, grad_a_plain, state_a_plain,
            optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2, eps=config.adamw_eps,
        )
        b_plain_new = _apply_step(
            b_plain, grad_b_plain, state_b_plain,
            optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2, eps=config.adamw_eps,
        )
        a_masked_new = _apply_step(
            a_masked, grad_a_masked, state_a_masked,
            optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2, eps=config.adamw_eps,
        )
        b_masked_new = _apply_step(
            b_masked, grad_b_masked, state_b_masked,
            optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2, eps=config.adamw_eps,
        )
        update_a_err = float((a_plain_new - a_masked_new).abs().max().item())
        update_b_err = float((b_plain_new - b_masked_new).abs().max().item())

        per_step.append({
            "step": step,
            "loss_plain": loss_plain,
            "loss_masked": loss_masked,
            "loss_diff_abs": loss_diff,
            "forward_max_abs_err": forward_err,
            "upstream_gradient_invariance_err": upstream_invariance_err,
            "grad_a_max_abs_err": grad_a_err,
            "grad_b_max_abs_err": grad_b_err,
            "grad_x_max_abs_err": grad_x_err,
            "adapter_a_update_max_abs_err": update_a_err,
            "adapter_b_update_max_abs_err": update_b_err,
            "lora_state_fingerprint": (
                lora_state_fingerprint(lora_state) if lora_state is not None else None
            ),
        })
        a_plain, b_plain = a_plain_new, b_plain_new
        a_masked, b_masked = a_masked_new, b_masked_new

    # ---- Final summary ----
    final_y_plain = plain_lora_linear_forward(
        x_private, w_public, a_plain, b_plain, bias_public, alpha=lora_cfg.alpha,
    )
    final_y_masked, _ = _masked_forward_recover(
        x_private, w_public, a_masked, b_masked, bias_public,
        lora_cfg, fcfg, state=lora_state, generator=gen,
    )
    final_output_err = float((final_y_plain - final_y_masked).abs().max().item())
    final_a_err = float((a_plain - a_masked).abs().max().item())
    final_b_err = float((b_plain - b_masked).abs().max().item())

    if config.dtype == "float64":
        tol_forward = 1e-9
        tol_grad = 1e-9
        tol_update = 1e-9
        tol_invariance = 1e-9
    else:
        tol_forward = 5e-4
        tol_grad = 5e-3
        tol_update = 5e-3
        tol_invariance = 1e-3

    max_grad_a_err = max((r["grad_a_max_abs_err"] for r in per_step), default=0.0)
    max_grad_b_err = max((r["grad_b_max_abs_err"] for r in per_step), default=0.0)
    max_invariance_err = max(
        (r["upstream_gradient_invariance_err"] for r in per_step), default=0.0,
    )
    allclose = (
        max((r["forward_max_abs_err"] for r in per_step), default=0.0) <= tol_forward
        and max_grad_a_err <= tol_grad
        and max_grad_b_err <= tol_grad
        and final_a_err <= tol_update
        and final_b_err <= tol_update
        and max_invariance_err <= tol_invariance
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
        "autograd_vs_analytic_step0": autograd_vs_analytic,
        "masked_backward_correctness": {
            "num_steps": config.num_steps,
            "per_step": per_step,
            "max_loss_diff": max(
                (r["loss_diff_abs"] for r in per_step), default=0.0,
            ),
            "max_grad_a_err": max_grad_a_err,
            "max_grad_b_err": max_grad_b_err,
            "max_grad_x_err": max(
                (r["grad_x_max_abs_err"] for r in per_step
                 if r["grad_x_max_abs_err"] is not None),
                default=0.0,
            ),
            "max_upstream_gradient_invariance_err": max_invariance_err,
            "final_adapter_a_update_err": final_a_err,
            "final_adapter_b_update_err": final_b_err,
            "final_output_err": final_output_err,
            "tolerance_forward": tol_forward,
            "tolerance_grad": tol_grad,
            "tolerance_update": tol_update,
            "tolerance_invariance": tol_invariance,
            "allclose": bool(allclose),
            "masked_backward_allclose": bool(
                max_grad_a_err <= tol_grad
                and max_grad_b_err <= tol_grad
            ),
        },
        "loss_handling": {
            "location": "trusted",
            "stage_7_1_status": "trusted_loss",
            "note": (
                "Trusted side computes L = MSE(Y_recovered, Y_target) and"
                " G = dL/dY = (2/N)(Y - Y_target). Only G_tilde = G N_out^{-T}"
                " is sent to the GPU backward."
            ),
        },
        "gradient_handling": {
            "backward_location": "masked_gpu",
            "stage_7_1_status": "masked_backward_prototype",
            "gpu_visibility": {
                "x_tilde": True,
                "w_tilde": True,
                "a_tilde": True,
                "b_tilde": True,
                "grad_y_tilde": True,
                "grad_a_tilde": True,
                "grad_b_tilde": True,
                "grad_x_tilde": bool(config.recover_grad_x),
                "raw_x": False,
                "raw_a": False,
                "raw_b": False,
                "raw_grad_a": False,
                "raw_grad_b": False,
                "raw_upstream_gradient_g": False,
                "optimizer_state": False,
                "private_target_y": False,
            },
        },
        "optimizer_handling": {
            "location": "trusted",
            "stage_7_1_status": "trusted_optimizer",
            "optimizer": optimizer,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "note": (
                "Optimizer state (SGD momentum / AdamW m, v) lives entirely"
                " on the trusted side and is never exposed to the GPU or to"
                " JSON / CSV / Markdown reports."
            ),
        },
        "pad_compensation": {
            "use_pad": config.use_pad,
            "pad_scale": config.pad_scale,
            "grad_a_compensation_formula": (
                "(alpha / r) T_in^T G B^T   (trusted side, plain space)"
            ),
            "grad_b_compensation_formula": (
                "(alpha / r) A^T T_in^T G   (trusted side, plain space)"
            ),
            "is_trusted_only": True,
        },
        "masked_backward_formula": {
            "upstream_gradient_mask": "G_tilde = G N_out^{-T}",
            "grad_a_tilde": "s X_tilde^T (G_tilde B_tilde^T)",
            "grad_b_tilde": "s (X_tilde A_tilde)^T G_tilde",
            "grad_x_tilde": "G_tilde W_tilde^T + s G_tilde B_tilde^T A_tilde^T",
            "grad_a_recovery": "grad_A = N_in^{-T} grad_A_tilde U^T (+ trusted pad compensation)",
            "grad_b_recovery": "grad_B = U^{-T}    grad_B_tilde N_out^T (+ trusted pad compensation)",
            "grad_x_recovery": "grad_X = grad_X_tilde N_in^T",
            "chain_rule_invariance": "tr(G^T dY) = tr(G_tilde^T dY_tilde) for Y_tilde = Y N_out",
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_backward": (
            "masked-gradient-proxy-evaluated, not formal"
        ),
        "lora_private_training_status": "prototype",
        "lora_backward_status": "masked_backward_prototype",
        "lora_loss_status": "trusted_loss",
        "lora_optimizer_status": "trusted_optimizer",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.2 — rank padding to hide r from A_tilde / B_tilde / grad_A_tilde / grad_B_tilde shapes.",
            "Stage 7.3 — multi-layer LoRA in a tiny transformer block end-to-end; calibrated LoRA workload + LoRA timing-side proxy.",
        ],
    }


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------


def backward_probe_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        rows.append({
            "section": "config", "step": "n/a", "metric": k, "value": v,
        })
    for k, v in report["autograd_vs_analytic_step0"].items():
        rows.append({
            "section": "autograd_cross_check", "step": "0",
            "metric": k, "value": v,
        })
    mb = report["masked_backward_correctness"]
    for r in mb["per_step"]:
        for k, v in r.items():
            if k == "lora_state_fingerprint":
                continue
            rows.append({
                "section": "per_step", "step": str(r["step"]),
                "metric": k, "value": v,
            })
    for k in (
        "max_loss_diff", "max_grad_a_err", "max_grad_b_err", "max_grad_x_err",
        "max_upstream_gradient_invariance_err",
        "final_adapter_a_update_err", "final_adapter_b_update_err",
        "final_output_err", "allclose", "masked_backward_allclose",
    ):
        rows.append({
            "section": "summary", "step": "final", "metric": k, "value": mb[k],
        })
    for section_key in (
        "loss_handling", "gradient_handling", "optimizer_handling",
        "pad_compensation", "masked_backward_formula",
    ):
        s = report[section_key]
        for k, v in s.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    rows.append({
                        "section": section_key, "step": "n/a",
                        "metric": f"{k}.{k2}", "value": v2,
                    })
            else:
                rows.append({
                    "section": section_key, "step": "n/a", "metric": k, "value": v,
                })
    return rows


__all__ = [
    "LoRABackwardProbeConfig",
    "VALID_OPTIMIZERS",
    "backward_probe_csv_rows",
    "normalize_optimizer",
    "run_lora_backward_probe",
]
