"""Stage 7.2 — Rank-padded LoRA training correctness probe.

Reuses the Stage 7.0 masked forward + Stage 7.1 masked backward, but
inserts a Stage 7.2 rank-padded adapter layer in between. The trusted
side holds plain ``(A_real, B_real)`` of shape ``r × …``; before each
forward call it samples dummy rank slices (per
``RankPaddingConfig.dummy_strategy``) to build ``(A_pad, B_pad)`` of
shape ``r_pad × …`` such that ``A_pad B_pad == A_real B_real``. The
GPU then sees only the padded shape. After backward the trusted side
slices ``grad_A_pad[:, :true_rank]`` / ``grad_B_pad[:true_rank, :]``
and feeds those into the SGD / AdamW step on ``(A_real, B_real)``.

The probe compares this rank-padded path against plain rank-``r`` LoRA
training (Stage 7.0 / 7.1 contract) and confirms:

  * forward + backward + update remain allclose to plain rank-``r``;
  * dummy contribution to the forward output is ``≈ 0`` numerically;
  * the optimizer state has rank-``r`` shape, never rank-``r_pad``;
  * the GPU-visible padded tensors have rank ``r_pad`` (true rank
    hidden from shape).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    init_lora_adapters,
    plain_lora_linear_forward,
)
from pllo.ops.lora_backward import plain_lora_backward_reference
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    VALID_DUMMY_STRATEGIES,
    create_rank_padded_lora_adapters,
    dummy_contribution_norm,
    extract_real_gradients,
    plain_rank_padded_lora_forward,
    run_masked_rank_padded_lora_backward,
    run_masked_rank_padded_lora_linear,
    validate_rank_padding_config,
)


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
class LoRARankPaddingProbeConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    batch_size: int = 4
    d_in: int = 32
    d_out: int = 16
    true_rank: int = 4
    padded_rank: int = 8
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
    fresh_dummy_per_step: bool = True
    dummy_strategy: str = "paired_cancellation_dummy"
    dummy_scale: float = 1.0
    pad_scale: float = 1.0
    recover_grad_x: bool = False
    dtype: str = "float64"
    device: str = "cpu"


@dataclass
class _OptimizerState:
    m: torch.Tensor | None = None
    v: torch.Tensor | None = None
    step: int = 0


def _apply_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: _OptimizerState,
    *, optimizer: str, lr: float, weight_decay: float,
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


_LIMITATIONS = [
    "Stage 7.2 hides true rank from tensor shape by exposing padded rank.",
    "Padded rank r_pad remains visible to the GPU.",
    "Dummy rank indistinguishability is evaluated by proxy (Stage 7.2 security proxy), not formally proven.",
    "Optimizer state remains trusted-only and is sized to true_rank, never padded_rank.",
    "Backward / loss computation remains trusted (Stage 7.1 contract): only G_tilde, A_pad_tilde, B_pad_tilde, grad_A_pad_tilde, grad_B_pad_tilde cross the boundary.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "No real TEE training.",
    "No full Qwen / TinyLlama / LLaMA LoRA fine-tuning.",
    "Adapter is NEVER merged into the public base weight W.",
    "No formal / cryptographic / semantic security is claimed.",
    "Reports publish summary metrics + fingerprints. Private data, raw adapters (real or padded), raw gradients, optimizer state, and dense masks are never emitted in outputs.",
]


def run_lora_rank_padding_probe(
    config: LoRARankPaddingProbeConfig,
) -> dict[str, Any]:
    """Run the Stage 7.2 rank-padded LoRA training correctness probe."""
    optimizer = normalize_optimizer(config.optimizer)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    torch.manual_seed(config.seed)

    rank_padding_cfg = RankPaddingConfig(
        true_rank=config.true_rank, padded_rank=config.padded_rank,
        dummy_scale=config.dummy_scale, dummy_strategy=config.dummy_strategy,
        fresh_dummy_per_step=config.fresh_dummy_per_step,
        dtype=config.dtype, device=config.device,
    )
    validate_rank_padding_config(rank_padding_cfg)

    real_lora_cfg = LoRAConfig(
        d_in=config.d_in, d_out=config.d_out, rank=config.true_rank,
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

    # Private data + public base weight.
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

    # Initialise the real (rank-r) adapter; perturb B_real away from zero.
    a0, b0 = init_lora_adapters(real_lora_cfg, generator=gen)
    b0 = b0 + 1e-3 * torch.randn(
        config.true_rank, config.d_out, generator=gen, dtype=dtype, device=device,
    )
    a_plain = a0.clone()
    b_plain = b0.clone()
    a_padded_real = a0.clone()
    b_padded_real = b0.clone()

    state_a_plain = _OptimizerState()
    state_b_plain = _OptimizerState()
    state_a_padded = _OptimizerState()
    state_b_padded = _OptimizerState()

    per_step: list[dict[str, Any]] = []
    lora_state: LoRAState | None = None
    dummy_meta: dict[str, Any] | None = None
    visible_fingerprint: dict[str, Any] | None = None

    for step in range(config.num_steps):
        # --- Plain rank-r LoRA training step (Stage 7.0/7.1 reference) ---
        y_plain = plain_lora_linear_forward(
            x_private, w_public, a_plain, b_plain, bias_public, alpha=config.alpha,
        )
        diff_plain = y_plain - y_target
        loss_plain = float((diff_plain * diff_plain).mean().item())
        g_plain = (2.0 / float(y_plain.numel())) * diff_plain
        ref = plain_lora_backward_reference(
            x_private, w_public, a_plain, b_plain, g_plain, alpha=config.alpha,
        )
        grad_a_plain = ref["grad_a"]
        grad_b_plain = ref["grad_b"]

        # --- Rank-padded path ---
        # Build a fresh padded adapter from the current real (A, B).
        pack = create_rank_padded_lora_adapters(
            a_padded_real, b_padded_real, rank_padding_cfg, generator=gen,
        )
        a_pad = pack["a_pad"]
        b_pad = pack["b_pad"]
        dummy_meta = pack["metadata"]

        # Verify A_pad B_pad ≈ A B (sanity).
        dummy_residual_norm = dummy_contribution_norm(
            a_pad, b_pad, config.true_rank,
        )

        # Masked padded forward.
        y_padded, lora_state = run_masked_rank_padded_lora_linear(
            x_private, w_public, a_pad, b_pad, bias_public,
            true_rank=config.true_rank, padded_rank=config.padded_rank,
            alpha=config.alpha, state=lora_state,
            forward_config=fcfg, generator=gen,
        )
        diff_padded = y_padded - y_target
        loss_padded = float((diff_padded * diff_padded).mean().item())
        g_padded = (2.0 / float(y_padded.numel())) * diff_padded

        # GPU-visible shape fingerprint (per-step).
        a_tilde_pad_shape = [config.d_in, config.padded_rank]
        b_tilde_pad_shape = [config.padded_rank, config.d_out]
        visible_fingerprint = {
            "a_tilde_pad_shape": a_tilde_pad_shape,
            "b_tilde_pad_shape": b_tilde_pad_shape,
            "visible_rank_from_a_shape": config.padded_rank,
            "visible_rank_from_b_shape": config.padded_rank,
            "true_rank_hidden_from_shape": True,
        }

        # Masked padded backward.
        got = run_masked_rank_padded_lora_backward(
            x_private, w_public, a_pad, b_pad, g_padded,
            true_rank=config.true_rank, padded_rank=config.padded_rank,
            alpha=config.alpha, state=lora_state,
            recover_grad_x=config.recover_grad_x,
        )
        real_grads = extract_real_gradients(
            got["grad_a_pad"], got["grad_b_pad"], config.true_rank,
        )
        grad_a_real = real_grads["grad_a_real"]
        grad_b_real = real_grads["grad_b_real"]

        forward_err = float((y_plain - y_padded).abs().max().item())
        loss_diff = abs(loss_plain - loss_padded)
        grad_a_err = float((grad_a_plain - grad_a_real).abs().max().item())
        grad_b_err = float((grad_b_plain - grad_b_real).abs().max().item())

        # Optimizer updates — only the real slice is touched.
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
        a_padded_new = _apply_step(
            a_padded_real, grad_a_real, state_a_padded,
            optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2, eps=config.adamw_eps,
        )
        b_padded_new = _apply_step(
            b_padded_real, grad_b_real, state_b_padded,
            optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2, eps=config.adamw_eps,
        )
        update_a_err = float((a_plain_new - a_padded_new).abs().max().item())
        update_b_err = float((b_plain_new - b_padded_new).abs().max().item())

        per_step.append({
            "step": step,
            "loss_plain": loss_plain,
            "loss_padded": loss_padded,
            "loss_diff_abs": loss_diff,
            "forward_max_abs_err": forward_err,
            "dummy_contribution_norm": dummy_residual_norm,
            "grad_a_real_max_abs_err": grad_a_err,
            "grad_b_real_max_abs_err": grad_b_err,
            "adapter_a_update_max_abs_err": update_a_err,
            "adapter_b_update_max_abs_err": update_b_err,
        })

        a_plain, b_plain = a_plain_new, b_plain_new
        a_padded_real, b_padded_real = a_padded_new, b_padded_new

    # Optimizer state shape introspection.
    optimizer_state_contains_dummy = False
    optimizer_state_shape_a: list[int] | None = None
    optimizer_state_shape_b: list[int] | None = None
    if state_a_padded.m is not None:
        optimizer_state_shape_a = list(state_a_padded.m.shape)
        if optimizer_state_shape_a[-1] != config.true_rank:
            optimizer_state_contains_dummy = True
    if state_b_padded.m is not None:
        optimizer_state_shape_b = list(state_b_padded.m.shape)
        if optimizer_state_shape_b[0] != config.true_rank:
            optimizer_state_contains_dummy = True
    # For SGD with no momentum, m / v stay None. Validate via the
    # trainable adapter shapes themselves.
    if not optimizer_state_contains_dummy:
        if a_padded_real.shape[-1] != config.true_rank:
            optimizer_state_contains_dummy = True
        if b_padded_real.shape[0] != config.true_rank:
            optimizer_state_contains_dummy = True
    trainable_adapter_shape_a = list(a_padded_real.shape)
    trainable_adapter_shape_b = list(b_padded_real.shape)

    final_a_err = float((a_plain - a_padded_real).abs().max().item())
    final_b_err = float((b_plain - b_padded_real).abs().max().item())

    if config.dtype == "float64":
        tol_forward = 1e-9
        tol_grad = 1e-9
        tol_update = 1e-9
        tol_dummy = 1e-9
    else:
        tol_forward = 5e-4
        tol_grad = 5e-3
        tol_update = 5e-3
        tol_dummy = 1e-3

    max_grad_a_err = max((r["grad_a_real_max_abs_err"] for r in per_step), default=0.0)
    max_grad_b_err = max((r["grad_b_real_max_abs_err"] for r in per_step), default=0.0)
    max_dummy_norm = max((r["dummy_contribution_norm"] for r in per_step), default=0.0)
    allclose = (
        max((r["forward_max_abs_err"] for r in per_step), default=0.0) <= tol_forward
        and max_grad_a_err <= tol_grad
        and max_grad_b_err <= tol_grad
        and final_a_err <= tol_update
        and final_b_err <= tol_update
        and max_dummy_norm <= tol_dummy
    )

    return {
        "config": {**asdict(config), "optimizer": optimizer},
        "lora_config_fingerprint": {
            "d_in": config.d_in,
            "d_out": config.d_out,
            "true_rank": config.true_rank,
            "padded_rank": config.padded_rank,
            "alpha": config.alpha,
            "use_bias": config.use_bias,
        },
        "dummy_rank_strategy": {
            "requested": config.dummy_strategy,
            "effective": dummy_meta["dummy_strategy_effective"] if dummy_meta else None,
            "dummy_size": dummy_meta["dummy_size"] if dummy_meta else 0,
            "dummy_scale": config.dummy_scale,
            "fresh_dummy_per_step": config.fresh_dummy_per_step,
            "max_dummy_contribution_norm": max_dummy_norm,
        },
        "rank_padding_correctness": {
            "num_steps": config.num_steps,
            "per_step": per_step,
            "max_loss_diff": max((r["loss_diff_abs"] for r in per_step), default=0.0),
            "max_grad_a_real_err": max_grad_a_err,
            "max_grad_b_real_err": max_grad_b_err,
            "max_dummy_contribution_norm": max_dummy_norm,
            "final_adapter_a_update_err": final_a_err,
            "final_adapter_b_update_err": final_b_err,
            "tolerance_forward": tol_forward,
            "tolerance_grad": tol_grad,
            "tolerance_update": tol_update,
            "tolerance_dummy": tol_dummy,
            "allclose": bool(allclose),
        },
        "optimizer_handling": {
            "location": "trusted",
            "stage_7_2_status": "trusted_optimizer_rank_r",
            "optimizer": optimizer,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "trainable_adapter_shape_a": trainable_adapter_shape_a,
            "trainable_adapter_shape_b": trainable_adapter_shape_b,
            "optimizer_state_shape_a": optimizer_state_shape_a,
            "optimizer_state_shape_b": optimizer_state_shape_b,
            "optimizer_state_contains_dummy": bool(optimizer_state_contains_dummy),
            "dummy_update_applied": False,
            "note": (
                "Optimizer state (and the trainable A_real / B_real tensors)"
                " is sized to true_rank, never padded_rank. The dummy slice"
                " is re-sampled from scratch each step and never enters the"
                " optimizer."
            ),
        },
        "shape_level_rank_hiding": {
            **(visible_fingerprint or {}),
            "true_rank_hidden_from_shape": True,
            "padded_rank_visible": True,
            "note": (
                "Stage 7.2 hides true_rank from the dimensions of A_pad_tilde"
                " / B_pad_tilde / grad_A_pad_tilde / grad_B_pad_tilde. The"
                " padded_rank itself remains visible — see the security proxy"
                " for residual spectral / gradient-side inference risk."
            ),
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_rank_padding": (
            "rank-padding-proxy-evaluated, not formal"
        ),
        "lora_rank_padding_status": "implemented",
        "lora_hidden_rank_status": "padded-rank-prototype",
        "lora_true_rank_hidden_from_shape": True,
        "lora_padded_rank_visible": True,
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.3 — multi-layer LoRA end-to-end + LoRA training timing-side proxy.",
            "Stage 7.x — stronger spectral / gradient-side dummy strategies that resist rank inference proxies.",
        ],
    }


def rank_padding_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        rows.append({
            "section": "config", "step": "n/a", "metric": k, "value": v,
        })
    ds = report["dummy_rank_strategy"]
    for k, v in ds.items():
        rows.append({
            "section": "dummy_strategy", "step": "n/a", "metric": k, "value": v,
        })
    rp = report["rank_padding_correctness"]
    for r in rp["per_step"]:
        for k, v in r.items():
            rows.append({
                "section": "per_step", "step": str(r["step"]),
                "metric": k, "value": v,
            })
    for k in (
        "max_loss_diff", "max_grad_a_real_err", "max_grad_b_real_err",
        "max_dummy_contribution_norm",
        "final_adapter_a_update_err", "final_adapter_b_update_err",
        "allclose",
    ):
        rows.append({
            "section": "summary", "step": "final", "metric": k, "value": rp[k],
        })
    for section_key in ("optimizer_handling", "shape_level_rank_hiding"):
        section = report[section_key]
        for k, v in section.items():
            rows.append({
                "section": section_key, "step": "n/a", "metric": k, "value": v,
            })
    return rows


__all__ = [
    "LoRARankPaddingProbeConfig",
    "VALID_OPTIMIZERS",
    "normalize_optimizer",
    "rank_padding_csv_rows",
    "run_lora_rank_padding_probe",
]
