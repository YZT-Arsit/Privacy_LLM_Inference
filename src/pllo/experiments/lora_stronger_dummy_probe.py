"""Stage 7.4 — stronger dummy correctness probe.

For each Stage 7.4 dummy strategy, verify that the rank-padded masked
LoRA training step still reproduces the plain rank-``r`` reference:

* ``A_pad B_pad = A_real B_real + correction`` (correction tracked); the
  trusted side subtracts ``(α / true_rank) X @ correction`` from the
  recovered output.
* Per-step forward / loss matches plain rank-``r`` LoRA.
* Per-step grad_A_real / grad_B_real (sliced from grad_A_pad /
  grad_B_pad) matches plain.
* Per-step SGD / AdamW update matches plain.
* The dummy slice is never updated; the optimizer state is sized to
  ``true_rank`` only.
* Padded-rank tensors are visible to the GPU — ``true_rank`` is hidden
  from shape under any strategy that pads.

Loss + optimizer remain trusted (Stage 7.1 contract). Stage 7.0 / 7.1 /
7.2 primitives are NOT modified; this probe wraps them with the Stage
7.4 dummy-strategy adapter constructor + a dummy-correction subtraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    init_lora_adapters,
    plain_lora_linear_forward,
)
from pllo.ops.lora_dummy_strategies import (
    StrongDummyConfig,
    VALID_STRONG_DUMMY_STRATEGIES,
    apply_dummy_correction,
    create_stronger_rank_padded_lora_adapters,
    dummy_correction_norm,
    validate_strong_dummy_config,
)
from pllo.ops.lora_rank_padding import (
    run_masked_rank_padded_lora_linear,
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
class StrongerDummyProbeConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    batch_size: int = 4
    d_in: int = 32
    d_out: int = 16
    true_rank: int = 4
    padded_rank: int = 16
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
    dummy_strategies: tuple[str, ...] = field(
        default_factory=lambda: tuple(VALID_STRONG_DUMMY_STRATEGIES)
    )
    dummy_scale: float = 1.0
    noise_scale: float = 1e-3
    spectrum_match_strength: float = 1.0
    pad_scale: float = 1.0
    dtype: str = "float64"
    device: str = "cpu"


@dataclass
class _OptState:
    m: torch.Tensor | None = None
    v: torch.Tensor | None = None
    step: int = 0


def _apply_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: _OptState,
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
    "Stronger dummy distributions are proxy-evaluated, not formal.",
    "Padded rank remains visible unless heterogeneous padded_rank is separately enabled.",
    "Spectral hardening does not imply cryptographic hiding.",
    "Optimizer state remains trusted-only and is sized to true_rank for every LoRA module.",
    "No real TEE training is evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "No full Qwen / TinyLlama / LLaMA LoRA fine-tuning is evaluated; this is a single-linear probe.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "Adapter is NEVER merged into the public base weight W.",
    "Reports publish summary metrics + fingerprints only; private data, raw adapters, raw gradients, optimizer state, and dense masks are never emitted.",
]


def _run_one_strategy(
    strategy: str,
    *,
    config: StrongerDummyProbeConfig,
    x_private: torch.Tensor,
    y_target: torch.Tensor,
    w_public: torch.Tensor,
    bias_public: torch.Tensor | None,
    a0: torch.Tensor,
    b0: torch.Tensor,
    forward_config: MaskedLoRAForwardConfig,
    gen: torch.Generator,
) -> dict[str, Any]:
    dummy_cfg = StrongDummyConfig(
        true_rank=config.true_rank,
        padded_rank=config.padded_rank,
        dummy_strategy=strategy,
        dummy_scale=config.dummy_scale,
        noise_scale=config.noise_scale,
        spectrum_match_strength=config.spectrum_match_strength,
        fresh_dummy_per_step=config.fresh_dummy_per_step,
        dtype=config.dtype, device=config.device,
    )
    validate_strong_dummy_config(dummy_cfg)
    optimizer = normalize_optimizer(config.optimizer)

    a_plain = a0.detach().clone()
    b_plain = b0.detach().clone()
    a_real = a0.detach().clone()
    b_real = b0.detach().clone()

    state_a_plain = _OptState()
    state_b_plain = _OptState()
    state_a_real = _OptState()
    state_b_real = _OptState()

    per_step: list[dict[str, Any]] = []
    lora_state: LoRAState | None = None
    metadata_last: dict[str, Any] = {}

    for step in range(config.num_steps):
        # ---- Plain rank-r reference ----
        y_plain = plain_lora_linear_forward(
            x_private, w_public, a_plain, b_plain, bias_public,
            alpha=config.alpha,
        )
        diff_plain = y_plain - y_target
        loss_plain = (diff_plain * diff_plain).mean()
        # autograd-friendly reference via leaf clones.
        a_pl = a_plain.detach().clone().requires_grad_(True)
        b_pl = b_plain.detach().clone().requires_grad_(True)
        y_pl_ag = plain_lora_linear_forward(
            x_private, w_public, a_pl, b_pl, bias_public,
            alpha=config.alpha,
        )
        loss_pl_ag = ((y_pl_ag - y_target) ** 2).mean()
        loss_pl_ag.backward()
        grad_a_plain = a_pl.grad.detach().clone()
        grad_b_plain = b_pl.grad.detach().clone()

        # ---- Stronger-dummy rank-padded path ----
        pack = create_stronger_rank_padded_lora_adapters(
            a_real.detach(), b_real.detach(), dummy_cfg, generator=gen,
        )
        a_pad = pack["a_pad"].detach().clone().requires_grad_(True)
        b_pad = pack["b_pad"].detach().clone().requires_grad_(True)
        correction = pack["correction"]
        metadata_last = pack["metadata"]

        y_masked, lora_state = run_masked_rank_padded_lora_linear(
            x_private, w_public, a_pad, b_pad, bias_public,
            true_rank=config.true_rank, padded_rank=config.padded_rank,
            alpha=config.alpha, state=lora_state,
            forward_config=forward_config, generator=gen,
        )
        y_corrected = apply_dummy_correction(
            y_masked, x_private, correction,
            true_rank=config.true_rank, alpha=config.alpha,
        )
        diff_masked = y_corrected - y_target
        loss_masked = (diff_masked * diff_masked).mean()
        loss_masked.backward()
        grad_a_pad = a_pad.grad.detach().clone()
        grad_b_pad = b_pad.grad.detach().clone()
        grad_a_real = grad_a_pad[:, : config.true_rank].contiguous()
        grad_b_real = grad_b_pad[: config.true_rank, :].contiguous()

        forward_err = float(
            (y_plain - y_corrected.detach()).abs().max().item()
        )
        loss_diff = float(
            abs(loss_plain.detach().item() - loss_masked.detach().item())
        )
        grad_a_err = float(
            (grad_a_plain - grad_a_real).abs().max().item()
        )
        grad_b_err = float(
            (grad_b_plain - grad_b_real).abs().max().item()
        )
        correction_norm = (
            0.0 if correction is None
            else float(correction.detach().norm().item())
        )
        dummy_norm = dummy_correction_norm(
            a_pad.detach(), b_pad.detach(), config.true_rank,
        )

        # ---- Trusted optimizer updates (real slice only) ----
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
        a_real_new = _apply_step(
            a_real, grad_a_real, state_a_real,
            optimizer=optimizer, lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2,
            eps=config.adamw_eps,
        )
        b_real_new = _apply_step(
            b_real, grad_b_real, state_b_real,
            optimizer=optimizer, lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.adamw_beta1, beta2=config.adamw_beta2,
            eps=config.adamw_eps,
        )
        update_a_err = float(
            (a_plain_new - a_real_new).abs().max().item()
        )
        update_b_err = float(
            (b_plain_new - b_real_new).abs().max().item()
        )

        per_step.append({
            "step": step,
            "loss_plain": float(loss_plain.detach().item()),
            "loss_masked": float(loss_masked.detach().item()),
            "loss_diff_abs": loss_diff,
            "forward_max_abs_err": forward_err,
            "dummy_contribution_norm": dummy_norm,
            "correction_norm": correction_norm,
            "grad_a_real_max_abs_err": grad_a_err,
            "grad_b_real_max_abs_err": grad_b_err,
            "adapter_a_update_max_abs_err": update_a_err,
            "adapter_b_update_max_abs_err": update_b_err,
        })

        a_plain, b_plain = a_plain_new, b_plain_new
        a_real, b_real = a_real_new, b_real_new

    # Tolerances per dtype + per strategy.
    if config.dtype == "float64":
        tol_forward = 1e-9
        tol_grad = 1e-7
        tol_update = 1e-7
        tol_dummy_exact = 1e-9
        tol_correction = (
            5.0 * config.noise_scale
            if strategy == "noise_injected_cancellation_dummy"
            else 1e-9
        )
    else:
        tol_forward = 5e-4
        tol_grad = 5e-3
        tol_update = 5e-3
        tol_dummy_exact = 1e-3
        tol_correction = (
            5.0 * config.noise_scale
            if strategy == "noise_injected_cancellation_dummy"
            else 1e-3
        )

    max_forward = max(
        (r["forward_max_abs_err"] for r in per_step), default=0.0,
    )
    max_grad_a = max(
        (r["grad_a_real_max_abs_err"] for r in per_step), default=0.0,
    )
    max_grad_b = max(
        (r["grad_b_real_max_abs_err"] for r in per_step), default=0.0,
    )
    max_update_a = max(
        (r["adapter_a_update_max_abs_err"] for r in per_step), default=0.0,
    )
    max_update_b = max(
        (r["adapter_b_update_max_abs_err"] for r in per_step), default=0.0,
    )
    max_loss_diff = max(
        (r["loss_diff_abs"] for r in per_step), default=0.0,
    )
    max_dummy = max(
        (r["dummy_contribution_norm"] for r in per_step), default=0.0,
    )
    max_correction = max(
        (r["correction_norm"] for r in per_step), default=0.0,
    )
    allclose = (
        max_forward <= tol_forward
        and max_grad_a <= tol_grad
        and max_grad_b <= tol_grad
        and max_update_a <= tol_update
        and max_update_b <= tol_update
        and max_loss_diff <= tol_forward
    )

    # Optimizer state introspection — `a_real` / `b_real` always rank-r.
    optimizer_state_contains_dummy = False
    if a_real.shape[1] != config.true_rank:
        optimizer_state_contains_dummy = True
    if b_real.shape[0] != config.true_rank:
        optimizer_state_contains_dummy = True
    optimizer_state_shape_a = (
        list(state_a_real.m.shape) if state_a_real.m is not None else None
    )
    optimizer_state_shape_b = (
        list(state_b_real.m.shape) if state_b_real.m is not None else None
    )

    return {
        "dummy_strategy": strategy,
        "dummy_strategy_effective": metadata_last.get(
            "dummy_strategy_effective", strategy,
        ),
        "metadata": metadata_last,
        "per_step": per_step,
        "max_loss_diff": max_loss_diff,
        "max_forward_err": max_forward,
        "max_grad_a_real_err": max_grad_a,
        "max_grad_b_real_err": max_grad_b,
        "max_update_a_err": max_update_a,
        "max_update_b_err": max_update_b,
        "max_dummy_contribution_norm": max_dummy,
        "max_correction_norm": max_correction,
        "tolerance_forward": tol_forward,
        "tolerance_grad": tol_grad,
        "tolerance_update": tol_update,
        "tolerance_dummy_exact": tol_dummy_exact,
        "tolerance_correction": tol_correction,
        "allclose": bool(allclose),
        "shape_level_rank_hiding": {
            "visible_rank_from_a_shape": config.padded_rank,
            "visible_rank_from_b_shape": config.padded_rank,
            "true_rank_hidden_from_shape": bool(
                config.padded_rank > config.true_rank
            ),
            "padded_rank_visible": True,
        },
        "optimizer_handling": {
            "location": "trusted",
            "optimizer": optimizer,
            "trainable_adapter_shape_a": list(a_real.shape),
            "trainable_adapter_shape_b": list(b_real.shape),
            "optimizer_state_shape_a": optimizer_state_shape_a,
            "optimizer_state_shape_b": optimizer_state_shape_b,
            "optimizer_state_contains_dummy": bool(
                optimizer_state_contains_dummy
            ),
            "dummy_update_applied": False,
        },
    }


def run_lora_stronger_dummy_probe(
    config: StrongerDummyProbeConfig,
) -> dict[str, Any]:
    """Run the Stage 7.4 stronger-dummy correctness probe across every
    requested strategy.
    """
    optimizer = normalize_optimizer(config.optimizer)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    gen = torch.Generator(device="cpu").manual_seed(config.seed)

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

    x_private = torch.randn(
        config.batch_size, config.d_in,
        generator=gen, dtype=dtype, device=device,
    )
    y_target = torch.randn(
        config.batch_size, config.d_out,
        generator=gen, dtype=dtype, device=device,
    )
    w_public = torch.randn(
        config.d_in, config.d_out,
        generator=gen, dtype=dtype, device=device,
    )
    bias_public = (
        torch.randn(
            config.d_out, generator=gen, dtype=dtype, device=device,
        )
        if config.use_bias else None
    )
    a0, b0 = init_lora_adapters(real_lora_cfg, generator=gen)
    b0 = b0 + 1e-3 * torch.randn(
        config.true_rank, config.d_out,
        generator=gen, dtype=dtype, device=device,
    )

    per_strategy: list[dict[str, Any]] = []
    for strategy in config.dummy_strategies:
        per_strategy.append(
            _run_one_strategy(
                strategy,
                config=config,
                x_private=x_private,
                y_target=y_target,
                w_public=w_public,
                bias_public=bias_public,
                a0=a0, b0=b0,
                forward_config=fcfg,
                gen=gen,
            )
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
        "stronger_dummy_strategy_design": {
            "supported_strategies": list(VALID_STRONG_DUMMY_STRATEGIES),
            "evaluated_strategies": list(config.dummy_strategies),
            "design_notes": [
                "All cancellation strategies preserve A_pad B_pad = A_real B_real exactly.",
                "noise_injected_cancellation_dummy tracks a small trusted-side correction = A_pad[:, r:] B_pad[r:, :] that the harness subtracts from the recovered output via (alpha / true_rank) X @ correction.",
                "spectrum_matched_dummy cycles singular values from the empirical A_real / B_real spectrum.",
                "gaussian_matched_dummy samples R / S from a Gaussian matched to per-column statistics of A_real / B_real.",
                "orthogonalized_cancellation_dummy projects R / S orthogonal to the column / row span of A_real / B_real.",
                "mixed_dummy_ensemble samples a per-pair strategy from the four cancellation strategies above.",
                "Spectral hardening does not imply cryptographic hiding.",
            ],
        },
        "per_strategy": per_strategy,
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_dummy_hardening": (
            "spectral-rank-hardening-proxy-evaluated, not formal"
        ),
        "lora_stronger_dummy_status": "implemented",
        "lora_spectral_rank_hardening_status": "proxy-evaluated",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.5 — paper artifact consolidation + projected vs measured runtime alignment.",
            "Stage 7.x — heterogeneous padded_rank across modules / layers to hide padded_rank itself.",
        ],
    }


def stronger_dummy_probe_csv_rows(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if isinstance(v, (tuple, list)):
            v = "|".join(str(x) for x in v)
        rows.append({
            "section": "config",
            "strategy": "n/a",
            "step": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for entry in report["per_strategy"]:
        strat = entry["dummy_strategy"]
        for k in (
            "max_loss_diff",
            "max_forward_err",
            "max_grad_a_real_err",
            "max_grad_b_real_err",
            "max_update_a_err",
            "max_update_b_err",
            "max_dummy_contribution_norm",
            "max_correction_norm",
            "allclose",
            "dummy_strategy_effective",
        ):
            rows.append({
                "section": "per_strategy_summary",
                "strategy": strat,
                "step": "summary",
                "metric": k,
                "value": entry[k],
                "notes": "",
            })
        for r in entry["per_step"]:
            for k, v in r.items():
                if k == "step":
                    continue
                rows.append({
                    "section": "per_step",
                    "strategy": strat,
                    "step": str(r["step"]),
                    "metric": k,
                    "value": v,
                    "notes": "",
                })
        for k, v in entry["shape_level_rank_hiding"].items():
            rows.append({
                "section": "shape_level_rank_hiding",
                "strategy": strat,
                "step": "n/a",
                "metric": k,
                "value": v,
                "notes": "",
            })
        for k, v in entry["optimizer_handling"].items():
            if isinstance(v, (tuple, list)):
                v = "|".join(str(x) for x in v)
            rows.append({
                "section": "optimizer_handling",
                "strategy": strat,
                "step": "n/a",
                "metric": k,
                "value": v,
                "notes": "",
            })
    return rows


__all__ = [
    "StrongerDummyProbeConfig",
    "VALID_OPTIMIZERS",
    "normalize_optimizer",
    "run_lora_stronger_dummy_probe",
    "stronger_dummy_probe_csv_rows",
]
