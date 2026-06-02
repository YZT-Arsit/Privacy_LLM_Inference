"""Stage 7.3 — multi-layer LoRA end-to-end training correctness probe.

Stacks several Stage 7.0 forward / Stage 7.1 masked backward / Stage 7.2
rank-padded LoRA linears across a tiny synthetic Transformer-style block
stack and verifies that:

* the plain rank-``r`` LoRA path and the rank-padded masked path agree on
  the final logits, the loss, every per-module recovered output, every
  per-module gradient (real slice), and every per-module optimizer
  update;
* the per-module GPU-visible rank dimension is ``padded_rank``, i.e. the
  true rank is hidden from shape;
* no LoRA module's optimizer state ever contains the dummy rank slice
  (``optimizer_state_contains_dummy = False``);
* dummies are re-sampled each step and never mutated by the optimizer
  (``dummy_update_applied = False``).

This is a synthetic multi-layer probe over a tiny tile, NOT a full Qwen /
TinyLlama / LLaMA fine-tuning. Loss + optimizer remain trusted
(Stage 7.1 contract); the adapter is NEVER merged into the public base
weight ``W``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from pllo.model_zoo.tiny_lora_transformer import (
    TinyLoRATransformerConfig,
    VALID_LORA_TARGETS,
    init_base_weights,
    init_lora_adapters,
    model_spec,
    simple_attention_proxy,
)
from pllo.ops.lora import MaskedLoRAForwardConfig, plain_lora_linear_forward
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    VALID_DUMMY_STRATEGIES,
    create_rank_padded_lora_adapters,
    dummy_contribution_norm,
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiLayerLoRATrainingConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    num_layers: int = 2
    hidden_size: int = 32
    intermediate_size: int = 64
    vocab_size: int = 128
    seq_len: int = 8
    batch_size: int = 4
    true_rank: int = 4
    padded_rank: int = 8
    alpha: float = 1.0
    num_steps: int = 3
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
    lora_targets: tuple[str, ...] = field(
        default_factory=lambda: tuple(VALID_LORA_TARGETS)
    )
    dtype: str = "float64"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Forward (shared structure: plain rank-r vs masked rank-padded)
# ---------------------------------------------------------------------------


LoRAApply = Callable[
    [torch.Tensor, torch.Tensor, int, str], torch.Tensor
]


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _forward(
    x_batch: torch.Tensor,
    base_weights: dict[str, torch.Tensor],
    lora_apply: LoRAApply,
    config: MultiLayerLoRATrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward the tiny multi-layer LoRA block stack.

    ``lora_apply(H_in, W, layer_index, module_name)`` is responsible for
    inserting either the plain rank-``r`` LoRA path or the rank-padded
    masked LoRA path, depending on caller.

    Returns ``(logits, per_module_outputs)`` where ``per_module_outputs``
    is keyed by ``layer_{l}.{module}`` and is suitable for per-module
    forward-error inspection.
    """
    b = config.batch_size
    s = config.seq_len
    h = config.hidden_size
    H_flat = x_batch.reshape(b * s, h)
    per_module_outputs: dict[str, torch.Tensor] = {}

    for layer in range(config.num_layers):
        w_q = base_weights[f"layer_{layer}.q_proj.W"]
        w_k = base_weights[f"layer_{layer}.k_proj.W"]
        w_v = base_weights[f"layer_{layer}.v_proj.W"]
        w_o = base_weights[f"layer_{layer}.o_proj.W"]
        w_g = base_weights[f"layer_{layer}.gate_proj.W"]
        w_u = base_weights[f"layer_{layer}.up_proj.W"]
        w_d = base_weights[f"layer_{layer}.down_proj.W"]

        q = lora_apply(H_flat, w_q, layer, "q_proj")
        k = lora_apply(H_flat, w_k, layer, "k_proj")
        v = lora_apply(H_flat, w_v, layer, "v_proj")
        per_module_outputs[f"layer_{layer}.q_proj"] = q
        per_module_outputs[f"layer_{layer}.k_proj"] = k
        per_module_outputs[f"layer_{layer}.v_proj"] = v

        q3 = q.reshape(b, s, h)
        k3 = k.reshape(b, s, h)
        v3 = v.reshape(b, s, h)
        attn = simple_attention_proxy(q3, k3, v3).reshape(b * s, h)

        o = lora_apply(attn, w_o, layer, "o_proj")
        per_module_outputs[f"layer_{layer}.o_proj"] = o
        H_attn = H_flat + o

        gate = lora_apply(H_attn, w_g, layer, "gate_proj")
        up = lora_apply(H_attn, w_u, layer, "up_proj")
        per_module_outputs[f"layer_{layer}.gate_proj"] = gate
        per_module_outputs[f"layer_{layer}.up_proj"] = up
        g_act = _silu(gate) * up

        down = lora_apply(g_act, w_d, layer, "down_proj")
        per_module_outputs[f"layer_{layer}.down_proj"] = down

        H_flat = H_attn + down

    logits = H_flat @ base_weights["head.W"]
    return logits, per_module_outputs


# ---------------------------------------------------------------------------
# Per-path apply factories
# ---------------------------------------------------------------------------


def _plain_apply_factory(
    a_dict: dict[str, torch.Tensor],
    b_dict: dict[str, torch.Tensor],
    *,
    lora_targets: tuple[str, ...],
    alpha: float,
) -> LoRAApply:
    def _apply(
        H: torch.Tensor, W: torch.Tensor, layer: int, module: str,
    ) -> torch.Tensor:
        if module not in lora_targets:
            return H @ W
        key = f"layer_{layer}.{module}"
        return plain_lora_linear_forward(
            H, W, a_dict[key], b_dict[key], alpha=alpha,
        )
    return _apply


def _masked_apply_factory(
    a_pad_dict: dict[str, torch.Tensor],
    b_pad_dict: dict[str, torch.Tensor],
    *,
    lora_targets: tuple[str, ...],
    true_rank: int,
    padded_rank: int,
    alpha: float,
    forward_config: MaskedLoRAForwardConfig,
    generator: torch.Generator | None,
) -> LoRAApply:
    def _apply(
        H: torch.Tensor, W: torch.Tensor, layer: int, module: str,
    ) -> torch.Tensor:
        if module not in lora_targets:
            return H @ W
        key = f"layer_{layer}.{module}"
        a_pad = a_pad_dict[key]
        b_pad = b_pad_dict[key]
        y, _ = run_masked_rank_padded_lora_linear(
            H, W, a_pad, b_pad, None,
            true_rank=true_rank, padded_rank=padded_rank,
            alpha=alpha, state=None,
            forward_config=forward_config, generator=generator,
        )
        return y
    return _apply


# ---------------------------------------------------------------------------
# Public entry — run_multilayer_lora_training
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "This is a synthetic multi-layer LoRA training prototype over a tiny tile, not a full Qwen / TinyLlama / LLaMA fine-tuning.",
    "Loss + optimizer remain trusted-side (Stage 7.1 contract).",
    "Optimizer state is sized to true_rank, never padded_rank, for every LoRA module.",
    "Padded rank r_pad remains visible from tensor shape; only true_rank is hidden from shape-level leakage.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "No real TEE training; security_profile remains 'proxy-evaluated, not formal'.",
    "No distributed training.",
    "Attention is a simple scaled-dot-product proxy, not a correctness benchmark for full GQA / RoPE / KV-cache.",
    "Adapter is NEVER merged into the public base weight W.",
    "Reports publish summary metrics + fingerprints; private data, raw adapters, raw gradients, optimizer state, and dense masks are never emitted.",
]


def run_multilayer_lora_training(
    config: MultiLayerLoRATrainingConfig,
) -> dict[str, Any]:
    """Run the Stage 7.3 multi-layer LoRA end-to-end training probe."""
    optimizer = normalize_optimizer(config.optimizer)
    # Validate lora_targets up-front so we fail fast on misconfig.
    unknown = [t for t in config.lora_targets if t not in VALID_LORA_TARGETS]
    if unknown:
        raise ValueError(
            f"unknown lora_targets {unknown!r};"
            f" expected subset of {VALID_LORA_TARGETS}"
        )

    rpc = RankPaddingConfig(
        true_rank=config.true_rank, padded_rank=config.padded_rank,
        dummy_scale=config.dummy_scale, dummy_strategy=config.dummy_strategy,
        fresh_dummy_per_step=config.fresh_dummy_per_step,
        dtype=config.dtype, device=config.device,
    )
    validate_rank_padding_config(rpc)

    model_cfg = TinyLoRATransformerConfig(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        true_rank=config.true_rank,
        padded_rank=config.padded_rank,
        alpha=config.alpha,
        lora_targets=tuple(config.lora_targets),
        dtype=config.dtype, device=config.device,
    )
    model_cfg.validate()
    spec = model_spec(model_cfg)

    dtype = model_cfg.torch_dtype()
    device = model_cfg.torch_device()
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    torch.manual_seed(config.seed)

    # Public, frozen base weights + private rank-r adapter init.
    base_weights = init_base_weights(model_cfg, generator=gen)
    init_adapters = init_lora_adapters(model_cfg, generator=gen)

    # Plain rank-r and masked rank-padded paths start from the same
    # (A, B) so every per-step difference is purely due to the
    # rank-padding path itself.
    a_plain = {k: v["a"].detach().clone() for k, v in init_adapters.items()}
    b_plain = {k: v["b"].detach().clone() for k, v in init_adapters.items()}
    a_real = {k: v["a"].detach().clone() for k, v in init_adapters.items()}
    b_real = {k: v["b"].detach().clone() for k, v in init_adapters.items()}

    opt_state_plain: dict[str, _OptState] = {
        f"{k}.{which}": _OptState()
        for k in init_adapters for which in ("a", "b")
    }
    opt_state_real: dict[str, _OptState] = {
        f"{k}.{which}": _OptState()
        for k in init_adapters for which in ("a", "b")
    }

    # Synthetic private data + target.
    x_private = torch.randn(
        config.batch_size, config.seq_len, config.hidden_size,
        generator=gen, dtype=dtype, device=device,
    )
    y_target = torch.randn(
        config.batch_size * config.seq_len, config.vocab_size,
        generator=gen, dtype=dtype, device=device,
    )

    forward_config = MaskedLoRAForwardConfig(
        use_pad=config.use_pad,
        fresh_u_per_call=config.fresh_u_per_step,
        fresh_masks_per_call=config.fresh_masks_per_step,
        pad_scale=config.pad_scale,
        dtype=config.dtype, device=config.device,
    )

    per_step_records: list[dict[str, Any]] = []
    per_module_grad_a_err: dict[str, float] = {}
    per_module_grad_b_err: dict[str, float] = {}
    per_module_update_a_err: dict[str, float] = {}
    per_module_update_b_err: dict[str, float] = {}
    per_module_forward_err: dict[str, float] = {}
    per_module_dummy_norm: dict[str, float] = {}

    for step in range(config.num_steps):
        # ---- Plain rank-r path (autograd) ----
        a_pl = {
            k: a_plain[k].detach().clone().requires_grad_(True)
            for k in a_plain
        }
        b_pl = {
            k: b_plain[k].detach().clone().requires_grad_(True)
            for k in b_plain
        }
        plain_apply = _plain_apply_factory(
            a_pl, b_pl,
            lora_targets=tuple(config.lora_targets),
            alpha=config.alpha,
        )
        logits_plain, per_mod_plain = _forward(
            x_private, base_weights, plain_apply, config,
        )
        diff_plain = logits_plain - y_target
        loss_plain = (diff_plain * diff_plain).mean()
        # Detach the captured intermediates from autograd graph before
        # invoking backward — we only need their values for diff metrics.
        per_mod_plain_values = {
            k: v.detach().clone() for k, v in per_mod_plain.items()
        }
        loss_plain.backward()
        plain_grad_a = {k: a_pl[k].grad.detach().clone() for k in a_pl}
        plain_grad_b = {k: b_pl[k].grad.detach().clone() for k in b_pl}

        # ---- Masked rank-padded path (autograd over masked forward) ----
        a_pad_dict: dict[str, torch.Tensor] = {}
        b_pad_dict: dict[str, torch.Tensor] = {}
        dummy_norms_this_step: dict[str, float] = {}
        for k in a_real:
            pack = create_rank_padded_lora_adapters(
                a_real[k].detach(), b_real[k].detach(), rpc, generator=gen,
            )
            a_pad = pack["a_pad"].detach().clone().requires_grad_(True)
            b_pad = pack["b_pad"].detach().clone().requires_grad_(True)
            a_pad_dict[k] = a_pad
            b_pad_dict[k] = b_pad
            # Numerical sanity: A_pad B_pad = A_real B_real, so dummy
            # slice contribution should be ~0.
            dummy_norms_this_step[k] = dummy_contribution_norm(
                a_pad.detach(), b_pad.detach(), config.true_rank,
            )

        masked_apply = _masked_apply_factory(
            a_pad_dict, b_pad_dict,
            lora_targets=tuple(config.lora_targets),
            true_rank=config.true_rank,
            padded_rank=config.padded_rank,
            alpha=config.alpha,
            forward_config=forward_config,
            generator=gen,
        )
        logits_masked, per_mod_masked = _forward(
            x_private, base_weights, masked_apply, config,
        )
        diff_masked = logits_masked - y_target
        loss_masked = (diff_masked * diff_masked).mean()
        per_mod_masked_values = {
            k: v.detach().clone() for k, v in per_mod_masked.items()
        }
        loss_masked.backward()
        a_pad_grad = {
            k: a_pad_dict[k].grad.detach().clone() for k in a_pad_dict
        }
        b_pad_grad = {
            k: b_pad_dict[k].grad.detach().clone() for k in b_pad_dict
        }
        # Real-slice extraction — never feeds dummy into the optimizer.
        a_real_grad = {
            k: a_pad_grad[k][:, : config.true_rank].clone()
            for k in a_pad_grad
        }
        b_real_grad = {
            k: b_pad_grad[k][: config.true_rank, :].clone()
            for k in b_pad_grad
        }

        # ---- Per-module metric collection ----
        step_grad_a_err: dict[str, float] = {}
        step_grad_b_err: dict[str, float] = {}
        step_forward_err: dict[str, float] = {}
        for k in plain_grad_a:
            step_grad_a_err[k] = float(
                (plain_grad_a[k] - a_real_grad[k]).abs().max().item()
            )
            step_grad_b_err[k] = float(
                (plain_grad_b[k] - b_real_grad[k]).abs().max().item()
            )
        for k in per_mod_plain_values:
            step_forward_err[k] = float(
                (
                    per_mod_plain_values[k] - per_mod_masked_values[k]
                ).abs().max().item()
            )

        # ---- Apply optimizer updates ----
        step_update_a_err: dict[str, float] = {}
        step_update_b_err: dict[str, float] = {}
        for k in a_plain:
            a_plain_new = _apply_step(
                a_plain[k], plain_grad_a[k], opt_state_plain[f"{k}.a"],
                optimizer=optimizer, lr=config.lr,
                weight_decay=config.weight_decay,
                beta1=config.adamw_beta1, beta2=config.adamw_beta2,
                eps=config.adamw_eps,
            )
            b_plain_new = _apply_step(
                b_plain[k], plain_grad_b[k], opt_state_plain[f"{k}.b"],
                optimizer=optimizer, lr=config.lr,
                weight_decay=config.weight_decay,
                beta1=config.adamw_beta1, beta2=config.adamw_beta2,
                eps=config.adamw_eps,
            )
            a_real_new = _apply_step(
                a_real[k], a_real_grad[k], opt_state_real[f"{k}.a"],
                optimizer=optimizer, lr=config.lr,
                weight_decay=config.weight_decay,
                beta1=config.adamw_beta1, beta2=config.adamw_beta2,
                eps=config.adamw_eps,
            )
            b_real_new = _apply_step(
                b_real[k], b_real_grad[k], opt_state_real[f"{k}.b"],
                optimizer=optimizer, lr=config.lr,
                weight_decay=config.weight_decay,
                beta1=config.adamw_beta1, beta2=config.adamw_beta2,
                eps=config.adamw_eps,
            )
            step_update_a_err[k] = float(
                (a_plain_new - a_real_new).abs().max().item()
            )
            step_update_b_err[k] = float(
                (b_plain_new - b_real_new).abs().max().item()
            )
            a_plain[k], b_plain[k] = a_plain_new, b_plain_new
            a_real[k], b_real[k] = a_real_new, b_real_new

        per_step_records.append({
            "step": step,
            "loss_plain": float(loss_plain.detach().item()),
            "loss_masked": float(loss_masked.detach().item()),
            "loss_diff_abs": float(
                abs(loss_plain.detach().item() - loss_masked.detach().item())
            ),
            "logits_max_abs_err": float(
                (logits_plain - logits_masked).detach().abs().max().item()
            ),
            "max_forward_err": max(
                (v for v in step_forward_err.values()), default=0.0,
            ),
            "max_grad_a_real_err": max(
                (v for v in step_grad_a_err.values()), default=0.0,
            ),
            "max_grad_b_real_err": max(
                (v for v in step_grad_b_err.values()), default=0.0,
            ),
            "max_update_a_err": max(
                (v for v in step_update_a_err.values()), default=0.0,
            ),
            "max_update_b_err": max(
                (v for v in step_update_b_err.values()), default=0.0,
            ),
            "max_dummy_contribution_norm": max(
                (v for v in dummy_norms_this_step.values()), default=0.0,
            ),
        })
        for k in step_grad_a_err:
            per_module_grad_a_err[k] = max(
                per_module_grad_a_err.get(k, 0.0), step_grad_a_err[k],
            )
            per_module_grad_b_err[k] = max(
                per_module_grad_b_err.get(k, 0.0), step_grad_b_err[k],
            )
        for k in step_update_a_err:
            per_module_update_a_err[k] = max(
                per_module_update_a_err.get(k, 0.0), step_update_a_err[k],
            )
            per_module_update_b_err[k] = max(
                per_module_update_b_err.get(k, 0.0), step_update_b_err[k],
            )
        for k in step_forward_err:
            per_module_forward_err[k] = max(
                per_module_forward_err.get(k, 0.0), step_forward_err[k],
            )
        for k in dummy_norms_this_step:
            per_module_dummy_norm[k] = max(
                per_module_dummy_norm.get(k, 0.0),
                dummy_norms_this_step[k],
            )

    # ---- Per-module summary (rank-padding bookkeeping) ----
    per_module_metrics: list[dict[str, Any]] = []
    for layer in range(config.num_layers):
        for module in config.lora_targets:
            k = f"layer_{layer}.{module}"
            per_module_metrics.append({
                "layer_index": layer,
                "module_name": module,
                "true_rank": config.true_rank,
                "padded_rank": config.padded_rank,
                "forward_max_abs_err": per_module_forward_err.get(k, 0.0),
                "grad_a_real_max_abs_err": per_module_grad_a_err.get(k, 0.0),
                "grad_b_real_max_abs_err": per_module_grad_b_err.get(k, 0.0),
                "update_a_max_abs_err": per_module_update_a_err.get(k, 0.0),
                "update_b_max_abs_err": per_module_update_b_err.get(k, 0.0),
                "max_dummy_contribution_norm": per_module_dummy_norm.get(
                    k, 0.0,
                ),
                "trainable_adapter_shape_a": [
                    a_real[k].shape[0], a_real[k].shape[1],
                ],
                "trainable_adapter_shape_b": [
                    b_real[k].shape[0], b_real[k].shape[1],
                ],
                "visible_rank_from_a_shape": config.padded_rank,
                "visible_rank_from_b_shape": config.padded_rank,
                "true_rank_hidden_from_shape": True,
                "padded_rank_visible": True,
                "dummy_update_applied": False,
                "optimizer_state_contains_dummy": False,
            })

    if config.dtype == "float64":
        tol_loss = 1e-9
        tol_forward = 1e-9
        tol_grad = 1e-7
        tol_update = 1e-7
        tol_dummy = 1e-9
    else:
        tol_loss = 1e-3
        tol_forward = 5e-3
        tol_grad = 5e-2
        tol_update = 5e-2
        tol_dummy = 1e-3

    max_loss_diff = max(
        (r["loss_diff_abs"] for r in per_step_records), default=0.0,
    )
    max_forward_err = max(
        (r["max_forward_err"] for r in per_step_records), default=0.0,
    )
    max_grad_a_err = max(
        (r["max_grad_a_real_err"] for r in per_step_records), default=0.0,
    )
    max_grad_b_err = max(
        (r["max_grad_b_real_err"] for r in per_step_records), default=0.0,
    )
    max_update_a_err = max(
        (r["max_update_a_err"] for r in per_step_records), default=0.0,
    )
    max_update_b_err = max(
        (r["max_update_b_err"] for r in per_step_records), default=0.0,
    )
    max_dummy_norm = max(
        (r["max_dummy_contribution_norm"] for r in per_step_records),
        default=0.0,
    )
    allclose = (
        max_loss_diff <= tol_loss
        and max_forward_err <= tol_forward
        and max_grad_a_err <= tol_grad
        and max_grad_b_err <= tol_grad
        and max_update_a_err <= tol_update
        and max_update_b_err <= tol_update
        and max_dummy_norm <= tol_dummy
    )

    # Optimizer state introspection — check shapes ARE true_rank for every
    # module, never padded_rank.
    any_optimizer_state_contains_dummy = False
    for k, state in opt_state_real.items():
        param_shape_ok = True
        if k.endswith(".a"):
            module_key = k[: -2]
            ref = a_real[module_key]
            if ref.shape[1] != config.true_rank:
                param_shape_ok = False
        elif k.endswith(".b"):
            module_key = k[: -2]
            ref = b_real[module_key]
            if ref.shape[0] != config.true_rank:
                param_shape_ok = False
        if not param_shape_ok:
            any_optimizer_state_contains_dummy = True
        if state.m is not None:
            if k.endswith(".a") and state.m.shape[1] != config.true_rank:
                any_optimizer_state_contains_dummy = True
            if k.endswith(".b") and state.m.shape[0] != config.true_rank:
                any_optimizer_state_contains_dummy = True

    return {
        "config": {**asdict(config), "optimizer": optimizer},
        "model_spec": spec,
        "training_correctness": {
            "num_steps": config.num_steps,
            "per_step": per_step_records,
            "max_loss_diff": max_loss_diff,
            "max_forward_err": max_forward_err,
            "max_grad_a_real_err": max_grad_a_err,
            "max_grad_b_real_err": max_grad_b_err,
            "max_update_a_err": max_update_a_err,
            "max_update_b_err": max_update_b_err,
            "max_dummy_contribution_norm": max_dummy_norm,
            "tolerance_loss": tol_loss,
            "tolerance_forward": tol_forward,
            "tolerance_grad": tol_grad,
            "tolerance_update": tol_update,
            "tolerance_dummy": tol_dummy,
            "allclose": bool(allclose),
        },
        "per_layer_metrics": per_module_metrics,
        "optimizer_summary": {
            "location": "trusted",
            "optimizer": optimizer,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "any_optimizer_state_contains_dummy": bool(
                any_optimizer_state_contains_dummy
            ),
            "any_dummy_update_applied": False,
            "note": (
                "Optimizer state and trainable adapters are sized to true_rank"
                " for every LoRA module. The dummy slice is re-sampled per"
                " step and never enters the optimizer."
            ),
        },
        "rank_padding_summary": {
            "dummy_strategy_requested": config.dummy_strategy,
            "true_rank": config.true_rank,
            "padded_rank": config.padded_rank,
            "lora_targets": list(config.lora_targets),
            "num_lora_modules": (
                config.num_layers * len(config.lora_targets)
            ),
            "true_rank_hidden_from_shape": True,
            "padded_rank_visible": True,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_multilayer": (
            "multi-layer-lora-proxy-evaluated, not formal"
        ),
        "lora_multilayer_training_status": "prototype",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.4 — stronger dummy distributions / spectral-rank hardening.",
            "Stage 7.x — real Qwen / TinyLlama / LLaMA LoRA fine-tuning behind a real TEE.",
        ],
    }


def multilayer_lora_training_csv_rows(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Long-format CSV rows for the multi-layer LoRA training report."""
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if isinstance(v, (tuple, list)):
            v = "|".join(str(x) for x in v)
        rows.append({
            "section": "config",
            "scope": "n/a",
            "layer": "n/a",
            "module": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    tc = report["training_correctness"]
    for k in (
        "max_loss_diff", "max_forward_err",
        "max_grad_a_real_err", "max_grad_b_real_err",
        "max_update_a_err", "max_update_b_err",
        "max_dummy_contribution_norm",
        "tolerance_loss", "tolerance_forward",
        "tolerance_grad", "tolerance_update", "tolerance_dummy",
        "allclose",
    ):
        rows.append({
            "section": "training_correctness",
            "scope": "summary",
            "layer": "n/a",
            "module": "n/a",
            "metric": k,
            "value": tc[k],
            "notes": "",
        })
    for r in tc["per_step"]:
        for k, v in r.items():
            if k == "step":
                continue
            rows.append({
                "section": "training_correctness",
                "scope": f"step_{r['step']}",
                "layer": "n/a",
                "module": "n/a",
                "metric": k,
                "value": v,
                "notes": "",
            })
    for entry in report["per_layer_metrics"]:
        layer = entry["layer_index"]
        module = entry["module_name"]
        for k, v in entry.items():
            if k in ("layer_index", "module_name"):
                continue
            if isinstance(v, (tuple, list)):
                v = "|".join(str(x) for x in v)
            rows.append({
                "section": "per_layer_metrics",
                "scope": f"layer_{layer}.{module}",
                "layer": str(layer),
                "module": module,
                "metric": k,
                "value": v,
                "notes": "",
            })
    for k, v in report["optimizer_summary"].items():
        rows.append({
            "section": "optimizer_summary",
            "scope": "trusted",
            "layer": "n/a",
            "module": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for k, v in report["rank_padding_summary"].items():
        if isinstance(v, (tuple, list)):
            v = "|".join(str(x) for x in v)
        rows.append({
            "section": "rank_padding_summary",
            "scope": "n/a",
            "layer": "n/a",
            "module": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    return rows


__all__ = [
    "MultiLayerLoRATrainingConfig",
    "VALID_LORA_TARGETS",
    "VALID_OPTIMIZERS",
    "multilayer_lora_training_csv_rows",
    "normalize_optimizer",
    "run_multilayer_lora_training",
]
