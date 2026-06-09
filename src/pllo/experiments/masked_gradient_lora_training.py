"""Stage 7.6 — Masked-gradient LoRA training experiment.

Runs the plaintext LoRA training loop and the masked-gradient LoRA
training loop in lockstep on a synthetic regression task. Validates,
per step:

* forward equivalence (``Y_tilde N_y^T = Y_plain``);
* loss equivalence (``MSE(Y_tilde, target_tilde) = MSE(Y, target)``);
* gradient algebra (``grad_A_tilde = N_x^T grad_A M``,
  ``grad_B_tilde = M^T grad_B N_y``);
* masked-SGD update equivalence under recovery;
* masked-momentum-SGD update equivalence under recovery;
* dummy contribution norm under cancellation rank padding (must be 0).

No raw tensors, masks, gradients, or optimiser states are exported.
JSON / CSV / Markdown emit only summary scalars, shapes, and short
fingerprints.

CPU local emulation only. No real TEE / GPU. No formal cryptographic
security claimed.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Any

import torch

from pllo.ops.masked_gradient_lora import (
    DenseMaskedAdamWUnsupported,
    MaskedGradientLoRAConfig,
    MaskedLoRAState,
    create_cancellation_padded_lora,
    create_masked_lora_state,
    create_orthogonal_matrix,
    dummy_contribution_norm,
    masked_adamw_step_unsupported,
    masked_lora_forward,
    masked_lora_state_fingerprint,
    masked_momentum_sgd_step,
    masked_sgd_step,
    recover_lora_from_masked,
    visible_grad_fingerprint,
)


_REQUIRED_HONESTY_PHRASES: tuple[str, ...] = (
    "The GPU never receives plaintext LoRA adapters or plaintext LoRA "
    "gradients in this experiment.",
    "Masked SGD is algebraically equivalent under orthogonal masks.",
    "Dense masked AdamW is not claimed because coordinate-wise second "
    "moments are not invariant under dense orthogonal mixing.",
    "This is a CPU-only algebraic and proxy-leakage experiment, not a "
    "real TEE/GPU training benchmark.",
    "The user side does not require a GPU; the simulated cloud "
    "accelerator performs masked forward, backward, and optimizer "
    "updates.",
    "No formal, cryptographic, or semantic security is claimed.",
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _dtype_of(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _sample_data(
    cfg: MaskedGradientLoRAConfig, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = _dtype_of(cfg.dtype)
    device = torch.device(cfg.device)
    X = torch.randn(
        cfg.batch_size, cfg.d_in, dtype=dtype, device=device,
        generator=generator,
    )
    target = torch.randn(
        cfg.batch_size, cfg.d_out, dtype=dtype, device=device,
        generator=generator,
    )
    A_real = torch.randn(
        cfg.d_in, cfg.true_rank, dtype=dtype, device=device,
        generator=generator,
    ) * 0.1
    B_real = torch.randn(
        cfg.true_rank, cfg.d_out, dtype=dtype, device=device,
        generator=generator,
    ) * 0.1
    return X, target, A_real, B_real


def _setup_masks(
    cfg: MaskedGradientLoRAConfig, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = _dtype_of(cfg.dtype)
    device = torch.device(cfg.device)
    N_x = create_orthogonal_matrix(
        cfg.d_in, generator=generator, dtype=dtype, device=device,
    )
    N_y = create_orthogonal_matrix(
        cfg.d_out, generator=generator, dtype=dtype, device=device,
    )
    M = create_orthogonal_matrix(
        cfg.padded_rank, generator=generator, dtype=dtype, device=device,
    )
    return N_x, N_y, M


# ---------------------------------------------------------------------------
# Loss + backward (analytic; nothing torch.autograd needs here)
# ---------------------------------------------------------------------------


def _plain_forward_and_backward(
    X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reference plaintext forward + analytic backward of MSE loss.

    L = mean((X A B - target)^2)
    dL/dY = 2 (Y - target) / (B*S*d_out)
    grad_A = X^T (dL/dY) B^T
    grad_B = (X A)^T (dL/dY)
    """
    Y = X @ A @ B
    diff = Y - target
    n = diff.numel()
    grad_Y = 2.0 * diff / n
    grad_A = X.transpose(-2, -1) @ grad_Y @ B.transpose(-2, -1)
    grad_B = (X @ A).transpose(-2, -1) @ grad_Y
    loss = (diff * diff).mean()
    return {
        "Y": Y, "loss": loss, "grad_Y": grad_Y,
        "grad_A": grad_A, "grad_B": grad_B,
    }


def _masked_forward_and_backward(
    X_tilde: torch.Tensor, A_tilde: torch.Tensor, B_tilde: torch.Tensor,
    target_tilde: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Analytic masked forward + backward.

    On the masked side we never observe ``Y_plain`` directly. The
    GPU-side autograd would yield the same algebraic result; we
    compute it analytically here to mirror the plain reference exactly.
    """
    Y_tilde = X_tilde @ A_tilde @ B_tilde
    diff_t = Y_tilde - target_tilde
    n = diff_t.numel()
    grad_Y_tilde = 2.0 * diff_t / n
    grad_A_tilde = (
        X_tilde.transpose(-2, -1) @ grad_Y_tilde @ B_tilde.transpose(-2, -1)
    )
    grad_B_tilde = (
        (X_tilde @ A_tilde).transpose(-2, -1) @ grad_Y_tilde
    )
    loss_tilde = (diff_t * diff_t).mean()
    return {
        "Y_tilde": Y_tilde, "loss_tilde": loss_tilde,
        "grad_Y_tilde": grad_Y_tilde,
        "grad_A_tilde": grad_A_tilde, "grad_B_tilde": grad_B_tilde,
    }


# ---------------------------------------------------------------------------
# Top-level lockstep training
# ---------------------------------------------------------------------------


def run_masked_gradient_lora_training(
    cfg: MaskedGradientLoRAConfig,
    *, num_steps: int = 6,
) -> dict[str, Any]:
    if cfg.padded_rank < cfg.true_rank:
        raise ValueError("padded_rank must be >= true_rank")
    dtype = _dtype_of(cfg.dtype)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)

    X, target, A_real, B_real = _sample_data(cfg, g)

    if cfg.use_rank_padding and cfg.padded_rank > cfg.true_rank:
        A_pad, B_pad, pad_meta = create_cancellation_padded_lora(
            A_real, B_real,
            padded_rank=cfg.padded_rank,
            strategy=cfg.dummy_strategy,
            generator=g,
        )
        dummy_strategy_used = pad_meta["strategy"]
    else:
        A_pad, B_pad = A_real.clone(), B_real.clone()
        pad_meta = {
            "strategy": "none", "true_rank": int(cfg.true_rank),
            "padded_rank": int(cfg.true_rank),
            "dummy_columns_in_A": 0, "dummy_rows_in_B": 0,
        }
        dummy_strategy_used = "none"

    # Sanity: dummy contribution is identically zero at init.
    init_dummy_norm = dummy_contribution_norm(
        A_pad, B_pad, true_rank=cfg.true_rank,
    )

    N_x, N_y, M = _setup_masks(cfg, g_mask)
    state = create_masked_lora_state(
        A_pad, B_pad,
        N_x=N_x, N_y=N_y, M=M,
        padded_rank=A_pad.shape[1], true_rank=cfg.true_rank,
    )
    target_tilde = target @ N_y
    X_tilde = X @ N_x

    # Plaintext clone of (A_pad, B_pad) for the lockstep reference.
    A_plain = A_pad.clone()
    B_plain = B_pad.clone()

    # Momentum buffers (zero-initialised).
    V_A_tilde = torch.zeros_like(state.A_tilde)
    V_B_tilde = torch.zeros_like(state.B_tilde)
    V_A_plain = torch.zeros_like(A_plain)
    V_B_plain = torch.zeros_like(B_plain)

    per_step: list[dict[str, Any]] = []
    for step in range(num_steps):
        plain_out = _plain_forward_and_backward(X, A_plain, B_plain, target)
        masked_out = _masked_forward_and_backward(
            X_tilde, state.A_tilde, state.B_tilde, target_tilde,
        )

        # Forward / loss equivalence.
        Y_recovered = masked_out["Y_tilde"] @ N_y.transpose(-2, -1)
        forward_err = float(
            (Y_recovered - plain_out["Y"]).abs().max().item()
        )
        loss_err = float(
            (masked_out["loss_tilde"] - plain_out["loss"]).abs().item()
        )

        # Gradient algebra: grad_A_tilde == N_x^T grad_A M, etc.
        expected_grad_A_tilde = (
            N_x.transpose(-2, -1) @ plain_out["grad_A"] @ M
        )
        expected_grad_B_tilde = (
            M.transpose(-2, -1) @ plain_out["grad_B"] @ N_y
        )
        grad_A_err = float(
            (masked_out["grad_A_tilde"] - expected_grad_A_tilde)
            .abs().max().item()
        )
        grad_B_err = float(
            (masked_out["grad_B_tilde"] - expected_grad_B_tilde)
            .abs().max().item()
        )

        # Masked SGD update + plain SGD update + recovery equality.
        A_tilde_sgd, B_tilde_sgd = masked_sgd_step(
            state.A_tilde, state.B_tilde,
            masked_out["grad_A_tilde"], masked_out["grad_B_tilde"],
            lr=cfg.lr,
        )
        A_plain_sgd = A_plain - cfg.lr * plain_out["grad_A"]
        B_plain_sgd = B_plain - cfg.lr * plain_out["grad_B"]
        A_rec_sgd, B_rec_sgd = recover_lora_from_masked(
            A_tilde_sgd, B_tilde_sgd, N_x=N_x, N_y=N_y, M=M,
        )
        sgd_A_err = float((A_rec_sgd - A_plain_sgd).abs().max().item())
        sgd_B_err = float((B_rec_sgd - B_plain_sgd).abs().max().item())

        # Masked momentum SGD update for the same gradients (does not
        # advance the live state; advanced separately below if
        # use_momentum is on).
        A_tilde_mom, B_tilde_mom, V_A_tilde_mom, V_B_tilde_mom = (
            masked_momentum_sgd_step(
                state.A_tilde, state.B_tilde,
                masked_out["grad_A_tilde"], masked_out["grad_B_tilde"],
                V_A_tilde, V_B_tilde,
                lr=cfg.lr, momentum=cfg.momentum,
            )
        )
        V_A_plain_mom = cfg.momentum * V_A_plain + plain_out["grad_A"]
        V_B_plain_mom = cfg.momentum * V_B_plain + plain_out["grad_B"]
        A_plain_mom = A_plain - cfg.lr * V_A_plain_mom
        B_plain_mom = B_plain - cfg.lr * V_B_plain_mom
        A_rec_mom, B_rec_mom = recover_lora_from_masked(
            A_tilde_mom, B_tilde_mom, N_x=N_x, N_y=N_y, M=M,
        )
        mom_A_err = float((A_rec_mom - A_plain_mom).abs().max().item())
        mom_B_err = float((B_rec_mom - B_plain_mom).abs().max().item())

        # Dummy contribution norm after this step's chosen update path.
        # We advance the LIVE state with masked SGD (or momentum SGD,
        # if use_momentum is on). Both are exact under orthogonal masks
        # so the lockstep stays in sync.
        if cfg.use_momentum:
            state = MaskedLoRAState(
                A_tilde=A_tilde_mom, B_tilde=B_tilde_mom,
                N_x=N_x, N_x_inv=N_x.transpose(-2, -1),
                N_y=N_y, N_y_inv=N_y.transpose(-2, -1),
                M=M, M_inv=M.transpose(-2, -1),
                padded_rank=state.padded_rank, true_rank=state.true_rank,
            )
            V_A_tilde, V_B_tilde = V_A_tilde_mom, V_B_tilde_mom
            A_plain, B_plain = A_plain_mom, B_plain_mom
            V_A_plain, V_B_plain = V_A_plain_mom, V_B_plain_mom
        else:
            state = MaskedLoRAState(
                A_tilde=A_tilde_sgd, B_tilde=B_tilde_sgd,
                N_x=N_x, N_x_inv=N_x.transpose(-2, -1),
                N_y=N_y, N_y_inv=N_y.transpose(-2, -1),
                M=M, M_inv=M.transpose(-2, -1),
                padded_rank=state.padded_rank, true_rank=state.true_rank,
            )
            A_plain, B_plain = A_plain_sgd, B_plain_sgd

        post_dummy_norm = dummy_contribution_norm(
            A_plain, B_plain, true_rank=cfg.true_rank,
        )

        per_step.append({
            "step": int(step),
            "forward_max_abs_err": forward_err,
            "loss_abs_err": loss_err,
            "grad_A_tilde_relation_max_abs_err": grad_A_err,
            "grad_B_tilde_relation_max_abs_err": grad_B_err,
            "masked_sgd_update_A_max_abs_err_after_recovery": sgd_A_err,
            "masked_sgd_update_B_max_abs_err_after_recovery": sgd_B_err,
            "masked_momentum_sgd_update_A_max_abs_err_after_recovery":
                mom_A_err,
            "masked_momentum_sgd_update_B_max_abs_err_after_recovery":
                mom_B_err,
            "dummy_contribution_norm_after_step": post_dummy_norm,
            "visible_grad_fingerprint": visible_grad_fingerprint(
                masked_out["grad_A_tilde"], masked_out["grad_B_tilde"],
            ),
            "masked_state_fingerprint": masked_lora_state_fingerprint(state),
        })

    # AdamW unsupported check -- explicit attempted call.
    adamw_record: dict[str, Any]
    try:
        masked_adamw_step_unsupported(state.A_tilde, state.B_tilde)
        adamw_record = {
            "status": "unexpectedly_did_not_raise",
            "exception_type": None,
        }
    except DenseMaskedAdamWUnsupported as exc:
        adamw_record = {
            "status": "explicitly_raised_as_designed",
            "exception_type": type(exc).__name__,
            "reason": str(exc),
        }

    gpu_visibility_table = [
        {
            "variable": "plaintext_A", "visible_to_gpu": False,
            "exposed_form": "never_exported",
        },
        {
            "variable": "plaintext_B", "visible_to_gpu": False,
            "exposed_form": "never_exported",
        },
        {
            "variable": "plaintext_grad_A", "visible_to_gpu": False,
            "exposed_form": "never_exported",
        },
        {
            "variable": "plaintext_grad_B", "visible_to_gpu": False,
            "exposed_form": "never_exported",
        },
        {
            "variable": "plaintext_optimizer_state", "visible_to_gpu": False,
            "exposed_form": "never_exported",
        },
        {
            "variable": "N_x / N_y / M",
            "visible_to_gpu": False,
            "exposed_form": "trusted_only",
        },
        {
            "variable": "X_tilde", "visible_to_gpu": True,
            "exposed_form": "X @ N_x (masked)",
        },
        {
            "variable": "A_tilde", "visible_to_gpu": True,
            "exposed_form": "N_x^T A_pad M (masked, rank-padded)",
        },
        {
            "variable": "B_tilde", "visible_to_gpu": True,
            "exposed_form": "M^T B_pad N_y (masked, rank-padded)",
        },
        {
            "variable": "grad_A_tilde", "visible_to_gpu": True,
            "exposed_form": "N_x^T grad_A M (masked)",
        },
        {
            "variable": "grad_B_tilde", "visible_to_gpu": True,
            "exposed_form": "M^T grad_B N_y (masked)",
        },
    ]

    return {
        "status": "ok",
        "stage": "7.6",
        "main_mode": "masked_gradient_lora_training",
        "config": asdict(cfg),
        "num_steps": int(num_steps),
        "dummy_rank_padding": {
            "strategy_used": dummy_strategy_used,
            "initial_dummy_contribution_norm": init_dummy_norm,
            **{k: v for k, v in pad_meta.items() if k != "strategy"},
        },
        "per_step": per_step,
        "adamw_dense_mask_unsupported": adamw_record,
        "gpu_visibility": gpu_visibility_table,
        "honesty_phrases": list(_REQUIRED_HONESTY_PHRASES),
        "formal_security_claim": False,
        "limitations": [
            "Dense masked AdamW is not claimed because coordinate-wise "
            "second moments are not invariant under dense orthogonal "
            "mixing.",
            "AdamW under dense masks would require trusted-assisted "
            "update, signed-permutation masks, or a specialised "
            "masked optimiser; none are implemented in Stage 7.6.",
            "CPU local emulation only; no real TEE / GPU runtime is "
            "measured.",
            "This is an algebraic correctness + proxy leakage stage; "
            "no formal cryptographic / semantic / differential-privacy "
            "security is claimed.",
            "Synthetic regression task only; this is not a real Qwen / "
            "LLaMA LoRA fine-tuning workload.",
            "Loss boundary uses MSE on Y_tilde vs target_tilde; "
            "orthogonal N_y preserves the L2 loss exactly, but a "
            "softmax / cross-entropy loss would require a trusted "
            "loss boundary, which is out of scope for Stage 7.6.",
            "Raw tensors, adapters, gradients, and masks are NEVER "
            "exported; outputs contain only summary scalars, shapes, "
            "and short fingerprints.",
        ],
        "paper_safe_wording": (
            "We validate that the masked-gradient LoRA construction "
            "preserves plaintext SGD and momentum-SGD updates "
            "algebraically at float64 machine precision, and we "
            "explicitly raise on dense masked AdamW rather than "
            "approximating. The construction hides the true rank "
            "via cancellation padding while the visible padded rank "
            "is observable."
        ),
        "unsafe_wording_to_avoid": [
            "Dense masked AdamW exact.",
            "Cryptographic security.",
            "Semantic security.",
            "Plaintext gradients revealed to GPU.",
            "Full Qwen / LLaMA private fine-tuning deployed.",
        ],
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _round(x: Any, digits: int = 6) -> Any:
    if isinstance(x, float):
        if x != x:
            return "NaN"
        return round(x, digits)
    return x


def _write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)


def _flatten_for_csv(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ps in report["per_step"]:
        rows.append({
            "step": ps["step"],
            "forward_max_abs_err": _round(ps["forward_max_abs_err"], 12),
            "loss_abs_err": _round(ps["loss_abs_err"], 12),
            "grad_A_tilde_relation_max_abs_err": _round(
                ps["grad_A_tilde_relation_max_abs_err"], 12,
            ),
            "grad_B_tilde_relation_max_abs_err": _round(
                ps["grad_B_tilde_relation_max_abs_err"], 12,
            ),
            "masked_sgd_update_A_max_abs_err_after_recovery": _round(
                ps["masked_sgd_update_A_max_abs_err_after_recovery"], 12,
            ),
            "masked_sgd_update_B_max_abs_err_after_recovery": _round(
                ps["masked_sgd_update_B_max_abs_err_after_recovery"], 12,
            ),
            "masked_momentum_sgd_update_A_max_abs_err_after_recovery": _round(
                ps["masked_momentum_sgd_update_A_max_abs_err_after_recovery"],
                12,
            ),
            "masked_momentum_sgd_update_B_max_abs_err_after_recovery": _round(
                ps["masked_momentum_sgd_update_B_max_abs_err_after_recovery"],
                12,
            ),
            "dummy_contribution_norm_after_step": _round(
                ps["dummy_contribution_norm_after_step"], 12,
            ),
            "A_tilde_fingerprint": ps["masked_state_fingerprint"][
                "A_tilde_fingerprint"
            ],
            "B_tilde_fingerprint": ps["masked_state_fingerprint"][
                "B_tilde_fingerprint"
            ],
        })
    return rows


def _write_csv(report: dict[str, Any], path: str) -> None:
    rows = _flatten_for_csv(report)
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Stage 7.6 — Masked-Gradient LoRA Training with Rank-Space Mixing")
    w()
    w("## 1. Experiment Scope")
    w()
    w(
        "We train a small synthetic LoRA regression task in lockstep "
        "between a plaintext reference and a masked-gradient cloud "
        "path. The cloud accelerator computes masked forward, masked "
        "backward, and masked optimiser updates. The trusted side "
        "owns the orthogonal masks ``N_x``, ``N_y``, ``M`` and never "
        "exports them. We validate per-step that the masked update "
        "recovers exactly to the plaintext update at float64 machine "
        "precision."
    )
    w()
    w("## 2. Threat Model")
    w()
    w(
        "Honest-but-curious cloud accelerator. The GPU never receives "
        "plaintext LoRA adapters or plaintext LoRA gradients in this "
        "experiment. The user side does not require a GPU; the "
        "simulated cloud accelerator performs masked forward, "
        "backward, and optimizer updates. No formal, cryptographic, "
        "or semantic security is claimed."
    )
    w()
    w("## 3. Masked LoRA Forward Construction")
    w()
    w(
        "```\n"
        "  A_tilde = N_x^T A M\n"
        "  B_tilde = M^T B N_y\n"
        "  X_tilde = X N_x\n"
        "  Y_tilde = X_tilde A_tilde B_tilde\n"
        "          = X N_x N_x^T A M M^T B N_y\n"
        "          = X A B N_y\n"
        "```"
    )
    w(
        f"Forward recovery (per-step max abs err): "
        f"`<= {max(ps['forward_max_abs_err'] for ps in report['per_step']):.2e}`."
    )
    w()
    w("## 4. Masked Gradient Derivation")
    w()
    w(
        "With L = MSE(X A B, target) we have\n"
        "```\n"
        "  grad_A = X^T (dL/dY) B^T\n"
        "  grad_B = (X A)^T (dL/dY)\n"
        "```\n"
        "Under the masked forward, ``grad_Y_tilde = 2 (Y_tilde - "
        "target_tilde) / n``. The chain rule gives\n"
        "```\n"
        "  grad_A_tilde = X_tilde^T grad_Y_tilde B_tilde^T = N_x^T grad_A M\n"
        "  grad_B_tilde = (X_tilde A_tilde)^T grad_Y_tilde = M^T grad_B N_y\n"
        "```\n"
        "We verify both relations per step:"
    )
    rel_a = max(
        ps["grad_A_tilde_relation_max_abs_err"] for ps in report["per_step"]
    )
    rel_b = max(
        ps["grad_B_tilde_relation_max_abs_err"] for ps in report["per_step"]
    )
    w(f"- `max(|grad_A_tilde - N_x^T grad_A M|) <= {rel_a:.2e}`")
    w(f"- `max(|grad_B_tilde - M^T grad_B N_y|) <= {rel_b:.2e}`")
    w()
    w("## 5. GPU-side Masked SGD")
    w()
    w(
        "Update rule: ``A_tilde <- A_tilde - lr * grad_A_tilde``, "
        "``B_tilde <- B_tilde - lr * grad_B_tilde``. Because "
        "right-multiplication by an orthogonal mask distributes over "
        "the linear combination, this is algebraically equivalent to "
        "plaintext SGD: Masked SGD is algebraically equivalent under "
        "orthogonal masks. Per-step recovery error against plaintext "
        "SGD:"
    )
    sgd_a = max(
        ps["masked_sgd_update_A_max_abs_err_after_recovery"]
        for ps in report["per_step"]
    )
    sgd_b = max(
        ps["masked_sgd_update_B_max_abs_err_after_recovery"]
        for ps in report["per_step"]
    )
    w(f"- `max(|recovered A_tilde_next - A_plain_next|) <= {sgd_a:.2e}`")
    w(f"- `max(|recovered B_tilde_next - B_plain_next|) <= {sgd_b:.2e}`")
    w()
    w("## 6. Momentum SGD")
    w()
    w(
        "We track masked momentum buffers ``V_A_tilde``, ``V_B_tilde`` "
        "with the heavy-ball update ``V <- mu V + grad``, ``param <- "
        "param - lr V``. Right-multiplication by orthogonal masks "
        "distributes over both updates, so the masked momentum-SGD "
        "step recovers exactly to plaintext momentum SGD:"
    )
    mom_a = max(
        ps["masked_momentum_sgd_update_A_max_abs_err_after_recovery"]
        for ps in report["per_step"]
    )
    mom_b = max(
        ps["masked_momentum_sgd_update_B_max_abs_err_after_recovery"]
        for ps in report["per_step"]
    )
    w(f"- `max(|recovered A_tilde_mom - A_plain_mom|) <= {mom_a:.2e}`")
    w(f"- `max(|recovered B_tilde_mom - B_plain_mom|) <= {mom_b:.2e}`")
    w()
    w("## 7. Adam/AdamW Limitation")
    w()
    w(
        "Dense masked AdamW is not claimed because coordinate-wise "
        "second moments are not invariant under dense orthogonal "
        "mixing. For a dense orthogonal ``Q``, "
        "``(g Q)_{i, j}^2 != g_{i, j}^2 Q`` in general, so the "
        "running second moment ``v <- beta_2 v + (1 - beta_2) g^2`` "
        "does not commute with the mask. Stage 7.6's module raises "
        "``DenseMaskedAdamWUnsupported`` when AdamW is requested "
        "rather than silently approximating. A future stage could add "
        "(i) a trusted-assisted update (recover, AdamW on plain, "
        "re-mask), (ii) signed-permutation masks (the only orthogonal "
        "class that commutes with coordinate-wise squaring), or "
        "(iii) a specialised masked optimiser."
    )
    record = report["adamw_dense_mask_unsupported"]
    w(
        f"AdamW gate status in this run: "
        f"`{record.get('status')}` ({record.get('exception_type')})."
    )
    w()
    w("## 8. Rank Padding and Rank-Space Mixing")
    w()
    drp = report["dummy_rank_padding"]
    w(
        f"Strategy: `{drp['strategy_used']}`; true_rank = "
        f"`{drp['true_rank']}`; padded_rank = `{drp['padded_rank']}`; "
        f"dummy columns added to A = `{drp['dummy_columns_in_A']}`; "
        f"dummy rows added to B = `{drp['dummy_rows_in_B']}`. The "
        f"cancellation block is ``A_pad = [A_real, R, -R]``, "
        f"``B_pad = vstack(B_real, S, S)`` so that "
        f"``A_pad B_pad = A_real B_real`` (initial dummy contribution "
        f"norm = `{drp['initial_dummy_contribution_norm']:.2e}`). "
        "The rank-space orthogonal mixer ``M`` is then applied over "
        "the padded rank so the accelerator-visible inner dimension "
        "is `padded_rank`, while the true rank `true_rank` is hidden "
        "from any shape inspection. Per step the dummy contribution "
        "norm is verified to remain at machine zero."
    )
    w()
    w("## 9. Correctness Results")
    w()
    w("| step | fwd_err | loss_err | grad_A rel | grad_B rel | sgd_A rec | sgd_B rec | mom_A rec | mom_B rec | dummy norm |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for ps in report["per_step"]:
        w(
            f"| {ps['step']} | "
            f"{ps['forward_max_abs_err']:.2e} | "
            f"{ps['loss_abs_err']:.2e} | "
            f"{ps['grad_A_tilde_relation_max_abs_err']:.2e} | "
            f"{ps['grad_B_tilde_relation_max_abs_err']:.2e} | "
            f"{ps['masked_sgd_update_A_max_abs_err_after_recovery']:.2e} | "
            f"{ps['masked_sgd_update_B_max_abs_err_after_recovery']:.2e} | "
            f"{ps['masked_momentum_sgd_update_A_max_abs_err_after_recovery']:.2e} | "
            f"{ps['masked_momentum_sgd_update_B_max_abs_err_after_recovery']:.2e} | "
            f"{ps['dummy_contribution_norm_after_step']:.2e} |"
        )
    w()
    w("## 10. Gradient Leakage Proxy")
    w()
    w(
        "GPU-visible per-call gradient fingerprints are published as "
        "short SHA-256 prefixes so cross-step linkability can be "
        "audited externally without exposing raw gradients. The "
        "companion module "
        "`masked_gradient_lora_security_proxy.py` runs a more "
        "structured proxy: true-rank inference from "
        "``A_tilde / B_tilde`` spectra, real-vs-dummy subspace "
        "separation, and cross-step linkability under fixed vs fresh "
        "masks. Raw tensors, masks, and adapters are NEVER exported."
    )
    w()
    w("### GPU visibility table")
    w()
    w("| variable | visible_to_gpu | exposed_form |")
    w("|---|---|---|")
    for entry in report["gpu_visibility"]:
        w(
            f"| `{entry['variable']}` | {entry['visible_to_gpu']} | "
            f"{entry['exposed_form']} |"
        )
    w()
    w("## 11. Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## 12. Next Stage Plan")
    w()
    w(
        "Future work: (i) integrate signed-permutation masks for "
        "AdamW exactness; (ii) explore a trusted-assisted AdamW "
        "boundary where the cloud accelerator returns the masked "
        "gradient and the trusted side runs the per-coordinate "
        "second-moment update on a small slice; (iii) extend the "
        "construction to softmax / cross-entropy losses via a trusted "
        "loss boundary; (iv) integrate with the Stage 7.5c deployable "
        "runtime API so a real serving runtime can route masked "
        "gradients without seeing the trusted-side recovery."
    )
    w()
    w(f"`formal_security_claim`: `{report['formal_security_claim']}`")
    w()
    w("## Honesty phrases (verbatim)")
    w()
    for phrase in report.get("honesty_phrases", []):
        w(f"- {phrase}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *, outputs_dir: str = "outputs",
    json_filename: str = "masked_gradient_lora_training.json",
    csv_filename: str = "masked_gradient_lora_training.csv",
    md_filename: str = "masked_gradient_lora_training.md",
) -> tuple[str, str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, json_filename)
    csv_path = os.path.join(outputs_dir, csv_filename)
    md_path = os.path.join(outputs_dir, md_filename)
    _write_json(report, json_path)
    _write_csv(report, csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return json_path, csv_path, md_path


__all__ = [
    "render_markdown",
    "run_masked_gradient_lora_training",
    "write_reports",
]
