"""Stage 7.2 — Rank-padded LoRA rank-leakage security proxy.

Three sub-attacks evaluate residual rank leakage when Stage 7.2 rank
padding is enabled, compared to the Stage 7.0 / 7.1 baseline without
padding:

1. **Shape-level rank leakage** — what an attacker reads directly off the
   GPU-visible tensor dimensions of ``A_tilde`` / ``B_tilde`` /
   ``grad_A_tilde`` / ``grad_B_tilde``. Without padding this is the true
   ``r``; with padding this is ``r_pad ≥ r`` and ``r`` is hidden from
   shape.

2. **Spectral rank inference** — an attacker who sees ``A_tilde_pad`` /
   ``B_tilde_pad`` / ``grad_A_tilde_pad`` / ``grad_B_tilde_pad`` runs
   SVD and looks for the singular-value drop. Reports the inferred rank,
   the gap between consecutive singular values around the drop, and an
   accuracy metric over ``true_ranks``. ``zero_dummy`` is expected to
   leak the true rank exactly (the dummy rows of ``B_pad`` are zero);
   ``paired_cancellation_dummy`` reduces this signal — but the
   pair-induced rank-1 contributions remain detectable, so the proxy
   reports `needs_more_evaluation` rather than `low` risk.

3. **Membership-style linkability** — for two private samples
   ``X1, X2``, compare visible padded-adapter / padded-gradient
   transcripts under "fixed masks" vs "fresh masks". This reuses the
   Stage 7.0 / 7.1 proxy methodology but inside the padded shape.

All metrics are illustrative. The masking + padding equations alone do
not bound a real attacker; this is a proxy for ranking strategies, not
a security proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    create_masked_lora_state,
    init_lora_adapters,
)
from pllo.ops.lora_backward import (
    masked_lora_backward,
    transform_upstream_gradient,
)
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    VALID_DUMMY_STRATEGIES,
    _effective_alpha,
    create_rank_padded_lora_adapters,
    validate_rank_padding_config,
)


VALID_RANK_STRATEGIES: tuple[str, ...] = ("no_padding", "rank_padding")


@dataclass
class LoRARankSecurityProxyConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    d_in: int = 32
    d_out: int = 16
    true_ranks: tuple[int, ...] = (2, 4, 8)
    padded_rank: int = 16
    alpha: float = 1.0
    use_bias: bool = True
    num_trials: int = 64
    pad_scale: float = 1.0
    dummy_scale: float = 1.0
    dummy_strategy: str = "paired_cancellation_dummy"
    use_pad: bool = True
    dtype: str = "float64"
    device: str = "cpu"
    membership_trials_per_sample: int = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sv_threshold_rank(t: torch.Tensor, *, tol: float = 1e-6) -> int:
    """Inferred numerical rank via singular-value threshold."""
    if t.numel() == 0:
        return 0
    sv = torch.linalg.svdvals(t)
    if sv.numel() == 0:
        return 0
    sv_max = sv.max().item()
    if sv_max <= 0.0:
        return 0
    return int((sv > sv_max * tol).sum().item())


def _largest_sv_gap_index(t: torch.Tensor) -> tuple[int, float]:
    """Position of the largest relative drop ``σ_{i+1} / σ_i`` in the
    singular-value spectrum. Returns ``(inferred_rank, gap_ratio)`` where
    ``inferred_rank = i + 1`` (number of "above-cliff" singular values).
    """
    if t.numel() == 0:
        return 0, 0.0
    sv = torch.linalg.svdvals(t)
    if sv.numel() < 2:
        return int(sv.numel()), 1.0
    # Drop ratio between successive σ_i, σ_{i+1}.
    ratios = (sv[1:] / sv[:-1].clamp_min(1e-30)).cpu()
    # We want the most aggressive drop: smallest ratio (closest to 0)
    # → that's the cliff.
    drop_idx = int(torch.argmin(ratios).item())
    inferred_rank = drop_idx + 1
    cliff_ratio = float(ratios[drop_idx].item())
    return inferred_rank, cliff_ratio


def _build_padded_state(
    d_in: int, d_out: int, padded_rank: int,
    dtype: torch.dtype, device: torch.device,
    *, use_pad: bool, pad_scale: float, seq_len: int,
    generator: torch.Generator,
    fresh_state: LoRAState | None,
    fresh_masks_per_call: bool, fresh_u_per_call: bool,
) -> LoRAState:
    inner_cfg = LoRAConfig(
        d_in=d_in, d_out=d_out, rank=padded_rank, alpha=1.0,
        dtype="float64" if dtype == torch.float64 else "float32",
        device=device.type,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=use_pad, fresh_u_per_call=fresh_u_per_call,
        fresh_masks_per_call=fresh_masks_per_call,
        pad_scale=pad_scale,
        dtype=inner_cfg.dtype, device=inner_cfg.device,
    )
    return create_masked_lora_state(
        inner_cfg, fcfg, seq_len=seq_len, state=fresh_state, generator=generator,
    )


def _mask_padded_adapter(
    a_pad: torch.Tensor, b_pad: torch.Tensor, state: LoRAState,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_tilde = state.n_in_inv @ a_pad @ state.u
    b_tilde = state.u_inv @ b_pad @ state.n_out
    return a_tilde, b_tilde


# ---------------------------------------------------------------------------
# Proxy 1 — shape-level rank leakage
# ---------------------------------------------------------------------------


def _shape_level_rank_leakage(
    true_ranks: tuple[int, ...], padded_rank: int,
) -> dict[str, Any]:
    no_padding: list[dict[str, Any]] = []
    rank_padding: list[dict[str, Any]] = []
    for r in true_ranks:
        no_padding.append({
            "true_rank": r,
            "rank_visible_from_a_shape": r,
            "rank_visible_from_b_shape": r,
            "rank_visible_from_grad_shape": r,
            "exposed_rank_value": r,
            "true_rank_hidden_from_shape": False,
        })
        rank_padding.append({
            "true_rank": r,
            "rank_visible_from_a_shape": padded_rank,
            "rank_visible_from_b_shape": padded_rank,
            "rank_visible_from_grad_shape": padded_rank,
            "exposed_rank_value": padded_rank,
            "true_rank_hidden_from_shape": True,
        })
    return {
        "no_padding": no_padding,
        "rank_padding": rank_padding,
        "interpretation": (
            "Without padding, the rank dimension of A_tilde / B_tilde /"
            " grad_A_tilde / grad_B_tilde equals the true LoRA rank r."
            " With rank padding, the GPU-visible rank dimension is the"
            " padded rank r_pad and r is hidden from shape."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 2 — spectral rank inference
# ---------------------------------------------------------------------------


def _spectral_rank_inference(
    config: LoRARankSecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    for true_rank in config.true_ranks:
        gen = torch.Generator(device="cpu").manual_seed(
            config.seed * 31 + true_rank
        )
        a_real_cfg = LoRAConfig(
            d_in=config.d_in, d_out=config.d_out, rank=true_rank,
            alpha=config.alpha, use_bias=config.use_bias,
            dtype=config.dtype, device=config.device,
        )
        rpc = RankPaddingConfig(
            true_rank=true_rank, padded_rank=config.padded_rank,
            dummy_scale=config.dummy_scale, dummy_strategy=config.dummy_strategy,
            fresh_dummy_per_step=True,
            dtype=config.dtype, device=config.device,
        )
        validate_rank_padding_config(rpc)

        no_padding_infer: list[int] = []
        no_padding_cliff: list[float] = []
        rank_padding_a_infer: list[int] = []
        rank_padding_b_infer: list[int] = []
        rank_padding_a_cliff: list[float] = []
        rank_padding_b_cliff: list[float] = []
        rank_padding_grad_b_infer: list[int] = []
        rank_padding_grad_b_cliff: list[float] = []

        seq_len = max(4, config.d_in // 4)
        for _ in range(config.num_trials):
            a_real, b_real = init_lora_adapters(a_real_cfg, generator=gen)
            b_real = b_real + 0.1 * torch.randn(
                true_rank, config.d_out, generator=gen,
                dtype=dtype, device=device,
            )
            # No padding baseline — masked rank-r adapter is just the
            # Stage 7.0 path, here with rank=true_rank.
            state_np = _build_padded_state(
                config.d_in, config.d_out, true_rank,
                dtype, device,
                use_pad=config.use_pad, pad_scale=config.pad_scale,
                seq_len=seq_len, generator=gen,
                fresh_state=None,
                fresh_masks_per_call=True, fresh_u_per_call=True,
            )
            a_tilde_np = state_np.n_in_inv @ a_real @ state_np.u
            b_tilde_np = state_np.u_inv @ b_real @ state_np.n_out
            inferred_np_b, cliff_np_b = _largest_sv_gap_index(b_tilde_np)
            no_padding_infer.append(inferred_np_b)
            no_padding_cliff.append(cliff_np_b)

            # Rank padding.
            pack = create_rank_padded_lora_adapters(
                a_real, b_real, rpc, generator=gen,
            )
            a_pad, b_pad = pack["a_pad"], pack["b_pad"]
            state_pad = _build_padded_state(
                config.d_in, config.d_out, config.padded_rank,
                dtype, device,
                use_pad=config.use_pad, pad_scale=config.pad_scale,
                seq_len=seq_len, generator=gen,
                fresh_state=None,
                fresh_masks_per_call=True, fresh_u_per_call=True,
            )
            a_tilde_pad, b_tilde_pad = _mask_padded_adapter(
                a_pad, b_pad, state_pad,
            )
            inferred_a, cliff_a = _largest_sv_gap_index(a_tilde_pad)
            inferred_b, cliff_b = _largest_sv_gap_index(b_tilde_pad)
            rank_padding_a_infer.append(inferred_a)
            rank_padding_b_infer.append(inferred_b)
            rank_padding_a_cliff.append(cliff_a)
            rank_padding_b_cliff.append(cliff_b)

            # Gradient-side spectral inference.
            x = torch.randn(seq_len, config.d_in, generator=gen, dtype=dtype, device=device)
            g = torch.randn(seq_len, config.d_out, generator=gen, dtype=dtype, device=device)
            if state_pad.pad is None:
                x_tilde = x @ state_pad.n_in
            else:
                x_tilde = (x - state_pad.pad) @ state_pad.n_in
            grad_y_tilde = transform_upstream_gradient(g, state_pad.n_out)
            masked = masked_lora_backward(
                x_tilde, a_tilde_pad, b_tilde_pad, grad_y_tilde,
                alpha=_effective_alpha(config.alpha, true_rank, config.padded_rank),
                recover_grad_x=False,
            )
            grad_b_tilde_pad = masked["grad_b_tilde"]
            inferred_gb, cliff_gb = _largest_sv_gap_index(grad_b_tilde_pad)
            rank_padding_grad_b_infer.append(inferred_gb)
            rank_padding_grad_b_cliff.append(cliff_gb)

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / max(1, len(xs)))

        def _mean_int(xs: list[int]) -> float:
            return _mean([float(v) for v in xs])

        rank_padding_inferred_min_ab = [
            min(a, b) for a, b in zip(rank_padding_a_infer, rank_padding_b_infer)
        ]
        rank_padding_acc = sum(
            1 for v in rank_padding_inferred_min_ab if v == true_rank
        ) / max(1, len(rank_padding_inferred_min_ab))
        no_padding_acc = sum(
            1 for v in no_padding_infer if v == true_rank
        ) / max(1, len(no_padding_infer))

        # Build the risk verdict conservatively per requirement 12.
        if config.dummy_strategy == "zero_dummy":
            # zero_dummy makes spectral rank readable from B exactly.
            risk_level = "high"
            verdict = (
                "zero_dummy: B_pad has zero dummy rows, so the spectral"
                " rank of B_pad_tilde equals true rank exactly. Spectral"
                " attacker recovers true rank."
            )
        elif rank_padding_acc >= 0.5:
            risk_level = "high"
            verdict = (
                "Spectral attacker recovers true rank in ≥ 50% of trials"
                " under this dummy strategy. Padding does not hide rank"
                " from spectral inference."
            )
        elif rank_padding_acc >= 0.20:
            risk_level = "medium"
            verdict = (
                "Spectral attacker recovers true rank between 20-50% of"
                " trials; needs_more_evaluation — dummy strategy partially"
                " obscures but does not eliminate rank inference."
            )
        else:
            # Even with low accuracy, the inferred rank is an *upper*
            # bound that may still leak structural info (e.g. r + #pairs).
            risk_level = "needs_more_evaluation"
            verdict = (
                "Spectral attacker recovers true rank in fewer than 20% of"
                " trials. The inferred rank may still be an upper bound"
                " constrained by paired-cancellation structure; this is a"
                " proxy result, not a formal claim."
            )

        rows.append({
            "true_rank": true_rank,
            "no_padding": {
                "inferred_rank_mean": _mean_int(no_padding_infer),
                "rank_inference_accuracy": float(no_padding_acc),
                "mean_rank_error": _mean_int(
                    [abs(v - true_rank) for v in no_padding_infer]
                ),
                "confidence_gap": _mean(no_padding_cliff),
            },
            "rank_padding": {
                "inferred_rank_from_a_tilde_pad_mean": _mean_int(rank_padding_a_infer),
                "inferred_rank_from_b_tilde_pad_mean": _mean_int(rank_padding_b_infer),
                "inferred_rank_from_grad_b_tilde_pad_mean": _mean_int(
                    rank_padding_grad_b_infer
                ),
                "rank_inference_accuracy": float(rank_padding_acc),
                "mean_rank_error": _mean_int(
                    [abs(v - true_rank) for v in rank_padding_inferred_min_ab]
                ),
                "confidence_gap_a": _mean(rank_padding_a_cliff),
                "confidence_gap_b": _mean(rank_padding_b_cliff),
                "confidence_gap_grad_b": _mean(rank_padding_grad_b_cliff),
                "risk_level": risk_level,
                "verdict": verdict,
            },
        })
    return {
        "rows": rows,
        "interpretation": (
            "Spectral rank inference threshold: largest σ_{i+1}/σ_i drop"
            " in the singular-value spectrum. The inferred rank is the"
            " index just before the cliff. For zero_dummy this aligns"
            " with the true rank exactly (B_pad has zero rows). For"
            " paired_cancellation_dummy the inferred rank is bounded by"
            " true_rank + ⌊(r_pad - r) / 2⌋ + leftover_zero."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 3 — gradient-side spectral inference (same shape attacker
#         but driving the analysis from grad_A / grad_B)
# ---------------------------------------------------------------------------


def _gradient_rank_inference(
    config: LoRARankSecurityProxyConfig,
) -> dict[str, Any]:
    """Gradient-only spectral inference.

    Mirrors the spectral inference proxy above but reports the
    grad-tensor-only metrics in a separate section so the Markdown can
    surface them under "Gradient Rank Inference".
    """
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    for true_rank in config.true_ranks:
        gen = torch.Generator(device="cpu").manual_seed(
            config.seed * 53 + true_rank
        )
        a_real_cfg = LoRAConfig(
            d_in=config.d_in, d_out=config.d_out, rank=true_rank,
            alpha=config.alpha, use_bias=config.use_bias,
            dtype=config.dtype, device=config.device,
        )
        rpc = RankPaddingConfig(
            true_rank=true_rank, padded_rank=config.padded_rank,
            dummy_scale=config.dummy_scale, dummy_strategy=config.dummy_strategy,
            fresh_dummy_per_step=True,
            dtype=config.dtype, device=config.device,
        )
        seq_len = max(4, config.d_in // 4)
        grad_a_inferred: list[int] = []
        grad_b_inferred: list[int] = []
        for _ in range(config.num_trials):
            a_real, b_real = init_lora_adapters(a_real_cfg, generator=gen)
            b_real = b_real + 0.1 * torch.randn(
                true_rank, config.d_out, generator=gen,
                dtype=dtype, device=device,
            )
            pack = create_rank_padded_lora_adapters(
                a_real, b_real, rpc, generator=gen,
            )
            state_pad = _build_padded_state(
                config.d_in, config.d_out, config.padded_rank,
                dtype, device,
                use_pad=config.use_pad, pad_scale=config.pad_scale,
                seq_len=seq_len, generator=gen,
                fresh_state=None,
                fresh_masks_per_call=True, fresh_u_per_call=True,
            )
            a_tilde, b_tilde = _mask_padded_adapter(
                pack["a_pad"], pack["b_pad"], state_pad,
            )
            x = torch.randn(seq_len, config.d_in, generator=gen, dtype=dtype, device=device)
            g = torch.randn(seq_len, config.d_out, generator=gen, dtype=dtype, device=device)
            if state_pad.pad is None:
                x_tilde = x @ state_pad.n_in
            else:
                x_tilde = (x - state_pad.pad) @ state_pad.n_in
            grad_y_tilde = transform_upstream_gradient(g, state_pad.n_out)
            masked = masked_lora_backward(
                x_tilde, a_tilde, b_tilde, grad_y_tilde,
                alpha=_effective_alpha(config.alpha, true_rank, config.padded_rank),
                recover_grad_x=False,
            )
            inferred_a, _ = _largest_sv_gap_index(masked["grad_a_tilde"])
            inferred_b, _ = _largest_sv_gap_index(masked["grad_b_tilde"])
            grad_a_inferred.append(inferred_a)
            grad_b_inferred.append(inferred_b)

        def _mean_int(xs: list[int]) -> float:
            return float(sum(xs) / max(1, len(xs)))

        acc = sum(
            1 for a, b in zip(grad_a_inferred, grad_b_inferred)
            if min(a, b) == true_rank
        ) / max(1, len(grad_a_inferred))
        if config.dummy_strategy == "zero_dummy" or acc >= 0.5:
            risk = "high"
        elif acc >= 0.20:
            risk = "medium"
        else:
            risk = "needs_more_evaluation"
        rows.append({
            "true_rank": true_rank,
            "inferred_rank_from_grad_a_tilde_pad_mean": _mean_int(grad_a_inferred),
            "inferred_rank_from_grad_b_tilde_pad_mean": _mean_int(grad_b_inferred),
            "rank_inference_accuracy": float(acc),
            "mean_rank_error_min_ab": _mean_int(
                [abs(min(a, b) - true_rank)
                 for a, b in zip(grad_a_inferred, grad_b_inferred)]
            ),
            "risk_level": risk,
        })
    return {
        "rows": rows,
        "interpretation": (
            "Gradient-side spectral rank inference uses the same SVD-cliff"
            " detector but on grad_A_tilde_pad / grad_B_tilde_pad. Because"
            " gradients depend on (X, A, B, G), the spectrum may differ"
            " from the static adapter spectrum and warrants a separate"
            " section."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 4 — membership / linkability over the padded transcript
# ---------------------------------------------------------------------------


def _membership_linkability_padded(
    config: LoRARankSecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    for true_rank in config.true_ranks:
        gen = torch.Generator(device="cpu").manual_seed(
            config.seed * 71 + true_rank
        )
        rpc = RankPaddingConfig(
            true_rank=true_rank, padded_rank=config.padded_rank,
            dummy_scale=config.dummy_scale, dummy_strategy=config.dummy_strategy,
            fresh_dummy_per_step=True,
            dtype=config.dtype, device=config.device,
        )
        validate_rank_padding_config(rpc)
        a_real_cfg = LoRAConfig(
            d_in=config.d_in, d_out=config.d_out, rank=true_rank,
            alpha=config.alpha, use_bias=config.use_bias,
            dtype=config.dtype, device=config.device,
        )
        a_real, b_real = init_lora_adapters(a_real_cfg, generator=gen)
        b_real = b_real + 0.1 * torch.randn(
            true_rank, config.d_out, generator=gen, dtype=dtype, device=device,
        )
        seq_len = max(4, config.d_in // 4)
        x1 = torch.randn(seq_len, config.d_in, generator=gen, dtype=dtype, device=device)
        x2 = torch.randn(seq_len, config.d_in, generator=gen, dtype=dtype, device=device)
        g1 = torch.randn(seq_len, config.d_out, generator=gen, dtype=dtype, device=device)
        g2 = torch.randn(seq_len, config.d_out, generator=gen, dtype=dtype, device=device)

        def _visible_transcript(x: torch.Tensor, g: torch.Tensor):
            pack = create_rank_padded_lora_adapters(
                a_real, b_real, rpc, generator=gen,
            )
            state = _build_padded_state(
                config.d_in, config.d_out, config.padded_rank,
                dtype, device,
                use_pad=config.use_pad, pad_scale=config.pad_scale,
                seq_len=seq_len, generator=gen,
                fresh_state=None,
                fresh_masks_per_call=True, fresh_u_per_call=True,
            )
            a_tilde, b_tilde = _mask_padded_adapter(
                pack["a_pad"], pack["b_pad"], state,
            )
            if state.pad is None:
                x_tilde = x @ state.n_in
            else:
                x_tilde = (x - state.pad) @ state.n_in
            grad_y_tilde = transform_upstream_gradient(g, state.n_out)
            masked = masked_lora_backward(
                x_tilde, a_tilde, b_tilde, grad_y_tilde,
                alpha=_effective_alpha(config.alpha, true_rank, config.padded_rank),
                recover_grad_x=False,
            )
            return a_tilde, b_tilde, masked["grad_a_tilde"], masked["grad_b_tilde"]

        def _distance(p1, p2) -> float:
            return float(
                sum(((p1[k] - p2[k]).norm() ** 2).item() for k in range(4))
                ** 0.5
            )

        trials_per_sample = config.membership_trials_per_sample
        p1 = [_visible_transcript(x1, g1) for _ in range(trials_per_sample)]
        p2 = [_visible_transcript(x2, g2) for _ in range(trials_per_sample)]
        same: list[float] = []
        diff: list[float] = []
        for i in range(trials_per_sample):
            for j in range(i + 1, trials_per_sample):
                same.append(_distance(p1[i], p1[j]))
                same.append(_distance(p2[i], p2[j]))
        for i in range(trials_per_sample):
            for j in range(trials_per_sample):
                diff.append(_distance(p1[i], p2[j]))

        wins = 0
        total = 0
        for s in same:
            for d in diff:
                if s < d:
                    wins += 1
                elif s == d:
                    wins += 0.5
                total += 1
        auc = float(wins / max(1, total))
        linkability_rank = 2.0 * abs(auc - 0.5)
        if auc > 0.85:
            risk = "high"
        elif auc > 0.65:
            risk = "medium"
        else:
            risk = "low"
        rows.append({
            "true_rank": true_rank,
            "same_sample_distance_mean": float(sum(same) / max(1, len(same))),
            "different_sample_distance_mean": float(sum(diff) / max(1, len(diff))),
            "membership_auc_proxy": auc,
            "linkability_rank": linkability_rank,
            "risk_level": risk,
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are rank-leakage proxy attacks, not formal security proofs.",
    "True rank is hidden only from shape-level leakage when rank padding is enabled. Padded rank r_pad remains visible.",
    "Spectral / statistical rank inference may still be possible depending on dummy strategy. zero_dummy explicitly leaks true rank via SVD of B_pad.",
    "paired_cancellation_dummy reduces obvious zero-norm leakage but the spectral upper bound `true_rank + ⌊(r_pad - r) / 2⌋` still narrows the attacker's range; this is reported as needs_more_evaluation, not low.",
    "No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "Optimizer state remains trusted-only and is sized to true_rank.",
    "Hardware side-channels (cache / power / EM) are NOT evaluated.",
    "No full model LoRA fine-tuning is evaluated; this is a single-linear, tiny-dimension proxy.",
    "Adapter is NEVER merged into the public base weight W.",
]


def run_lora_rank_security_proxy(
    config: LoRARankSecurityProxyConfig,
) -> dict[str, Any]:
    if config.dummy_strategy not in VALID_DUMMY_STRATEGIES:
        raise ValueError(
            f"unknown dummy_strategy {config.dummy_strategy!r};"
            f" expected one of {VALID_DUMMY_STRATEGIES}"
        )
    for r in config.true_ranks:
        if r <= 0 or r > config.padded_rank:
            raise ValueError(
                f"each true_rank must satisfy 0 < r <= padded_rank"
                f" ({config.padded_rank}), got {r}"
            )

    shape = _shape_level_rank_leakage(config.true_ranks, config.padded_rank)
    spectral = _spectral_rank_inference(config)
    gradient = _gradient_rank_inference(config)
    membership = _membership_linkability_padded(config)

    # Summary verdict — pick the strongest of the per-true-rank verdicts.
    spectral_risks = [row["rank_padding"]["risk_level"] for row in spectral["rows"]]
    gradient_risks = [row["risk_level"] for row in gradient["rows"]]
    overall_spectral_risk = (
        "high" if "high" in spectral_risks
        else "medium" if "medium" in spectral_risks
        else "needs_more_evaluation" if "needs_more_evaluation" in spectral_risks
        else "low"
    )
    overall_gradient_risk = (
        "high" if "high" in gradient_risks
        else "medium" if "medium" in gradient_risks
        else "needs_more_evaluation" if "needs_more_evaluation" in gradient_risks
        else "low"
    )

    return {
        "config": asdict(config),
        "scope": (
            "single linear + LoRA, tiny dimensions, synthetic adapter +"
            " synthetic upstream gradient, rank padding via dummy_strategy"
        ),
        "shape_level_rank_leakage": shape,
        "spectral_rank_inference": spectral,
        "gradient_rank_inference": gradient,
        "membership_style_linkability": membership,
        "interpretation": {
            "shape_level_summary": (
                "Rank padding hides true_rank from tensor shape; padded_rank"
                " remains visible."
            ),
            "spectral_inference_summary": (
                f"Across true_ranks={list(config.true_ranks)} with"
                f" dummy_strategy={config.dummy_strategy!r}, the spectral"
                f" rank inference risk is **{overall_spectral_risk}**."
            ),
            "gradient_inference_summary": (
                f"Gradient-side spectral inference risk under the same"
                f" dummy strategy is **{overall_gradient_risk}**."
            ),
            "padded_rank_visibility_note": (
                "Padded rank r_pad is still visible from tensor shape;"
                " hiding it is out of Stage 7.2 scope."
            ),
            "merge_adapter_into_w": False,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_rank_padding": (
            "rank-padding-proxy-evaluated, not formal"
        ),
        "lora_rank_security_proxy_status": "implemented",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.3 — multi-layer LoRA + LoRA training timing-side proxy.",
            "Stage 7.x — stronger dummy distributions that resist spectral and gradient rank inference.",
            "Stage 7.x — explore hiding padded_rank itself (e.g. tiled padding across multiple linears).",
        ],
    }


def rank_security_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if isinstance(v, (tuple, list)):
            v = "|".join(str(x) for x in v)
        rows.append({
            "section": "config", "scope": "n/a", "metric": k, "value": v,
        })
    shape = report["shape_level_rank_leakage"]
    for strategy in ("no_padding", "rank_padding"):
        for entry in shape[strategy]:
            for k, v in entry.items():
                rows.append({
                    "section": "shape_level_rank_leakage",
                    "scope": f"{strategy}.true_rank_{entry['true_rank']}",
                    "metric": k, "value": v,
                })
    for entry in report["spectral_rank_inference"]["rows"]:
        true_rank = entry["true_rank"]
        for sub_key in ("no_padding", "rank_padding"):
            for k, v in entry[sub_key].items():
                rows.append({
                    "section": "spectral_rank_inference",
                    "scope": f"{sub_key}.true_rank_{true_rank}",
                    "metric": k, "value": v,
                })
    for entry in report["gradient_rank_inference"]["rows"]:
        for k, v in entry.items():
            if k == "true_rank":
                continue
            rows.append({
                "section": "gradient_rank_inference",
                "scope": f"true_rank_{entry['true_rank']}",
                "metric": k, "value": v,
            })
    for entry in report["membership_style_linkability"]["rows"]:
        for k, v in entry.items():
            if k == "true_rank":
                continue
            rows.append({
                "section": "membership_linkability",
                "scope": f"true_rank_{entry['true_rank']}",
                "metric": k, "value": v,
            })
    for k, v in report["interpretation"].items():
        rows.append({
            "section": "interpretation", "scope": "summary",
            "metric": k, "value": v,
        })
    return rows


__all__ = [
    "LoRARankSecurityProxyConfig",
    "VALID_RANK_STRATEGIES",
    "rank_security_csv_rows",
    "run_lora_rank_security_proxy",
]
