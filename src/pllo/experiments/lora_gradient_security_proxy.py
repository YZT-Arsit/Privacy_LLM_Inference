"""Stage 7.1 — LoRA gradient-side security proxy.

Adds gradient tensors to the GPU-visible transcript of the Stage 7.0
forward proxy and re-evaluates leakage. Five strategies map the same
way as Stage 7.0:

  * ``unmasked_gradient_baseline``     — GPU sees ``grad_A`` / ``grad_B`` /
    ``G`` directly (no masking at all).
  * ``fixed_masks_fixed_u``            — one global ``(N_in, N_out, U)``.
  * ``fresh_u_only``                   — fixed ``(N_in, N_out)``; fresh ``U``.
  * ``fresh_masks_fresh_u``            — fresh ``(N_in, N_out, U)`` per trial.
  * ``fresh_masks_fresh_u_with_pad``   — same as above + fresh pad ``T``.

Sub-attacks
-----------

1. **Gradient extraction proxy** — what can a GPU-side attacker recover
   about ``grad_A`` / ``grad_B`` / ``G`` from ``grad_A_tilde`` /
   ``grad_B_tilde`` / ``G_tilde``?

2. **Gradient membership-style linkability** — for two private samples
   ``X1, X2`` (and matching ``Y_target_1, Y_target_2``), run each through
   one masked LoRA backward step many times and check whether
   ``grad_A_tilde`` / ``grad_B_tilde`` lets an attacker decide
   "same sample" vs "different sample".

3. **Gradient leakage accounting** — a static per-variable table covering
   the Stage 7.1 contract (loss / optimizer remain trusted; backward
   arithmetic runs on masked tensors; raw gradients never leave the
   trusted side).

All metrics are illustrative. The masking equations alone do not bound a
real attacker; this is a proxy for ranking strategies, not a security
proof. Outputs publish summary statistics + fingerprints only.
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
    plain_lora_backward_reference,
    recover_lora_gradients,
    transform_upstream_gradient,
)


VALID_STRATEGIES: tuple[str, ...] = (
    "unmasked_gradient_baseline",
    "fixed_masks_fixed_u",
    "fresh_u_only",
    "fresh_masks_fresh_u",
    "fresh_masks_fresh_u_with_pad",
)


def _strategy_to_forward_config(
    strategy: str, base_dtype: str, device: str, pad_scale: float,
) -> MaskedLoRAForwardConfig:
    if strategy == "fresh_u_only":
        return MaskedLoRAForwardConfig(
            use_pad=False, fresh_u_per_call=True, fresh_masks_per_call=False,
            pad_scale=pad_scale, dtype=base_dtype, device=device,
        )
    if strategy == "fresh_masks_fresh_u":
        return MaskedLoRAForwardConfig(
            use_pad=False, fresh_u_per_call=True, fresh_masks_per_call=True,
            pad_scale=pad_scale, dtype=base_dtype, device=device,
        )
    if strategy == "fresh_masks_fresh_u_with_pad":
        return MaskedLoRAForwardConfig(
            use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
            pad_scale=pad_scale, dtype=base_dtype, device=device,
        )
    # baseline / fixed default
    return MaskedLoRAForwardConfig(
        use_pad=False, fresh_u_per_call=False, fresh_masks_per_call=False,
        pad_scale=pad_scale, dtype=base_dtype, device=device,
    )


@dataclass
class LoRAGradientSecurityProxyConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    d_in: int = 32
    d_out: int = 16
    rank: int = 4
    alpha: float = 1.0
    use_bias: bool = True
    num_trials: int = 64
    pad_scale: float = 1.0
    dtype: str = "float64"
    device: str = "cpu"
    strategies: tuple[str, ...] = field(
        default_factory=lambda: tuple(VALID_STRATEGIES),
    )
    membership_trials_per_sample: int = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).norm().item()
    den = b.norm().item()
    if den < 1e-12:
        return float(num)
    return float(num / den)


def _subspace_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    qa, _ = torch.linalg.qr(a)
    qb, _ = torch.linalg.qr(b)
    sv = torch.linalg.svdvals(qa.transpose(0, 1) @ qb)
    return float(sv.mean().item())


def _singular_value_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = torch.linalg.svdvals(a)
    sb = torch.linalg.svdvals(b)
    k = min(sa.numel(), sb.numel())
    sa = sa[:k]
    sb = sb[:k]
    na = sa.norm().item()
    nb = sb.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((sa * sb).sum().item() / (na * nb))


def _norm_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm().item()
    nb = b.norm().item()
    if max(na, nb) < 1e-12:
        return 1.0
    return float(min(na, nb) / max(na, nb))


def _matrix_rank_signature(t: torch.Tensor, tol: float = 1e-8) -> int:
    sv = torch.linalg.svdvals(t)
    if sv.numel() == 0:
        return 0
    threshold = max(sv.max().item() * tol, tol)
    return int((sv > threshold).sum().item())


def _make_state(
    cfg: LoRAConfig, fcfg: MaskedLoRAForwardConfig,
    seq_len: int, gen: torch.Generator,
) -> LoRAState:
    return create_masked_lora_state(cfg, fcfg, seq_len=seq_len, generator=gen)


def _compute_masked_gradients(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    state: LoRAState,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_A_tilde, grad_B_tilde)`` for one masked backward."""
    if state.pad is None:
        x_tilde = x @ state.n_in
    else:
        x_tilde = (x - state.pad) @ state.n_in
    a_tilde = state.n_in_inv @ a @ state.u
    b_tilde = state.u_inv @ b @ state.n_out
    g_tilde = transform_upstream_gradient(g, state.n_out)
    masked = masked_lora_backward(
        x_tilde, a_tilde, b_tilde, g_tilde,
        alpha=alpha, recover_grad_x=False,
    )
    return masked["grad_a_tilde"], masked["grad_b_tilde"]


# ---------------------------------------------------------------------------
# Gradient extraction proxy
# ---------------------------------------------------------------------------


def _run_gradient_extraction_for_strategy(
    strategy: str,
    cfg: LoRAConfig,
    base_dtype: torch.dtype,
    device: torch.device,
    num_trials: int,
    pad_scale: float,
    generator: torch.Generator,
    seq_len: int,
) -> dict[str, Any]:
    a_plain, b_plain = init_lora_adapters(cfg, generator=generator)
    b_plain = b_plain + 0.1 * torch.randn(
        cfg.rank, cfg.d_out, generator=generator,
        dtype=base_dtype, device=device,
    )
    w_plain = torch.randn(
        cfg.d_in, cfg.d_out, generator=generator,
        dtype=base_dtype, device=device,
    )

    fcfg = _strategy_to_forward_config(
        strategy, cfg.dtype, cfg.device, pad_scale,
    )
    persistent_state: LoRAState | None = None
    if strategy in ("fixed_masks_fixed_u", "fresh_u_only"):
        persistent_state = _make_state(cfg, fcfg, seq_len, generator)

    grad_a_errs: list[float] = []
    grad_b_errs: list[float] = []
    grad_a_subspace: list[float] = []
    grad_b_subspace: list[float] = []
    grad_a_sv_sim: list[float] = []
    grad_b_sv_sim: list[float] = []
    grad_a_norm_sim: list[float] = []
    grad_b_norm_sim: list[float] = []
    rank_signatures_a: list[int] = []
    rank_signatures_b: list[int] = []

    for _ in range(num_trials):
        x_t = torch.randn(
            seq_len, cfg.d_in, generator=generator,
            dtype=base_dtype, device=device,
        )
        g_t = torch.randn(
            seq_len, cfg.d_out, generator=generator,
            dtype=base_dtype, device=device,
        )
        plain = plain_lora_backward_reference(
            x_t, w_plain, a_plain, b_plain, g_t, alpha=cfg.alpha,
        )
        if strategy == "unmasked_gradient_baseline":
            grad_a_tilde = plain["grad_a"]
            grad_b_tilde = plain["grad_b"]
        else:
            if persistent_state is not None:
                # Strategy chooses which axes refresh.
                fresh_state = create_masked_lora_state(
                    cfg, fcfg, seq_len=seq_len,
                    state=persistent_state, generator=generator,
                )
                if strategy == "fresh_u_only":
                    state = LoRAState(
                        n_in=persistent_state.n_in,
                        n_in_inv=persistent_state.n_in_inv,
                        n_out=persistent_state.n_out,
                        n_out_inv=persistent_state.n_out_inv,
                        u=fresh_state.u, u_inv=fresh_state.u_inv,
                        pad=fresh_state.pad, rank=cfg.rank, alpha=cfg.alpha,
                    )
                    persistent_state = state
                else:
                    state = persistent_state
            else:
                state = create_masked_lora_state(
                    cfg, fcfg, seq_len=seq_len, generator=generator,
                )
            grad_a_tilde, grad_b_tilde = _compute_masked_gradients(
                x_t, a_plain, b_plain, g_t, state, alpha=cfg.alpha,
            )

        grad_a_errs.append(_rel_l2(grad_a_tilde, plain["grad_a"]))
        grad_b_errs.append(_rel_l2(grad_b_tilde, plain["grad_b"]))
        grad_a_subspace.append(_subspace_similarity(grad_a_tilde, plain["grad_a"]))
        grad_b_subspace.append(
            _subspace_similarity(
                grad_b_tilde.transpose(0, 1),
                plain["grad_b"].transpose(0, 1),
            )
        )
        grad_a_sv_sim.append(_singular_value_similarity(grad_a_tilde, plain["grad_a"]))
        grad_b_sv_sim.append(_singular_value_similarity(grad_b_tilde, plain["grad_b"]))
        grad_a_norm_sim.append(_norm_similarity(grad_a_tilde, plain["grad_a"]))
        grad_b_norm_sim.append(_norm_similarity(grad_b_tilde, plain["grad_b"]))
        rank_signatures_a.append(_matrix_rank_signature(grad_a_tilde))
        rank_signatures_b.append(_matrix_rank_signature(grad_b_tilde))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    rank_visible = (
        max(rank_signatures_a) >= cfg.rank
        and max(rank_signatures_b) >= cfg.rank
    )
    return {
        "strategy": strategy,
        "grad_a_recovery_rel_l2_mean": _mean(grad_a_errs),
        "grad_a_recovery_rel_l2_min": float(min(grad_a_errs)),
        "grad_b_recovery_rel_l2_mean": _mean(grad_b_errs),
        "grad_b_recovery_rel_l2_min": float(min(grad_b_errs)),
        "grad_a_subspace_similarity_mean": _mean(grad_a_subspace),
        "grad_b_subspace_similarity_mean": _mean(grad_b_subspace),
        "grad_a_singular_value_similarity_mean": _mean(grad_a_sv_sim),
        "grad_b_singular_value_similarity_mean": _mean(grad_b_sv_sim),
        "gradient_norm_similarity_a_mean": _mean(grad_a_norm_sim),
        "gradient_norm_similarity_b_mean": _mean(grad_b_norm_sim),
        "rank_signature_a": int(max(rank_signatures_a)),
        "rank_signature_b": int(max(rank_signatures_b)),
        "configured_rank": int(cfg.rank),
        "rank_visible_from_grad_shape": rank_visible,
        "num_trials": num_trials,
        "exact_recovery": strategy == "unmasked_gradient_baseline",
        "interpretation": _interpret_extraction(
            strategy,
            grad_a_mean=_mean(grad_a_errs),
            subspace_mean=_mean(grad_a_subspace),
        ),
    }


def _interpret_extraction(
    strategy: str, *, grad_a_mean: float, subspace_mean: float,
) -> str:
    if strategy == "unmasked_gradient_baseline":
        return (
            "Gradient is fully exposed; attacker recovers grad_A / grad_B"
            " exactly."
        )
    if grad_a_mean < 1e-6:
        return (
            "Recovery essentially exact (possible mask bug or N / U"
            " cancellation)."
        )
    if grad_a_mean < 0.5 or subspace_mean > 0.5:
        return (
            "Gradient partially leaks via subspace structure; needs review."
        )
    return (
        "Recovery error is large (rel L2 > 0.5) and subspace similarity is"
        " low; under this proxy, mask makes naive gradient extraction"
        " unreliable."
    )


# ---------------------------------------------------------------------------
# Gradient leakage accounting
# ---------------------------------------------------------------------------


def _gradient_leakage_accounting(strategy: str) -> list[dict[str, Any]]:
    unmasked = strategy == "unmasked_gradient_baseline"
    fixed = strategy in ("unmasked_gradient_baseline", "fixed_masks_fixed_u")
    return [
        {
            "name": "private_input_X",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "high" if unmasked else "low",
            "mitigation": "Right-mask + optional input pad before backward.",
            "stage_7_1_status": "covered",
        },
        {
            "name": "private_target_Y",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Stays trusted; loss G = dL/dY computed inside trusted side.",
            "stage_7_1_status": "covered",
        },
        {
            "name": "adapter_A / adapter_B",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "high" if unmasked else "low",
            "mitigation": "Adapter masked by A_tilde / B_tilde; fresh U per call.",
            "stage_7_1_status": "covered",
        },
        {
            "name": "grad_A / grad_B (plain)",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": (
                "Recovered only on trusted side after multiplying by"
                " N_in^{-T} / U^{-T} / N_out^T."
            ),
            "stage_7_1_status": "trusted",
        },
        {
            "name": "G (plain upstream gradient)",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Computed from trusted Y_recovered and Y_target.",
            "stage_7_1_status": "trusted_loss",
        },
        {
            "name": "G_tilde (masked upstream gradient)",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": "high" if unmasked else ("medium" if fixed else "low"),
            "mitigation": "G_tilde = G N_out^{-T}; fresh N_out per call.",
            "stage_7_1_status": "covered",
        },
        {
            "name": "grad_A_tilde / grad_B_tilde",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": "high" if unmasked else ("medium" if fixed else "low"),
            "mitigation": "Masked by U / N_in / N_out; fresh U per call.",
            "stage_7_1_status": "covered",
        },
        {
            "name": "X_tilde A_tilde (intermediate)",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": "medium" if fixed else "low",
            "mitigation": (
                "Lives in the rank space of U; recovered grad_A / grad_B never"
                " surfaced on GPU."
            ),
            "stage_7_1_status": "covered",
        },
        {
            "name": "optimizer_state (SGD momentum / AdamW m, v)",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Trusted-only; never exported to JSON/CSV/Markdown.",
            "stage_7_1_status": "trusted_optimizer",
        },
        {
            "name": "masks N_in / N_out / U / pad T_in",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Sampled inside trusted side; never exported.",
            "stage_7_1_status": "covered",
        },
        {
            "name": "X_tilde / Y_tilde (forward, inherited from Stage 7.0)",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": "low" if not fixed else "medium",
            "mitigation": "Right-mask N_in / N_out; fresh per call recommended.",
            "stage_7_1_status": "covered",
        },
    ]


# ---------------------------------------------------------------------------
# Gradient membership-style linkability
# ---------------------------------------------------------------------------


def _membership_linkability_for_strategy(
    strategy: str,
    cfg: LoRAConfig,
    base_dtype: torch.dtype,
    device: torch.device,
    pad_scale: float,
    trials_per_sample: int,
    generator: torch.Generator,
    seq_len: int,
) -> dict[str, Any]:
    a_plain, b_plain = init_lora_adapters(cfg, generator=generator)
    b_plain = b_plain + 0.1 * torch.randn(
        cfg.rank, cfg.d_out, generator=generator,
        dtype=base_dtype, device=device,
    )

    x1 = torch.randn(seq_len, cfg.d_in, generator=generator, dtype=base_dtype, device=device)
    x2 = torch.randn(seq_len, cfg.d_in, generator=generator, dtype=base_dtype, device=device)
    y_target1 = torch.randn(seq_len, cfg.d_out, generator=generator, dtype=base_dtype, device=device)
    y_target2 = torch.randn(seq_len, cfg.d_out, generator=generator, dtype=base_dtype, device=device)
    w_plain = torch.randn(cfg.d_in, cfg.d_out, generator=generator, dtype=base_dtype, device=device)

    def _g_for(x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # MSE upstream gradient for one (x, y_target) sample.
        y = x @ w_plain + (cfg.alpha / max(cfg.rank, 1)) * (x @ a_plain) @ b_plain
        diff = y - y_target
        return (2.0 / float(y.numel())) * diff

    fcfg = _strategy_to_forward_config(
        strategy, cfg.dtype, cfg.device, pad_scale,
    )
    persistent_state: LoRAState | None = None
    if strategy in ("fixed_masks_fixed_u", "fresh_u_only"):
        persistent_state = _make_state(cfg, fcfg, seq_len, generator)

    def _next_state() -> LoRAState | None:
        nonlocal persistent_state
        if strategy == "unmasked_gradient_baseline":
            return None
        if strategy == "fixed_masks_fixed_u":
            return persistent_state
        if strategy == "fresh_u_only" and persistent_state is not None:
            fresh = create_masked_lora_state(
                cfg, fcfg, seq_len=seq_len, state=persistent_state, generator=generator,
            )
            persistent_state = LoRAState(
                n_in=persistent_state.n_in,
                n_in_inv=persistent_state.n_in_inv,
                n_out=persistent_state.n_out,
                n_out_inv=persistent_state.n_out_inv,
                u=fresh.u, u_inv=fresh.u_inv,
                pad=fresh.pad, rank=cfg.rank, alpha=cfg.alpha,
            )
            return persistent_state
        return create_masked_lora_state(
            cfg, fcfg, seq_len=seq_len, generator=generator,
        )

    def _grad_tilde(x: torch.Tensor, y_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = _g_for(x, y_target)
        if strategy == "unmasked_gradient_baseline":
            plain = plain_lora_backward_reference(
                x, w_plain, a_plain, b_plain, g, alpha=cfg.alpha,
            )
            return plain["grad_a"], plain["grad_b"]
        state = _next_state()
        assert state is not None
        return _compute_masked_gradients(x, a_plain, b_plain, g, state, alpha=cfg.alpha)

    grads_1 = [_grad_tilde(x1, y_target1) for _ in range(trials_per_sample)]
    grads_2 = [_grad_tilde(x2, y_target2) for _ in range(trials_per_sample)]

    def _gpair_distance(p1: tuple[torch.Tensor, torch.Tensor], p2: tuple[torch.Tensor, torch.Tensor]) -> float:
        return float(
            ((p1[0] - p2[0]).norm() ** 2 + (p1[1] - p2[1]).norm() ** 2).sqrt().item()
        )

    same_d: list[float] = []
    for i in range(trials_per_sample):
        for j in range(i + 1, trials_per_sample):
            same_d.append(_gpair_distance(grads_1[i], grads_1[j]))
            same_d.append(_gpair_distance(grads_2[i], grads_2[j]))
    diff_d: list[float] = []
    for i in range(trials_per_sample):
        for j in range(trials_per_sample):
            diff_d.append(_gpair_distance(grads_1[i], grads_2[j]))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    if same_d and diff_d:
        wins = 0
        total = 0
        for s in same_d:
            for d in diff_d:
                if s < d:
                    wins += 1
                elif s == d:
                    wins += 0.5
                total += 1
        auc = float(wins / max(1, total))
    else:
        auc = 0.5
    linkability_rank = 2.0 * abs(auc - 0.5)

    if strategy == "unmasked_gradient_baseline":
        risk_level = "high"
    elif auc > 0.85:
        risk_level = "high"
    elif auc > 0.65:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "strategy": strategy,
        "same_sample_gradient_distance_mean": _mean(same_d),
        "different_sample_gradient_distance_mean": _mean(diff_d),
        "membership_gradient_auc_proxy": auc,
        "gradient_linkability_rank": linkability_rank,
        "risk_level": risk_level,
        "trials_per_sample": trials_per_sample,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are gradient-side proxy attacks, not formal security proofs.",
    "Gradient tensors may leak rank (`grad_A_tilde` is (d_in × r); `grad_B_tilde` is (r × d_out)). Rank padding is NOT implemented in Stage 7.1 (deferred to Stage 7.2).",
    "Optimizer state remains trusted-only in Stage 7.1; SGD momentum / AdamW (m, v) are never exposed to the GPU.",
    "Loss / upstream gradient computation remains trusted-only; only G_tilde crosses the boundary.",
    "No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "No full model LoRA fine-tuning is evaluated; this is a single-linear, tiny-dimension gradient proxy.",
    "Adversary model is a passive GPU observer of (X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde, G_tilde, grad_A_tilde, grad_B_tilde) transcripts plus the dimensions; active / adaptive attackers and hardware side-channels are NOT evaluated.",
    "Adapter is NEVER merged into the public base weight W.",
]


def run_lora_gradient_security_proxy(
    config: LoRAGradientSecurityProxyConfig,
) -> dict[str, Any]:
    for s in config.strategies:
        if s not in VALID_STRATEGIES:
            raise ValueError(
                f"unknown strategy {s!r}; expected one of {VALID_STRATEGIES}"
            )
    base_dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    cfg = LoRAConfig(
        d_in=config.d_in, d_out=config.d_out, rank=config.rank,
        alpha=config.alpha, use_bias=config.use_bias,
        dtype=config.dtype, device=config.device,
    )
    seq_len = max(4, cfg.d_in // 4)

    extraction: list[dict[str, Any]] = []
    accounting: dict[str, list[dict[str, Any]]] = {}
    membership: list[dict[str, Any]] = []
    for strategy in config.strategies:
        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        extraction.append(
            _run_gradient_extraction_for_strategy(
                strategy, cfg, base_dtype, device,
                config.num_trials, config.pad_scale, gen, seq_len,
            )
        )
        accounting[strategy] = _gradient_leakage_accounting(strategy)
        gen2 = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        membership.append(
            _membership_linkability_for_strategy(
                strategy, cfg, base_dtype, device,
                config.pad_scale, config.membership_trials_per_sample,
                gen2, seq_len,
            )
        )

    baseline_auc = next(
        (m["membership_gradient_auc_proxy"] for m in membership
         if m["strategy"] == "fixed_masks_fixed_u"), None,
    )
    fresh_auc = next(
        (m["membership_gradient_auc_proxy"] for m in membership
         if m["strategy"] == "fresh_masks_fresh_u"), None,
    )
    fresh_pad_auc = next(
        (m["membership_gradient_auc_proxy"] for m in membership
         if m["strategy"] == "fresh_masks_fresh_u_with_pad"), None,
    )
    if baseline_auc is not None and (
        fresh_auc is not None or fresh_pad_auc is not None
    ):
        best = min(v for v in (fresh_auc, fresh_pad_auc) if v is not None)
        delta = baseline_auc - best
        if delta > 0.10:
            link_interp = (
                f"fresh masks reduce gradient-side membership linkability"
                f" (Δ AUC = {delta:+.3f} vs fixed_masks_fixed_u)."
            )
        else:
            link_interp = (
                "fresh masks did NOT clearly reduce gradient-side"
                " linkability under this proxy; needs_more_evaluation."
            )
    else:
        link_interp = "insufficient strategies enabled to compare."

    return {
        "config": asdict(config),
        "scope": (
            "single linear + LoRA, tiny dimensions, synthetic adapter and"
            " synthetic upstream gradient"
        ),
        "strategies": list(config.strategies),
        "gradient_extraction_proxy": extraction,
        "gradient_leakage_accounting": accounting,
        "gradient_membership_style_linkability_proxy": membership,
        "interpretation": {
            "linkability_summary": link_interp,
            "rank_visibility_note": (
                "LoRA rank r is visible from the shape of grad_A_tilde"
                " (d_in × r) and grad_B_tilde (r × d_out). Rank padding is"
                " NOT implemented in Stage 7.1."
            ),
            "loss_status": "trusted_loss",
            "optimizer_status": "trusted_optimizer",
            "merge_adapter_into_w": False,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_backward": (
            "masked-gradient-proxy-evaluated, not formal"
        ),
        "lora_gradient_security_proxy_status": "implemented",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.2 — rank padding to hide r from gradient shape.",
            "Stage 7.3 — multi-layer + cross-layer adapter / gradient linkability proxy.",
        ],
    }


# ---------------------------------------------------------------------------
# Runner helper — CSV rows
# ---------------------------------------------------------------------------


def gradient_security_proxy_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if k == "strategies":
            v = "|".join(v)
        rows.append({
            "section": "config", "strategy": "n/a", "metric": k, "value": v,
        })
    for entry in report["gradient_extraction_proxy"]:
        for k, v in entry.items():
            if k == "strategy":
                continue
            rows.append({
                "section": "gradient_extraction",
                "strategy": entry["strategy"],
                "metric": k, "value": v,
            })
    for strategy, table in report["gradient_leakage_accounting"].items():
        for entry in table:
            for k, v in entry.items():
                if k == "name":
                    continue
                rows.append({
                    "section": "gradient_leakage", "strategy": strategy,
                    "metric": f"{entry['name']}.{k}", "value": v,
                })
    for entry in report["gradient_membership_style_linkability_proxy"]:
        for k, v in entry.items():
            if k == "strategy":
                continue
            rows.append({
                "section": "gradient_membership_linkability",
                "strategy": entry["strategy"],
                "metric": k, "value": v,
            })
    for k, v in report["interpretation"].items():
        rows.append({
            "section": "interpretation", "strategy": "n/a",
            "metric": k, "value": v,
        })
    return rows


__all__ = [
    "LoRAGradientSecurityProxyConfig",
    "VALID_STRATEGIES",
    "gradient_security_proxy_csv_rows",
    "run_lora_gradient_security_proxy",
]
