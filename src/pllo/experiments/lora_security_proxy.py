"""Stage 7.0 — LoRA security proxy.

Three proxy attacks on the GPU-visible transcript of one LoRA-augmented
linear under five masking strategies:

* ``unmasked_adapter_baseline``    — GPU sees A, B in plaintext.
* ``fixed_masks_fixed_u``          — one global ``(N_in, N_out, U)`` across
  all trials; no pad.
* ``fresh_u_only``                 — fixed ``(N_in, N_out)``; fresh ``U`` per
  trial.
* ``fresh_masks_fresh_u``          — fresh ``(N_in, N_out, U)`` per trial.
* ``fresh_masks_fresh_u_with_pad`` — same as above + fresh input pad ``T``.

Sub-attacks
-----------

1. **Adapter extraction proxy** — what can a GPU-side attacker recover
   about ``A``, ``B``, ``ΔW = A B``, the LoRA rank ``r`` and the singular
   structure of the adapter from observing ``A_tilde / B_tilde`` (and
   optionally a transcript of ``X_tilde / Y_tilde``)?

2. **Gradient leakage accounting** — a static table of what the GPU sees
   versus what stays trusted in Stage 7.0. Backward / optimizer remain
   trusted (this matches the training-probe contract).

3. **Membership-style linkability proxy** — for two private samples
   ``X1, X2``, run each through the masked forward many times and check
   whether the visible ``X_tilde`` transcript lets an attacker decide
   "same sample" vs "different sample".

All metrics are illustrative. The masking equations alone do not bound a
real attacker; this is a proxy for ranking strategies, not a security
proof. The output is JSON/CSV/Markdown-safe — only summary statistics and
fingerprints leak; never raw masks, adapter tensors, or private data.
"""

from __future__ import annotations

import json
import math
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
    make_lora_pad_compensation,
    masked_lora_linear_forward,
    obfuscate_lora_input,
    plain_lora_linear_forward,
    recover_masked_output,
    transform_linear_weight_lora,
    transform_lora_adapter,
)


# ---------------------------------------------------------------------------
# Config + strategy table
# ---------------------------------------------------------------------------


VALID_STRATEGIES: tuple[str, ...] = (
    "unmasked_adapter_baseline",
    "fixed_masks_fixed_u",
    "fresh_u_only",
    "fresh_masks_fresh_u",
    "fresh_masks_fresh_u_with_pad",
)


def _strategy_to_forward_config(
    strategy: str, base_dtype: str, device: str, pad_scale: float,
) -> MaskedLoRAForwardConfig:
    if strategy == "unmasked_adapter_baseline":
        # Synthetic "no-op" mask: we keep the same dataclass but bypass
        # actual masking inside the proxy.
        return MaskedLoRAForwardConfig(
            use_pad=False, fresh_u_per_call=False, fresh_masks_per_call=False,
            pad_scale=pad_scale, dtype=base_dtype, device=device,
        )
    if strategy == "fixed_masks_fixed_u":
        return MaskedLoRAForwardConfig(
            use_pad=False, fresh_u_per_call=False, fresh_masks_per_call=False,
            pad_scale=pad_scale, dtype=base_dtype, device=device,
        )
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
    raise ValueError(f"unknown strategy {strategy!r}")


@dataclass
class LoRASecurityProxyConfig:
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
    # Membership proxy: how many trials per sample.
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
    """Average principal angle cosine between column spaces of A and B.

    Both inputs must be ``(d, r)`` with rank ≤ r. Returns 1.0 when the
    column spaces coincide, 0.0 when orthogonal.
    """
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    qa, _ = torch.linalg.qr(a)
    qb, _ = torch.linalg.qr(b)
    m = qa.transpose(0, 1) @ qb
    sv = torch.linalg.svdvals(m)
    return float(sv.mean().item())


def _singular_value_similarity(
    a: torch.Tensor, b: torch.Tensor,
) -> float:
    """Cosine similarity of singular-value spectra."""
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


def _matrix_rank_signature(t: torch.Tensor, tol: float = 1e-8) -> int:
    """Numerical rank via SVD threshold."""
    sv = torch.linalg.svdvals(t)
    if sv.numel() == 0:
        return 0
    threshold = max(sv.max().item() * tol, tol)
    return int((sv > threshold).sum().item())


def _make_strategy_state(
    strategy: str, cfg: LoRAConfig, fcfg: MaskedLoRAForwardConfig,
    seq_len: int, gen: torch.Generator,
) -> LoRAState:
    if strategy in ("fixed_masks_fixed_u", "fresh_u_only"):
        # Sample once and re-use n_in / n_out.
        return create_masked_lora_state(cfg, fcfg, seq_len=seq_len, generator=gen)
    # Fresh strategies will sample fresh state per trial anyway.
    return create_masked_lora_state(cfg, fcfg, seq_len=seq_len, generator=gen)


# ---------------------------------------------------------------------------
# Adapter extraction proxy
# ---------------------------------------------------------------------------


def _run_adapter_extraction_for_strategy(
    strategy: str,
    cfg: LoRAConfig,
    base_dtype: torch.dtype,
    device: torch.device,
    num_trials: int,
    pad_scale: float,
    generator: torch.Generator,
    seq_len: int,
) -> dict[str, Any]:
    """Per-trial adapter extraction summary for one masking strategy."""
    a_plain, b_plain = init_lora_adapters(cfg, generator=generator)
    # Perturb B so ΔW != 0 (matters for ΔW recovery proxy).
    b_plain = b_plain + 0.1 * torch.randn(
        cfg.rank, cfg.d_out, generator=generator,
        dtype=base_dtype, device=device,
    )
    delta_w = a_plain @ b_plain

    fcfg = _strategy_to_forward_config(
        strategy, cfg.dtype, cfg.device, pad_scale,
    )
    persistent_state: LoRAState | None = None
    if strategy in ("fixed_masks_fixed_u", "fresh_u_only"):
        persistent_state = _make_strategy_state(
            strategy, cfg, fcfg, seq_len, generator,
        )

    delta_w_errs: list[float] = []
    a_errs: list[float] = []
    b_errs: list[float] = []
    rank_signatures_a: list[int] = []
    rank_signatures_b: list[int] = []
    subspace_a_sims: list[float] = []
    subspace_b_sims: list[float] = []
    sv_a_sims: list[float] = []
    sv_b_sims: list[float] = []

    for _ in range(num_trials):
        if strategy == "unmasked_adapter_baseline":
            a_tilde = a_plain
            b_tilde = b_plain
        else:
            if persistent_state is not None:
                state = create_masked_lora_state(
                    cfg, fcfg, seq_len=seq_len,
                    state=persistent_state, generator=generator,
                )
                # Persist any non-fresh axes if fresh_u_only.
                if strategy == "fresh_u_only":
                    persistent_state = LoRAState(
                        n_in=persistent_state.n_in,
                        n_in_inv=persistent_state.n_in_inv,
                        n_out=persistent_state.n_out,
                        n_out_inv=persistent_state.n_out_inv,
                        u=state.u, u_inv=state.u_inv,
                        pad=state.pad, rank=cfg.rank, alpha=cfg.alpha,
                    )
                else:
                    persistent_state = state
            else:
                state = create_masked_lora_state(
                    cfg, fcfg, seq_len=seq_len, generator=generator,
                )
            a_tilde, b_tilde = transform_lora_adapter(
                a_plain, b_plain,
                state.n_in_inv, state.n_out,
                state.u, state.u_inv, alpha=cfg.alpha,
            )

        # Attacker hypotheses (assume no masking):
        # ΔW_hat = A_tilde @ B_tilde.
        delta_w_hat = a_tilde @ b_tilde
        delta_w_errs.append(_rel_l2(delta_w_hat, delta_w))
        # For A / B itself: shape-match.
        a_errs.append(_rel_l2(a_tilde, a_plain))
        b_errs.append(_rel_l2(b_tilde, b_plain))
        rank_signatures_a.append(_matrix_rank_signature(a_tilde))
        rank_signatures_b.append(_matrix_rank_signature(b_tilde))
        subspace_a_sims.append(_subspace_similarity(a_tilde, a_plain))
        subspace_b_sims.append(
            _subspace_similarity(
                b_tilde.transpose(0, 1), b_plain.transpose(0, 1),
            )
        )
        sv_a_sims.append(_singular_value_similarity(a_tilde, a_plain))
        sv_b_sims.append(_singular_value_similarity(b_tilde, b_plain))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    rank_visible = (
        max(rank_signatures_a) >= cfg.rank
        and max(rank_signatures_b) >= cfg.rank
    )
    return {
        "strategy": strategy,
        "delta_w_recovery_rel_l2_mean": _mean(delta_w_errs),
        "delta_w_recovery_rel_l2_min": float(min(delta_w_errs)),
        "adapter_a_recovery_rel_l2_mean": _mean(a_errs),
        "adapter_b_recovery_rel_l2_mean": _mean(b_errs),
        "rank_signature_a": int(max(rank_signatures_a)),
        "rank_signature_b": int(max(rank_signatures_b)),
        "configured_rank": int(cfg.rank),
        "rank_visible_in_a_tilde_shape": rank_visible,
        "subspace_similarity_a_mean": _mean(subspace_a_sims),
        "subspace_similarity_b_mean": _mean(subspace_b_sims),
        "singular_value_similarity_a_mean": _mean(sv_a_sims),
        "singular_value_similarity_b_mean": _mean(sv_b_sims),
        "num_trials": num_trials,
        "exact_recovery": strategy == "unmasked_adapter_baseline",
        "interpretation": _interpret_extraction(
            strategy,
            delta_w_mean=_mean(delta_w_errs),
            subspace_mean=_mean(subspace_a_sims),
        ),
    }


def _interpret_extraction(
    strategy: str, *, delta_w_mean: float, subspace_mean: float,
) -> str:
    if strategy == "unmasked_adapter_baseline":
        return (
            "Adapter is fully exposed; ΔW = A B reconstructible exactly."
        )
    if delta_w_mean < 1e-6:
        return "Recovery essentially exact (possible mask bug or n_in/n_out cancellation)."
    if delta_w_mean < 0.5 or subspace_mean > 0.5:
        return (
            "Adapter partially leaks via subspace structure; needs review."
        )
    return (
        "Recovery error is large (rel L2 > 0.5) and subspace similarity is"
        " low; under this proxy, mask makes naive extraction unreliable."
    )


# ---------------------------------------------------------------------------
# Gradient leakage accounting
# ---------------------------------------------------------------------------


def _gradient_leakage_accounting(strategy: str) -> list[dict[str, Any]]:
    """Static table of variable visibility under the Stage 7.0 contract."""
    table: list[dict[str, Any]] = [
        {
            "name": "private_input_X",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low" if strategy != "unmasked_adapter_baseline" else "high",
            "mitigation": (
                "Right-mask + optional input pad before crossing the boundary."
            ),
            "stage_7_0_status": "covered",
        },
        {
            "name": "private_target_Y",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Stays trusted; loss computed inside the trusted side.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "adapter_A",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low" if strategy != "unmasked_adapter_baseline" else "high",
            "mitigation": "A_tilde = N_in^{-1} A U; fresh U per call.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "adapter_B",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low" if strategy != "unmasked_adapter_baseline" else "high",
            "mitigation": "B_tilde = U^{-1} B N_out; fresh U per call.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "grad_A",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": (
                "Backward remains trusted in Stage 7.0; gradient never crosses"
                " the boundary."
            ),
            "stage_7_0_status": "trusted_backward_prototype",
        },
        {
            "name": "grad_B",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": (
                "Backward remains trusted in Stage 7.0; gradient never crosses"
                " the boundary."
            ),
            "stage_7_0_status": "trusted_backward_prototype",
        },
        {
            "name": "optimizer_state (SGD momentum / AdamW m, v)",
            "visibility": "trusted",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Trusted-only; never exported to JSON/CSV/Markdown.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "base_weight_W",
            "visibility": "public",
            "contains_plaintext": True,
            "leakage_risk": "low",
            "mitigation": "Public model weight; ΔW = A B does NOT merge into W.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "X_tilde",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": "medium" if strategy == "fixed_masks_fixed_u" else "low",
            "mitigation": "Right-mask + optional pad; fresh N_in per call.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "A_tilde / B_tilde",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": (
                "high" if strategy == "unmasked_adapter_baseline" else "medium"
            ),
            "mitigation": "U-mask in rank space; fresh U recommended.",
            "stage_7_0_status": "covered",
        },
        {
            "name": "Y_tilde",
            "visibility": "gpu",
            "contains_plaintext": False,
            "leakage_risk": "low",
            "mitigation": "Right-mask via N_out; recovered only on trusted side.",
            "stage_7_0_status": "covered",
        },
    ]
    return table


# ---------------------------------------------------------------------------
# Membership-style linkability proxy
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
    """Pairwise distance over visible X_tilde for two private samples."""
    x1 = torch.randn(seq_len, cfg.d_in, generator=generator, dtype=base_dtype, device=device)
    x2 = torch.randn(seq_len, cfg.d_in, generator=generator, dtype=base_dtype, device=device)

    fcfg = _strategy_to_forward_config(
        strategy, cfg.dtype, cfg.device, pad_scale,
    )
    persistent_state: LoRAState | None = None
    if strategy in ("fixed_masks_fixed_u", "fresh_u_only"):
        persistent_state = _make_strategy_state(
            strategy, cfg, fcfg, seq_len, generator,
        )

    def _next_state() -> LoRAState:
        nonlocal persistent_state
        if strategy == "unmasked_adapter_baseline":
            return None  # type: ignore[return-value]
        if persistent_state is not None and strategy == "fixed_masks_fixed_u":
            return persistent_state
        if persistent_state is not None and strategy == "fresh_u_only":
            new_u_state = create_masked_lora_state(
                cfg, fcfg, seq_len=seq_len, state=persistent_state,
                generator=generator,
            )
            persistent_state = LoRAState(
                n_in=persistent_state.n_in,
                n_in_inv=persistent_state.n_in_inv,
                n_out=persistent_state.n_out,
                n_out_inv=persistent_state.n_out_inv,
                u=new_u_state.u, u_inv=new_u_state.u_inv,
                pad=new_u_state.pad, rank=cfg.rank, alpha=cfg.alpha,
            )
            return persistent_state
        return create_masked_lora_state(
            cfg, fcfg, seq_len=seq_len, generator=generator,
        )

    def _x_tilde(x: torch.Tensor) -> torch.Tensor:
        if strategy == "unmasked_adapter_baseline":
            return x
        s = _next_state()
        return obfuscate_lora_input(x, s.n_in, s.pad)

    x_tildes_1 = [_x_tilde(x1) for _ in range(trials_per_sample)]
    x_tildes_2 = [_x_tilde(x2) for _ in range(trials_per_sample)]

    # Same-sample pairwise L2 distance (mean over (i, j))
    same_d: list[float] = []
    for i in range(trials_per_sample):
        for j in range(i + 1, trials_per_sample):
            same_d.append(float((x_tildes_1[i] - x_tildes_1[j]).norm().item()))
            same_d.append(float((x_tildes_2[i] - x_tildes_2[j]).norm().item()))
    # Different-sample pairwise L2 distance
    diff_d: list[float] = []
    for i in range(trials_per_sample):
        for j in range(trials_per_sample):
            diff_d.append(float((x_tildes_1[i] - x_tildes_2[j]).norm().item()))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    same_mean = _mean(same_d)
    diff_mean = _mean(diff_d)
    # Linkability rank proxy: how often is a same-sample pair closer than the
    # nearest different-sample pair? 1.0 = perfectly linkable; ~0.5 random.
    same_d_sorted = sorted(same_d)
    diff_d_sorted = sorted(diff_d)
    if same_d_sorted and diff_d_sorted:
        # AUC-style proxy: P(same_distance < diff_distance) under uniform sampling
        # over (s, d).
        wins = 0
        total = 0
        for s in same_d_sorted:
            for d in diff_d_sorted:
                if s < d:
                    wins += 1
                elif s == d:
                    wins += 0.5
                total += 1
        auc = float(wins / max(1, total))
    else:
        auc = 0.5

    # Linkability rank: 1.0 = best link signal; 0.0 = random.
    linkability_rank = 2.0 * abs(auc - 0.5)

    if strategy == "unmasked_adapter_baseline":
        risk_level = "high"
    elif auc > 0.85:
        risk_level = "high"
    elif auc > 0.65:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "strategy": strategy,
        "same_sample_distance_mean": same_mean,
        "different_sample_distance_mean": diff_mean,
        "membership_auc_proxy": auc,
        "linkability_rank": linkability_rank,
        "risk_level": risk_level,
        "trials_per_sample": trials_per_sample,
        "num_same_pairs": len(same_d),
        "num_diff_pairs": len(diff_d),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are proxy attacks, not formal security proofs.",
    "LoRA rank ``r`` remains visible from the shape of A_tilde / B_tilde unless explicit rank padding is implemented (NOT in Stage 7.0).",
    "Optimizer state (SGD momentum / AdamW moments) remains trusted-only in Stage 7.0 and is never exported.",
    "No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "No full private fine-tuning workload is evaluated; this is a single-linear, tiny-dimension proxy.",
    "Adversary model is a passive GPU observer of (X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde) transcripts plus the dimensions; active boundary-attack and adaptive-attack proxies are deferred to later stages.",
    "Hardware side-channel attacks (cache / power / EM) are NOT evaluated.",
    "Adapter is NEVER merged into the public base weight W (constraint 7).",
]


def run_lora_security_proxy(
    config: LoRASecurityProxyConfig,
) -> dict[str, Any]:
    """Run all three Stage 7.0 LoRA security proxies and return one report."""
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
            _run_adapter_extraction_for_strategy(
                strategy, cfg, base_dtype, device,
                config.num_trials, config.pad_scale, gen, seq_len,
            )
        )
        accounting[strategy] = _gradient_leakage_accounting(strategy)
        gen_membership = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        membership.append(
            _membership_linkability_for_strategy(
                strategy, cfg, base_dtype, device, config.pad_scale,
                config.membership_trials_per_sample, gen_membership, seq_len,
            )
        )

    # ---- Interpretation: compare strategies head-to-head -----
    baseline_membership_auc = next(
        (m["membership_auc_proxy"] for m in membership
         if m["strategy"] == "fixed_masks_fixed_u"),
        None,
    )
    fresh_membership_auc = next(
        (m["membership_auc_proxy"] for m in membership
         if m["strategy"] == "fresh_masks_fresh_u"),
        None,
    )
    fresh_with_pad_membership_auc = next(
        (m["membership_auc_proxy"] for m in membership
         if m["strategy"] == "fresh_masks_fresh_u_with_pad"),
        None,
    )
    if baseline_membership_auc is not None and (
        fresh_membership_auc is not None
        or fresh_with_pad_membership_auc is not None
    ):
        best_auc = min(
            v for v in (fresh_membership_auc, fresh_with_pad_membership_auc)
            if v is not None
        )
        link_drop = baseline_membership_auc - best_auc
        if link_drop > 0.10:
            link_interp = (
                f"fresh masks reduce membership-style linkability"
                f" (Δ AUC = {link_drop:+.3f} vs fixed_masks_fixed_u)."
            )
        else:
            link_interp = (
                "fresh masks did NOT clearly reduce linkability under this"
                " proxy; needs_more_evaluation."
            )
    else:
        link_interp = "insufficient strategies enabled to compare."

    rank_visibility_note = (
        "LoRA rank r is visible from the shape of A_tilde / B_tilde under"
        " all current strategies; rank padding is NOT implemented in"
        " Stage 7.0."
    )

    return {
        "config": asdict(config),
        "scope": "single linear + LoRA, tiny dimensions, synthetic adapter",
        "strategies": list(config.strategies),
        "adapter_extraction_proxy": extraction,
        "gradient_leakage_accounting": accounting,
        "membership_style_linkability_proxy": membership,
        "interpretation": {
            "linkability_summary": link_interp,
            "rank_visibility_note": rank_visibility_note,
            "trusted_backward_status": (
                "training backward remains trusted in Stage 7.0 prototype"
            ),
            "merge_adapter_into_w": False,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora": (
            "private-adapter-trusted-backward, not formal"
        ),
        "lora_security_proxy_status": "implemented",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.1 — masked backward / gradient-side obfuscation.",
            "Stage 7.2 — rank padding to hide r.",
            "Stage 7.3 — multi-layer LoRA + cross-layer adapter linkability.",
        ],
    }


# ---------------------------------------------------------------------------
# Runner helper — CSV rows
# ---------------------------------------------------------------------------


def security_proxy_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if k == "strategies":
            v = "|".join(v)
        rows.append({
            "section": "config", "strategy": "n/a", "metric": k, "value": v,
        })
    for entry in report["adapter_extraction_proxy"]:
        for k, v in entry.items():
            if k == "strategy":
                continue
            rows.append({
                "section": "adapter_extraction",
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
    for entry in report["membership_style_linkability_proxy"]:
        for k, v in entry.items():
            if k == "strategy":
                continue
            rows.append({
                "section": "membership_linkability",
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
    "LoRASecurityProxyConfig",
    "VALID_STRATEGIES",
    "run_lora_security_proxy",
    "security_proxy_csv_rows",
]
