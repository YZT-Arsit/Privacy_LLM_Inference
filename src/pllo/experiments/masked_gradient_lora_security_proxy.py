"""Stage 7.6 — Masked-gradient LoRA security proxy.

Evaluates leakage from GPU-visible masked parameters and masked
gradients under the Stage 7.6 construction. This is a PROXY audit;
no formal cryptographic / semantic / differential-privacy security is
claimed. Risk labels are conservative.

Probes:

* True-rank inference from ``A_tilde`` / ``B_tilde`` spectra and from
  ``grad_A_tilde`` / ``grad_B_tilde`` spectra.
* Real-vs-dummy subspace separation: how distinguishable is the true
  ``r_real``-dimensional column span of ``A_pad`` from the dummy
  cancellation block under the mixer ``M``?
* Cross-step linkability: under a fixed-mask policy, two visible
  gradients at different steps share the same mask -- this enables
  same-sample / same-step linkage. Under a fresh-mask policy the
  visible fingerprint changes per call.

CPU local emulation only. Raw tensors, masks, gradients, and adapters
are NEVER exported; only summary scalars, shapes, and short
fingerprints appear in JSON / CSV / Markdown.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.ops.masked_gradient_lora import (
    MaskedGradientLoRAConfig,
    create_cancellation_padded_lora,
    create_masked_lora_state,
    create_orthogonal_matrix,
)


_REQUIRED_HONESTY_PHRASES: tuple[str, ...] = (
    "The GPU never receives plaintext LoRA adapters or plaintext LoRA "
    "gradients in this experiment.",
    "This is a CPU-only algebraic and proxy-leakage experiment, not a "
    "real TEE/GPU training benchmark.",
    "No formal, cryptographic, or semantic security is claimed.",
)


@dataclass(frozen=True)
class MaskedGradientLoRASecurityProxyConfig:
    base: MaskedGradientLoRAConfig = MaskedGradientLoRAConfig()
    num_trials: int = 4
    num_steps: int = 4
    fresh_masks_per_step: bool = True
    fixed_masks_baseline: bool = True


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _svdvals(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(x.to(torch.float64))


def _spectrum_rank_estimate(
    sv: torch.Tensor, *, energy_threshold: float = 1 - 1e-6,
) -> dict[str, Any]:
    """Effective rank estimate from singular values.

    Returns the smallest ``k`` such that ``sum(sv[:k]^2) / sum(sv^2)
    >= energy_threshold``, plus the gap structure.
    """
    sv2 = sv.pow(2)
    total = float(sv2.sum().item())
    if total <= 0.0:
        return {
            "energy_rank": int(sv.numel()),
            "spectral_decay_ratio_top2": None,
            "spectral_decay_ratio_top_pad": None,
            "num_values": int(sv.numel()),
            "top_two_sorted": [],
        }
    cum = torch.cumsum(sv2, dim=0) / total
    energy_rank = int((cum < energy_threshold).sum().item()) + 1
    energy_rank = min(energy_rank, int(sv.numel()))
    top2 = sv[:2].tolist() if sv.numel() >= 2 else [float(sv[0].item())]
    if sv.numel() >= 2 and sv[0].item() > 0:
        decay2 = float((sv[1] / sv[0]).item())
    else:
        decay2 = None
    return {
        "energy_rank": energy_rank,
        "spectral_decay_ratio_top2": decay2,
        "num_values": int(sv.numel()),
        "top_two_sorted": top2,
    }


def _true_rank_proxy(
    A_tilde: torch.Tensor, B_tilde: torch.Tensor,
    grad_A_tilde: torch.Tensor, grad_B_tilde: torch.Tensor,
    *, true_rank: int, padded_rank: int,
) -> dict[str, Any]:
    """Try to recover the true rank from visible parameter and grad
    spectra.

    With cancellation padding ``A_pad = [A_real, R, -R]`` and a dense
    orthogonal mixer ``M``, the visible ``A_tilde`` spectrum has the
    SAME singular values as ``A_pad`` (because the SVD is invariant
    under right-multiplication by ``M``). The dummy block ``[R, -R]``
    in general has its own non-zero singular values, so the visible
    ``A_tilde`` spectrum will typically NOT reveal a clean gap at
    ``true_rank``. We measure the gap explicitly.
    """
    sv_A = _svdvals(A_tilde)
    sv_B = _svdvals(B_tilde)
    sv_gA = _svdvals(grad_A_tilde)
    sv_gB = _svdvals(grad_B_tilde)
    spec_A = _spectrum_rank_estimate(sv_A)
    spec_B = _spectrum_rank_estimate(sv_B)
    spec_gA = _spectrum_rank_estimate(sv_gA)
    spec_gB = _spectrum_rank_estimate(sv_gB)
    estimates = {
        "A_tilde": spec_A,
        "B_tilde": spec_B,
        "grad_A_tilde": spec_gA,
        "grad_B_tilde": spec_gB,
    }
    # Any estimate that lands exactly at ``true_rank`` (and not at
    # ``padded_rank``) signals a successful recovery.
    recovered_at_true_rank = any(
        e["energy_rank"] == true_rank for e in estimates.values()
    )
    revealed_padded_only = all(
        e["energy_rank"] >= true_rank for e in estimates.values()
    )
    return {
        "estimates": estimates,
        "true_rank": int(true_rank),
        "padded_rank": int(padded_rank),
        "any_estimate_matches_true_rank": bool(recovered_at_true_rank),
        "all_estimates_above_or_equal_true_rank": bool(revealed_padded_only),
    }


def _real_vs_dummy_subspace_proxy(
    A_tilde: torch.Tensor, B_tilde: torch.Tensor,
) -> dict[str, Any]:
    """Try to separate the real-vs-dummy column / row subspaces of
    ``A_tilde / B_tilde``.

    The attacker does not know ``M``; the best they can do without it
    is run a clustering / spectral split on the rank axis. We measure
    the spectral-gap-based two-cluster split confidence: a clean gap
    after ``true_rank`` columns would indicate recovery.
    """
    sv_A = _svdvals(A_tilde)
    sv_B = _svdvals(B_tilde)
    # Largest gap between consecutive singular values.
    if sv_A.numel() < 2:
        gap_A = 0.0
        gap_A_idx = 0
    else:
        diffs_A = sv_A[:-1] - sv_A[1:]
        gap_A_idx = int(diffs_A.argmax().item()) + 1
        gap_A = float(diffs_A.max().item())
    if sv_B.numel() < 2:
        gap_B = 0.0
        gap_B_idx = 0
    else:
        diffs_B = sv_B[:-1] - sv_B[1:]
        gap_B_idx = int(diffs_B.argmax().item()) + 1
        gap_B = float(diffs_B.max().item())
    return {
        "A_tilde_largest_spectral_gap_idx": gap_A_idx,
        "A_tilde_largest_spectral_gap": gap_A,
        "B_tilde_largest_spectral_gap_idx": gap_B_idx,
        "B_tilde_largest_spectral_gap": gap_B,
    }


def _cross_step_linkability(
    grads_per_step: list[tuple[torch.Tensor, torch.Tensor]],
    *, fresh_masks: bool,
) -> dict[str, Any]:
    """Aggregate similarity between consecutive masked-gradient pairs.

    Under fresh masks per step, the bilinear conjugation by
    ``(N_x, N_y, M)`` is independent across steps, so the
    accelerator-visible gradients should look uncorrelated. Under
    fixed masks the gradients share the same conjugation -- in
    particular ``cos(grad_A_tilde_{t}, grad_A_tilde_{t+1}) =
    cos(grad_A_t, grad_A_{t+1})``.
    """
    cos_sims_A: list[float] = []
    cos_sims_B: list[float] = []
    for t in range(len(grads_per_step) - 1):
        gA_t, gB_t = grads_per_step[t]
        gA_n, gB_n = grads_per_step[t + 1]
        cos_sims_A.append(float(
            torch.nn.functional.cosine_similarity(
                gA_t.flatten(), gA_n.flatten(), dim=0,
            ).item()
        ))
        cos_sims_B.append(float(
            torch.nn.functional.cosine_similarity(
                gB_t.flatten(), gB_n.flatten(), dim=0,
            ).item()
        ))
    if not cos_sims_A:
        return {
            "fresh_masks": bool(fresh_masks),
            "mean_consecutive_cos_A": None,
            "mean_consecutive_cos_B": None,
            "abs_mean_consecutive_cos_A": None,
            "abs_mean_consecutive_cos_B": None,
        }
    mean_A = sum(cos_sims_A) / len(cos_sims_A)
    mean_B = sum(cos_sims_B) / len(cos_sims_B)
    abs_mean_A = sum(abs(c) for c in cos_sims_A) / len(cos_sims_A)
    abs_mean_B = sum(abs(c) for c in cos_sims_B) / len(cos_sims_B)
    return {
        "fresh_masks": bool(fresh_masks),
        "mean_consecutive_cos_A": mean_A,
        "mean_consecutive_cos_B": mean_B,
        "abs_mean_consecutive_cos_A": abs_mean_A,
        "abs_mean_consecutive_cos_B": abs_mean_B,
    }


def _dummy_strategy_classification(
    A_tilde: torch.Tensor,
) -> dict[str, Any]:
    """Heuristically classify the dummy-padding strategy from the
    visible ``A_tilde`` spectrum / column-norm distribution.

    Under paired-cancellation the column-norm distribution typically
    contains paired magnitudes; under no padding the spectrum drops
    sharply at ``true_rank``. We expose only summary scalars; the
    attacker should not be able to distinguish strategies from
    summary statistics alone in expectation.
    """
    col_norms = A_tilde.norm(dim=0)
    return {
        "col_norm_mean": float(col_norms.mean().item()),
        "col_norm_std": float(col_norms.std(unbiased=False).item()),
        "col_norm_min": float(col_norms.min().item()),
        "col_norm_max": float(col_norms.max().item()),
    }


# ---------------------------------------------------------------------------
# Risk labels
# ---------------------------------------------------------------------------


def _label_risk(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low_proxy_risk"
    if value < high:
        return "medium_proxy_risk"
    return "high_proxy_risk"


def _label_rank_proxy(rank_proxy: dict[str, Any]) -> str:
    if rank_proxy["any_estimate_matches_true_rank"]:
        return "high_proxy_risk"
    return "low_proxy_risk"


def _label_linkability(cos_record: dict[str, Any]) -> str:
    if cos_record["abs_mean_consecutive_cos_A"] is None:
        return "needs_more_evaluation"
    avg = (
        cos_record["abs_mean_consecutive_cos_A"]
        + cos_record["abs_mean_consecutive_cos_B"]
    ) / 2
    return _label_risk(avg, low=0.2, high=0.6)


# ---------------------------------------------------------------------------
# Top-level proxy run
# ---------------------------------------------------------------------------


def _generate_grads(
    cfg: MaskedGradientLoRAConfig, generator: torch.Generator,
    *,
    num_steps: int, fresh_masks: bool,
    A_pad: torch.Tensor, B_pad: torch.Tensor,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Produce ``num_steps`` consecutive masked-gradient pairs for a
    fresh-mask or fixed-mask policy."""
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    device = torch.device(cfg.device)

    # Fixed masks: one (N_x, N_y, M) reused across steps.
    if not fresh_masks:
        N_x = create_orthogonal_matrix(
            cfg.d_in, generator=generator, dtype=dtype, device=device,
        )
        N_y = create_orthogonal_matrix(
            cfg.d_out, generator=generator, dtype=dtype, device=device,
        )
        M = create_orthogonal_matrix(
            cfg.padded_rank, generator=generator, dtype=dtype, device=device,
        )

    grads: list[tuple[torch.Tensor, torch.Tensor]] = []
    A_tilde_last = None
    B_tilde_last = None
    A_plain = A_pad.clone()
    B_plain = B_pad.clone()
    for _ in range(num_steps):
        if fresh_masks:
            N_x = create_orthogonal_matrix(
                cfg.d_in, generator=generator, dtype=dtype, device=device,
            )
            N_y = create_orthogonal_matrix(
                cfg.d_out, generator=generator, dtype=dtype, device=device,
            )
            M = create_orthogonal_matrix(
                cfg.padded_rank, generator=generator,
                dtype=dtype, device=device,
            )
        # Synthetic batch + target -- different per step to provide
        # signal for the linkability proxy.
        X = torch.randn(
            cfg.batch_size, cfg.d_in, dtype=dtype, device=device,
            generator=generator,
        )
        target = torch.randn(
            cfg.batch_size, cfg.d_out, dtype=dtype, device=device,
            generator=generator,
        )
        X_tilde = X @ N_x
        target_tilde = target @ N_y
        state = create_masked_lora_state(
            A_plain, B_plain,
            N_x=N_x, N_y=N_y, M=M,
            padded_rank=A_plain.shape[1], true_rank=cfg.true_rank,
        )
        Y_tilde = X_tilde @ state.A_tilde @ state.B_tilde
        diff_t = Y_tilde - target_tilde
        n = diff_t.numel()
        grad_Y_tilde = 2.0 * diff_t / n
        grad_A_tilde = (
            X_tilde.transpose(-2, -1) @ grad_Y_tilde
            @ state.B_tilde.transpose(-2, -1)
        )
        grad_B_tilde = (
            (X_tilde @ state.A_tilde).transpose(-2, -1) @ grad_Y_tilde
        )
        grads.append((grad_A_tilde, grad_B_tilde))
        A_tilde_last = state.A_tilde
        B_tilde_last = state.B_tilde
        # Advance plain (lockstep) for the next step.
        Y = X @ A_plain @ B_plain
        diff = Y - target
        grad_Y_plain = 2.0 * diff / n
        grad_A = X.transpose(-2, -1) @ grad_Y_plain @ B_plain.transpose(-2, -1)
        grad_B = (X @ A_plain).transpose(-2, -1) @ grad_Y_plain
        A_plain = A_plain - cfg.lr * grad_A
        B_plain = B_plain - cfg.lr * grad_B
    return grads, A_tilde_last, B_tilde_last


def run_masked_gradient_lora_security_proxy(
    cfg: MaskedGradientLoRASecurityProxyConfig,
) -> dict[str, Any]:
    base = cfg.base
    torch.manual_seed(base.seed)
    g = torch.Generator(device="cpu").manual_seed(base.seed)
    dtype = torch.float64 if base.dtype == "float64" else torch.float32
    device = torch.device(base.device)

    A_real = torch.randn(
        base.d_in, base.true_rank, dtype=dtype, device=device, generator=g,
    ) * 0.1
    B_real = torch.randn(
        base.true_rank, base.d_out, dtype=dtype, device=device, generator=g,
    ) * 0.1
    if base.use_rank_padding and base.padded_rank > base.true_rank:
        A_pad, B_pad, _ = create_cancellation_padded_lora(
            A_real, B_real,
            padded_rank=base.padded_rank,
            strategy=base.dummy_strategy,
            generator=g,
        )
    else:
        A_pad, B_pad = A_real.clone(), B_real.clone()

    trials_fresh: list[dict[str, Any]] = []
    trials_fixed: list[dict[str, Any]] = []
    for trial in range(cfg.num_trials):
        g_t = torch.Generator(device="cpu").manual_seed(base.seed + trial + 7)
        grads_fresh, A_tilde, B_tilde = _generate_grads(
            base, g_t,
            num_steps=cfg.num_steps, fresh_masks=True,
            A_pad=A_pad, B_pad=B_pad,
        )
        rank_proxy_fresh = _true_rank_proxy(
            A_tilde, B_tilde, grads_fresh[-1][0], grads_fresh[-1][1],
            true_rank=base.true_rank,
            padded_rank=A_pad.shape[1],
        )
        subspace_fresh = _real_vs_dummy_subspace_proxy(A_tilde, B_tilde)
        link_fresh = _cross_step_linkability(grads_fresh, fresh_masks=True)
        dummy_strat = _dummy_strategy_classification(A_tilde)
        trials_fresh.append({
            "trial": int(trial),
            "policy": "fresh_masks_per_step",
            "rank_proxy": rank_proxy_fresh,
            "subspace_proxy": subspace_fresh,
            "linkability": link_fresh,
            "dummy_strategy_classification": dummy_strat,
            "rank_proxy_label": _label_rank_proxy(rank_proxy_fresh),
            "linkability_label": _label_linkability(link_fresh),
        })

        if cfg.fixed_masks_baseline:
            g_t2 = torch.Generator(device="cpu").manual_seed(
                base.seed + trial + 13,
            )
            grads_fixed, A_t_fixed, B_t_fixed = _generate_grads(
                base, g_t2,
                num_steps=cfg.num_steps, fresh_masks=False,
                A_pad=A_pad, B_pad=B_pad,
            )
            rank_proxy_fixed = _true_rank_proxy(
                A_t_fixed, B_t_fixed,
                grads_fixed[-1][0], grads_fixed[-1][1],
                true_rank=base.true_rank,
                padded_rank=A_pad.shape[1],
            )
            subspace_fixed = _real_vs_dummy_subspace_proxy(A_t_fixed, B_t_fixed)
            link_fixed = _cross_step_linkability(grads_fixed, fresh_masks=False)
            trials_fixed.append({
                "trial": int(trial),
                "policy": "fixed_masks_baseline",
                "rank_proxy": rank_proxy_fixed,
                "subspace_proxy": subspace_fixed,
                "linkability": link_fixed,
                "rank_proxy_label": _label_rank_proxy(rank_proxy_fixed),
                "linkability_label": _label_linkability(link_fixed),
            })

    return {
        "status": "ok",
        "stage": "7.6",
        "main_mode": "masked_gradient_lora_security_proxy",
        "config": {
            "base": asdict(base),
            "num_trials": cfg.num_trials,
            "num_steps": cfg.num_steps,
            "fresh_masks_per_step": cfg.fresh_masks_per_step,
            "fixed_masks_baseline": cfg.fixed_masks_baseline,
        },
        "trials_fresh_masks": trials_fresh,
        "trials_fixed_masks_baseline": trials_fixed,
        "honesty_phrases": list(_REQUIRED_HONESTY_PHRASES),
        "formal_security_claim": False,
        "limitations": [
            "Proxy attacks only -- NOT a formal security proof.",
            "True-rank inference depends on the dummy strategy and on "
            "the mixer's invariance properties; the cancellation-"
            "padded paired-strategy used here is not guaranteed to "
            "hide the rank against all spectral attacks.",
            "Fixed-mask baseline is included for reference; the "
            "default Stage 7.6 policy is fresh masks per step.",
            "No real TEE / GPU runtime; no hardware side-channel "
            "evaluation.",
            "No formal cryptographic / semantic / differential-"
            "privacy security is claimed.",
            "Raw tensors, masks, adapters, and gradients are NEVER "
            "exported.",
        ],
        "paper_safe_wording": (
            "We probe the masked-gradient surface with spectral and "
            "linkability proxies. Risk labels are conservative; "
            "no formal security is claimed."
        ),
        "unsafe_wording_to_avoid": [
            "GPU has zero information about adapters.",
            "Cryptographic privacy.",
            "Semantic security.",
            "Adapter-extraction failed implies security.",
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
    for trial in report["trials_fresh_masks"] + report["trials_fixed_masks_baseline"]:
        rp = trial["rank_proxy"]
        link = trial["linkability"]
        rows.append({
            "trial": trial["trial"],
            "policy": trial["policy"],
            "rank_proxy_any_match_true_rank": rp[
                "any_estimate_matches_true_rank"
            ],
            "rank_proxy_label": trial["rank_proxy_label"],
            "linkability_abs_mean_cos_A": _round(
                link["abs_mean_consecutive_cos_A"], 6,
            ) if link["abs_mean_consecutive_cos_A"] is not None else None,
            "linkability_abs_mean_cos_B": _round(
                link["abs_mean_consecutive_cos_B"], 6,
            ) if link["abs_mean_consecutive_cos_B"] is not None else None,
            "linkability_label": trial["linkability_label"],
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

    w("# Stage 7.6 — Masked-Gradient LoRA Security Proxy")
    w()
    w(
        "Proxy audit of leakage from GPU-visible masked parameters "
        "and masked gradients. The GPU never receives plaintext "
        "LoRA adapters or plaintext LoRA gradients in this "
        "experiment. No formal, cryptographic, or semantic security "
        "is claimed. This is a CPU-only algebraic and proxy-leakage "
        "experiment, not a real TEE/GPU training benchmark."
    )
    w()
    w("## Fresh-masks-per-step trials")
    w()
    w("| trial | rank_proxy_match_true_rank | rank_proxy_label | abs_mean_cos_A | abs_mean_cos_B | linkability_label |")
    w("|---|---|---|---|---|---|")
    for t in report["trials_fresh_masks"]:
        link = t["linkability"]
        w(
            f"| {t['trial']} | "
            f"{t['rank_proxy']['any_estimate_matches_true_rank']} | "
            f"{t['rank_proxy_label']} | "
            f"{_round(link['abs_mean_consecutive_cos_A'], 4)} | "
            f"{_round(link['abs_mean_consecutive_cos_B'], 4)} | "
            f"{t['linkability_label']} |"
        )
    w()
    if report["trials_fixed_masks_baseline"]:
        w("## Fixed-masks baseline trials")
        w()
        w("| trial | rank_proxy_match_true_rank | rank_proxy_label | abs_mean_cos_A | abs_mean_cos_B | linkability_label |")
        w("|---|---|---|---|---|---|")
        for t in report["trials_fixed_masks_baseline"]:
            link = t["linkability"]
            w(
                f"| {t['trial']} | "
                f"{t['rank_proxy']['any_estimate_matches_true_rank']} | "
                f"{t['rank_proxy_label']} | "
                f"{_round(link['abs_mean_consecutive_cos_A'], 4)} | "
                f"{_round(link['abs_mean_consecutive_cos_B'], 4)} | "
                f"{t['linkability_label']} |"
            )
        w()
    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w(f"`formal_security_claim`: `{report['formal_security_claim']}`")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *,
    outputs_dir: str = "outputs",
    json_filename: str = "masked_gradient_lora_security_proxy.json",
    csv_filename: str = "masked_gradient_lora_security_proxy.csv",
    md_filename: str = "masked_gradient_lora_security_proxy.md",
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
    "MaskedGradientLoRASecurityProxyConfig",
    "render_markdown",
    "run_masked_gradient_lora_security_proxy",
    "write_reports",
]
