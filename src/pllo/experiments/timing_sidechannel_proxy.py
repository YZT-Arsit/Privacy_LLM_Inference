"""Stage 5.6 — Timing / boundary-call side-channel proxy.

A **model-based** latency proxy that drives the Stage 5.2c workload
profiler's per-forward op-count formulas with the Stage 5.2c cost model:

  latency_ms ≈ boundary_calls * tee_call_overhead_ms
             + gpu_ops / gpu_flops_per_ms
             + trusted_ops / (gpu_flops_per_ms / tee_to_gpu_flops_ratio)
             + bytes / tee_bytes_per_ms
             + Gaussian noise

The output is a leakage-risk assessment: which observable variables can an
attacker who only sees latency timing recover? This is **NOT a real TEE
wall-time measurement** — `wall_time_source` stays
``projected_from_op_counts``; nothing in the wider profile is reclassified.

Sub-attacks (kNN classifier / Pearson correlation over the simulated
latency timeline):

1. ``prompt_length_leakage`` — across prompt lengths, regress / classify
   the length bucket from simulated latency.
2. ``decode_step_leakage`` — across decode-step indices (position offset),
   classify step bucket.
3. ``method_distinguishability`` — across methods
   (``ours_current`` / ``ours_compatible_nonlinear_islands`` /
   ``tslp_trusted_nonlinear_baseline``), classify method from latency.
4. ``mitigation_distinguishability`` — within
   ``ours_compatible_nonlinear_islands``, classify mitigation bundle.
5. ``boundary_call_pattern_compare`` — per-method per-forward boundary-call
   count comparison (a static structural leakage measure independent of
   noise).

Findings are reported as proxy risk levels:

  - ``low``    : classification accuracy ≤ 1.5 × random chance
                 AND |correlation| ≤ 0.30
  - ``medium`` : accuracy in (1.5 × rc, 3 × rc] OR |corr| in (0.30, 0.70]
  - ``high``   : accuracy > 3 × random chance OR |corr| > 0.70
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.experiments.workload_profiler import (
    METHOD_BY_NAME,
    _per_forward_boundary_calls,
    _per_forward_gpu_ops,
    _per_forward_trusted_compute_ops,
    _per_forward_trusted_transfer_bytes,
    _project_wall_time_ms,
)
from pllo.experiments.experiment_registry import CostModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TimingSidechannelConfig:
    seed: int = 2026
    # Workload constants — mirror Stage 5.2c defaults.
    hidden_size: int = 64
    intermediate_size: int = 128
    num_attention_heads: int = 4
    head_dim: int = 16
    layers: int = 2
    vocab_size: int = 64
    batch_size: int = 1
    # Sweep bins.
    prompt_lengths: tuple[int, ...] = (4, 8, 16, 32)
    decode_steps: tuple[int, ...] = (0, 1, 2, 3)
    methods: tuple[str, ...] = (
        "ours_current",
        "ours_compatible_nonlinear_islands",
        "tslp_trusted_nonlinear_baseline",
    )
    mitigation_bundles_under_ours_compatible: tuple[str, ...] = (
        "fresh_perm_only",
        "fresh_perm_plus_sandwich_plus_pad",
    )
    samples_per_bin: int = 16
    timing_noise_std: float = 0.05   # fractional noise (gaussian, σ = std × mean)
    # Cost model — mirror Stage 5.2c defaults.
    gpu_flops_per_ms: float = 1.0e6
    tee_to_gpu_flops_ratio: float = 0.05
    tee_call_overhead_ms: float = 0.5
    tee_bytes_per_ms: float = 1.0e6


def _cost_model_from_config(c: TimingSidechannelConfig) -> CostModel:
    return CostModel(
        tee_to_gpu_flops_ratio=c.tee_to_gpu_flops_ratio,
        tee_call_overhead_ms=c.tee_call_overhead_ms,
        tee_bytes_per_ms=c.tee_bytes_per_ms,
    )


def _consts(c: TimingSidechannelConfig) -> dict[str, int]:
    # Keys must mirror what _per_forward_*_ops in workload_profiler expects.
    return {
        "hidden": c.hidden_size,
        "inter": c.intermediate_size,
        "heads": c.num_attention_heads,
        "head_dim": c.head_dim,
        "layers": c.layers,
        "vocab": c.vocab_size,
        "dtype_bytes": 4,
    }


# ---------------------------------------------------------------------------
# Simulated latency for one (method, prompt_len, decode_step) tuple
# ---------------------------------------------------------------------------


def _simulate_latency(
    method_name: str,
    consts: dict[str, int],
    *, prompt_length: int, decode_step: int,
    cost_model: CostModel, gpu_flops_per_ms: float,
    batch_size: int,
) -> dict[str, float]:
    """Return ``{"latency_ms_no_noise", "online_boundary_calls",
    "online_gpu_ops", "online_trusted_compute_ops",
    "online_trusted_transfer_bytes"}`` for the synthetic forward."""
    method = METHOD_BY_NAME[method_name]
    # Prefill forward: tokens = prompt_length, q_len = prompt_length, k_len = prompt_length.
    # Decode step (if any): tokens = 1, q_len = 1, k_len = prompt_length + step.
    forwards: list[tuple[int, int, int]] = [(prompt_length, prompt_length, prompt_length)]
    for s in range(decode_step):
        forwards.append((1, 1, prompt_length + s + 1))
    online_boundary = 0
    online_transfer_bytes = 0
    online_trusted_compute = 0
    online_gpu = 0
    for tokens, q_len, k_len in forwards:
        online_boundary += _per_forward_boundary_calls(method, consts["layers"])
        online_transfer_bytes += _per_forward_trusted_transfer_bytes(
            method, consts, tokens,
        )
        online_trusted_compute += _per_forward_trusted_compute_ops(
            method, consts, tokens, q_len, k_len,
        )
        online_gpu += _per_forward_gpu_ops(method, consts, tokens, q_len, k_len)
    agg = {
        "online_boundary_calls": online_boundary,
        "online_trusted_compute_ops": online_trusted_compute * batch_size,
        "online_trusted_transfer_bytes": online_transfer_bytes * batch_size,
        "online_gpu_ops": online_gpu * batch_size,
    }
    latency = _project_wall_time_ms(agg, gpu_flops_per_ms, cost_model)
    return {
        "latency_ms_no_noise": float(latency),
        **{k: int(v) for k, v in agg.items()},
    }


# ---------------------------------------------------------------------------
# kNN-style classification + Pearson correlation helpers
# ---------------------------------------------------------------------------


def _knn_accuracy(
    X: torch.Tensor, y: list[int | str],
) -> tuple[float, float]:
    """Leave-one-out 1-NN classification on a feature vector ``X[:, F]``.

    Returns ``(accuracy, random_chance)``.
    """
    n = X.shape[0]
    if n < 2:
        return 0.0, 0.0
    diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs()
    dist = diff.sum(dim=-1)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.argmin(dim=-1)
    correct = sum(1 for i in range(n) if y[nn_idx[i].item()] == y[i])
    # Random-chance = sum_c (n_c (n_c - 1)) / (n (n - 1)).
    counts: dict[Any, int] = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    rc = sum(v * (v - 1) for v in counts.values()) / max(1, n * (n - 1))
    return float(correct / n), float(rc)


def _pearson(x: torch.Tensor, z: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    xz = (x - x.mean()) * (z - z.mean())
    denom = (x.std(unbiased=False) * z.std(unbiased=False)).clamp_min(1e-30)
    return float((xz.mean() / denom).item())


def _risk(accuracy: float, random_chance: float, correlation: float) -> str:
    rc = max(random_chance, 1e-12)
    ratio = accuracy / rc
    abs_corr = abs(correlation)
    if ratio > 3.0 or abs_corr > 0.70:
        return "high"
    if ratio > 1.5 or abs_corr > 0.30:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "Timing results are model-based proxies, not real TEE wall-time measurements.",
    "Latency is computed from Stage 5.2c op-count formulas plus Gaussian noise; the cost model is illustrative, not calibrated to a deployment.",
    "No hardware side-channel attack (cache, power, EM) is implemented.",
    "No real TEE isolation is evaluated; `security_profile` stays `proxy-evaluated, not formal`.",
    "Boundary-call counts are a structural property of each method and are inherently observable from any latency-aware timing channel; constant-time mitigations are NOT implemented in Stage 5.6.",
]


def run_timing_sidechannel_proxy(
    config: TimingSidechannelConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    consts = _consts(config)
    cost = _cost_model_from_config(config)
    gen = torch.Generator(device="cpu").manual_seed(config.seed + 1)

    # ----- 1. Build the dataset -----
    rows: list[dict[str, Any]] = []
    for method in config.methods:
        for L in config.prompt_lengths:
            for step in config.decode_steps:
                base = _simulate_latency(
                    method, consts,
                    prompt_length=int(L), decode_step=int(step),
                    cost_model=cost, gpu_flops_per_ms=config.gpu_flops_per_ms,
                    batch_size=config.batch_size,
                )
                mean = base["latency_ms_no_noise"]
                for _ in range(config.samples_per_bin):
                    noise = (
                        torch.randn(1, generator=gen).item()
                        * (config.timing_noise_std * abs(mean))
                    )
                    rows.append({
                        **base,
                        "method": method,
                        "prompt_length": int(L),
                        "decode_step": int(step),
                        "latency_ms": float(mean + noise),
                    })

    # ----- 2. Sub-attacks -----
    latency_t = torch.tensor([r["latency_ms"] for r in rows], dtype=torch.float64)
    length_t = torch.tensor([r["prompt_length"] for r in rows], dtype=torch.float64)
    step_t = torch.tensor([r["decode_step"] for r in rows], dtype=torch.float64)

    # Length leakage: kNN over latency feature → length-bucket label.
    acc_len, rc_len = _knn_accuracy(
        latency_t.unsqueeze(-1).to(torch.float32),
        [r["prompt_length"] for r in rows],
    )
    corr_len = _pearson(latency_t, length_t)

    # Decode-step leakage.
    acc_step, rc_step = _knn_accuracy(
        latency_t.unsqueeze(-1).to(torch.float32),
        [r["decode_step"] for r in rows],
    )
    corr_step = _pearson(latency_t, step_t)

    # Method distinguishability across all rows.
    acc_method, rc_method = _knn_accuracy(
        latency_t.unsqueeze(-1).to(torch.float32),
        [r["method"] for r in rows],
    )

    # ----- Mitigation distinguishability inside ours_compatible_nonlinear_islands -----
    mit_rows: list[dict[str, Any]] = []
    for bundle in config.mitigation_bundles_under_ours_compatible:
        for L in config.prompt_lengths:
            for step in config.decode_steps:
                base = _simulate_latency(
                    "ours_compatible_nonlinear_islands", consts,
                    prompt_length=int(L), decode_step=int(step),
                    cost_model=cost, gpu_flops_per_ms=config.gpu_flops_per_ms,
                    batch_size=config.batch_size,
                )
                mean = base["latency_ms_no_noise"]
                for _ in range(config.samples_per_bin):
                    noise = (
                        torch.randn(1, generator=gen).item()
                        * (config.timing_noise_std * abs(mean))
                    )
                    mit_rows.append({
                        **base,
                        "mitigation_bundle": bundle,
                        "prompt_length": int(L),
                        "decode_step": int(step),
                        "latency_ms": float(mean + noise),
                    })
    mit_latency_t = torch.tensor(
        [r["latency_ms"] for r in mit_rows], dtype=torch.float64,
    )
    acc_mit, rc_mit = _knn_accuracy(
        mit_latency_t.unsqueeze(-1).to(torch.float32),
        [r["mitigation_bundle"] for r in mit_rows],
    )

    # ----- Boundary-call pattern static compare -----
    boundary_pattern: list[dict[str, Any]] = []
    for method in config.methods:
        m = METHOD_BY_NAME[method]
        boundary_pattern.append({
            "method": method,
            "per_forward_boundary_calls": int(_per_forward_boundary_calls(
                m, consts["layers"],
            )),
            "boundary_call_formula": (
                f"L={consts['layers']} formula: {_describe_formula(method, consts['layers'])}"
            ),
        })

    risk_len = _risk(acc_len, rc_len, corr_len)
    risk_step = _risk(acc_step, rc_step, corr_step)
    risk_method = _risk(acc_method, rc_method, 0.0)
    risk_mit = _risk(acc_mit, rc_mit, 0.0)

    overall_max_risk = _max_risk([risk_len, risk_step, risk_method, risk_mit])

    return {
        "config": asdict(config),
        "cost_model_note": (
            "Stage 5.2c illustrative cost model + per-forward op-count "
            "formulas. NOT a real TEE wall-time measurement; "
            "wall_time_source remains 'projected_from_op_counts'."
        ),
        "prompt_length_leakage": {
            "length_bucket_accuracy": acc_len,
            "random_chance_baseline": rc_len,
            "correlation_latency_length": corr_len,
            "risk_level": risk_len,
        },
        "decode_step_leakage": {
            "step_accuracy": acc_step,
            "random_chance_baseline": rc_step,
            "correlation_latency_step": corr_step,
            "risk_level": risk_step,
        },
        "method_distinguishability": {
            "method_accuracy": acc_method,
            "random_chance_baseline": rc_method,
            "risk_level": risk_method,
        },
        "mitigation_distinguishability": {
            "mitigation_accuracy": acc_mit,
            "random_chance_baseline": rc_mit,
            "risk_level": risk_mit,
        },
        "boundary_call_pattern": boundary_pattern,
        "overall_max_risk_level": overall_max_risk,
        "limitations": list(_LIMITATIONS),
    }


def _describe_formula(method_name: str, layers: int) -> str:
    if method_name == "ours_current":
        return f"4L + 1 = {4 * layers + 1}"
    if method_name == "ours_compatible_nonlinear_islands":
        return f"L + 2 = {layers + 2}"
    if method_name == "tslp_trusted_nonlinear_baseline":
        return f"3L + 2 = {3 * layers + 2}"
    if method_name == "plain_hf_gpu":
        return "0"
    return "n/a"


def _max_risk(risks: list[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    inv = {0: "low", 1: "medium", 2: "high"}
    return inv[max(order[r] for r in risks)]


__all__ = [
    "TimingSidechannelConfig",
    "run_timing_sidechannel_proxy",
]
