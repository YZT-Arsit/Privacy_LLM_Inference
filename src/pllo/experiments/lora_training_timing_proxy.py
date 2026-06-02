"""Stage 7.3 — LoRA training timing side-channel proxy.

A cost-model proxy for the wall-time of one LoRA training step under
Stage 7.0 / 7.1 / 7.2 (rank-padded masked forward + masked backward +
trusted optimizer). The proxy is intentionally NOT a real TEE
wall-clock measurement and does NOT sleep. Its job is to rank
configurations by their *projected* latency profile and ask whether a
passive timing observer can infer secret inputs (batch_size, seq_len,
true_rank, padded_rank, number of LoRA modules, optimizer type,
dummy_strategy / rank padding enabled) from latency alone.

Mitigation:

* ``constant_time_training_mode="off"`` — report leakage as it stands.
* ``constant_time_training_mode="proxy_equalized"`` — every step is
  padded to the upper bucket of each sensitive dimension. We do NOT
  actually sleep; the proxy reports the resulting upper-bound latency
  + overhead ratio so the paper can quote "the proxy equalization buys
  X% extra latency for Y bits of leakage reduction".

Limitations:

* Timing results are proxy estimates, not real TEE wall-time.
* Constant-time training mode is simulated and does not sleep.
* No hardware side-channel is evaluated.
* The cost model uses coarse FLOP / boundary constants; absolute
  numbers are illustrative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


VALID_CONSTANT_TIME_MODES: tuple[str, ...] = (
    "off",
    "proxy_equalized",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainingTimingProxyConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    seq_lens: tuple[int, ...] = (4, 8, 16)
    true_ranks: tuple[int, ...] = (2, 4, 8)
    padded_ranks: tuple[int, ...] = (8, 16)
    num_lora_modules: tuple[int, ...] = (2, 4, 7, 14)
    optimizers: tuple[str, ...] = ("sgd", "adamw")
    timing_noise_std: float = 0.05
    constant_time_training_mode: str = "off"
    samples_per_config: int = 16
    base_hidden: int = 64
    base_intermediate: int = 128
    # Cost-model constants. All are unitless proxies — we do not claim
    # real wall-time.
    gpu_cost_per_flop_ms: float = 1e-9
    trusted_cost_per_flop_ms: float = 4e-9
    mask_gen_cost_per_flop_ms: float = 4e-9
    boundary_call_cost_ms: float = 0.02
    base_overhead_ms: float = 0.10
    rank_padding_dummy_cost_ms: float = 0.01
    dtype: str = "float64"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def _module_io_proxy(num_modules: int, hidden: int, inter: int) -> list[
    tuple[int, int]
]:
    """Return (d_in, d_out) per module for a cost-model sweep.

    Distribution: roughly mirrors the Stage 7.3 multi-layer model
    (q/k/v/o = hidden→hidden, gate/up = hidden→inter, down = inter→hidden).
    """
    pattern = [
        (hidden, hidden),
        (hidden, hidden),
        (hidden, hidden),
        (hidden, hidden),
        (hidden, inter),
        (hidden, inter),
        (inter, hidden),
    ]
    out = []
    for i in range(num_modules):
        out.append(pattern[i % len(pattern)])
    return out


def _forward_ops_per_module(
    batch: int, seq: int, d_in: int, d_out: int,
    rank: int, rank_padding_on: bool,
) -> int:
    base = 2 * batch * seq * d_in * d_out
    lora_a = 2 * batch * seq * d_in * rank
    lora_b = 2 * batch * seq * rank * d_out
    return base + lora_a + lora_b


def _backward_ops_per_module(
    batch: int, seq: int, d_in: int, d_out: int, rank: int,
) -> int:
    grad_a = 2 * batch * seq * d_in * rank + 2 * d_in * batch * seq * rank
    grad_b = 2 * batch * seq * rank * d_out + 2 * rank * batch * seq * d_out
    grad_x = 2 * batch * seq * d_in * d_out
    return grad_a + grad_b + grad_x


def _trusted_optimizer_ops_per_module(
    optimizer: str, d_in: int, d_out: int, true_rank: int,
) -> int:
    base = 2 * (d_in * true_rank + true_rank * d_out)
    if optimizer == "sgd":
        return base
    if optimizer == "adamw":
        return 8 * base
    return base


def _mask_generation_ops_per_module(
    d_in: int, d_out: int, padded_rank: int,
) -> int:
    return d_in ** 3 + d_out ** 3 + padded_rank ** 3 + 2 * d_in * d_out


def _boundary_calls_per_module(rank_padding_on: bool) -> int:
    return 2 if not rank_padding_on else 3


def _step_latency_ms(
    *,
    batch_size: int,
    seq_len: int,
    true_rank: int,
    padded_rank: int,
    num_modules: int,
    optimizer: str,
    dummy_strategy: str,
    rank_padding_on: bool,
    hidden: int,
    intermediate: int,
    config: LoRATrainingTimingProxyConfig,
    noise: float,
) -> dict[str, float]:
    """Return per-step latency components for one config."""
    rank_for_compute = padded_rank if rank_padding_on else true_rank
    modules = _module_io_proxy(num_modules, hidden, intermediate)
    forward_ops = 0
    backward_ops = 0
    trusted_optimizer_ops = 0
    mask_generation_ops = 0
    boundary_calls = 0
    for d_in, d_out in modules:
        forward_ops += _forward_ops_per_module(
            batch_size, seq_len, d_in, d_out,
            rank_for_compute, rank_padding_on,
        )
        backward_ops += _backward_ops_per_module(
            batch_size, seq_len, d_in, d_out, rank_for_compute,
        )
        trusted_optimizer_ops += _trusted_optimizer_ops_per_module(
            optimizer, d_in, d_out, true_rank,
        )
        mask_generation_ops += _mask_generation_ops_per_module(
            d_in, d_out, padded_rank if rank_padding_on else true_rank,
        )
        boundary_calls += _boundary_calls_per_module(rank_padding_on)
    dummy_cost = 0.0
    if rank_padding_on:
        dummy_factor = 1.0 if dummy_strategy == "zero_dummy" else 2.0
        dummy_cost = (
            num_modules
            * dummy_factor
            * config.rank_padding_dummy_cost_ms
        )
    forward_ms = forward_ops * config.gpu_cost_per_flop_ms
    backward_ms = backward_ops * config.gpu_cost_per_flop_ms
    optimizer_ms = trusted_optimizer_ops * config.trusted_cost_per_flop_ms
    mask_gen_ms = mask_generation_ops * config.mask_gen_cost_per_flop_ms
    boundary_ms = boundary_calls * config.boundary_call_cost_ms
    total_ms_clean = (
        config.base_overhead_ms
        + forward_ms + backward_ms + optimizer_ms + mask_gen_ms
        + boundary_ms + dummy_cost
    )
    total_ms_with_noise = max(0.0, total_ms_clean * (1.0 + noise))
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "mask_generation_ms": mask_gen_ms,
        "boundary_ms": boundary_ms,
        "rank_padding_dummy_ms": dummy_cost,
        "base_overhead_ms": config.base_overhead_ms,
        "total_ms_clean": total_ms_clean,
        "total_ms_with_noise": total_ms_with_noise,
        "forward_ops": int(forward_ops),
        "backward_ops": int(backward_ops),
        "trusted_optimizer_ops": int(trusted_optimizer_ops),
        "mask_generation_ops": int(mask_generation_ops),
        "boundary_calls": int(boundary_calls),
    }


# ---------------------------------------------------------------------------
# Leakage classifier (nearest-bucket-mean)
# ---------------------------------------------------------------------------


def _classify_task(
    samples: list[dict[str, Any]],
    *,
    target_key: str,
    targets: tuple[Any, ...],
) -> dict[str, Any]:
    """Given samples (each with latency + true target), report how often
    the latency alone reveals the target bucket.

    The classifier is *nearest-bucket-mean* on the noisy latency: compute
    the mean latency per bucket from the samples themselves, then assign
    each sample to its closest bucket mean. This is a generous attacker
    (it sees the per-bucket means computed from training labels), so the
    accuracy is an upper bound on what a black-box timing attacker could
    do without those labels.
    """
    bucket_to_latencies: dict[Any, list[float]] = {t: [] for t in targets}
    for s in samples:
        bucket_to_latencies[s[target_key]].append(s["total_ms_with_noise"])
    bucket_means = {
        b: float(sum(v) / max(1, len(v)))
        for b, v in bucket_to_latencies.items() if v
    }
    if not bucket_means:
        return {
            "classification_accuracy": 0.0,
            "random_chance_baseline": 1.0 / max(1, len(targets)),
            "bucket_separation": 0.0,
            "risk_level": "low",
        }
    correct = 0
    total = 0
    for s in samples:
        latency = s["total_ms_with_noise"]
        best = None
        best_diff = float("inf")
        for b, m in bucket_means.items():
            diff = abs(latency - m)
            if diff < best_diff:
                best_diff = diff
                best = b
        total += 1
        if best == s[target_key]:
            correct += 1
    accuracy = correct / max(1, total)
    chance = 1.0 / max(1, len(targets))
    # Inter-bucket / intra-bucket spread proxy.
    means = list(bucket_means.values())
    if len(means) <= 1:
        separation = 0.0
    else:
        mean_of_means = sum(means) / len(means)
        spread = max(means) - min(means)
        separation = float(spread / max(abs(mean_of_means), 1e-12))
    if accuracy >= chance + 0.35:
        risk = "high"
    elif accuracy >= chance + 0.15:
        risk = "medium"
    elif accuracy >= chance + 0.05:
        risk = "low"
    else:
        risk = "low"
    return {
        "classification_accuracy": float(accuracy),
        "random_chance_baseline": float(chance),
        "bucket_separation": separation,
        "risk_level": risk,
    }


# ---------------------------------------------------------------------------
# Sweep + leakage tasks
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "Timing results are proxy estimates from an FLOP-and-boundary cost model, not real TEE wall-time.",
    "Constant-time training mode is simulated (upper-bound latency padding); the proxy does NOT sleep, does NOT modify real runtime.",
    "Hardware side-channels (cache / power / EM) are NOT evaluated.",
    "Cost-model constants are coarse; absolute latencies are illustrative.",
    "Leakage classifier uses bucket-mean labels as the attacker; this is a generous attacker model.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "No real Qwen / TinyLlama / LLaMA fine-tuning; this is a multi-linear cost-model proxy.",
    "Adapter is NEVER merged into the public base weight W.",
    "No formal / cryptographic / semantic security is claimed.",
]


def _sample_noise(
    generator: torch.Generator, sigma: float,
) -> float:
    if sigma <= 0.0:
        return 0.0
    return float(
        torch.randn(
            (1,), generator=generator, dtype=torch.float64,
        ).item() * sigma
    )


def _equalize_to_upper(
    config: LoRATrainingTimingProxyConfig,
) -> dict[str, Any]:
    """Compute the upper-bound latency for the proxy_equalized mode.

    Pads every sensitive dimension to its max bucket so the resulting
    latency is independent of the secret.
    """
    upper_bs = max(config.batch_sizes)
    upper_sl = max(config.seq_lens)
    upper_pr = max(config.padded_ranks)
    upper_nm = max(config.num_lora_modules)
    upper_optimizer = "adamw"  # adamw is more expensive than sgd
    # Worst-case rank for compute = upper_pr (rank padding always on).
    samples: dict[str, float] = {}
    for dummy_strategy in ("zero_dummy", "paired_cancellation_dummy"):
        latency = _step_latency_ms(
            batch_size=upper_bs, seq_len=upper_sl,
            true_rank=min(config.true_ranks),
            padded_rank=upper_pr,
            num_modules=upper_nm,
            optimizer=upper_optimizer,
            dummy_strategy=dummy_strategy,
            rank_padding_on=True,
            hidden=config.base_hidden,
            intermediate=config.base_intermediate,
            config=config,
            noise=0.0,
        )["total_ms_clean"]
        samples[dummy_strategy] = latency
    upper_latency_ms = max(samples.values())
    return {
        "upper_batch_size": upper_bs,
        "upper_seq_len": upper_sl,
        "upper_padded_rank": upper_pr,
        "upper_num_modules": upper_nm,
        "upper_optimizer": upper_optimizer,
        "upper_latency_ms": float(upper_latency_ms),
    }


def run_lora_training_timing_proxy(
    config: LoRATrainingTimingProxyConfig,
) -> dict[str, Any]:
    if config.constant_time_training_mode not in VALID_CONSTANT_TIME_MODES:
        raise ValueError(
            f"unknown constant_time_training_mode"
            f" {config.constant_time_training_mode!r};"
            f" expected one of {VALID_CONSTANT_TIME_MODES}"
        )
    if not config.batch_sizes or not config.seq_lens:
        raise ValueError("batch_sizes and seq_lens must be non-empty")
    if not config.true_ranks or not config.padded_ranks:
        raise ValueError("true_ranks and padded_ranks must be non-empty")
    if not config.num_lora_modules or not config.optimizers:
        raise ValueError("num_lora_modules and optimizers must be non-empty")

    rng = torch.Generator(device="cpu").manual_seed(config.seed)

    # ---- Generate the sweep (rank padding ON; default dummy strategy) ----
    samples_default: list[dict[str, Any]] = []
    samples_rank_padding_off: list[dict[str, Any]] = []
    samples_zero_dummy: list[dict[str, Any]] = []
    samples_paired_dummy: list[dict[str, Any]] = []

    def _gen_block(
        rank_padding_on: bool, dummy_strategy: str,
        target_list: list[dict[str, Any]],
    ) -> None:
        for batch in config.batch_sizes:
            for seq in config.seq_lens:
                for tr in config.true_ranks:
                    for pr in config.padded_ranks:
                        if pr < tr:
                            continue
                        for nm in config.num_lora_modules:
                            for opt in config.optimizers:
                                for _ in range(config.samples_per_config):
                                    noise = _sample_noise(
                                        rng, config.timing_noise_std,
                                    )
                                    lat = _step_latency_ms(
                                        batch_size=batch,
                                        seq_len=seq,
                                        true_rank=tr,
                                        padded_rank=pr,
                                        num_modules=nm,
                                        optimizer=opt,
                                        dummy_strategy=dummy_strategy,
                                        rank_padding_on=rank_padding_on,
                                        hidden=config.base_hidden,
                                        intermediate=config.base_intermediate,
                                        config=config,
                                        noise=noise,
                                    )
                                    record = {
                                        "batch_size": batch,
                                        "seq_len": seq,
                                        "true_rank": tr,
                                        "padded_rank": pr,
                                        "num_modules": nm,
                                        "optimizer": opt,
                                        "dummy_strategy": dummy_strategy,
                                        "rank_padding_on": rank_padding_on,
                                        **lat,
                                    }
                                    target_list.append(record)

    _gen_block(True, "paired_cancellation_dummy", samples_default)
    _gen_block(False, "paired_cancellation_dummy", samples_rank_padding_off)
    _gen_block(True, "zero_dummy", samples_zero_dummy)
    _gen_block(True, "paired_cancellation_dummy", samples_paired_dummy)

    # ---- Leakage tasks under constant_time_training_mode="off" ----
    leakage_tasks_off: dict[str, dict[str, Any]] = {}
    leakage_tasks_off["batch_size"] = _classify_task(
        samples_default, target_key="batch_size", targets=tuple(config.batch_sizes),
    )
    leakage_tasks_off["seq_len"] = _classify_task(
        samples_default, target_key="seq_len", targets=tuple(config.seq_lens),
    )
    leakage_tasks_off["true_rank"] = _classify_task(
        samples_default, target_key="true_rank", targets=tuple(config.true_ranks),
    )
    leakage_tasks_off["padded_rank"] = _classify_task(
        samples_default, target_key="padded_rank", targets=tuple(config.padded_ranks),
    )
    leakage_tasks_off["num_modules"] = _classify_task(
        samples_default, target_key="num_modules",
        targets=tuple(config.num_lora_modules),
    )
    leakage_tasks_off["optimizer"] = _classify_task(
        samples_default, target_key="optimizer", targets=tuple(config.optimizers),
    )
    # Rank padding enabled vs off
    rank_padding_samples = samples_default + samples_rank_padding_off
    leakage_tasks_off["rank_padding_on"] = _classify_task(
        rank_padding_samples, target_key="rank_padding_on", targets=(True, False),
    )
    # Dummy strategy zero vs paired
    dummy_samples = samples_zero_dummy + samples_paired_dummy
    leakage_tasks_off["dummy_strategy"] = _classify_task(
        dummy_samples, target_key="dummy_strategy",
        targets=("zero_dummy", "paired_cancellation_dummy"),
    )

    # ---- Constant-time training proxy ----
    constant_time = _equalize_to_upper(config)
    if config.constant_time_training_mode == "proxy_equalized":
        # Re-emit samples padded to the upper-bound latency with a FRESH
        # noise sample drawn from the upper bucket. The original
        # noise carries the original latency's amplitude so we cannot
        # reuse it here; resampling cuts the secret-dependent signal.
        upper = constant_time["upper_latency_ms"]

        def _equalize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for s in samples:
                fresh = _sample_noise(rng, config.timing_noise_std)
                out.append({
                    **s,
                    "total_ms_clean": upper,
                    "total_ms_with_noise": max(0.0, upper * (1.0 + fresh)),
                })
            return out

        equalized_samples_default = _equalize(samples_default)
        equalized_samples_rank_off = _equalize(samples_rank_padding_off)
        equalized_samples_zero = _equalize(samples_zero_dummy)
        equalized_samples_paired = _equalize(samples_paired_dummy)
        leakage_tasks_eq: dict[str, dict[str, Any]] = {}
        leakage_tasks_eq["batch_size"] = _classify_task(
            equalized_samples_default, target_key="batch_size",
            targets=tuple(config.batch_sizes),
        )
        leakage_tasks_eq["seq_len"] = _classify_task(
            equalized_samples_default, target_key="seq_len",
            targets=tuple(config.seq_lens),
        )
        leakage_tasks_eq["true_rank"] = _classify_task(
            equalized_samples_default, target_key="true_rank",
            targets=tuple(config.true_ranks),
        )
        leakage_tasks_eq["padded_rank"] = _classify_task(
            equalized_samples_default, target_key="padded_rank",
            targets=tuple(config.padded_ranks),
        )
        leakage_tasks_eq["num_modules"] = _classify_task(
            equalized_samples_default, target_key="num_modules",
            targets=tuple(config.num_lora_modules),
        )
        leakage_tasks_eq["optimizer"] = _classify_task(
            equalized_samples_default, target_key="optimizer",
            targets=tuple(config.optimizers),
        )
        leakage_tasks_eq["rank_padding_on"] = _classify_task(
            equalized_samples_default + equalized_samples_rank_off,
            target_key="rank_padding_on", targets=(True, False),
        )
        leakage_tasks_eq["dummy_strategy"] = _classify_task(
            equalized_samples_zero + equalized_samples_paired,
            target_key="dummy_strategy",
            targets=("zero_dummy", "paired_cancellation_dummy"),
        )
    else:
        leakage_tasks_eq = {}

    # ---- Overhead proxy ----
    mean_native = float(
        sum(s["total_ms_clean"] for s in samples_default)
        / max(1, len(samples_default))
    )
    overhead_ratio = float(
        (constant_time["upper_latency_ms"] / max(mean_native, 1e-12)) - 1.0
    )
    overhead_proxy = {
        "mean_native_latency_ms": mean_native,
        "upper_latency_ms": constant_time["upper_latency_ms"],
        "overhead_ratio": overhead_ratio,
        "overhead_pct": float(overhead_ratio * 100.0),
        "note": (
            "Proxy estimate; no real wall-clock measurement. The"
            " equalization pads every step to the upper bucket latency"
            " across sensitive dimensions, so the overhead is the maximum"
            " of the original variance."
        ),
    }

    # ---- Aggregate ----
    summary_off = {
        k: v["classification_accuracy"]
        for k, v in leakage_tasks_off.items()
    }
    summary_eq = {
        k: v["classification_accuracy"]
        for k, v in leakage_tasks_eq.items()
    } if leakage_tasks_eq else {}
    max_acc_off = max(summary_off.values(), default=0.0)
    max_acc_eq = max(summary_eq.values(), default=0.0) if summary_eq else None
    leakage_reduction = (
        None if max_acc_eq is None
        else float(max(0.0, max_acc_off - max_acc_eq))
    )

    return {
        "config": asdict(config),
        "scope": (
            "FLOP / boundary cost-model proxy for LoRA training step"
            " latency; not real TEE wall-time"
        ),
        "training_timing_model": {
            "components": [
                "forward_ms",
                "backward_ms",
                "optimizer_ms",
                "mask_generation_ms",
                "boundary_ms",
                "rank_padding_dummy_ms",
                "base_overhead_ms",
            ],
            "cost_model_constants": {
                "gpu_cost_per_flop_ms": config.gpu_cost_per_flop_ms,
                "trusted_cost_per_flop_ms": config.trusted_cost_per_flop_ms,
                "mask_gen_cost_per_flop_ms": config.mask_gen_cost_per_flop_ms,
                "boundary_call_cost_ms": config.boundary_call_cost_ms,
                "base_overhead_ms": config.base_overhead_ms,
                "rank_padding_dummy_cost_ms": config.rank_padding_dummy_cost_ms,
                "timing_noise_std": config.timing_noise_std,
            },
            "num_samples_default": len(samples_default),
            "num_samples_rank_padding_off": len(samples_rank_padding_off),
            "num_samples_zero_dummy": len(samples_zero_dummy),
            "num_samples_paired_dummy": len(samples_paired_dummy),
            "note": (
                "Each sample = one training step latency under one"
                " (batch_size, seq_len, true_rank, padded_rank, num_modules,"
                " optimizer, dummy_strategy, rank_padding_on) configuration,"
                " with Gaussian timing noise applied."
            ),
        },
        "leakage_tasks_off": leakage_tasks_off,
        "leakage_tasks_proxy_equalized": leakage_tasks_eq,
        "constant_time_training_proxy": {
            "constant_time_training_mode": config.constant_time_training_mode,
            "upper_bucket": constant_time,
            "did_actually_sleep": False,
            "note": (
                "proxy_equalized pads every step to the upper-bucket"
                " latency; we never invoke real sleep / runtime gating."
            ),
        },
        "overhead_proxy": overhead_proxy,
        "summary": {
            "max_classification_accuracy_off": max_acc_off,
            "max_classification_accuracy_proxy_equalized": max_acc_eq,
            "leakage_reduction_after_equalization": leakage_reduction,
            "leakage_summary_off": summary_off,
            "leakage_summary_proxy_equalized": summary_eq,
        },
        "interpretation": {
            "headline": (
                "Cost-model proxy: under constant_time_training_mode='off'"
                f" the worst-case classifier accuracy is"
                f" {max_acc_off:.3f}; under proxy_equalized it is"
                f" {max_acc_eq if max_acc_eq is not None else 'N/A'}."
            ),
            "overhead_summary": (
                f"Proxy overhead from equalization is"
                f" {overhead_proxy['overhead_pct']:.1f}%."
            ),
            "merge_adapter_into_w": False,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_timing": (
            "lora-training-timing-proxy-evaluated, not formal"
        ),
        "lora_training_timing_proxy_status": "implemented",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.4 — stronger dummy distributions / spectral-rank hardening.",
            "Stage 7.x — real TEE wall-time integration with actual constant-time gating.",
            "Stage 7.x — hardware side-channel (cache / power) proxies.",
        ],
    }


def lora_training_timing_proxy_csv_rows(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if isinstance(v, (tuple, list)):
            v = "|".join(str(x) for x in v)
        rows.append({
            "section": "config",
            "attack": "n/a",
            "strategy": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for k, v in report["training_timing_model"]["cost_model_constants"].items():
        rows.append({
            "section": "cost_model_constants",
            "attack": "n/a",
            "strategy": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for task, payload in report["leakage_tasks_off"].items():
        for k, v in payload.items():
            rows.append({
                "section": "leakage_tasks_off",
                "attack": task,
                "strategy": "constant_time_off",
                "metric": k,
                "value": v,
                "notes": "",
            })
    for task, payload in report["leakage_tasks_proxy_equalized"].items():
        for k, v in payload.items():
            rows.append({
                "section": "leakage_tasks_proxy_equalized",
                "attack": task,
                "strategy": "constant_time_proxy_equalized",
                "metric": k,
                "value": v,
                "notes": "",
            })
    cttp = report["constant_time_training_proxy"]
    rows.append({
        "section": "constant_time_training_proxy",
        "attack": "n/a",
        "strategy": cttp["constant_time_training_mode"],
        "metric": "did_actually_sleep",
        "value": cttp["did_actually_sleep"],
        "notes": cttp["note"],
    })
    for k, v in cttp["upper_bucket"].items():
        rows.append({
            "section": "constant_time_training_proxy",
            "attack": "upper_bucket",
            "strategy": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for k, v in report["overhead_proxy"].items():
        rows.append({
            "section": "overhead_proxy",
            "attack": "n/a",
            "strategy": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for k, v in report["summary"].items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append({
                    "section": "summary",
                    "attack": k,
                    "strategy": k2,
                    "metric": "accuracy",
                    "value": v2,
                    "notes": "",
                })
        else:
            rows.append({
                "section": "summary",
                "attack": "summary",
                "strategy": "summary",
                "metric": k,
                "value": v,
                "notes": "",
            })
    return rows


__all__ = [
    "LoRATrainingTimingProxyConfig",
    "VALID_CONSTANT_TIME_MODES",
    "lora_training_timing_proxy_csv_rows",
    "run_lora_training_timing_proxy",
]
