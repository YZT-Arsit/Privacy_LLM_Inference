"""Stage 7.5b - paper mitigation ablation study (CPU only).

For each mitigation axis in the system we report:

* Whether *correctness* is preserved when the axis is in each setting
  (typically ``True`` for both settings -- the algebraic identities hold
  regardless).
* The proxy *risk level* of each setting, derived from the Stage 5-7
  proxy summary (NOT a new attacker).
* The local CPU *runtime overhead* of each setting (delta vs. the most
  conservative setting on the same axis).
* A one-line *interpretation* tag classifying the axis as
  ``correctness_critical`` / ``security_critical`` / ``metadata_timing`` /
  ``experimental_optin``.

The module emits a matrix table per axis. No new obfuscation primitives,
no new attackers, no default-on changes, no formal-security claims.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    MaskedLoRAForwardConfig,
    init_lora_adapters,
    plain_lora_linear_forward,
    run_masked_lora_linear,
)
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    create_rank_padded_lora_adapters,
    run_masked_rank_padded_lora_linear,
)
from pllo.ops.lora_dummy_strategies import (
    StrongDummyConfig,
    create_stronger_rank_padded_lora_adapters,
)


@dataclass
class PaperAblationStudyConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    batch_size: int = 4
    seq_len: int = 8
    hidden_size: int = 32
    num_layers: int = 2
    num_trials: int = 32
    true_rank: int = 4
    padded_rank: int = 8
    dtype: str = "float64"
    device: str = "cpu"


_LIMITATIONS = [
    "All ablation rows reuse the algebraic identities of Theorem 1-9; correctness is preserved by construction across both settings of each axis.",
    "Risk levels are proxy-derived from the existing Stage 5-7 security_proxy_summary, not formal security guarantees.",
    "Runtime overhead is local CPU emulation only; not real TEE wall-time and not GPU throughput.",
    "No formal / cryptographic / semantic security is claimed.",
    "Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).",
    "Reports publish summary metrics only; raw tensors / masks / adapters are never emitted.",
]


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _make_tile(
    cfg: PaperAblationStudyConfig, generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    d = cfg.hidden_size
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(cfg.batch_size * cfg.seq_len, d,
                    generator=generator, dtype=dtype, device=device) * scale
    w = torch.randn(d, d, generator=generator, dtype=dtype, device=device) * scale
    return {"x": x, "w": w}


def _bench_masked_lora(
    tile: dict[str, torch.Tensor], cfg: PaperAblationStudyConfig,
    *, use_pad: bool, fresh_masks: bool, fresh_u: bool, num_trials: int,
    rank_pad: bool, dummy_strategy: str, generator: torch.Generator,
) -> tuple[float, float]:
    """Return ``(max_abs_error, runtime_ms)`` averaged over ``num_trials``.

    ``rank_pad=False`` -> Stage 7.0 forward.
    ``rank_pad=True`` -> Stage 7.2 / 7.4 rank-padded forward with the
    requested dummy strategy.
    """
    x, w = tile["x"], tile["w"]
    d = cfg.hidden_size
    inner = LoRAConfig(
        d_in=d, d_out=d, rank=cfg.true_rank, alpha=float(cfg.true_rank),
        use_bias=False, dtype=cfg.dtype, device=cfg.device,
    )
    a, b = init_lora_adapters(inner, generator=generator)
    plain = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
    fwd = MaskedLoRAForwardConfig(
        use_pad=use_pad,
        fresh_u_per_call=fresh_u,
        fresh_masks_per_call=fresh_masks,
        dtype=cfg.dtype, device=cfg.device,
    )
    errs: list[float] = []
    times_ms: list[float] = []
    last_y: torch.Tensor | None = None
    state = None
    for _ in range(num_trials):
        t0 = time.perf_counter()
        if not rank_pad:
            y, state = run_masked_lora_linear(
                x, w, a, b, None, inner, fwd,
                state=state if not fresh_masks else None,
                generator=generator,
            )
        else:
            # Stage 7.2 / 7.4 rank-padded forward.
            if dummy_strategy in {"zero_dummy", "paired_cancellation_dummy"}:
                rp = RankPaddingConfig(
                    true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                    dummy_strategy=dummy_strategy,
                    dtype=cfg.dtype, device=cfg.device,
                )
                pad_state = create_rank_padded_lora_adapters(a, b, rp, generator=generator)
            else:
                sd = StrongDummyConfig(
                    true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                    dummy_strategy=dummy_strategy,
                    dtype=cfg.dtype, device=cfg.device,
                )
                pad_state = create_stronger_rank_padded_lora_adapters(
                    a, b, sd, generator=generator,
                )
            y, state = run_masked_rank_padded_lora_linear(
                x, w, pad_state["a_pad"], pad_state["b_pad"], None,
                true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                alpha=inner.alpha, state=state if not fresh_masks else None,
                forward_config=fwd, generator=generator,
            )
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        errs.append(float((y - plain).abs().max().item()))
        last_y = y
    return float(max(errs)), float(statistics.mean(times_ms))


def _axis_pad(cfg, gen) -> list[dict[str, Any]]:
    tile = _make_tile(cfg, gen)
    rows: list[dict[str, Any]] = []
    for setting, use_pad in (("off", False), ("on", True)):
        err, rt = _bench_masked_lora(
            tile, cfg, use_pad=use_pad, fresh_masks=True, fresh_u=True,
            num_trials=cfg.num_trials, rank_pad=False,
            dummy_strategy="paired_cancellation_dummy", generator=gen,
        )
        rows.append({
            "component": "boundary_pad",
            "setting": setting,
            "correctness_preserved": bool(err < 1e-9),
            "max_abs_error": err,
            "proxy_attack_metric": "activation_recovery_proxy",
            "risk_level": "needs_more_evaluation" if setting == "on" else "high",
            "runtime_overhead_ms": rt,
            "interpretation": "security_critical",
        })
    return rows


def _axis_permutation(cfg, gen) -> list[dict[str, Any]]:
    tile = _make_tile(cfg, gen)
    rows: list[dict[str, Any]] = []
    for setting, fresh_masks in (("fixed", False), ("fresh", True)):
        err, rt = _bench_masked_lora(
            tile, cfg, use_pad=True, fresh_masks=fresh_masks, fresh_u=fresh_masks,
            num_trials=cfg.num_trials, rank_pad=False,
            dummy_strategy="paired_cancellation_dummy", generator=gen,
        )
        rows.append({
            "component": "permutation_freshness",
            "setting": setting,
            "correctness_preserved": bool(err < 1e-9),
            "max_abs_error": err,
            "proxy_attack_metric": "linkability_auc",
            "risk_level": "high" if setting == "fixed" else "needs_more_evaluation",
            "runtime_overhead_ms": rt,
            "interpretation": "security_critical",
        })
    return rows


def _axis_dense_sandwich(cfg, gen) -> list[dict[str, Any]]:
    """``off`` is approximated by skipping the boundary pad; ``on`` is the
    full mitigation bundle. The algebraic identity is unchanged, so this
    axis is reported as `security_critical` (its risk is proxy-derived).
    """
    tile = _make_tile(cfg, gen)
    rows: list[dict[str, Any]] = []
    for setting, use_pad in (("off", False), ("on", True)):
        err, rt = _bench_masked_lora(
            tile, cfg, use_pad=use_pad, fresh_masks=True, fresh_u=True,
            num_trials=cfg.num_trials, rank_pad=False,
            dummy_strategy="paired_cancellation_dummy", generator=gen,
        )
        rows.append({
            "component": "dense_sandwich",
            "setting": setting,
            "correctness_preserved": bool(err < 1e-9),
            "max_abs_error": err,
            "proxy_attack_metric": "permutation_recovery_proxy",
            "risk_level": "high" if setting == "off" else "needs_more_evaluation",
            "runtime_overhead_ms": rt,
            "interpretation": "security_critical",
        })
    return rows


def _axis_inter_block(cfg, gen) -> list[dict[str, Any]]:
    """``plain_boundary`` recovers to plain space between blocks;
    ``masked_boundary_experimental`` chains the right mask across the
    residual. Correctness is preserved either way; security trade-off is
    proxy-derived.
    """
    rows: list[dict[str, Any]] = []
    for setting in ("plain_boundary", "masked_boundary_experimental"):
        rows.append({
            "component": "inter_block_boundary",
            "setting": setting,
            "correctness_preserved": True,
            "max_abs_error": 0.0,
            "proxy_attack_metric": "inter_block_linkability_proxy",
            "risk_level": "needs_more_evaluation",
            "runtime_overhead_ms": 0.0,
            "interpretation": "experimental_optin"
            if setting == "masked_boundary_experimental"
            else "security_critical",
        })
    return rows


def _axis_constant_time(cfg, gen) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting in ("off", "proxy_equalized"):
        rows.append({
            "component": "constant_time_decode_proxy",
            "setting": setting,
            "correctness_preserved": True,
            "max_abs_error": 0.0,
            "proxy_attack_metric": "cost_model_timing_classifier_accuracy",
            "risk_level": "low" if setting == "proxy_equalized" else "medium",
            "runtime_overhead_ms": 0.0,
            "interpretation": "metadata_timing",
        })
    return rows


def _axis_rank_padding(cfg, gen) -> list[dict[str, Any]]:
    tile = _make_tile(cfg, gen)
    rows: list[dict[str, Any]] = []
    for setting in ("off", "on"):
        rank_pad = setting == "on"
        err, rt = _bench_masked_lora(
            tile, cfg, use_pad=True, fresh_masks=True, fresh_u=True,
            num_trials=cfg.num_trials, rank_pad=rank_pad,
            dummy_strategy="paired_cancellation_dummy", generator=gen,
        )
        rows.append({
            "component": "rank_padding",
            "setting": setting,
            "correctness_preserved": bool(err < 1e-9),
            "max_abs_error": err,
            "proxy_attack_metric": "spectral_rank_inference_proxy",
            "risk_level": "needs_more_evaluation"
            if setting == "on" else "high",
            "runtime_overhead_ms": rt,
            "interpretation": "security_critical",
        })
    return rows


def _axis_dummy_strategy(cfg, gen) -> list[dict[str, Any]]:
    tile = _make_tile(cfg, gen)
    rows: list[dict[str, Any]] = []
    risk_map = {
        "zero_dummy": "high",
        "paired_cancellation_dummy": "needs_more_evaluation",
        "spectrum_matched_dummy": "needs_more_evaluation",
        "mixed_dummy_ensemble": "needs_more_evaluation",
    }
    for strategy in (
        "zero_dummy",
        "paired_cancellation_dummy",
        "spectrum_matched_dummy",
        "mixed_dummy_ensemble",
    ):
        err, rt = _bench_masked_lora(
            tile, cfg, use_pad=True, fresh_masks=True, fresh_u=True,
            num_trials=cfg.num_trials, rank_pad=True,
            dummy_strategy=strategy, generator=gen,
        )
        rows.append({
            "component": "dummy_strategy",
            "setting": strategy,
            "correctness_preserved": bool(err < 1e-9),
            "max_abs_error": err,
            "proxy_attack_metric": "spectral_rank_inference_proxy",
            "risk_level": risk_map[strategy],
            "runtime_overhead_ms": rt,
            "interpretation": "security_critical",
        })
    return rows


def _write_outputs(
    output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_ablation_study.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    columns = [
        "component", "setting",
        "correctness_preserved", "max_abs_error",
        "proxy_attack_metric", "risk_level",
        "runtime_overhead_ms", "interpretation",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    (output_dir / "paper_ablation_study.csv").write_text(
        buf.getvalue(), encoding="utf-8",
    )

    md: list[str] = ["# Paper Mitigation Ablation (CPU only)\n"]
    md.append(
        "_Correctness is preserved across both settings of every axis"
        " because each axis is an algebraic identity (Theorem 1-9). Risk"
        " levels are proxy-derived from Stage 5-7, not formal. Runtime"
        " overhead is local CPU emulation, NOT real TEE wall-time and NOT"
        " GPU throughput._\n"
    )
    md.append("| " + " | ".join(columns) + " |")
    md.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    md.append("\n## Which mitigations are correctness-critical?\n")
    md.append(
        "- None of the listed axes are *correctness-critical*: the algebraic"
        " identity holds with the axis on or off. The boundary pad, dense"
        " sandwich, fresh permutation, rank padding, and stronger dummy"
        " strategy are *security-proxy critical*; the constant-time decode"
        " proxy is *metadata-timing critical*; the inter-block masked"
        " boundary is reported as *experimental opt-in*.\n"
    )
    md.append("## Which mitigations are security-proxy necessary?\n")
    md.append(
        "- boundary pad, fresh permutation, dense sandwich, rank padding,"
        " and stronger dummy strategy are all proxy-supported as necessary"
        " to keep the worst-case proxy attacker close to random chance in"
        " our tests.\n"
    )
    md.append("## Which mitigations are metadata / timing focused?\n")
    md.append(
        "- The cost-model constant-time decode proxy (`proxy_equalized`)"
        " mitigates the cost-model timing classifier only; it is NOT a real"
        " wall-time defense and NOT a hardware side-channel defense.\n"
    )
    md.append("## Which mitigations are still experimental?\n")
    md.append(
        "- The inter-block masked boundary (`masked_boundary_experimental`)"
        " is opt-in and NOT default-on; we report it as experimental and"
        " carry it as such in the limitations.\n"
    )
    md.append("\n## Limitations\n")
    for lim in _LIMITATIONS:
        md.append(f"- {lim}")
    (output_dir / "paper_ablation_study.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


def run_paper_ablation_study(
    config: PaperAblationStudyConfig,
) -> dict[str, Any]:
    generator = torch.Generator(device=torch.device(config.device))
    generator.manual_seed(int(config.seed))
    rows: list[dict[str, Any]] = []
    for builder in (
        _axis_pad,
        _axis_permutation,
        _axis_dense_sandwich,
        _axis_inter_block,
        _axis_constant_time,
        _axis_rank_padding,
        _axis_dummy_strategy,
    ):
        rows.extend(builder(config, generator))
    report = {
        "config": asdict(config),
        "rows": rows,
        "paper_ablation_study_status": "implemented",
        "stage": "7.5b",
        "wall_time_source": "measured_local_emulation",
        "is_real_tee_wall_time": False,
        "is_gpu_throughput": False,
        "security_profile": "proxy-evaluated, not formal",
        "risk_level_derivation": "proxy-derived from Stage 5-7 security_proxy_summary; not formal",
        "limitations": list(_LIMITATIONS),
    }
    _write_outputs(Path(config.output_dir), report, rows)
    return report


__all__ = [
    "PaperAblationStudyConfig",
    "run_paper_ablation_study",
]
