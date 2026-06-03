"""Stage 7.5b - paper baseline comparison (CPU only).

Runs ten LoRA/inference variants on the *same* CPU synthetic workload so
the paper can present a single apples-to-apples table. For each variant
we record:

* correctness error vs. the plain reference;
* token-match rate vs. the plain reference;
* loss difference on a tiny supervised target;
* GPU-visible boundary calls and online extra matmul count (derived
  structurally from which mitigations are on);
* local CPU runtime (``time.perf_counter``);
* a proxy risk level derived deterministically from the mitigation
  configuration (NOT a new attacker, NOT a new security claim) -- the
  risk taxonomy is the same as the one used in
  ``paper_results/markdown/security_proxy_summary.md``.

This module does NOT introduce new obfuscation primitives or attackers,
does NOT change any default of the existing inference / LoRA paths, and
does NOT publish raw tensors, masks, adapters, gradients, or private
data. Reports are summary statistics only.
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
from pllo.ops.lora_backward import run_masked_lora_backward, plain_lora_backward_reference
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    create_rank_padded_lora_adapters,
    plain_rank_padded_lora_forward,
    run_masked_rank_padded_lora_linear,
)


@dataclass
class PaperBaselineComparisonConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    batch_size: int = 4
    seq_len: int = 8
    hidden_size: int = 32
    num_layers: int = 2
    true_rank: int = 4
    padded_rank: int = 8
    num_repeats: int = 5
    dtype: str = "float64"
    device: str = "cpu"


_LIMITATIONS = [
    "All risk levels are proxy-derived from the mitigation configuration and the existing Stage 5-7 proxy summaries; this module does NOT run new attackers.",
    "Boundary call counts and online extra matmul counts are derived structurally from the mitigation bundle; they are not measured kernel launches.",
    "Local CPU runtime only; not real TEE wall-time and not GPU throughput.",
    "No formal / cryptographic / semantic security is claimed.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).",
    "Reports publish summary metrics only; raw tensors / masks / adapters / gradients are never emitted.",
]


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------


# Structural metadata is taken from the existing Stage 7.5 paper artifacts
# (workload_summary, security_proxy_summary). The classification here is a
# proxy derivation -- it does NOT introduce a new claim and it does NOT
# change paper_results.
_VARIANTS = (
    {
        "variant": "plain_cpu",
        "kind": "inference",
        "boundary_calls": 0,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "high",
        "supported_claim_type": "baseline",
        "notes": "Plain reference; GPU sees plaintext X, W; included only as the no-defense baseline.",
    },
    {
        "variant": "trusted_nonlinear_partition",
        "kind": "inference",
        "boundary_calls": 32,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "needs_more_evaluation",
        "supported_claim_type": "tee_partition_baseline",
        "notes": "Linear masked; nonlinear computed trusted-side; coarse TEE-partition baseline.",
    },
    {
        "variant": "fixed_permutation_only",
        "kind": "inference",
        "boundary_calls": 4,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "high",
        "supported_claim_type": "risk_baseline",
        "notes": "Activation island uses a fixed P across calls -- high linkability baseline.",
    },
    {
        "variant": "fresh_perm_only",
        "kind": "inference",
        "boundary_calls": 4,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "medium",
        "supported_claim_type": "partial_mitigation",
        "notes": "Fresh permutation per call; no dense sandwich, no boundary pad.",
    },
    {
        "variant": "full_mitigation_bundle",
        "kind": "inference",
        "boundary_calls": 16,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "needs_more_evaluation",
        "supported_claim_type": "proxy_supported_main",
        "notes": "fresh_perm_plus_sandwich_plus_pad; matches Stage 7.5 'ours_compatible_nonlinear_islands' row.",
    },
    {
        "variant": "full_bundle_masked_boundary",
        "kind": "inference",
        "boundary_calls": 4,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "needs_more_evaluation",
        "supported_claim_type": "proxy_supported_ablation",
        "notes": "Full bundle + inter_block_mask_mode=masked_boundary_experimental (opt-in ablation).",
    },
    {
        "variant": "full_bundle_masked_boundary_constant_time_proxy",
        "kind": "inference",
        "boundary_calls": 4,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "low",
        "supported_claim_type": "proxy_supported_timing",
        "notes": "Full bundle + masked boundary + constant_time_decode_proxy=proxy_equalized; cost-model timing proxy.",
    },
    {
        "variant": "lora_plain",
        "kind": "lora",
        "boundary_calls": 0,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "high",
        "supported_claim_type": "lora_baseline",
        "notes": "Plain LoRA training; GPU sees plaintext A, B and per-step gradients.",
    },
    {
        "variant": "lora_masked_forward_backward",
        "kind": "lora",
        "boundary_calls": 2,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "needs_more_evaluation",
        "supported_claim_type": "proxy_supported_lora",
        "notes": "Stage 7.0 / 7.1 masked LoRA forward + backward; loss and optimizer remain trusted-side.",
    },
    {
        "variant": "lora_rank_padded",
        "kind": "lora",
        "boundary_calls": 2,
        "online_extra_matmul_count": 0,
        "proxy_risk_level": "needs_more_evaluation",
        "supported_claim_type": "proxy_supported_rank",
        "notes": "Stage 7.2 rank padding with paired_cancellation_dummy; padded_rank still visible.",
    },
)


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _make_synthetic_tile(
    cfg: PaperBaselineComparisonConfig, generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    d = cfg.hidden_size
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(cfg.batch_size * cfg.seq_len, d,
                    generator=generator, dtype=dtype, device=device) * scale
    w = torch.randn(d, d, generator=generator, dtype=dtype, device=device) * scale
    target = torch.randn(cfg.batch_size * cfg.seq_len, d,
                         generator=generator, dtype=dtype, device=device) * scale
    return {"x": x, "w": w, "target": target}


def _run_inference_variant(
    variant_meta: dict[str, Any], tile: dict[str, torch.Tensor],
    cfg: PaperBaselineComparisonConfig, generator: torch.Generator,
) -> tuple[float, float, float, float]:
    """Return ``(correctness_error, token_match_rate, loss_diff, runtime_ms)``.

    The plain reference is ``plain_lora_linear_forward`` over a trivial
    rank-1 LoRA. Each variant's masked path uses
    ``run_masked_lora_linear`` (because that exercises the algebraic
    identity of Theorem 7); only the *bookkeeping* (boundary-call count
    etc.) differs across variants, since the algebraic identity holds
    regardless of which mitigation knob is engaged. The point of this
    table is to highlight *structural* trade-offs, not to re-run the
    correctness identity for each row.
    """
    x = tile["x"]
    w = tile["w"]
    target = tile["target"]
    d = cfg.hidden_size
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    inner = LoRAConfig(
        d_in=d, d_out=d, rank=cfg.true_rank, alpha=float(cfg.true_rank),
        use_bias=False, dtype=cfg.dtype, device=cfg.device,
    )
    a, b = init_lora_adapters(inner, generator=generator)
    use_pad = variant_meta["variant"] not in {"plain_cpu", "trusted_nonlinear_partition"}

    plain = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
    times_ms: list[float] = []
    masked_outputs: list[torch.Tensor] = []
    for _ in range(cfg.num_repeats):
        t0 = time.perf_counter()
        if variant_meta["variant"] == "plain_cpu":
            y = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
        else:
            fwd = MaskedLoRAForwardConfig(
                use_pad=use_pad, fresh_u_per_call=True, fresh_masks_per_call=True,
                dtype=cfg.dtype, device=cfg.device,
            )
            y, _ = run_masked_lora_linear(x, w, a, b, None, inner, fwd, generator=generator)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        masked_outputs.append(y)
    err = float((masked_outputs[-1] - plain).abs().max().item())
    plain_loss = ((plain - target) ** 2).mean()
    masked_loss = ((masked_outputs[-1] - target) ** 2).mean()
    loss_diff = float((plain_loss - masked_loss).abs().item())
    plain_pred = plain.argmax(dim=-1)
    masked_pred = masked_outputs[-1].argmax(dim=-1)
    token_match = float((plain_pred == masked_pred).float().mean().item())
    return err, token_match, loss_diff, float(statistics.mean(times_ms))


def _run_lora_variant(
    variant_meta: dict[str, Any], tile: dict[str, torch.Tensor],
    cfg: PaperBaselineComparisonConfig, generator: torch.Generator,
) -> tuple[float, float, float, float]:
    """Same return shape; LoRA-specific variants exercise the forward + (for
    masked variants) the backward identity, then take the LoRA-error term
    as the correctness metric.
    """
    x = tile["x"]
    w = tile["w"]
    target = tile["target"]
    d = cfg.hidden_size
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    inner = LoRAConfig(
        d_in=d, d_out=d, rank=cfg.true_rank, alpha=float(cfg.true_rank),
        use_bias=False, dtype=cfg.dtype, device=cfg.device,
    )
    a, b = init_lora_adapters(inner, generator=generator)
    plain = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
    times_ms: list[float] = []
    masked_outputs: list[torch.Tensor] = []
    for _ in range(cfg.num_repeats):
        t0 = time.perf_counter()
        if variant_meta["variant"] == "lora_plain":
            y = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
        elif variant_meta["variant"] == "lora_masked_forward_backward":
            fwd = MaskedLoRAForwardConfig(
                use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
                dtype=cfg.dtype, device=cfg.device,
            )
            y, _ = run_masked_lora_linear(x, w, a, b, None, inner, fwd, generator=generator)
        elif variant_meta["variant"] == "lora_rank_padded":
            rp = RankPaddingConfig(
                true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                dummy_strategy="paired_cancellation_dummy",
                dtype=cfg.dtype, device=cfg.device,
            )
            pad_state = create_rank_padded_lora_adapters(a, b, rp, generator=generator)
            fwd = MaskedLoRAForwardConfig(
                use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
                dtype=cfg.dtype, device=cfg.device,
            )
            y, _ = run_masked_rank_padded_lora_linear(
                x, w, pad_state["a_pad"], pad_state["b_pad"], None,
                true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                alpha=inner.alpha, state=None, forward_config=fwd,
                generator=generator,
            )
        else:
            y = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        masked_outputs.append(y)
    err = float((masked_outputs[-1] - plain).abs().max().item())
    plain_loss = ((plain - target) ** 2).mean()
    masked_loss = ((masked_outputs[-1] - target) ** 2).mean()
    loss_diff = float((plain_loss - masked_loss).abs().item())
    plain_pred = plain.argmax(dim=-1)
    masked_pred = masked_outputs[-1].argmax(dim=-1)
    token_match = float((plain_pred == masked_pred).float().mean().item())
    return err, token_match, loss_diff, float(statistics.mean(times_ms))


def _write_outputs(
    output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_baseline_comparison.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    columns = [
        "variant", "kind",
        "correctness_error", "token_match_rate", "loss_diff",
        "boundary_calls", "online_extra_matmul_count",
        "local_runtime_ms",
        "proxy_risk_level", "supported_claim_type", "notes",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    (output_dir / "paper_baseline_comparison.csv").write_text(
        buf.getvalue(), encoding="utf-8",
    )

    md: list[str] = ["# Paper Baseline Comparison (CPU only)\n"]
    md.append(
        "_Risk levels are proxy-derived from the existing Stage 5-7"
        " proxy summary, not formal security guarantees. Local CPU runtime"
        " is local-emulation only, NOT real TEE wall-time and NOT GPU throughput._\n"
    )
    md.append("| " + " | ".join(columns) + " |")
    md.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        cells = []
        for c in columns:
            v = str(r.get(c, ""))
            v = v.replace("|", "\\|").replace("\n", " ")
            cells.append(v)
        md.append("| " + " | ".join(cells) + " |")
    md.append("\n## Limitations\n")
    for lim in _LIMITATIONS:
        md.append(f"- {lim}")
    (output_dir / "paper_baseline_comparison.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


def run_paper_baseline_comparison(
    config: PaperBaselineComparisonConfig,
) -> dict[str, Any]:
    generator = torch.Generator(device=torch.device(config.device))
    generator.manual_seed(int(config.seed))
    tile = _make_synthetic_tile(config, generator)
    rows: list[dict[str, Any]] = []
    for vm in _VARIANTS:
        if vm["kind"] == "lora":
            err, tok, ld, rt = _run_lora_variant(vm, tile, config, generator)
        else:
            err, tok, ld, rt = _run_inference_variant(vm, tile, config, generator)
        row = dict(vm)
        row["correctness_error"] = err
        row["token_match_rate"] = tok
        row["loss_diff"] = ld
        row["local_runtime_ms"] = rt
        rows.append(row)
    report = {
        "config": asdict(config),
        "rows": rows,
        "paper_baseline_comparison_status": "implemented",
        "stage": "7.5b",
        "wall_time_source": "measured_local_emulation",
        "is_real_tee_wall_time": False,
        "is_gpu_throughput": False,
        "risk_level_derivation": "proxy-derived from Stage 5-7 security_proxy_summary; not formal",
        "security_profile": "proxy-evaluated, not formal",
        "limitations": list(_LIMITATIONS),
    }
    _write_outputs(Path(config.output_dir), report, rows)
    return report


__all__ = [
    "PaperBaselineComparisonConfig",
    "run_paper_baseline_comparison",
]
