"""Stage 7.5 — measured runtime evaluation (local emulation, NOT real TEE).

Drives existing inference / LoRA primitives under a small synthetic
workload and records wall-clock latency via ``time.perf_counter`` for
the paper's "projected vs measured" alignment figure. **This is local
runtime emulation only. It is NOT a real TEE wall-time measurement and
does NOT prove TEE performance.**

The components benchmarked here are (each independent, skipped on
failure with a recorded reason):

* ``plain_synthetic_linear`` — pure ``X W`` matmul on a synthetic tile.
* ``masked_linear`` — Stage 7.0 ``run_masked_lora_linear`` forward.
* ``masked_lora_backward`` — Stage 7.1 backward + recovery.
* ``rank_padded_forward`` — Stage 7.2 rank-padded masked forward.
* ``multi_layer_lora_training_step`` — Stage 7.3 ``run_multilayer_lora_training``.
* ``modern_decoder_model_wrapper`` — Stage 6.4c full forward via the
  obfuscated wrapper (synthetic tiny config), if available.

For each component we record:
``mean_ms``, ``median_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
``num_warmup``, ``num_repeats``, ``device``, ``dtype``,
``wall_time_source = "measured_local_emulation"``, and optional
``skipped_with_reason``.

This module never calls ``time.sleep``, never downloads models, and
never publishes raw tensors / masks / adapters / gradients.
"""

from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from pllo.experiments.multilayer_lora_training import (
    MultiLayerLoRATrainingConfig,
    run_multilayer_lora_training,
)
from pllo.ops.lora import (
    LoRAConfig,
    MaskedLoRAForwardConfig,
    init_lora_adapters,
    plain_lora_linear_forward,
    run_masked_lora_linear,
)
from pllo.ops.lora_backward import run_masked_lora_backward
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    create_rank_padded_lora_adapters,
    run_masked_rank_padded_lora_linear,
)


@dataclass
class MeasuredRuntimeEvaluationConfig:
    output_dir: str = "paper_results"
    seed: int = 2026
    num_warmup: int = 3
    num_repeats: int = 10
    device: str = "cpu"
    dtype: str = "float64"
    include_gpu_if_available: bool = False
    strict: bool = False
    # Workload sizes (kept small so pytest stays fast).
    linear_d_in: int = 64
    linear_d_out: int = 32
    linear_batch: int = 8
    lora_rank: int = 4
    lora_padded_rank: int = 8
    multilayer_num_layers: int = 2
    multilayer_hidden_size: int = 16
    multilayer_intermediate_size: int = 24
    multilayer_seq_len: int = 4
    multilayer_batch: int = 2
    multilayer_num_steps: int = 1
    include_modern_decoder_wrapper: bool = False


_LIMITATIONS = [
    "This is local runtime emulation, not real TEE wall-time.",
    "No real sleep, no real runtime gating; ``time.perf_counter`` only.",
    "Workload tiles are small for pytest stability — absolute numbers are illustrative.",
    "Modern decoder model-wrapper benchmark is opt-in and recorded as skipped when unavailable.",
    "No formal / cryptographic / semantic security is claimed.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).",
    "Reports publish timing statistics only — raw tensors, raw adapters, raw gradients, and dense masks are never emitted.",
]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _time_block(
    fn: Callable[[], Any], *, num_warmup: int, num_repeats: int,
) -> dict[str, float]:
    """Run ``fn`` warmup + repeat times and return latency statistics."""
    for _ in range(num_warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "std_ms": (
            float(statistics.stdev(times_ms)) if len(times_ms) >= 2 else 0.0
        ),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def _row(
    component: str, variant: str, config: MeasuredRuntimeEvaluationConfig,
    *, fn: Callable[[], Any] | None,
    skip_reason: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    base = {
        "component": component,
        "variant": variant,
        "num_warmup": config.num_warmup,
        "num_repeats": config.num_repeats,
        "device": config.device,
        "dtype": config.dtype,
        "wall_time_source": "measured_local_emulation",
        "notes": notes,
    }
    if fn is None or skip_reason is not None:
        base.update({
            "mean_ms": None,
            "median_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
            "skipped_with_reason": skip_reason or "no fn provided",
        })
        return base
    try:
        stats = _time_block(
            fn, num_warmup=config.num_warmup, num_repeats=config.num_repeats,
        )
        base.update(stats)
        base["skipped_with_reason"] = None
    except Exception as e:  # noqa: BLE001
        if config.strict:
            raise
        base.update({
            "mean_ms": None,
            "median_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
            "skipped_with_reason": (
                f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
            ),
        })
    return base


# ---------------------------------------------------------------------------
# Benchmark factories
# ---------------------------------------------------------------------------


def _bench_plain_linear(config: MeasuredRuntimeEvaluationConfig):
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    x = torch.randn(
        config.linear_batch, config.linear_d_in,
        generator=g, dtype=dtype, device=device,
    )
    w = torch.randn(
        config.linear_d_in, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )

    def _fn():
        _ = x @ w
    return _fn


def _bench_masked_lora_forward(config: MeasuredRuntimeEvaluationConfig):
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    lora_cfg = LoRAConfig(
        d_in=config.linear_d_in, d_out=config.linear_d_out,
        rank=config.lora_rank, alpha=1.0,
        dtype=config.dtype, device=config.device,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
        dtype=config.dtype, device=config.device,
    )
    a, b = init_lora_adapters(lora_cfg, generator=g)
    b = b + 1e-3 * torch.randn(
        config.lora_rank, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    w = torch.randn(
        config.linear_d_in, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    x = torch.randn(
        config.linear_batch, config.linear_d_in,
        generator=g, dtype=dtype, device=device,
    )

    def _fn():
        _ = run_masked_lora_linear(x, w, a, b, None, lora_cfg, fcfg)
    return _fn


def _bench_plain_lora_forward(config: MeasuredRuntimeEvaluationConfig):
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    lora_cfg = LoRAConfig(
        d_in=config.linear_d_in, d_out=config.linear_d_out,
        rank=config.lora_rank, alpha=1.0,
        dtype=config.dtype, device=config.device,
    )
    a, b = init_lora_adapters(lora_cfg, generator=g)
    b = b + 1e-3 * torch.randn(
        config.lora_rank, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    w = torch.randn(
        config.linear_d_in, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    x = torch.randn(
        config.linear_batch, config.linear_d_in,
        generator=g, dtype=dtype, device=device,
    )

    def _fn():
        _ = plain_lora_linear_forward(x, w, a, b, None, alpha=1.0)
    return _fn


def _bench_masked_lora_backward(config: MeasuredRuntimeEvaluationConfig):
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    lora_cfg = LoRAConfig(
        d_in=config.linear_d_in, d_out=config.linear_d_out,
        rank=config.lora_rank, alpha=1.0,
        dtype=config.dtype, device=config.device,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
        dtype=config.dtype, device=config.device,
    )
    a, b = init_lora_adapters(lora_cfg, generator=g)
    b = b + 1e-3 * torch.randn(
        config.lora_rank, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    w = torch.randn(
        config.linear_d_in, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    x = torch.randn(
        config.linear_batch, config.linear_d_in,
        generator=g, dtype=dtype, device=device,
    )
    grad_y = torch.randn(
        config.linear_batch, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    _, state = run_masked_lora_linear(x, w, a, b, None, lora_cfg, fcfg)

    def _fn():
        _ = run_masked_lora_backward(
            x, w, a, b, grad_y,
            alpha=1.0,
            n_in=state.n_in, n_in_inv=state.n_in_inv,
            n_out=state.n_out, u=state.u, u_inv=state.u_inv,
            pad=state.pad,
            recover_grad_x=False,
        )
    return _fn


def _bench_rank_padded_forward(config: MeasuredRuntimeEvaluationConfig):
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    rpc = RankPaddingConfig(
        true_rank=config.lora_rank, padded_rank=config.lora_padded_rank,
        dummy_strategy="paired_cancellation_dummy",
        dtype=config.dtype, device=config.device,
    )
    lora_cfg = LoRAConfig(
        d_in=config.linear_d_in, d_out=config.linear_d_out,
        rank=config.lora_rank, alpha=1.0,
        dtype=config.dtype, device=config.device,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
        dtype=config.dtype, device=config.device,
    )
    a, b = init_lora_adapters(lora_cfg, generator=g)
    b = b + 1e-3 * torch.randn(
        config.lora_rank, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    w = torch.randn(
        config.linear_d_in, config.linear_d_out,
        generator=g, dtype=dtype, device=device,
    )
    x = torch.randn(
        config.linear_batch, config.linear_d_in,
        generator=g, dtype=dtype, device=device,
    )

    def _fn():
        pack = create_rank_padded_lora_adapters(a, b, rpc, generator=g)
        _ = run_masked_rank_padded_lora_linear(
            x, w, pack["a_pad"], pack["b_pad"], None,
            true_rank=config.lora_rank,
            padded_rank=config.lora_padded_rank,
            alpha=1.0,
            state=None, forward_config=fcfg, generator=g,
        )
    return _fn


def _bench_multilayer_training(config: MeasuredRuntimeEvaluationConfig):
    train_cfg = MultiLayerLoRATrainingConfig(
        seed=config.seed,
        num_layers=config.multilayer_num_layers,
        hidden_size=config.multilayer_hidden_size,
        intermediate_size=config.multilayer_intermediate_size,
        vocab_size=32,
        seq_len=config.multilayer_seq_len,
        batch_size=config.multilayer_batch,
        true_rank=2, padded_rank=4,
        num_steps=config.multilayer_num_steps,
        optimizer="sgd", lr=1e-2,
        dtype=config.dtype, device=config.device,
    )

    def _fn():
        _ = run_multilayer_lora_training(train_cfg)
    return _fn


def _bench_modern_decoder_wrapper(config: MeasuredRuntimeEvaluationConfig):
    # Only opt-in; the wrapper depends on HF weights which we don't want
    # to load by default. Record skip otherwise.
    if not config.include_modern_decoder_wrapper:
        return None, "modern_decoder_wrapper is opt-in (include_modern_decoder_wrapper=False)"
    try:
        from pllo.hf_wrappers import ObfuscatedModernDecoderModelWrapper  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return None, f"modern decoder wrapper unavailable: {e}"
    return None, "modern_decoder_wrapper benchmark not wired (skipped)"


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_measured_runtime_evaluation(
    config: MeasuredRuntimeEvaluationConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "markdown").mkdir(parents=True, exist_ok=True)
    (output_dir / "latex").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    # Plain baseline.
    try:
        fn_plain = _bench_plain_linear(config)
    except Exception as e:  # noqa: BLE001
        fn_plain = None
        rows.append(_row(
            "plain_synthetic_linear", "X W", config, fn=None,
            skip_reason=f"{type(e).__name__}: {e}",
        ))
    else:
        rows.append(_row(
            "plain_synthetic_linear", "X W", config, fn=fn_plain,
            notes="Synthetic baseline; no obfuscation.",
        ))

    try:
        fn_plain_lora = _bench_plain_lora_forward(config)
    except Exception as e:  # noqa: BLE001
        fn_plain_lora = None
        rows.append(_row(
            "plain_lora_forward", "plain_rank_r", config, fn=None,
            skip_reason=f"{type(e).__name__}: {e}",
        ))
    else:
        rows.append(_row(
            "plain_lora_forward", "plain_rank_r", config, fn=fn_plain_lora,
            notes="Plain rank-r LoRA forward; no masking.",
        ))

    try:
        fn_masked_fwd = _bench_masked_lora_forward(config)
    except Exception as e:  # noqa: BLE001
        fn_masked_fwd = None
        rows.append(_row(
            "masked_lora_forward", "fresh_masks_fresh_u_with_pad",
            config, fn=None,
            skip_reason=f"{type(e).__name__}: {e}",
        ))
    else:
        rows.append(_row(
            "masked_lora_forward", "fresh_masks_fresh_u_with_pad",
            config, fn=fn_masked_fwd,
            notes="Stage 7.0 run_masked_lora_linear forward.",
        ))

    try:
        fn_masked_bwd = _bench_masked_lora_backward(config)
    except Exception as e:  # noqa: BLE001
        fn_masked_bwd = None
        rows.append(_row(
            "masked_lora_backward", "fresh_masks_fresh_u_with_pad",
            config, fn=None,
            skip_reason=f"{type(e).__name__}: {e}",
        ))
    else:
        rows.append(_row(
            "masked_lora_backward", "fresh_masks_fresh_u_with_pad",
            config, fn=fn_masked_bwd,
            notes="Stage 7.1 run_masked_lora_backward.",
        ))

    try:
        fn_rank_padded = _bench_rank_padded_forward(config)
    except Exception as e:  # noqa: BLE001
        fn_rank_padded = None
        rows.append(_row(
            "rank_padded_lora_forward", "paired_cancellation_dummy",
            config, fn=None,
            skip_reason=f"{type(e).__name__}: {e}",
        ))
    else:
        rows.append(_row(
            "rank_padded_lora_forward", "paired_cancellation_dummy",
            config, fn=fn_rank_padded,
            notes="Stage 7.2 rank-padded masked forward.",
        ))

    try:
        fn_multi = _bench_multilayer_training(config)
    except Exception as e:  # noqa: BLE001
        fn_multi = None
        rows.append(_row(
            "multi_layer_lora_training_step", "synthetic_tile",
            config, fn=None,
            skip_reason=f"{type(e).__name__}: {e}",
        ))
    else:
        rows.append(_row(
            "multi_layer_lora_training_step", "synthetic_tile",
            config, fn=fn_multi,
            notes="Stage 7.3 run_multilayer_lora_training (one training step).",
        ))

    fn_modern, skip = _bench_modern_decoder_wrapper(config)
    rows.append(_row(
        "modern_decoder_model_wrapper", "opt_in_only",
        config, fn=fn_modern, skip_reason=skip,
        notes="Opt-in benchmark; pytest defaults stay synthetic.",
    ))

    # Build CSV / Markdown / LaTeX.
    columns = [
        "component", "variant", "num_warmup", "num_repeats",
        "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms",
        "device", "dtype", "wall_time_source",
        "skipped_with_reason", "notes",
    ]
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    (output_dir / "csv" / "measured_runtime.csv").write_text(
        buf.getvalue(), encoding="utf-8",
    )

    # Markdown.
    md_lines: list[str] = ["# Measured Runtime (Local Emulation, NOT real TEE)\n"]
    md_lines.append(
        "_This is local runtime emulation, not real TEE wall-time. No real"
        " sleep, no real runtime gating._\n"
    )
    md_lines.append(
        "| " + " | ".join(columns) + " |"
    )
    md_lines.append(
        "|" + "|".join(["---"] * len(columns)) + "|"
    )
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            s = str(v).replace("|", "\\|").replace("\n", " ")
            if len(s) > 120:
                s = s[:117] + "..."
            cells.append(s)
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines.append("")
    md_lines.append("## Limitations\n")
    for lim in _LIMITATIONS:
        md_lines.append(f"- {lim}")
    md_lines.append("")
    (output_dir / "markdown" / "measured_runtime.md").write_text(
        "\n".join(md_lines), encoding="utf-8",
    )

    # LaTeX.
    def _esc(s: str) -> str:
        return (
            str(s).replace("\\", "\\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("$", r"\$")
            .replace("{", r"\{").replace("}", r"\}")
        )

    tex_lines = [
        "% Auto-generated by measured_runtime_evaluation",
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{Measured runtime (local emulation, NOT real TEE wall-time).}",
        r"\label{tab:measured_runtime}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(_esc(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            s = str(v)
            if len(s) > 60:
                s = s[:57] + "..."
            cells.append(_esc(s))
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    (output_dir / "latex" / "measured_runtime.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8",
    )

    # JSON.
    report = {
        "config": asdict(config),
        "rows": rows,
        "measured_runtime_evaluation_status": "implemented",
        "wall_time_source": "measured_local_emulation",
        "is_real_tee_wall_time": False,
        "security_profile": "proxy-evaluated, not formal",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.6 — heterogeneous padded_rank across modules to hide r_pad itself.",
            "Stage 7.7 — real TEE wall-time integration (deferred — out of scope).",
        ],
    }
    (output_dir / "json" / "measured_runtime.json").write_text(
        __import__("json").dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


__all__ = [
    "MeasuredRuntimeEvaluationConfig",
    "run_measured_runtime_evaluation",
]
