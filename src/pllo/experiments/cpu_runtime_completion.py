"""Stage 7.5b - extended CPU local-emulation runtime benchmark.

Complements ``measured_runtime_evaluation.py`` by sweeping
``(batch_size, seq_len, hidden_size)`` for the twelve components named in
the Stage 7.5b spec. Every row records mean / median / std / min / max
in milliseconds. Components that depend on the modern decoder full
generation path are simulated with a synthetic stand-in (single
Linear+permutation+Linear+Linear chain) so the benchmark stays CPU-only,
network-free, and PEFT/DeepSpeed/vLLM/FlashAttention-free.

This module never calls ``time.sleep`` and never publishes raw tensors,
masks, adapters, gradients, or private data. All measurements are
"local trusted-runtime emulation, NOT real TEE wall-time and NOT GPU
throughput."
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
from typing import Any, Callable

import torch

from pllo.ops.compatible_masks import (
    generate_orthogonal,
    generate_permutation,
)
from pllo.ops.nonlinear_islands import gelu_reference
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
from pllo.ops.lora_dummy_strategies import (
    StrongDummyConfig,
    create_stronger_rank_padded_lora_adapters,
)


@dataclass
class CPURuntimeCompletionConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    num_warmup: int = 3
    num_repeats: int = 20
    batch_sizes: tuple[int, ...] = (1, 4)
    seq_lens: tuple[int, ...] = (8, 16)
    hidden_sizes: tuple[int, ...] = (32, 64)
    true_rank: int = 4
    padded_rank: int = 8
    dtype: str = "float64"
    device: str = "cpu"


_LIMITATIONS = [
    "This is CPU / local trusted-runtime emulation, NOT real TEE wall-time and NOT GPU throughput.",
    "No ``time.sleep`` is used; ``time.perf_counter`` only.",
    "Modern-decoder rows use synthetic stand-ins; no Hugging Face model load and no network download.",
    "No PEFT / DeepSpeed / vLLM / FlashAttention integration.",
    "Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).",
    "Workload sizes are small for pytest stability -- absolute numbers are illustrative.",
    "No formal / cryptographic / semantic security is claimed.",
    "Reports publish summary statistics only; raw tensors / masks / adapters / gradients are never emitted.",
]


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _time_block(
    fn: Callable[[], Any], *, num_warmup: int, num_repeats: int,
) -> dict[str, float]:
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
        "std_ms": float(statistics.pstdev(times_ms)) if num_repeats > 1 else 0.0,
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def _make_xw(
    cfg: CPURuntimeCompletionConfig, bs: int, sl: int, hs: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    scale = 1.0 / math.sqrt(max(hs, 1))
    x = torch.randn(bs * sl, hs, generator=generator, dtype=dtype, device=device) * scale
    w = torch.randn(hs, hs, generator=generator, dtype=dtype, device=device) * scale
    return x, w


def _bench_linear_masked_forward(cfg, bs, sl, hs, gen):
    x, w = _make_xw(cfg, bs, sl, hs, gen)
    inner = LoRAConfig(d_in=hs, d_out=hs, rank=cfg.true_rank,
                       alpha=float(cfg.true_rank), use_bias=False,
                       dtype=cfg.dtype, device=cfg.device)
    a, b = init_lora_adapters(inner, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=cfg.dtype, device=cfg.device)
    return lambda: run_masked_lora_linear(x, w, a, b, None, inner, fwd, generator=gen)


def _bench_nonlinear_island_forward(cfg, bs, sl, hs, gen):
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    Z = torch.randn(bs * sl, hs, generator=gen, dtype=dtype, device=device)
    P = generate_permutation(hs, dtype, device)["matrix"]
    return lambda: gelu_reference(Z @ P)


def _bench_modern_decoder_synthetic(cfg, bs, sl, hs, gen, *, kind: str):
    """``kind`` selects ``full`` / ``prefill`` / ``decode_step`` / ``greedy_gen``."""
    dtype = _torch_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    x = torch.randn(bs * sl, hs, generator=gen, dtype=dtype, device=device)
    w1 = torch.randn(hs, hs, generator=gen, dtype=dtype, device=device)
    w2 = torch.randn(hs, hs, generator=gen, dtype=dtype, device=device)
    N = generate_orthogonal(hs, dtype, device)
    Ninv = N.transpose(0, 1)
    P = generate_permutation(hs, dtype, device)["matrix"]

    def _full():
        z = x @ w1 @ N
        return (gelu_reference(z @ P) @ P.transpose(0, 1)) @ Ninv @ w2

    if kind == "full":
        return _full
    if kind == "prefill":
        return _full  # synthetic; same shape as full forward.
    if kind == "decode_step":
        # One-token decode step: do the same arithmetic but over a single
        # row -- this is the per-step proxy.
        x_step = x[:1]

        def _step():
            z = x_step @ w1 @ N
            return (gelu_reference(z @ P) @ P.transpose(0, 1)) @ Ninv @ w2

        return _step
    if kind == "greedy_gen":
        # Three consecutive decode steps -- proxy for greedy generation.
        x_step = x[:1]

        def _gen():
            cur = x_step
            for _ in range(3):
                z = cur @ w1 @ N
                cur = (gelu_reference(z @ P) @ P.transpose(0, 1)) @ Ninv @ w2
            return cur

        return _gen
    raise ValueError(f"unknown kind {kind!r}")


def _bench_lora_forward(cfg, bs, sl, hs, gen):
    return _bench_linear_masked_forward(cfg, bs, sl, hs, gen)


def _bench_lora_backward(cfg, bs, sl, hs, gen):
    x, w = _make_xw(cfg, bs, sl, hs, gen)
    inner = LoRAConfig(d_in=hs, d_out=hs, rank=cfg.true_rank,
                       alpha=float(cfg.true_rank), use_bias=False,
                       dtype=cfg.dtype, device=cfg.device)
    a, b = init_lora_adapters(inner, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=cfg.dtype, device=cfg.device)
    grad_y = torch.randn(bs * sl, hs, generator=gen, dtype=_torch_dtype(cfg.dtype),
                         device=torch.device(cfg.device))

    def _bwd():
        _, state = run_masked_lora_linear(x, w, a, b, None, inner, fwd, generator=gen)
        return run_masked_lora_backward(
            x, w, a, b, grad_y, alpha=inner.alpha,
            n_in=state.n_in, n_in_inv=state.n_in_inv, n_out=state.n_out,
            u=state.u, u_inv=state.u_inv, pad=state.pad,
        )

    return _bwd


def _bench_lora_rank_padding(cfg, bs, sl, hs, gen):
    x, w = _make_xw(cfg, bs, sl, hs, gen)
    inner = LoRAConfig(d_in=hs, d_out=hs, rank=cfg.true_rank,
                       alpha=float(cfg.true_rank), use_bias=False,
                       dtype=cfg.dtype, device=cfg.device)
    a, b = init_lora_adapters(inner, generator=gen)
    rp = RankPaddingConfig(true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                           dummy_strategy="paired_cancellation_dummy",
                           dtype=cfg.dtype, device=cfg.device)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=cfg.dtype, device=cfg.device)

    def _rp():
        st = create_rank_padded_lora_adapters(a, b, rp, generator=gen)
        return run_masked_rank_padded_lora_linear(
            x, w, st["a_pad"], st["b_pad"], None,
            true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
            alpha=inner.alpha, state=None, forward_config=fwd, generator=gen,
        )

    return _rp


def _bench_multilayer_lora_training_step(cfg, bs, sl, hs, gen):
    x, w1 = _make_xw(cfg, bs, sl, hs, gen)
    _, w2 = _make_xw(cfg, bs, sl, hs, gen)
    inner = LoRAConfig(d_in=hs, d_out=hs, rank=cfg.true_rank,
                       alpha=float(cfg.true_rank), use_bias=False,
                       dtype=cfg.dtype, device=cfg.device)
    a1, b1 = init_lora_adapters(inner, generator=gen)
    a2, b2 = init_lora_adapters(inner, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=cfg.dtype, device=cfg.device)

    def _step():
        h, _ = run_masked_lora_linear(x, w1, a1, b1, None, inner, fwd, generator=gen)
        y, _ = run_masked_lora_linear(h, w2, a2, b2, None, inner, fwd, generator=gen)
        return y

    return _step


def _bench_stronger_dummy_generation(cfg, bs, sl, hs, gen):
    x, w = _make_xw(cfg, bs, sl, hs, gen)
    inner = LoRAConfig(d_in=hs, d_out=hs, rank=cfg.true_rank,
                       alpha=float(cfg.true_rank), use_bias=False,
                       dtype=cfg.dtype, device=cfg.device)
    a, b = init_lora_adapters(inner, generator=gen)
    sd = StrongDummyConfig(true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                           dummy_strategy="mixed_dummy_ensemble",
                           dtype=cfg.dtype, device=cfg.device)

    def _gen_step():
        return create_stronger_rank_padded_lora_adapters(a, b, sd, generator=gen)

    return _gen_step


def _bench_paper_toy_task_train_step(cfg, bs, sl, hs, gen):
    """Two-layer LoRA forward + scalar MSE -- a single train-step proxy."""
    x, w1 = _make_xw(cfg, bs, sl, hs, gen)
    _, w2 = _make_xw(cfg, bs, sl, hs, gen)
    target = torch.randn(bs * sl, hs, generator=gen,
                         dtype=_torch_dtype(cfg.dtype),
                         device=torch.device(cfg.device))
    inner = LoRAConfig(d_in=hs, d_out=hs, rank=cfg.true_rank,
                       alpha=float(cfg.true_rank), use_bias=False,
                       dtype=cfg.dtype, device=cfg.device)
    a1, b1 = init_lora_adapters(inner, generator=gen)
    a2, b2 = init_lora_adapters(inner, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=cfg.dtype, device=cfg.device)

    def _step():
        h, _ = run_masked_lora_linear(x, w1, a1, b1, None, inner, fwd, generator=gen)
        y, _ = run_masked_lora_linear(h, w2, a2, b2, None, inner, fwd, generator=gen)
        return ((y - target) ** 2).mean()

    return _step


_COMPONENT_BUILDERS = (
    ("linear_masked_forward", lambda c, b, s, h, g: _bench_linear_masked_forward(c, b, s, h, g),
     "Stage 7.0 LoRA forward used as the canonical masked Linear."),
    ("nonlinear_island_forward",
     lambda c, b, s, h, g: _bench_nonlinear_island_forward(c, b, s, h, g),
     "Permutation-island around GELU; Theorem 2."),
    ("modern_decoder_full_forward",
     lambda c, b, s, h, g: _bench_modern_decoder_synthetic(c, b, s, h, g, kind="full"),
     "Synthetic stand-in for the model-level wrapper (no HF model load)."),
    ("modern_decoder_prefill",
     lambda c, b, s, h, g: _bench_modern_decoder_synthetic(c, b, s, h, g, kind="prefill"),
     "Prefill proxy under the synthetic stand-in."),
    ("modern_decoder_decode_step",
     lambda c, b, s, h, g: _bench_modern_decoder_synthetic(c, b, s, h, g, kind="decode_step"),
     "Single-token decode-step proxy under the synthetic stand-in."),
    ("modern_decoder_greedy_generation",
     lambda c, b, s, h, g: _bench_modern_decoder_synthetic(c, b, s, h, g, kind="greedy_gen"),
     "Three-step greedy generation proxy under the synthetic stand-in."),
    ("lora_forward", lambda c, b, s, h, g: _bench_lora_forward(c, b, s, h, g),
     "Stage 7.0 masked LoRA forward."),
    ("lora_backward", lambda c, b, s, h, g: _bench_lora_backward(c, b, s, h, g),
     "Stage 7.1 masked LoRA backward."),
    ("lora_rank_padding", lambda c, b, s, h, g: _bench_lora_rank_padding(c, b, s, h, g),
     "Stage 7.2 rank-padded masked LoRA forward."),
    ("multilayer_lora_training_step",
     lambda c, b, s, h, g: _bench_multilayer_lora_training_step(c, b, s, h, g),
     "Stage 7.3 multi-layer LoRA training-step proxy."),
    ("stronger_dummy_generation",
     lambda c, b, s, h, g: _bench_stronger_dummy_generation(c, b, s, h, g),
     "Stage 7.4 stronger dummy generation."),
    ("paper_toy_task_train_step",
     lambda c, b, s, h, g: _bench_paper_toy_task_train_step(c, b, s, h, g),
     "Stage 7.5b toy-task single train-step proxy (MSE)."),
)


def _write_outputs(
    output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cpu_runtime_completion.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    columns = [
        "component", "variant",
        "batch_size", "seq_len", "hidden_size",
        "num_warmup", "num_repeats",
        "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms",
        "notes",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    (output_dir / "cpu_runtime_completion.csv").write_text(
        buf.getvalue(), encoding="utf-8",
    )

    md: list[str] = ["# CPU Runtime Completion (Local Emulation)\n"]
    md.append(
        "_This is CPU / local trusted-runtime emulation, NOT real TEE"
        " wall-time and NOT GPU throughput. ``time.perf_counter`` only;"
        " no ``time.sleep`` and no network downloads._\n"
    )
    md.append("| " + " | ".join(columns) + " |")
    md.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    md.append("\n## Limitations\n")
    for lim in _LIMITATIONS:
        md.append(f"- {lim}")
    (output_dir / "cpu_runtime_completion.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


def run_cpu_runtime_completion(
    config: CPURuntimeCompletionConfig,
) -> dict[str, Any]:
    generator = torch.Generator(device=torch.device(config.device))
    generator.manual_seed(int(config.seed))
    rows: list[dict[str, Any]] = []
    for bs in config.batch_sizes:
        for sl in config.seq_lens:
            for hs in config.hidden_sizes:
                for name, builder, notes in _COMPONENT_BUILDERS:
                    try:
                        fn = builder(config, int(bs), int(sl), int(hs), generator)
                        stats = _time_block(
                            fn,
                            num_warmup=config.num_warmup,
                            num_repeats=config.num_repeats,
                        )
                        skip = ""
                    except Exception as e:  # noqa: BLE001
                        stats = {"mean_ms": float("nan"), "median_ms": float("nan"),
                                 "std_ms": float("nan"), "min_ms": float("nan"),
                                 "max_ms": float("nan")}
                        skip = f"{type(e).__name__}: {e}"
                    rows.append({
                        "component": name,
                        "variant": f"bs{bs}_sl{sl}_hs{hs}",
                        "batch_size": int(bs),
                        "seq_len": int(sl),
                        "hidden_size": int(hs),
                        "num_warmup": int(config.num_warmup),
                        "num_repeats": int(config.num_repeats),
                        **stats,
                        "notes": notes + (f" -- skipped: {skip}" if skip else ""),
                    })
    report = {
        "config": asdict(config),
        "rows": rows,
        "cpu_runtime_completion_status": "implemented",
        "stage": "7.5b",
        "wall_time_source": "measured_local_emulation",
        "is_real_tee_wall_time": False,
        "is_gpu_throughput": False,
        "security_profile": "proxy-evaluated, not formal",
        "limitations": list(_LIMITATIONS),
    }
    _write_outputs(Path(config.output_dir), report, rows)
    return report


__all__ = [
    "CPURuntimeCompletionConfig",
    "run_cpu_runtime_completion",
]
