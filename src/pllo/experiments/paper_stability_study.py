"""Stage 7.5b - paper robustness/stability study (CPU only).

For each experiment x seed x batch x seq x hidden x rank combination, we
verify the masked-vs-plain correctness identity and record max abs error,
relative L2 error, allclose, and runtime. We then aggregate across all
trials per experiment into ``allclose_rate``, ``max_error_p95``,
``max_error_max``, ``runtime_mean``, ``runtime_std``, and
``failure_count``.

Experiments covered:

* ``modern_decoder_synthetic_forward`` (compatible-island stack -- runs a
  Linear+permutation island+Linear chain as a stand-in for the model-level
  wrapper);
* ``kv_cache_append`` (right-mask + token-axis append);
* ``nonlinear_island`` (paired-permutation island on a GELU-like
  pointwise activation);
* ``lora_forward`` (Stage 7.0);
* ``lora_backward`` (Stage 7.1);
* ``rank_padded_lora`` (Stage 7.2);
* ``multilayer_lora`` (Stage 7.3 single-step proxy).

No new attackers, no new ops, no real TEE, no GPU throughput.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
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
from pllo.ops.lora_backward import (
    plain_lora_backward_reference,
    run_masked_lora_backward,
)
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    create_rank_padded_lora_adapters,
    plain_rank_padded_lora_forward,
    run_masked_rank_padded_lora_linear,
)
from pllo.ops.compatible_masks import (
    generate_orthogonal,
    generate_permutation,
)
from pllo.ops.nonlinear_islands import gelu_reference


@dataclass
class PaperStabilityStudyConfig:
    output_dir: str = "outputs"
    seeds: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    seq_lens: tuple[int, ...] = (4, 8, 16)
    hidden_sizes: tuple[int, ...] = (16, 32, 64)
    true_ranks: tuple[int, ...] = (2, 4)
    padded_ranks: tuple[int, ...] = (8, 16)
    dtype: str = "float64"
    device: str = "cpu"
    atol: float = 1e-9
    rtol: float = 1e-9


_LIMITATIONS = [
    "All identities are exercised on synthetic tiles; no production fine-tune or external model.",
    "Sweeps are restricted to CPU local emulation; no GPU, no real TEE, no network downloads.",
    "Stability is measured by the spread of float64 round-off across seeds/shapes; it is NOT a security metric.",
    "Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).",
    "Reports publish summary statistics only; raw tensors / masks / adapters / gradients are never emitted.",
    "No formal / cryptographic / semantic security is claimed.",
]


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float((a - b).norm().item())
    den = float(b.norm().item())
    return num / max(den, 1e-30)


def _modern_decoder_synthetic_forward(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    """Linear -> orthogonal RMSNorm island -> Linear, plain vs. masked."""
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device) * scale
    w1 = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    w2 = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    t0 = time.perf_counter()
    z = (x @ w1) / max(math.sqrt(d), 1e-30)
    y_plain = z @ w2
    # Masked path: right-multiply w1 by orthogonal N, propagate, then absorb
    # the inverse into w2. This is the algebraic identity of Theorem 1+4.
    n = generate_orthogonal(d, dtype, device)
    n_inv = n.transpose(0, 1)
    w1_tilde = w1 @ n
    z_tilde = (x @ w1_tilde) / max(math.sqrt(d), 1e-30)
    w2_tilde = n_inv @ w2
    y_masked = z_tilde @ w2_tilde
    rt = (time.perf_counter() - t0) * 1000.0
    err = float((y_plain - y_masked).abs().max().item())
    rel = _rel_l2(y_masked, y_plain)
    allclose = bool(torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9))
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


def _kv_cache_append(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    # Per-head right-mask invariant: [K_1; K_2; ...; K_t] N = [K_1 N; ...; K_t N].
    K = torch.randn(seq, d, generator=gen, dtype=dtype, device=device)
    N = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=dtype, device=device))[0]
    t0 = time.perf_counter()
    masked_full = K @ N
    masked_step_wise = torch.cat([K[i:i + 1] @ N for i in range(seq)], dim=0)
    rt = (time.perf_counter() - t0) * 1000.0
    err = float((masked_full - masked_step_wise).abs().max().item())
    rel = _rel_l2(masked_step_wise, masked_full)
    allclose = bool(torch.allclose(masked_full, masked_step_wise, atol=1e-9, rtol=1e-9))
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


def _nonlinear_island(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    Z = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device)
    P = generate_permutation(d, dtype, device)["matrix"]
    t0 = time.perf_counter()
    plain = gelu_reference(Z) @ P
    masked = gelu_reference(Z @ P)
    rt = (time.perf_counter() - t0) * 1000.0
    err = float((plain - masked).abs().max().item())
    rel = _rel_l2(masked, plain)
    allclose = bool(torch.allclose(plain, masked, atol=1e-9, rtol=1e-9))
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


def _lora_forward(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    if rank > d:
        return {"max_abs_error": 0.0, "relative_l2_error": 0.0,
                "allclose": True, "runtime_ms": 0.0,
                "failure_reason": "rank>hidden_size_skipped"}
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device) * scale
    w = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    inner = LoRAConfig(d_in=d, d_out=d, rank=rank, alpha=float(rank), use_bias=False,
                       dtype=dtype_name, device=device_name)
    a, b = init_lora_adapters(inner, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=dtype_name, device=device_name)
    t0 = time.perf_counter()
    plain = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
    masked, _ = run_masked_lora_linear(x, w, a, b, None, inner, fwd, generator=gen)
    rt = (time.perf_counter() - t0) * 1000.0
    err = float((plain - masked).abs().max().item())
    rel = _rel_l2(masked, plain)
    allclose = bool(torch.allclose(plain, masked, atol=1e-9, rtol=1e-9))
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


def _lora_backward(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    if rank > d:
        return {"max_abs_error": 0.0, "relative_l2_error": 0.0,
                "allclose": True, "runtime_ms": 0.0,
                "failure_reason": "rank>hidden_size_skipped"}
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device) * scale
    w = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    inner = LoRAConfig(d_in=d, d_out=d, rank=rank, alpha=float(rank), use_bias=False,
                       dtype=dtype_name, device=device_name)
    a, b = init_lora_adapters(inner, generator=gen)
    grad_y = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device)
    # Sample a fresh masked-LoRA state via a forward call.
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=dtype_name, device=device_name)
    t0 = time.perf_counter()
    plain = plain_lora_backward_reference(x, w, a, b, grad_y, alpha=inner.alpha)
    _, state = run_masked_lora_linear(x, w, a, b, None, inner, fwd, generator=gen)
    recovered = run_masked_lora_backward(
        x, w, a, b, grad_y, alpha=inner.alpha,
        n_in=state.n_in, n_in_inv=state.n_in_inv, n_out=state.n_out,
        u=state.u, u_inv=state.u_inv, pad=state.pad,
    )
    rt = (time.perf_counter() - t0) * 1000.0
    err_a = float((plain["grad_a"] - recovered["grad_a"]).abs().max().item())
    err_b = float((plain["grad_b"] - recovered["grad_b"]).abs().max().item())
    err = max(err_a, err_b)
    rel = max(_rel_l2(recovered["grad_a"], plain["grad_a"]),
              _rel_l2(recovered["grad_b"], plain["grad_b"]))
    allclose = (
        bool(torch.allclose(plain["grad_a"], recovered["grad_a"], atol=1e-9, rtol=1e-9))
        and bool(torch.allclose(plain["grad_b"], recovered["grad_b"], atol=1e-9, rtol=1e-9))
    )
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


def _rank_padded_lora(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    if rank > d or padded < rank:
        return {"max_abs_error": 0.0, "relative_l2_error": 0.0,
                "allclose": True, "runtime_ms": 0.0,
                "failure_reason": "incompatible_rank_skipped"}
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device) * scale
    w = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    inner = LoRAConfig(d_in=d, d_out=d, rank=rank, alpha=float(rank), use_bias=False,
                       dtype=dtype_name, device=device_name)
    a, b = init_lora_adapters(inner, generator=gen)
    rp = RankPaddingConfig(true_rank=rank, padded_rank=padded,
                           dummy_strategy="paired_cancellation_dummy",
                           dtype=dtype_name, device=device_name)
    pad_state = create_rank_padded_lora_adapters(a, b, rp, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=dtype_name, device=device_name)
    t0 = time.perf_counter()
    plain = plain_lora_linear_forward(x, w, a, b, bias=None, alpha=inner.alpha)
    masked, _ = run_masked_rank_padded_lora_linear(
        x, w, pad_state["a_pad"], pad_state["b_pad"], None,
        true_rank=rank, padded_rank=padded, alpha=inner.alpha,
        state=None, forward_config=fwd, generator=gen,
    )
    rt = (time.perf_counter() - t0) * 1000.0
    err = float((plain - masked).abs().max().item())
    rel = _rel_l2(masked, plain)
    allclose = bool(torch.allclose(plain, masked, atol=1e-9, rtol=1e-9))
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


def _multilayer_lora(
    seed: int, bs: int, seq: int, hidden: int, rank: int, padded: int,
    dtype_name: str, device_name: str,
) -> dict[str, Any]:
    """Two stacked LoRA Linears -- a single-step proxy for Stage 7.3."""
    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    d = hidden
    if rank > d:
        return {"max_abs_error": 0.0, "relative_l2_error": 0.0,
                "allclose": True, "runtime_ms": 0.0,
                "failure_reason": "rank>hidden_size_skipped"}
    scale = 1.0 / math.sqrt(max(d, 1))
    x = torch.randn(bs * seq, d, generator=gen, dtype=dtype, device=device) * scale
    w1 = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    w2 = torch.randn(d, d, generator=gen, dtype=dtype, device=device) * scale
    inner = LoRAConfig(d_in=d, d_out=d, rank=rank, alpha=float(rank), use_bias=False,
                       dtype=dtype_name, device=device_name)
    a1, b1 = init_lora_adapters(inner, generator=gen)
    a2, b2 = init_lora_adapters(inner, generator=gen)
    fwd = MaskedLoRAForwardConfig(use_pad=True, fresh_u_per_call=True,
                                   fresh_masks_per_call=True,
                                   dtype=dtype_name, device=device_name)
    t0 = time.perf_counter()
    h_plain = plain_lora_linear_forward(x, w1, a1, b1, bias=None, alpha=inner.alpha)
    plain = plain_lora_linear_forward(h_plain, w2, a2, b2, bias=None, alpha=inner.alpha)
    h_masked, _ = run_masked_lora_linear(x, w1, a1, b1, None, inner, fwd, generator=gen)
    masked, _ = run_masked_lora_linear(h_masked, w2, a2, b2, None, inner, fwd, generator=gen)
    rt = (time.perf_counter() - t0) * 1000.0
    err = float((plain - masked).abs().max().item())
    rel = _rel_l2(masked, plain)
    allclose = bool(torch.allclose(plain, masked, atol=1e-7, rtol=1e-7))
    return {"max_abs_error": err, "relative_l2_error": rel,
            "allclose": allclose, "runtime_ms": rt, "failure_reason": ""}


_EXPERIMENTS = {
    "modern_decoder_synthetic_forward": _modern_decoder_synthetic_forward,
    "kv_cache_append": _kv_cache_append,
    "nonlinear_island": _nonlinear_island,
    "lora_forward": _lora_forward,
    "lora_backward": _lora_backward,
    "rank_padded_lora": _rank_padded_lora,
    "multilayer_lora": _multilayer_lora,
}


def _summarize(rows: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    keep = [r for r in rows if r["experiment"] == experiment]
    if not keep:
        return {
            "experiment": experiment, "trials": 0,
            "allclose_rate": float("nan"),
            "max_error_p95": float("nan"),
            "max_error_max": float("nan"),
            "runtime_mean": float("nan"),
            "runtime_std": float("nan"),
            "failure_count": 0,
        }
    n = len(keep)
    allclose_rate = float(sum(1 for r in keep if r["allclose"]) / n)
    errs = sorted([float(r["max_abs_error"]) for r in keep])
    p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)
    runtimes = [float(r["runtime_ms"]) for r in keep]
    failures = sum(1 for r in keep if r["failure_reason"])
    return {
        "experiment": experiment, "trials": n,
        "allclose_rate": allclose_rate,
        "max_error_p95": float(errs[p95_idx]),
        "max_error_max": float(errs[-1]),
        "runtime_mean": float(statistics.mean(runtimes)),
        "runtime_std": float(statistics.pstdev(runtimes)) if n > 1 else 0.0,
        "failure_count": int(failures),
    }


def _write_outputs(
    output_dir: Path, report: dict[str, Any],
    trial_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_stability_study.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    summary_cols = [
        "experiment", "trials", "allclose_rate",
        "max_error_p95", "max_error_max",
        "runtime_mean", "runtime_std", "failure_count",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=summary_cols, extrasaction="ignore")
    writer.writeheader()
    for r in summary_rows:
        writer.writerow({c: r.get(c, "") for c in summary_cols})
    (output_dir / "paper_stability_study.csv").write_text(
        buf.getvalue(), encoding="utf-8",
    )

    md: list[str] = ["# Paper Robustness / Stability Study (CPU only)\n"]
    md.append(
        "_Stability is measured by the spread of float64 round-off across"
        " seeds and shapes; it is NOT a security metric. CPU local"
        " emulation only -- NOT real TEE wall-time and NOT GPU throughput._\n"
    )
    md.append("| " + " | ".join(summary_cols) + " |")
    md.append("|" + "|".join(["---"] * len(summary_cols)) + "|")
    for r in summary_rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in summary_cols) + " |")
    md.append("\n## Limitations\n")
    for lim in _LIMITATIONS:
        md.append(f"- {lim}")
    (output_dir / "paper_stability_study.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


def run_paper_stability_study(
    config: PaperStabilityStudyConfig,
) -> dict[str, Any]:
    trial_rows: list[dict[str, Any]] = []
    for name, fn in _EXPERIMENTS.items():
        for seed in config.seeds:
            for bs in config.batch_sizes:
                for sl in config.seq_lens:
                    for hs in config.hidden_sizes:
                        for tr in config.true_ranks:
                            for pr in config.padded_ranks:
                                if pr < tr:
                                    continue
                                try:
                                    res = fn(int(seed), int(bs), int(sl), int(hs), int(tr),
                                             int(pr), config.dtype, config.device)
                                    failure = res.get("failure_reason", "")
                                except Exception as e:  # noqa: BLE001
                                    res = {"max_abs_error": float("nan"),
                                           "relative_l2_error": float("nan"),
                                           "allclose": False, "runtime_ms": 0.0,
                                           "failure_reason": f"{type(e).__name__}: {e}"}
                                    failure = res["failure_reason"]
                                trial_rows.append({
                                    "experiment": name,
                                    "seed": int(seed),
                                    "batch_size": int(bs),
                                    "seq_len": int(sl),
                                    "hidden_size": int(hs),
                                    "true_rank": int(tr),
                                    "padded_rank": int(pr),
                                    "max_abs_error": float(res["max_abs_error"]),
                                    "relative_l2_error": float(res["relative_l2_error"]),
                                    "allclose": bool(res["allclose"]),
                                    "runtime_ms": float(res["runtime_ms"]),
                                    "failure_reason": failure,
                                })
    summary_rows = [_summarize(trial_rows, name) for name in _EXPERIMENTS]
    report = {
        "config": asdict(config),
        "trial_rows": trial_rows,
        "summary_rows": summary_rows,
        "paper_stability_study_status": "implemented",
        "stage": "7.5b",
        "wall_time_source": "measured_local_emulation",
        "is_real_tee_wall_time": False,
        "is_gpu_throughput": False,
        "security_profile": "proxy-evaluated, not formal",
        "limitations": list(_LIMITATIONS),
    }
    _write_outputs(Path(config.output_dir), report, trial_rows, summary_rows)
    return report


__all__ = [
    "PaperStabilityStudyConfig",
    "run_paper_stability_study",
]
