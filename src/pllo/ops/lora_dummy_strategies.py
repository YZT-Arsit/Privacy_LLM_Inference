"""Stage 7.4 — stronger dummy distributions for rank-padded LoRA.

Builds on the Stage 7.2 ``paired_cancellation_dummy`` baseline. Each
strategy returns padded adapters ``A_pad / B_pad`` such that

    A_pad B_pad = A_real B_real + correction

with ``correction`` either exactly ``0`` (cancellation strategies) or a
small trusted-side compensation tensor that the experiment harness
subtracts from the recovered output. Stage 7.2 / 7.3 primitives are NOT
modified; this module is additive and the wrapper applies the
correction outside the Stage 7.2 ops boundary.

Supported strategies
--------------------

* ``zero_dummy`` — baseline; ``A_dummy`` random, ``B_dummy = 0``.
* ``paired_cancellation_dummy`` — Stage 7.2 baseline, exact zero
  contribution.
* ``gaussian_matched_dummy`` — paired cancellation with R / S sampled
  to match the per-column statistics of ``A_real`` / ``B_real``, so
  dummy column norms / means are statistically indistinguishable.
* ``spectrum_matched_dummy`` — paired cancellation where R / S are
  scaled by singular values cycled from the empirical
  ``A_real B_real`` spectrum, so the SVD cliff at ``true_rank`` is
  buried under a matched magnitude tail.
* ``noise_injected_cancellation_dummy`` — paired cancellation + small
  noise on the dummy slice. The residual
  ``correction = A_pad[:, r:] @ B_pad[r:, :]`` is non-zero but small;
  the trusted side subtracts ``(α / true_rank) X @ correction`` from
  the recovered output.
* ``orthogonalized_cancellation_dummy`` — paired cancellation with the
  dummy ``R`` / ``S`` projected orthogonal to the column / row span of
  ``A_real`` / ``B_real``, weakening cross-module identity linkage.
* ``mixed_dummy_ensemble`` — per-pair random selection from the four
  cancellation strategies above. Aggregate metadata only — per-pair
  choices are NOT published.

All strategies maintain Stage 7.4's contract:

1. ``A_pad B_pad = A_real B_real + correction`` (correction tracked).
2. Stage 7.0 / 7.1 / 7.2 primitives untouched.
3. Dummy slices are NEVER fed into the optimizer.
4. Reports publish summary metrics + fingerprints only; raw dummy
   tensors stay trusted.

These are PROXY hardenings. The spectral attacker may still recover an
upper bound on ``true_rank``; the security proxy reports residual risk
and explicitly does NOT make formal claims (see constraint 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.utils.validation import require_rank2, require_same_dtype_device


VALID_STRONG_DUMMY_STRATEGIES: tuple[str, ...] = (
    "zero_dummy",
    "paired_cancellation_dummy",
    "gaussian_matched_dummy",
    "spectrum_matched_dummy",
    "noise_injected_cancellation_dummy",
    "orthogonalized_cancellation_dummy",
    "mixed_dummy_ensemble",
)

# Cancellation strategies (used by mixed_dummy_ensemble).
_CANCELLATION_STRATEGIES: tuple[str, ...] = (
    "paired_cancellation_dummy",
    "gaussian_matched_dummy",
    "spectrum_matched_dummy",
    "orthogonalized_cancellation_dummy",
)


@dataclass
class StrongDummyConfig:
    """Stage 7.4 stronger-dummy configuration."""

    true_rank: int
    padded_rank: int
    dummy_strategy: str = "spectrum_matched_dummy"
    dummy_scale: float = 1.0
    noise_scale: float = 1e-3
    spectrum_match_strength: float = 1.0
    orthogonalize_dummy: bool = True
    fresh_dummy_per_step: bool = True
    dtype: str = "float64"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype {self.dtype!r}")

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def dummy_size(self) -> int:
        return max(0, self.padded_rank - self.true_rank)


def validate_strong_dummy_config(config: StrongDummyConfig) -> None:
    if config.true_rank <= 0:
        raise ValueError(f"true_rank must be > 0, got {config.true_rank}")
    if config.padded_rank < config.true_rank:
        raise ValueError(
            f"padded_rank ({config.padded_rank}) must be >= "
            f"true_rank ({config.true_rank})"
        )
    if config.dummy_strategy not in VALID_STRONG_DUMMY_STRATEGIES:
        raise ValueError(
            f"unknown dummy_strategy {config.dummy_strategy!r};"
            f" expected one of {VALID_STRONG_DUMMY_STRATEGIES}"
        )
    if config.dummy_scale <= 0:
        raise ValueError(f"dummy_scale must be > 0, got {config.dummy_scale}")
    if config.noise_scale < 0:
        raise ValueError(
            f"noise_scale must be >= 0, got {config.noise_scale}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randn(shape, generator=generator, dtype=dtype, device=device)


def _per_column_stats(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) over rows for each column. Shape ``(cols,)``."""
    mean = a.mean(dim=0)
    std = a.std(dim=0, unbiased=False).clamp_min(1e-12)
    return mean, std


def _per_row_stats(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) over cols for each row. Shape ``(rows,)``."""
    mean = b.mean(dim=1)
    std = b.std(dim=1, unbiased=False).clamp_min(1e-12)
    return mean, std


def _orthogonal_project_out(
    candidate: torch.Tensor, basis_cols: torch.Tensor,
) -> torch.Tensor:
    """Project ``candidate`` (shape ``(d_in,)``) orthogonal to the column
    span of ``basis_cols`` (shape ``(d_in, k)``).
    """
    if basis_cols.shape[1] == 0:
        return candidate
    q, _ = torch.linalg.qr(basis_cols)
    return candidate - q @ (q.transpose(0, 1) @ candidate)


def _orthogonal_project_out_row(
    candidate: torch.Tensor, basis_rows: torch.Tensor,
) -> torch.Tensor:
    """Project ``candidate`` (shape ``(d_out,)``) orthogonal to the row
    span of ``basis_rows`` (shape ``(k, d_out)``).
    """
    if basis_rows.shape[0] == 0:
        return candidate
    # Project candidate onto row span = column span of basis_rows^T.
    q, _ = torch.linalg.qr(basis_rows.transpose(0, 1))
    return candidate - q @ (q.transpose(0, 1) @ candidate)


# ---------------------------------------------------------------------------
# Per-pair builders for cancellation strategies
# ---------------------------------------------------------------------------


def _sample_pair_paired_cancellation(
    d_in: int, d_out: int, dummy_scale: float,
    a: torch.Tensor, b: torch.Tensor,
    dtype: torch.dtype, device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r_vec = _randn((d_in,), dtype, device, generator) * dummy_scale
    s_vec = _randn((d_out,), dtype, device, generator) * dummy_scale
    return r_vec, s_vec


def _sample_pair_gaussian_matched(
    d_in: int, d_out: int, dummy_scale: float,
    a: torch.Tensor, b: torch.Tensor,
    dtype: torch.dtype, device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample R / S whose per-entry stats match a random column of A_real
    / row of B_real.
    """
    a_mean, a_std = _per_column_stats(a)
    b_mean, b_std = _per_row_stats(b)
    # Pick a random column / row.
    col_idx = int(
        torch.randint(
            0, a.shape[1], (1,), generator=generator,
        ).item()
        if generator is not None
        else torch.randint(0, a.shape[1], (1,)).item()
    )
    row_idx = int(
        torch.randint(
            0, b.shape[0], (1,), generator=generator,
        ).item()
        if generator is not None
        else torch.randint(0, b.shape[0], (1,)).item()
    )
    r_vec = (
        a_mean[col_idx] + a_std[col_idx]
        * _randn((d_in,), dtype, device, generator)
    ) * dummy_scale
    s_vec = (
        b_mean[row_idx] + b_std[row_idx]
        * _randn((d_out,), dtype, device, generator)
    ) * dummy_scale
    return r_vec, s_vec


def _sample_pair_spectrum_matched(
    d_in: int, d_out: int, dummy_scale: float,
    a: torch.Tensor, b: torch.Tensor,
    sigmas_a: torch.Tensor, sigmas_b: torch.Tensor,
    pair_index: int,
    dtype: torch.dtype, device: torch.device,
    generator: torch.Generator | None,
    spectrum_match_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build R / S scaled to the cycled singular value at index pair_index.

    Singular values cycle from the empirical A_real / B_real spectrum.
    """
    sigma_a = float(sigmas_a[pair_index % sigmas_a.numel()].item())
    sigma_b = float(sigmas_b[pair_index % sigmas_b.numel()].item())
    r_raw = _randn((d_in,), dtype, device, generator)
    s_raw = _randn((d_out,), dtype, device, generator)
    r_norm = r_raw.norm().clamp_min(1e-12)
    s_norm = s_raw.norm().clamp_min(1e-12)
    target_r = sigma_a * spectrum_match_strength + 1.0 * (
        1.0 - spectrum_match_strength
    )
    target_s = sigma_b * spectrum_match_strength + 1.0 * (
        1.0 - spectrum_match_strength
    )
    r_vec = (r_raw / r_norm) * target_r * dummy_scale
    s_vec = (s_raw / s_norm) * target_s * dummy_scale
    return r_vec, s_vec


def _sample_pair_orthogonalized(
    d_in: int, d_out: int, dummy_scale: float,
    a: torch.Tensor, b: torch.Tensor,
    dtype: torch.dtype, device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample R / S then project orthogonal to A_real cols / B_real rows."""
    r_raw = _randn((d_in,), dtype, device, generator) * dummy_scale
    s_raw = _randn((d_out,), dtype, device, generator) * dummy_scale
    r_vec = _orthogonal_project_out(r_raw, a)
    s_vec = _orthogonal_project_out_row(s_raw, b)
    # If projection collapses (rare), fall back to the raw vector.
    if r_vec.norm().item() < 1e-12:
        r_vec = r_raw
    if s_vec.norm().item() < 1e-12:
        s_vec = s_raw
    return r_vec, s_vec


_PAIR_SAMPLER = {
    "paired_cancellation_dummy": _sample_pair_paired_cancellation,
    "gaussian_matched_dummy": _sample_pair_gaussian_matched,
    "orthogonalized_cancellation_dummy": _sample_pair_orthogonalized,
}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def create_stronger_rank_padded_lora_adapters(
    a: torch.Tensor,
    b: torch.Tensor,
    config: StrongDummyConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Build ``(A_pad, B_pad, correction)`` for a Stage 7.4 dummy strategy.

    Returned dict keys:
      * ``a_pad`` — ``d_in × padded_rank``
      * ``b_pad`` — ``padded_rank × d_out``
      * ``correction`` — ``None`` (for exact cancellation) or
        ``d_in × d_out`` trusted-side tensor (subtract
        ``(α / true_rank) X @ correction`` from the recovered output).
      * ``true_rank`` / ``padded_rank`` / ``real_slice`` / ``dummy_slice``
      * ``dummy_strategy`` / ``dummy_strategy_effective``
      * ``metadata`` — JSON-safe summary (no raw tensor values).
    """
    validate_strong_dummy_config(config)
    require_rank2("a", a)
    require_rank2("b", b)
    if a.shape[1] != config.true_rank:
        raise ValueError(
            f"a rank dimension must match true_rank={config.true_rank},"
            f" got {tuple(a.shape)}"
        )
    if b.shape[0] != config.true_rank:
        raise ValueError(
            f"b rank dimension must match true_rank={config.true_rank},"
            f" got {tuple(b.shape)}"
        )
    require_same_dtype_device("a", a, b=b)

    r = config.true_rank
    r_pad = config.padded_rank
    d_in = a.shape[0]
    d_out = b.shape[1]
    dummy_size = r_pad - r
    dtype = a.dtype
    device = a.device

    a_pad = torch.empty(d_in, r_pad, dtype=dtype, device=device)
    b_pad = torch.empty(r_pad, d_out, dtype=dtype, device=device)
    a_pad[:, :r] = a
    b_pad[:r, :] = b

    metadata: dict[str, Any] = {
        "dummy_strategy_requested": config.dummy_strategy,
        "dummy_size": int(dummy_size),
        "dummy_scale": float(config.dummy_scale),
        "noise_scale": float(config.noise_scale),
        "spectrum_match_strength": float(config.spectrum_match_strength),
        "fresh_dummy_per_step": bool(config.fresh_dummy_per_step),
    }
    correction: torch.Tensor | None = None
    dummy_strategy_effective = config.dummy_strategy

    if dummy_size == 0:
        dummy_strategy_effective = "no_padding"
    elif config.dummy_strategy == "zero_dummy":
        a_pad[:, r:] = _randn(
            (d_in, dummy_size), dtype, device, generator,
        ) * config.dummy_scale
        b_pad[r:, :] = 0.0
    elif config.dummy_strategy == "noise_injected_cancellation_dummy":
        _fill_paired_cancellation(
            a_pad, b_pad, r, dummy_size, config.dummy_scale,
            a, b, dtype, device, generator, sampler=_sample_pair_paired_cancellation,
            sampler_extra_args=(),
        )
        # Inject noise on top.
        noise_a = _randn(
            (d_in, dummy_size), dtype, device, generator,
        ) * config.noise_scale
        noise_b = _randn(
            (dummy_size, d_out), dtype, device, generator,
        ) * config.noise_scale
        a_pad[:, r:] = a_pad[:, r:] + noise_a
        b_pad[r:, :] = b_pad[r:, :] + noise_b
        correction = a_pad[:, r:] @ b_pad[r:, :]
    elif config.dummy_strategy == "spectrum_matched_dummy":
        sigmas_a = _safe_svdvals(a)
        sigmas_b = _safe_svdvals(b)
        _fill_paired_cancellation_spectrum_matched(
            a_pad, b_pad, r, dummy_size, config.dummy_scale,
            a, b, sigmas_a, sigmas_b,
            dtype, device, generator,
            spectrum_match_strength=config.spectrum_match_strength,
        )
    elif config.dummy_strategy == "mixed_dummy_ensemble":
        ensemble_counts = _fill_mixed_ensemble(
            a_pad, b_pad, r, dummy_size, config.dummy_scale,
            a, b, dtype, device, generator,
            spectrum_match_strength=config.spectrum_match_strength,
        )
        metadata["ensemble_counts"] = ensemble_counts
    else:
        sampler = _PAIR_SAMPLER[config.dummy_strategy]
        _fill_paired_cancellation(
            a_pad, b_pad, r, dummy_size, config.dummy_scale,
            a, b, dtype, device, generator, sampler=sampler,
            sampler_extra_args=(),
        )

    dummy_contrib = a_pad[:, r:] @ b_pad[r:, :]
    dummy_contribution_norm = float(dummy_contrib.norm().item())
    correction_norm = (
        0.0 if correction is None
        else float(correction.norm().item())
    )
    metadata["dummy_strategy_effective"] = dummy_strategy_effective
    metadata["dummy_contribution_norm"] = dummy_contribution_norm
    metadata["correction_applied"] = correction is not None
    metadata["correction_norm"] = correction_norm
    metadata["dummy_padding_size"] = int(dummy_size)

    return {
        "a_pad": a_pad,
        "b_pad": b_pad,
        "correction": correction,
        "true_rank": r,
        "padded_rank": r_pad,
        "real_slice": slice(0, r),
        "dummy_slice": slice(r, r_pad),
        "dummy_strategy": config.dummy_strategy,
        "dummy_strategy_effective": dummy_strategy_effective,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Paired-cancellation fillers
# ---------------------------------------------------------------------------


def _fill_paired_cancellation(
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    r: int,
    dummy_size: int,
    dummy_scale: float,
    a: torch.Tensor,
    b: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
    *,
    sampler,
    sampler_extra_args: tuple,
) -> None:
    num_pairs = dummy_size // 2
    leftover = dummy_size % 2
    d_in = a.shape[0]
    d_out = b.shape[1]
    for i in range(num_pairs):
        r_vec, s_vec = sampler(
            d_in, d_out, dummy_scale, a, b, dtype, device, generator,
            *sampler_extra_args,
        )
        a_pad[:, r + 2 * i] = r_vec
        a_pad[:, r + 2 * i + 1] = r_vec
        b_pad[r + 2 * i, :] = s_vec
        b_pad[r + 2 * i + 1, :] = -s_vec
    if leftover == 1:
        a_pad[:, -1] = _randn(
            (d_in,), dtype, device, generator,
        ) * dummy_scale
        b_pad[-1, :] = 0.0


def _fill_paired_cancellation_spectrum_matched(
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    r: int,
    dummy_size: int,
    dummy_scale: float,
    a: torch.Tensor,
    b: torch.Tensor,
    sigmas_a: torch.Tensor,
    sigmas_b: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
    *,
    spectrum_match_strength: float,
) -> None:
    num_pairs = dummy_size // 2
    leftover = dummy_size % 2
    d_in = a.shape[0]
    d_out = b.shape[1]
    for i in range(num_pairs):
        r_vec, s_vec = _sample_pair_spectrum_matched(
            d_in, d_out, dummy_scale, a, b,
            sigmas_a, sigmas_b, i,
            dtype, device, generator,
            spectrum_match_strength,
        )
        a_pad[:, r + 2 * i] = r_vec
        a_pad[:, r + 2 * i + 1] = r_vec
        b_pad[r + 2 * i, :] = s_vec
        b_pad[r + 2 * i + 1, :] = -s_vec
    if leftover == 1:
        # Scale the leftover column to the next cycled singular value.
        sigma_a = float(
            sigmas_a[num_pairs % max(1, sigmas_a.numel())].item()
        )
        r_raw = _randn((d_in,), dtype, device, generator)
        r_norm = r_raw.norm().clamp_min(1e-12)
        target = sigma_a * spectrum_match_strength + 1.0 * (
            1.0 - spectrum_match_strength
        )
        a_pad[:, -1] = (r_raw / r_norm) * target * dummy_scale
        b_pad[-1, :] = 0.0


def _fill_mixed_ensemble(
    a_pad: torch.Tensor,
    b_pad: torch.Tensor,
    r: int,
    dummy_size: int,
    dummy_scale: float,
    a: torch.Tensor,
    b: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
    *,
    spectrum_match_strength: float,
) -> dict[str, int]:
    num_pairs = dummy_size // 2
    leftover = dummy_size % 2
    d_in = a.shape[0]
    d_out = b.shape[1]
    counts: dict[str, int] = {s: 0 for s in _CANCELLATION_STRATEGIES}
    sigmas_a = _safe_svdvals(a)
    sigmas_b = _safe_svdvals(b)
    for i in range(num_pairs):
        # Pick a strategy index deterministically from the generator.
        if generator is None:
            idx = int(torch.randint(0, len(_CANCELLATION_STRATEGIES), (1,)).item())
        else:
            idx = int(
                torch.randint(
                    0, len(_CANCELLATION_STRATEGIES), (1,), generator=generator,
                ).item()
            )
        sub = _CANCELLATION_STRATEGIES[idx]
        counts[sub] += 1
        if sub == "spectrum_matched_dummy":
            r_vec, s_vec = _sample_pair_spectrum_matched(
                d_in, d_out, dummy_scale, a, b,
                sigmas_a, sigmas_b, i,
                dtype, device, generator,
                spectrum_match_strength,
            )
        else:
            sampler = _PAIR_SAMPLER[sub]
            r_vec, s_vec = sampler(
                d_in, d_out, dummy_scale, a, b, dtype, device, generator,
            )
        a_pad[:, r + 2 * i] = r_vec
        a_pad[:, r + 2 * i + 1] = r_vec
        b_pad[r + 2 * i, :] = s_vec
        b_pad[r + 2 * i + 1, :] = -s_vec
    if leftover == 1:
        a_pad[:, -1] = _randn(
            (d_in,), dtype, device, generator,
        ) * dummy_scale
        b_pad[-1, :] = 0.0
    return counts


def _safe_svdvals(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return torch.tensor([1.0], dtype=t.dtype, device=t.device)
    sv = torch.linalg.svdvals(t)
    if sv.numel() == 0:
        return torch.tensor([1.0], dtype=t.dtype, device=t.device)
    return sv.clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Forward + backward wrappers that honour the correction term
# ---------------------------------------------------------------------------


def apply_dummy_correction(
    y: torch.Tensor,
    x: torch.Tensor,
    correction: torch.Tensor | None,
    *,
    true_rank: int,
    alpha: float,
) -> torch.Tensor:
    """Subtract ``(α / true_rank) X @ correction`` from ``y`` if needed."""
    if correction is None:
        return y
    scale = float(alpha) / max(true_rank, 1)
    return y - scale * (x @ correction)


def dummy_correction_norm(
    a_pad: torch.Tensor, b_pad: torch.Tensor, true_rank: int,
) -> float:
    """Return ``‖A_pad[:, r:] B_pad[r:, :]‖_F`` — the trusted-side
    correction magnitude. 0 for cancellation strategies.
    """
    if a_pad.shape[1] == true_rank:
        return 0.0
    return float(
        (a_pad[:, true_rank:] @ b_pad[true_rank:, :]).norm().item()
    )


def visible_strong_dummy_fingerprint(
    a_pad_tilde: torch.Tensor,
    b_pad_tilde: torch.Tensor,
    *,
    dummy_strategy: str,
) -> dict[str, Any]:
    """Return shape-only fingerprint suitable for JSON publication."""
    return {
        "a_tilde_pad_shape": list(a_pad_tilde.shape),
        "b_tilde_pad_shape": list(b_pad_tilde.shape),
        "visible_rank_from_a_shape": int(a_pad_tilde.shape[1]),
        "visible_rank_from_b_shape": int(b_pad_tilde.shape[0]),
        "dummy_strategy": dummy_strategy,
    }


__all__ = [
    "StrongDummyConfig",
    "VALID_STRONG_DUMMY_STRATEGIES",
    "apply_dummy_correction",
    "create_stronger_rank_padded_lora_adapters",
    "dummy_correction_norm",
    "validate_strong_dummy_config",
    "visible_strong_dummy_fingerprint",
]
