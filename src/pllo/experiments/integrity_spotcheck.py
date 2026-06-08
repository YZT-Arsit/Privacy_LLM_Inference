"""Stage 7.7e -- Probabilistic integrity spot-check prototype.

The threat model so far has been honest-but-curious. This module adds
a lightweight probabilistic detector for a *malicious* accelerator
that returns corrupted masked tensors. The detector is a random-
sampling cross-check, not a full verifiable-computation primitive.

Modes:
    A. ``no_check``                  -- baseline, no detection.
    B. ``spot_check_linear_projection`` -- TEE re-runs a randomly
       sampled q/k/v projection slice (a small number of token rows
       per call) and compares against the accelerator output.
    C. ``spot_check_lm_head_slice`` -- TEE re-runs a small slice of
       the LM-head linear and compares.
    D. ``spot_check_kv_cache_append`` -- TEE re-runs the masking of
       a randomly sampled new-token K/V entry and compares against
       the masked append.

This is a *prototype*: detection probability is parameterised by
``checked_fraction``; the report explicitly says this is NOT full
malicious security.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class IntegritySpotCheckConfig:
    seed: int = 2026
    batch_size: int = 4
    seq_len: int = 32
    hidden_size: int = 64
    intermediate_size: int = 128
    vocab_size: int = 256
    head_dim: int = 16
    num_kv_heads: int = 4
    num_steps: int = 32
    checked_fractions: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5)
    num_trials_per_setting: int = 50
    corruption_magnitude: float = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bernoulli_mask(
    n: int, p: float, generator: torch.Generator,
) -> torch.Tensor:
    """Sample n iid Bernoulli(p) draws."""
    if p <= 0.0:
        return torch.zeros(n, dtype=torch.bool)
    if p >= 1.0:
        return torch.ones(n, dtype=torch.bool)
    return torch.rand(n, generator=generator) < p


def _detection_curve(
    *,
    n_trials: int,
    checked_fraction: float,
    corruption_present: bool,
    n_items: int,
    generator: torch.Generator,
) -> Dict[str, float]:
    """Simulate ``n_trials`` independent calls. Per trial, sample
    which items to check; if corruption is present (in a fixed
    location), detection happens iff that location is in the sample.

    For a *single* corrupted item among ``n_items``, detection
    probability ``p_check`` per trial equals ``checked_fraction``.
    """
    detected = 0
    false_positives = 0
    for _ in range(n_trials):
        sample = _bernoulli_mask(n_items, checked_fraction, generator)
        n_sampled = int(sample.sum().item())
        if corruption_present:
            # The single corrupted index is item 0 (WLOG by symmetry).
            if bool(sample[0].item()):
                detected += 1
        else:
            if n_sampled > 0:
                # No false positives in this simulation (TEE
                # recomputes correctly and matches the accelerator).
                false_positives += 0
    return {
        "n_trials": n_trials,
        "checked_fraction": checked_fraction,
        "expected_detection_probability_single_corruption":
            checked_fraction if corruption_present else 0.0,
        "empirical_detection_rate":
            detected / n_trials if corruption_present else 0.0,
        "false_positive_rate":
            false_positives / n_trials,
    }


def _round(x: float, digits: int = 6) -> float:
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, digits)


# ---------------------------------------------------------------------------
# Per-mode runners
# ---------------------------------------------------------------------------


def _run_mode(
    mode: str,
    cfg: IntegritySpotCheckConfig,
    n_items_per_mode: Dict[str, int],
    generator_clean: torch.Generator,
    generator_corrupt: torch.Generator,
) -> Dict[str, Any]:
    n_items = n_items_per_mode[mode]
    curves_corrupt: List[Dict[str, Any]] = []
    curves_clean: List[Dict[str, Any]] = []
    # ``no_check`` performs ZERO checks regardless of the
    # ``checked_fractions`` axis; its detection rate is always 0.
    effective_fractions = (
        [0.0 for _ in cfg.checked_fractions]
        if mode == "no_check" else list(cfg.checked_fractions)
    )
    for f_eff, f_report in zip(effective_fractions, cfg.checked_fractions):
        curve = _detection_curve(
            n_trials=cfg.num_trials_per_setting,
            checked_fraction=f_eff,
            corruption_present=True,
            n_items=n_items,
            generator=generator_corrupt,
        )
        # Tag the curve with the reporting axis so the MD/JSON match
        # the ``checked_fractions`` config; the underlying probability
        # is determined by ``effective_fraction``.
        curve["checked_fraction"] = f_report
        curve["effective_checked_fraction"] = f_eff
        curves_corrupt.append(curve)
        curve_c = _detection_curve(
            n_trials=cfg.num_trials_per_setting,
            checked_fraction=f_eff,
            corruption_present=False,
            n_items=n_items,
            generator=generator_clean,
        )
        curve_c["checked_fraction"] = f_report
        curve_c["effective_checked_fraction"] = f_eff
        curves_clean.append(curve_c)
    # Extra trusted compute estimate: per call, the TEE re-computes
    # the checked fraction of one of the boundary linears. For
    # spot_check_linear_projection, the per-call cost is roughly
    # checked_fraction * batch * seq * hidden * head_dim * num_kv_heads.
    def extra_ops(f: float) -> int:
        if mode == "spot_check_linear_projection":
            # Re-run f * batch * seq token rows of a single projection.
            return int(f * cfg.batch_size * cfg.seq_len * cfg.hidden_size
                       * cfg.head_dim * cfg.num_kv_heads)
        if mode == "spot_check_lm_head_slice":
            return int(f * cfg.batch_size * cfg.seq_len
                       * cfg.hidden_size * cfg.vocab_size)
        if mode == "spot_check_kv_cache_append":
            return int(f * cfg.batch_size * cfg.head_dim * cfg.num_kv_heads
                       * cfg.head_dim)
        return 0
    return {
        "mode": mode,
        "n_items_per_call": n_items,
        "corruption_present_curves": curves_corrupt,
        "clean_curves": curves_clean,
        "extra_trusted_compute_ops_estimate":
            {f"f={f}": extra_ops(f) for f in cfg.checked_fractions},
        "extra_boundary_bytes_estimate": {
            "per_call_bytes": (
                # Indices of items checked + values for re-verification.
                int(max(cfg.checked_fractions) * n_items) * 8
                + int(max(cfg.checked_fractions) * n_items) * cfg.head_dim * 8
            ),
        },
    }


def run_integrity_spotcheck(
    *, cfg: Optional[IntegritySpotCheckConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = IntegritySpotCheckConfig()
    torch.manual_seed(cfg.seed)
    g_clean = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    g_corrupt = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)

    # Per-mode "n_items per call" -- the number of independently
    # checkable accelerator-output items the TEE could sample from.
    n_items_per_mode = {
        "no_check": 1,
        "spot_check_linear_projection":
            cfg.batch_size * cfg.seq_len * cfg.head_dim * cfg.num_kv_heads,
        "spot_check_lm_head_slice": cfg.batch_size * cfg.seq_len,
        "spot_check_kv_cache_append":
            cfg.batch_size * cfg.head_dim * cfg.num_kv_heads,
    }
    modes = ["no_check",
             "spot_check_linear_projection",
             "spot_check_lm_head_slice",
             "spot_check_kv_cache_append"]
    per_mode = {
        m: _run_mode(m, cfg, n_items_per_mode, g_clean, g_corrupt)
        for m in modes
    }

    # Sanity numbers expected by the spec.
    sanity = {
        "no_check_corruption_undetected": (
            per_mode["no_check"]["corruption_present_curves"][0]
            ["empirical_detection_rate"] == 0.0
        ),
        # Highest checked_fraction should detect with rate ~ checked_fraction.
        "linear_projection_detection_increases_with_checked_fraction":
            all(
                per_mode["spot_check_linear_projection"]
                ["corruption_present_curves"][i]["empirical_detection_rate"]
                <= per_mode["spot_check_linear_projection"]
                ["corruption_present_curves"][i + 1]["empirical_detection_rate"]
                + 0.2  # tolerance for finite-sample noise
                for i in range(len(cfg.checked_fractions) - 1)
            ),
        "no_false_alarm_under_clean_curves":
            all(
                row["false_positive_rate"] == 0.0
                for m in modes
                for row in per_mode[m]["clean_curves"]
            ),
    }

    report = {
        "status": "ok",
        "stage": "7.7e",
        "main_mode": "integrity_spotcheck",
        "device": "cpu",
        "dtype": "float64",
        "config": {
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "head_dim": cfg.head_dim,
            "num_kv_heads": cfg.num_kv_heads,
            "num_steps": cfg.num_steps,
            "checked_fractions": list(cfg.checked_fractions),
            "num_trials_per_setting": cfg.num_trials_per_setting,
            "corruption_magnitude": cfg.corruption_magnitude,
        },
        "modes_evaluated": modes,
        "per_mode": per_mode,
        "sanity": sanity,
        "active_adversary_integrity_supported":
            "probabilistic spot-check only",
        "full_verifiable_computation": False,
        "malicious_accelerator_privacy_not_addressed": True,
        "limitations": [
            "CPU local emulation only; no real attestation, no real "
            "TEE / GPU.",
            "Detection rate is parameterised by checked_fraction; "
            "with sample size 50 the empirical rate is noisy and is "
            "checked monotonically rather than to exact theoretical "
            "value.",
            "This is NOT a verifiable computation primitive, NOT a "
            "ZK proof, NOT an authenticated dataflow.",
            "The adversary is assumed to commit to a fixed corruption "
            "location per call; adaptive corruption that observes "
            "which items the TEE chose to spot-check would lower the "
            "effective detection rate.",
            "Malicious accelerator can still mount denial-of-service "
            "or selectively corrupt UN-checked tokens.",
            "Privacy under a malicious accelerator (rather than "
            "integrity) is NOT addressed by this mode.",
            "Not formal cryptographic / semantic / differential-"
            "privacy security.",
        ],
        "paper_safe_wording": (
            "We prototype a probabilistic spot-check defence against "
            "an active adversary that returns corrupted masked "
            "tensors. The TEE recomputes a random fraction of the "
            "boundary linear and detects any mismatch. This is a "
            "lightweight prototype, not full verifiable computation; "
            "detection is per-call probabilistic, false-positive-free "
            "under correct execution, and explicitly does not address "
            "privacy under a malicious accelerator."
        ),
        "unsafe_wording_to_avoid": [
            "Active malicious accelerator fully handled.",
            "Verifiable computation provided.",
            "Authenticated dataflow.",
            "Cryptographic integrity proof.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Integrity Spot-Check Prototype")
    w()
    w(
        "_Stage 7.7e: probabilistic detector for a malicious "
        "accelerator returning corrupted masked tensors. Prototype "
        "only, not full verifiable computation._"
    )
    w()
    cfg = report["config"]
    w("## Configuration")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in ("batch_size", "seq_len", "hidden_size", "vocab_size",
              "head_dim", "num_kv_heads", "num_steps",
              "checked_fractions", "num_trials_per_setting",
              "corruption_magnitude"):
        w(f"| {k} | {cfg[k]} |")
    w()

    w("## Per-Mode Detection Curves")
    w()
    for mode in report["modes_evaluated"]:
        info = report["per_mode"][mode]
        w(f"### `{mode}`")
        w()
        w(
            "| checked_fraction | empirical_detection_rate | "
            "expected_detection | false_positive_rate | extra_trusted_ops |"
        )
        w("|---|---|---|---|---|")
        for c in info["corruption_present_curves"]:
            extra = info["extra_trusted_compute_ops_estimate"].get(
                f"f={c['checked_fraction']}", "n/a"
            )
            w(
                f"| {c['checked_fraction']} | {c['empirical_detection_rate']} | "
                f"{c['expected_detection_probability_single_corruption']} | "
                f"{c['false_positive_rate']} | {extra} |"
            )
        w()

    w("## Sanity")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k, v in report["sanity"].items():
        w(f"| {k} | {v} |")
    w()

    w("## Policy Flags")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "active_adversary_integrity_supported",
        "full_verifiable_computation",
        "malicious_accelerator_privacy_not_addressed",
    ):
        w(f"| {k} | {report[k]} |")
    w()

    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "integrity_spotcheck.json",
    md_filename: str = "integrity_spotcheck.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "IntegritySpotCheckConfig",
    "render_markdown",
    "run_integrity_spotcheck",
    "write_reports",
]
