"""Shared report emitter helpers for Stage 5.0 experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from pllo.utils.tensor_compare import compare_tensors


# ---------------------------------------------------------------------------
# Tensor metric extraction
# ---------------------------------------------------------------------------


METRIC_KEYS = (
    "max_abs_error",
    "mean_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "allclose",
)


def compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare two tensors and keep only the headline metric fields."""
    raw = compare_tensors(reference, candidate, atol=atol, rtol=rtol)
    return {key: raw[key] for key in METRIC_KEYS if key in raw}


def pick(metrics: dict[str, Any]) -> dict[str, Any]:
    """Subset a richer metric dict down to the headline fields."""
    return {key: metrics[key] for key in METRIC_KEYS if key in metrics}


# ---------------------------------------------------------------------------
# File emitters
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def fmt(value: Any) -> str:
    """Render a value as a compact Markdown cell."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1.0 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(fmt(c) for c in row) + " |")
    return "\n".join(out)
