"""Correctness metrics for obfuscated execution."""

from __future__ import annotations

import torch

from pllo.utils.tensor_compare import compare_tensors


def compute_correctness_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, float | bool | list[int]]:
    """Compute standard numerical correctness metrics."""
    return compare_tensors(reference, candidate, atol=atol, rtol=rtol)


def top1_match_rate(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    """Compute token-wise top-1 agreement for logits."""
    return float(
        (reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1))
        .to(torch.float64)
        .mean()
        .item()
    )


def token_match_rate(reference_tokens: torch.Tensor, candidate_tokens: torch.Tensor) -> float:
    """Compute elementwise token match rate."""
    if reference_tokens.shape != candidate_tokens.shape:
        return 0.0
    return float((reference_tokens == candidate_tokens).to(torch.float64).mean().item())


def sequence_exact_match(reference_tokens: torch.Tensor, candidate_tokens: torch.Tensor) -> float:
    """Compute batch-averaged exact sequence match."""
    if reference_tokens.shape != candidate_tokens.shape:
        return 0.0
    if reference_tokens.ndim == 1:
        return float(bool(torch.equal(reference_tokens, candidate_tokens)))
    return float((reference_tokens == candidate_tokens).all(dim=1).to(torch.float64).mean().item())
