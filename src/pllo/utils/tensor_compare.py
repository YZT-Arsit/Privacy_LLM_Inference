"""Tensor comparison utilities."""

from __future__ import annotations

import torch


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    eps: float = 1e-30,
) -> dict[str, float | bool | list[int]]:
    """Compare tensors and return scalar error metrics.

    Shape mismatches are reported without attempting invalid elementwise
    operations.
    """
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
            "cosine_similarity": float("nan"),
            "allclose": False,
        }

    ref = reference.detach()
    cand = candidate.detach()
    diff = cand - ref
    abs_diff = diff.abs()

    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    ref_norm = torch.linalg.vector_norm(ref_flat)
    diff_norm = torch.linalg.vector_norm(diff.reshape(-1))
    denom = torch.clamp(ref_norm, min=torch.as_tensor(eps, dtype=ref.dtype, device=ref.device))

    cand_norm = torch.linalg.vector_norm(cand_flat)
    cosine_denom = torch.clamp(
        ref_norm * cand_norm,
        min=torch.as_tensor(eps, dtype=ref.dtype, device=ref.device),
    )
    # Clamp to [-1, 1]: floating-point rounding can push nearly-identical
    # vectors slightly outside this range without clamping.
    cosine = (torch.dot(ref_flat, cand_flat) / cosine_denom).clamp(-1.0, 1.0)

    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "relative_l2_error": float((diff_norm / denom).item()),
        "cosine_similarity": float(cosine.item()),
        "allclose": bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
    }
