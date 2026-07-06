"""Frequency / distribution analysis attack (P2, statistical) — supplementary.

Low-cost statistical leakage: does the protected representation preserve a stable
per-token frequency / norm / cluster structure that matches a plaintext
reference? Permutation-based methods preserve token frequency (a relabeling);
fresh-pad / matrix-mixing defenses break it.
"""

from __future__ import annotations

import time

import torch

from .metrics import relative_l2_error
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "Frequency/distribution analysis (supplementary statistical attack)"


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    ra = a.argsort().argsort().to(torch.float32)
    rb = b.argsort().argsort().to(torch.float32)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return 0.0 if denom == 0 else float((ra @ rb / denom).item())


def run_frequency_distribution_attack(
    inputs: AttackInputs, *, target_method: str,
    threat_model: str = "closed_model_no_weight_access",
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="frequency_distribution", attack_name="Frequency/distribution analysis",
                attack_family="statistical", target_method=target_method,
                threat_model=threat_model, paper_source=_PAPER, implementation_level="full",
                model_family=model_family, model_name_or_path=model_name_or_path)
    p = inputs.plaintext_embeddings
    q = inputs.protected_embeddings
    if p is None or q is None:
        return blocked_result(**base, reason="needs plaintext + protected states")
    p = p.reshape(-1, p.shape[-1]).to(torch.float32)
    q = q.reshape(-1, q.shape[-1]).to(torch.float32)
    t0 = time.perf_counter()

    # per-token norm frequency: does the sorted norm profile rank-correlate?
    pn = p.norm(dim=-1)
    qn = q.norm(dim=-1)
    n = min(pn.numel(), qn.numel())
    freq_rank_corr = _spearman(pn[:n], qn[:n])

    # token frequency recovery via norm-multiset match (a relabeling preserves it)
    norm_multiset_err = relative_l2_error(qn[:n].sort().values, pn[:n].sort().values)
    freq_recovery = max(0.0, 1.0 - norm_multiset_err)

    ms = (time.perf_counter() - t0) * 1000.0
    metrics = empty_metrics()
    metrics["frequency_rank_correlation"] = freq_rank_corr
    metrics["attack_success_rate"] = freq_recovery
    metrics["multiset_leakage_score"] = freq_recovery
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(has_intermediate_activations=True),
                        input_info=make_input_info(num_samples=p.shape[0], hidden_size=p.shape[1]),
                        metrics=metrics, runtime=make_runtime(wall_time_ms=ms),
                        notes=f"norm-frequency rank corr={freq_rank_corr:.3f}, recovery={freq_recovery:.3f}. "
                              "Permutation preserves per-token frequency; fresh-pad/mixing breaks it.")


__all__ = ["run_frequency_distribution_attack"]
