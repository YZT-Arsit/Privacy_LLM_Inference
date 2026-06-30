"""Stepwise logits-parity diagnostic: plaintext trusted greedy vs folded_remote.

The folded path recovers logits trusted-side every step; this module compares those
recovered logits against the plaintext model's logits at the SAME step to localise a
divergence (the IFEval id=1005 degeneration). For each step it records the top-K
token ids/values on both sides and computes:

* ``top1_agree``     -- argmax token id matches;
* ``top5_overlap``   -- |top5(plain) ∩ top5(ours)| / 5;
* ``max_abs_error``  -- max |plain_logit - ours_logit| over the compared vocab;
* ``rank_error``     -- rank of the plaintext-argmax token in ours' ordering (0 == ours
  also ranks it first), the key signal for "ours picked a different token".

:func:`compare_step` works on two logit vectors; :func:`compare_run` walks a list of
(plain, ours) step pairs and reports where divergence first appears (``prefill`` ==
step 0, else ``decode`` at step >= 1). Pure / stdlib + (optional) lists of floats; no
torch required at the comparison layer (callers pass plain Python lists), so it is
fully unit-testable without a model.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "topk",
    "compare_step",
    "compare_run",
    "PARITY_SANITY_PROMPTS",
]

# small, neutral sanity prompts used to gate paper-facing runs (deterministic,
# no sensitive content); the diagnostic must show top1 parity on these.
PARITY_SANITY_PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Water is made of hydrogen and",
)


def topk(logits: Sequence[float], k: int = 10) -> list[dict[str, Any]]:
    """Top-k (id, value) by descending value, ties broken by smaller id."""
    idx = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))
    return [{"id": i, "value": round(float(logits[i]), 6)} for i in idx[:k]]


def _rank_of(token_id: int, logits: Sequence[float]) -> int:
    """0-based rank of ``token_id`` in descending-logit order."""
    order = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))
    return order.index(token_id)


def compare_step(plain: Sequence[float], ours: Sequence[float], *,
                 k: int = 10) -> dict[str, Any]:
    """Per-step parity metrics between two logit vectors (same vocab order)."""
    n = min(len(plain), len(ours))
    if n == 0:
        return {"top1_agree": False, "top5_overlap": 0.0,
                "max_abs_error": None, "rank_error": None,
                "plain_top1": None, "ours_top1": None, "compared_vocab": 0}
    p_top = topk(plain, max(k, 5))
    o_top = topk(ours, max(k, 5))
    p1, o1 = p_top[0]["id"], o_top[0]["id"]
    p5 = {e["id"] for e in p_top[:5]}
    o5 = {e["id"] for e in o_top[:5]}
    denom5 = float(min(5, n))                # tiny-vocab safe denominator
    max_abs = max(abs(float(plain[i]) - float(ours[i])) for i in range(n))
    return {
        "top1_agree": bool(p1 == o1),
        "top5_overlap": round(len(p5 & o5) / denom5, 4) if denom5 else 0.0,
        "max_abs_error": round(max_abs, 6),
        # rank of the plaintext-chosen token within ours' ordering: 0 == ours also
        # ranked it first (no divergence); larger == ours demoted the right token.
        "rank_error": _rank_of(p1, ours),
        "plain_top1": p1, "ours_top1": o1,
        "plain_topk": p_top[:k], "ours_topk": o_top[:k],
        "compared_vocab": n,
    }


def compare_run(steps: list[tuple[Sequence[float], Sequence[float]]], *,
                k: int = 10, max_abs_tol: float = 1e-2,
                require_top1: bool = True) -> dict[str, Any]:
    """Walk per-step (plain, ours) logit pairs; localise the first divergence.

    A step is a divergence when top1 disagrees (if ``require_top1``) or the
    ``max_abs_error`` exceeds ``max_abs_tol``. ``divergence_phase`` is ``prefill``
    when the first divergent step is index 0, else ``decode``; ``None`` when every
    step agrees. ``passed`` == no divergence at all."""
    per_step = []
    first_div = None
    for i, (plain, ours) in enumerate(steps):
        m = compare_step(plain, ours, k=k)
        m["step"] = i
        m["phase"] = "prefill" if i == 0 else "decode"
        diverged = (require_top1 and not m["top1_agree"]) or (
            m["max_abs_error"] is not None and m["max_abs_error"] > max_abs_tol)
        m["diverged"] = bool(diverged)
        if diverged and first_div is None:
            first_div = i
        per_step.append(m)
    n = len(per_step)
    top1_agree_rate = (sum(1 for m in per_step if m["top1_agree"]) / n
                       if n else None)
    return {
        "stage": "logits_parity_diagnostic",
        "num_steps": n,
        "passed": first_div is None and n > 0,
        "first_divergence_step": first_div,
        "divergence_phase": (None if first_div is None
                             else ("prefill" if first_div == 0 else "decode")),
        "top1_agreement_rate": (round(top1_agree_rate, 4)
                                if top1_agree_rate is not None else None),
        "mean_top5_overlap": (round(sum(m["top5_overlap"] for m in per_step) / n,
                                    4) if n else None),
        "max_abs_error_overall": (max((m["max_abs_error"] for m in per_step
                                       if m["max_abs_error"] is not None),
                                      default=None)),
        "max_rank_error": (max((m["rank_error"] for m in per_step
                                if m["rank_error"] is not None), default=None)),
        "max_abs_tol": max_abs_tol,
        "per_step": per_step,
    }
