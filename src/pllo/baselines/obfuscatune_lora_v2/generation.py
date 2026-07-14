"""Cache-aware deterministic generation helpers."""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_generate(step_fn, prompt_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
    """Generate from ``step_fn(ids, cache) -> (logits, cache)``."""
    if prompt_ids.ndim != 2 or max_new_tokens < 1:
        raise ValueError("prompt must be [batch, seq] and max_new_tokens positive")
    logits, cache = step_fn(prompt_ids, None)
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    values = [next_id]
    for _ in range(max_new_tokens - 1):
        logits, cache = step_fn(next_id, cache)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        values.append(next_id)
    return torch.cat(values, dim=1)
