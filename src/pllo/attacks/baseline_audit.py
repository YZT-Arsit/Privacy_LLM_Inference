"""Run the ObfuscaTune + STIP baseline correctness audits and collect reports."""

from __future__ import annotations

from typing import Any

import torch


def run_baseline_audits(*, model_name_or_path: str | None = None, seed: int = 0,
                        dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config
    from pllo.baselines.obfuscatune.audit import audit_obfuscatune_qwen
    from pllo.baselines.stip.audit import audit_stip_qwen

    if model_name_or_path:
        model = load_qwen(model_name_or_path=model_name_or_path, seed=seed, dtype=dtype)
        src = model_name_or_path
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=seed, dtype=dtype)
        src = "tiny_random_qwen"
    return {
        "model": src,
        "obfuscatune": audit_obfuscatune_qwen(model, seed=seed),
        "stip": audit_stip_qwen(model, seed=seed),
    }


__all__ = ["run_baseline_audits"]
