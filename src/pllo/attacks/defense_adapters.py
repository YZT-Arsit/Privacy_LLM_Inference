"""Build attack representations (AttackInputs) for each target defense method.

Unifies toy / tiny-Qwen / STIP / ObfuscaTune / ours into one entry point so the
attack runners are defense-agnostic. Never downloads a model.
"""

from __future__ import annotations

from typing import Any

import torch

from .representations import AttackInputs, tiny_qwen_representations, toy_representations


def build_representation(
    target_method: str, *, source: str = "toy", hidden_size: int = 32,
    vocab_size: int = 128, num_samples: int = 64, seed: int = 0,
    model_name_or_path: str | None = None,
) -> tuple[AttackInputs, dict[str, Any]]:
    """Return (AttackInputs, meta) for ``target_method``.

    source: "toy" (synthetic), "tiny_qwen" (hook capture), "stip"/"obfuscatune"
    (defense-specific tiny-Qwen representations).
    """
    if source == "toy" or target_method in ("toy", "plaintext_gpu"):
        obf = {"plaintext_gpu": "none", "toy": "orthogonal",
               "obfuscatune_qwen_orthogonal": "orthogonal",
               "obfuscatune_qwen_random": "random",
               "stip_qwen": "permutation",
               "ours_amulet_style": "fresh_pad",
               "ours_amulet_style_no_fresh_pad": "orthogonal"}.get(target_method, "orthogonal")
        ai = toy_representations(vocab_size=vocab_size, hidden_size=hidden_size,
                                 num_samples=num_samples, seed=seed, obfuscation=obf)
        return ai, {"source": "toy", "obfuscation": obf}

    if source == "stip" or target_method == "stip_qwen":
        from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config
        from pllo.baselines.stip.adapter import stip_attack_representation
        if model_name_or_path:
            model = load_qwen(model_name_or_path=model_name_or_path, seed=seed, dtype=torch.float32)
        else:
            model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=seed, dtype=torch.float32)
        ids = torch.randint(0, model.config.vocab_size, (1, min(num_samples, 12)))
        rep = stip_attack_representation(model, ids, seed=seed)
        return rep["attack_inputs"], {"source": "stip_qwen", "model": model_name_or_path or "tiny_random_qwen"}

    if source == "tiny_qwen":
        return tiny_qwen_representations(num_samples=min(num_samples, 12), seed=seed), \
            {"source": "tiny_qwen"}

    raise ValueError(f"unknown source {source!r} for target {target_method!r}")


__all__ = ["build_representation"]
