"""Forward-hook capture of Qwen2 intermediate representations for attacks.

Captures the tensors attacks consume: embed_tokens output, per-layer
input_layernorm / q,k,v,o_proj / post_attention_layernorm / mlp gate,up,down,
final norm, lm_head logits, and past_key_values. Works on a *tiny random* Qwen2
(no download) and on a local checkpoint; the same interface points at Qwen-7B on
a server. Default captures layer 0 only to stay cheap.
"""

from __future__ import annotations

from typing import Any

import torch


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    return None


class QwenActivationCapture:
    """Register forward hooks on a Qwen2 model and collect intermediates.

    Usage:
        cap = QwenActivationCapture(model, layers=[0])
        out = cap.run(input_ids)          # dict of captured tensors
        cap.remove()
    """

    def __init__(self, model, layers: list[int] | None = None) -> None:
        self.model = model
        self.tr = model.model
        n = len(self.tr.layers)
        self.layers = layers if layers is not None else [0]
        self.layers = [l for l in self.layers if 0 <= l < n]
        self._handles: list[Any] = []
        self.captured: dict[str, torch.Tensor] = {}

    def _hook(self, key):
        def fn(_module, inputs, output):
            t = _first_tensor(output)
            if t is not None:
                self.captured[key] = t.detach()
        return fn

    def _hook_input(self, key):
        def fn(_module, inputs, _output):
            t = _first_tensor(inputs)
            if t is not None:
                self.captured[key] = t.detach()
        return fn

    def register(self) -> "QwenActivationCapture":
        h = self._handles
        h.append(self.tr.embed_tokens.register_forward_hook(self._hook("embed_tokens")))
        h.append(self.tr.norm.register_forward_hook(self._hook("final_norm")))
        h.append(self.model.lm_head.register_forward_hook(self._hook("lm_head_logits")))
        for li in self.layers:
            layer = self.tr.layers[li]
            h.append(layer.register_forward_hook(self._hook_input(f"layer{li}_input")))
            h.append(layer.input_layernorm.register_forward_hook(self._hook(f"layer{li}_input_layernorm")))
            h.append(layer.post_attention_layernorm.register_forward_hook(self._hook(f"layer{li}_post_attention_layernorm")))
            a = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                h.append(getattr(a, name).register_forward_hook(self._hook(f"layer{li}_{name}")))
            m = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                h.append(getattr(m, name).register_forward_hook(self._hook(f"layer{li}_{name}")))
        return self

    def run(self, input_ids: torch.Tensor, *, use_cache: bool = True) -> dict[str, Any]:
        if not self._handles:
            self.register()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            out = self.model(input_ids, use_cache=use_cache)
        result = dict(self.captured)
        result["logits"] = out.logits.detach()
        result["past_key_values"] = getattr(out, "past_key_values", None)
        result["embedding_table"] = self.tr.embed_tokens.weight.detach()
        result["input_ids"] = input_ids.detach()
        return result

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.register()

    def __exit__(self, *exc):
        self.remove()


def capture_tiny_qwen(*, num_tokens: int = 8, seed: int = 0, layers=None,
                      dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """Build a tiny random Qwen2 and capture intermediates (no download)."""
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config
    cfg = make_tiny_qwen_config()
    model = load_qwen(tiny_random_qwen=cfg, seed=seed, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (1, num_tokens), generator=g)
    cap = QwenActivationCapture(model, layers=layers or [0])
    out = cap.run(ids)
    cap.remove()
    out["model"] = model
    return out


__all__ = ["QwenActivationCapture", "capture_tiny_qwen"]
