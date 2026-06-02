"""Stage 7.3 — synthetic tiny multi-layer LoRA Transformer.

A minimal multi-layer decoder-like block stack used by the Stage 7.3
multi-layer LoRA training probe. The model is intentionally small and
synthetic — it does NOT depend on Hugging Face, on a tokenizer, on a
real RoPE / KV cache, or on a real generation loop. The purpose is to
exercise the Stage 7.0 / 7.1 / 7.2 LoRA forward + backward + rank-padding
primitives across multiple LoRA-augmented linears in series.

Per layer:

    Attention-like path:
      Q = X W_q + LoRA_q(X)
      K = X W_k + LoRA_k(X)
      V = X W_v + LoRA_v(X)
      AttnProxy = scaled_dot_product(Q, K, V)
      O = AttnProxy W_o + LoRA_o(AttnProxy)
      H_attn = X + O

    SwiGLU MLP path:
      Gate = H_attn W_gate + LoRA_gate(H_attn)
      Up   = H_attn W_up   + LoRA_up(H_attn)
      G    = SiLU(Gate) * Up
      Down = G W_down + LoRA_down(G)
      H_{l+1} = H_attn + Down

Output:
      logits = H_L W_head

Base weights (W_q / W_k / W_v / W_o / W_gate / W_up / W_down / W_head)
are public and frozen — they are not modified by training. The LoRA
adapters (A / B per target module per layer) are the private trainable
parameters. The model is rank-r LoRA in the plain reference and uses
rank-padded LoRA in the masked path; this file does NOT decide between
plain / masked — the experiment harness does.

Reports must NOT publish private inputs, base weights, adapters, or
gradients. This module returns plain tensors; the harness is responsible
for tensor-free reporting (Stage 7.0 / 7.1 / 7.2 contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


VALID_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TinyLoRATransformerConfig:
    """Tiny multi-layer LoRA transformer shape config."""

    num_layers: int = 2
    hidden_size: int = 32
    intermediate_size: int = 64
    vocab_size: int = 128
    seq_len: int = 8
    batch_size: int = 4
    true_rank: int = 4
    padded_rank: int = 8
    alpha: float = 1.0
    lora_targets: tuple[str, ...] = field(
        default_factory=lambda: tuple(VALID_LORA_TARGETS)
    )
    dtype: str = "float64"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype {self.dtype!r}")

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def validate(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be > 0, got {self.hidden_size}"
            )
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be > 0,"
                f" got {self.intermediate_size}"
            )
        if self.true_rank <= 0:
            raise ValueError(f"true_rank must be > 0, got {self.true_rank}")
        if self.padded_rank < self.true_rank:
            raise ValueError(
                f"padded_rank ({self.padded_rank}) must be >= "
                f"true_rank ({self.true_rank})"
            )
        unknown = [t for t in self.lora_targets if t not in VALID_LORA_TARGETS]
        if unknown:
            raise ValueError(
                f"unknown lora_targets {unknown!r};"
                f" expected subset of {VALID_LORA_TARGETS}"
            )


def _module_io_shapes(
    module: str, hidden: int, inter: int
) -> tuple[int, int]:
    """Return (d_in, d_out) for one LoRA-augmented linear module."""
    if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return hidden, hidden
    if module in ("gate_proj", "up_proj"):
        return hidden, inter
    if module == "down_proj":
        return inter, hidden
    raise ValueError(f"unknown LoRA target {module!r}")


def model_spec(config: TinyLoRATransformerConfig) -> dict[str, Any]:
    """Return a JSON-safe summary of the model's shapes / module list.

    No tensor values are included.
    """
    config.validate()
    per_layer_modules: list[dict[str, Any]] = []
    for layer_index in range(config.num_layers):
        for module in config.lora_targets:
            d_in, d_out = _module_io_shapes(
                module, config.hidden_size, config.intermediate_size
            )
            per_layer_modules.append({
                "layer_index": layer_index,
                "module_name": module,
                "d_in": d_in,
                "d_out": d_out,
                "true_rank": config.true_rank,
                "padded_rank": config.padded_rank,
            })
    return {
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "true_rank": config.true_rank,
        "padded_rank": config.padded_rank,
        "alpha": config.alpha,
        "lora_targets": list(config.lora_targets),
        "modules_per_layer": len(config.lora_targets),
        "total_lora_modules": (
            config.num_layers * len(config.lora_targets)
        ),
        "per_layer_modules": per_layer_modules,
        "dtype": config.dtype,
        "device": config.device,
    }


# ---------------------------------------------------------------------------
# Base weights + LoRA adapter initialisers
# ---------------------------------------------------------------------------


def init_base_weights(
    config: TinyLoRATransformerConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Initialise the public, frozen base weights (per-layer + head).

    Returned dict keys:
      ``layer_{l}.{module}.W`` for each layer ``l`` and module in
      ``lora_targets``, plus ``head.W`` for the output projection.
    """
    config.validate()
    dtype = config.torch_dtype()
    device = config.torch_device()
    hidden = config.hidden_size
    inter = config.intermediate_size
    vocab = config.vocab_size
    weights: dict[str, torch.Tensor] = {}
    # Always materialise all 7 base linear weights per layer. Whether
    # each one has a LoRA adapter is decided separately by ``lora_targets``.
    for layer_index in range(config.num_layers):
        for module in VALID_LORA_TARGETS:
            d_in, d_out = _module_io_shapes(module, hidden, inter)
            key = f"layer_{layer_index}.{module}.W"
            scale = (1.0 / max(d_in, 1)) ** 0.5
            if generator is None:
                weights[key] = (
                    torch.randn(d_in, d_out, dtype=dtype, device=device) * scale
                )
            else:
                weights[key] = (
                    torch.randn(
                        d_in, d_out, generator=generator,
                        dtype=dtype, device=device,
                    ) * scale
                )
    scale_head = (1.0 / max(hidden, 1)) ** 0.5
    if generator is None:
        weights["head.W"] = (
            torch.randn(hidden, vocab, dtype=dtype, device=device) * scale_head
        )
    else:
        weights["head.W"] = (
            torch.randn(
                hidden, vocab, generator=generator,
                dtype=dtype, device=device,
            ) * scale_head
        )
    return weights


def init_lora_adapters(
    config: TinyLoRATransformerConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Initialise the private rank-``true_rank`` LoRA adapters.

    Returns a dict keyed by ``layer_{l}.{module}`` with a sub-dict
    ``{"a": <d_in × true_rank>, "b": <true_rank × d_out>}`` per module.
    """
    config.validate()
    dtype = config.torch_dtype()
    device = config.torch_device()
    hidden = config.hidden_size
    inter = config.intermediate_size
    rank = config.true_rank
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    for layer_index in range(config.num_layers):
        for module in config.lora_targets:
            d_in, d_out = _module_io_shapes(module, hidden, inter)
            scale = (1.0 / max(d_in, 1)) ** 0.5
            if generator is None:
                a = torch.randn(d_in, rank, dtype=dtype, device=device) * scale
            else:
                a = (
                    torch.randn(
                        d_in, rank, generator=generator,
                        dtype=dtype, device=device,
                    ) * scale
                )
            b = torch.zeros(rank, d_out, dtype=dtype, device=device)
            # Perturb B away from zero so the LoRA branch contributes a
            # non-trivial gradient at step 0 (otherwise grad_a = 0 trivially).
            if generator is None:
                b = b + 1e-3 * torch.randn(
                    rank, d_out, dtype=dtype, device=device,
                )
            else:
                b = b + 1e-3 * torch.randn(
                    rank, d_out, generator=generator,
                    dtype=dtype, device=device,
                )
            adapters[f"layer_{layer_index}.{module}"] = {"a": a, "b": b}
    return adapters


# ---------------------------------------------------------------------------
# Building blocks: SiLU + simple attention proxy
# ---------------------------------------------------------------------------


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def simple_attention_proxy(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    """Token-level scaled-dot-product attention proxy.

    Operates on a 2-D ``(seq_len, hidden)`` matrix per sample. The harness
    flattens ``(batch, seq, hidden)`` into ``(batch * seq, hidden)`` for
    the LoRA linears (Stage 7.0 contract requires a rank-2 ``X``); for the
    attention step we reshape back to per-sample sequences. This is NOT a
    correctness benchmark for attention; it is a non-trivial mixing
    operation between adjacent LoRA linears.
    """
    scale = 1.0 / max(q.shape[-1], 1) ** 0.5
    scores = q @ k.transpose(-2, -1) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


__all__ = [
    "TinyLoRATransformerConfig",
    "VALID_LORA_TARGETS",
    "init_base_weights",
    "init_lora_adapters",
    "model_spec",
    "simple_attention_proxy",
]
