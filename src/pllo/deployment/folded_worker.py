"""Worker-side application of a folded weight package (no masks on the worker).

The untrusted GPU worker loads pre-folded operators from a package shard and runs
a masked decoder block over masked activations. It holds **no** mask secrets:
the down projection is already folded (``wdown_tilde = down[perm] @ n_res``) and
attention/MLP use the folded ``*_tilde`` operators directly, so this reuses the
existing masked kernels (:func:`_masked_attention` / :func:`_masked_mlp`) with no
new mask math.

This is the incremental step toward the full 28-layer shard-streamed decode: it
runs ONE folded layer's masked prefill from a loaded package and matches the
in-process folded path bit-for-bit (fp tolerance). The public per-layer config +
RoPE caches are provided by the boundary (they are not secret).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pllo.hf_wrappers.llama_qwen_single_block import _masked_attention, _masked_mlp
from pllo.ops.nonlinear_islands import rmsnorm_core

__all__ = [
    "FOLDED_LAYER_KEYS",
    "load_folded_layer",
    "load_folded_head",
    "build_folded_layer_dict",
    "apply_folded_layer_prefill",
    "apply_folded_layer_decode",
    "apply_folded_prefill",
    "apply_folded_decode",
    "apply_folded_head",
]


def _empty_cache(device) -> None:
    try:
        import torch as _t
        if str(getattr(device, "type", device)) == "cuda" and \
                _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        pass

# Keys the masked kernels index on ``folded`` (biases may be None / absent).
FOLDED_LAYER_KEYS = (
    "wq_tilde", "wk_tilde", "wv_tilde", "wo_tilde",
    "bq_tilde", "bk_tilde", "bv_tilde", "bo_tilde",
    "wgate_tilde", "wup_tilde", "bgate_tilde", "bup_tilde",
    "wdown_tilde", "bdown_tilde",
)


def _shard_for_layer(package_dir: Path, layer_index: int) -> Path:
    from pllo.deployment.folded_package import list_package_shards
    name = f"layer_{layer_index:03d}"
    for p in list_package_shards(package_dir):
        if p.stem == name:
            return p
    raise FileNotFoundError(f"no shard {name!r} in {package_dir}")


def load_folded_layer(package_dir: str | Path, layer_index: int
                      ) -> dict[str, torch.Tensor]:
    """Load one layer's folded operators from a package directory."""
    from pllo.deployment.folded_package import load_shard
    return load_shard(_shard_for_layer(Path(package_dir), layer_index))


def load_folded_head(package_dir: str | Path) -> torch.Tensor:
    """Load the folded final-norm + LM-head operator (``w_lm_tilde``)."""
    from pllo.deployment.folded_package import list_package_shards, load_shard
    for p in list_package_shards(Path(package_dir)):
        if p.stem == "head":
            return load_shard(p)["w_lm_tilde"]
    raise FileNotFoundError(f"no head shard in {package_dir}")


def build_folded_layer_dict(layer_tensors: dict[str, torch.Tensor], *,
                            device: Any = None, dtype: Any = None
                            ) -> dict[str, Any]:
    """Assemble the ``folded`` dict the masked kernels expect, defaulting any
    absent bias to ``None`` and optionally moving tensors to device/dtype."""
    def conv(v):
        if v is None:
            return None
        if device is not None or dtype is not None:
            return v.to(device=device, dtype=dtype)
        return v
    return {k: conv(layer_tensors.get(k)) for k in FOLDED_LAYER_KEYS}


def apply_folded_layer_prefill(x_tilde: torch.Tensor,
                               layer_tensors: dict[str, torch.Tensor],
                               config: Any, cos: torch.Tensor, sin: torch.Tensor,
                               eps: float) -> dict[str, Any]:
    """Masked prefill of ONE folded layer from package tensors (no masks).

    ``x_tilde`` is the masked hidden ``[B, T, H]``; ``layer_tensors`` are the
    package's folded operators for the layer; ``config`` + ``cos``/``sin`` are the
    public per-layer block config + RoPE caches. Returns ``{"y_tilde": ...,
    "cache": {...}}`` -- the same shape the in-process masked block returns."""
    folded = build_folded_layer_dict(layer_tensors, device=x_tilde.device,
                                     dtype=x_tilde.dtype)
    r1 = rmsnorm_core(x_tilde, eps)
    a = _masked_attention(r1, folded, config, cos, sin, causal_offset=0)
    x1 = x_tilde + a["out"]
    r2 = rmsnorm_core(x1, eps)
    mlp = _masked_mlp(r2, folded)              # uses pre-folded wdown_tilde
    y_tilde = x1 + mlp["out"]
    return {"y_tilde": y_tilde,
            "cache": {"key_rope_tilde": a["key_rope_full"],
                      "value_tilde": a["value_full"]}}


def apply_folded_layer_decode(x_next_tilde: torch.Tensor,
                              layer_tensors: dict[str, torch.Tensor],
                              cache: dict, position: int, config: Any,
                              cos: torch.Tensor, sin: torch.Tensor,
                              eps: float) -> dict[str, Any]:
    """Masked one-token decode of ONE folded layer from package tensors (no
    masks), using the per-layer masked KV ``cache`` + absolute ``position``."""
    folded = build_folded_layer_dict(layer_tensors, device=x_next_tilde.device,
                                     dtype=x_next_tilde.dtype)
    pid = torch.tensor([position], device=x_next_tilde.device)
    r1 = rmsnorm_core(x_next_tilde, eps)
    a = _masked_attention(r1, folded, config, cos, sin, causal_offset=None,
                          position_ids=pid,
                          past_key_rope=cache["key_rope_tilde"],
                          past_value=cache["value_tilde"])
    x1 = x_next_tilde + a["out"]
    r2 = rmsnorm_core(x1, eps)
    mlp = _masked_mlp(r2, folded)
    y_tilde = x1 + mlp["out"]
    return {"y_tilde": y_tilde,
            "cache": {"key_rope_tilde": a["key_rope_full"],
                      "value_tilde": a["value_full"]}}


def apply_folded_prefill(h_tilde: torch.Tensor, package_dir, num_exec_layers: int,
                         config: Any, cos: torch.Tensor, sin: torch.Tensor,
                         eps: float, *, layer_configs=None,
                         empty_cache: bool = True) -> dict[str, Any]:
    """Stream the first ``num_exec_layers`` folded layers from the package over
    masked ``h_tilde`` (no masks on the worker). Returns the masked hidden after
    those layers + the per-layer masked KV. One shard is resident at a time."""
    h = h_tilde
    kv: list[dict[str, Any]] = []
    for ell in range(num_exec_layers):
        lt = load_folded_layer(package_dir, ell)
        cfg = layer_configs[ell] if layer_configs is not None else config
        out = apply_folded_layer_prefill(h, lt, cfg, cos, sin, eps)
        h = out["y_tilde"]
        kv.append(out["cache"])
        del lt
        if empty_cache:
            _empty_cache(h.device)
    return {"y_tilde": h, "kv": kv}


def apply_folded_decode(x_next_tilde: torch.Tensor, package_dir, kv: list,
                        position: int, num_exec_layers: int, config: Any,
                        cos: torch.Tensor, sin: torch.Tensor, eps: float, *,
                        layer_configs=None, empty_cache: bool = True
                        ) -> dict[str, Any]:
    """Stream a one-token masked decode over ``num_exec_layers`` folded layers
    from the package, threading the per-layer masked KV. Returns the masked hidden
    + updated KV."""
    h = x_next_tilde
    new_kv: list[dict[str, Any]] = []
    for ell in range(num_exec_layers):
        lt = load_folded_layer(package_dir, ell)
        cfg = layer_configs[ell] if layer_configs is not None else config
        out = apply_folded_layer_decode(h, lt, kv[ell], position, cfg, cos, sin,
                                        eps)
        h = out["y_tilde"]
        new_kv.append(out["cache"])
        del lt
        if empty_cache:
            _empty_cache(h.device)
    return {"y_tilde": h, "kv": new_kv}


def apply_folded_head(h_tilde: torch.Tensor, package_dir, eps: float
                      ) -> torch.Tensor:
    """Masked logits from the folded final-norm+LM-head operator:
    ``rmsnorm_core(h_tilde) @ w_lm_tilde`` (no masks; vocab mask is pre-folded)."""
    w = load_folded_head(package_dir).to(h_tilde.device, h_tilde.dtype)
    return rmsnorm_core(h_tilde, eps) @ w
