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

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from pllo.hf_wrappers.llama_qwen_single_block import (
    _linear, _masked_attention, _masked_mlp)
from pllo.ops.nonlinear_islands import rmsnorm_core

__all__ = [
    "FOLDED_LAYER_KEYS",
    "load_folded_layer",
    "load_folded_head",
    "load_folded_head_dict",
    "load_resident_head_dict",
    "build_folded_layer_dict",
    "build_resident_folded_layers",
    "load_resident_head",
    "folded_layers_nbytes",
    "apply_folded_layer_prefill",
    "apply_folded_layer_decode",
    "apply_folded_prefill",
    "apply_folded_decode",
    "apply_folded_head",
]


def _region(timer: Any, name: str):
    """A timing region from an optional ``WorkerTimer`` (no-op when ``timer`` is
    None -- the default path is byte-for-byte unchanged)."""
    return timer.region(name) if timer is not None else nullcontext()


def _layer_ctx(timer: Any):
    return timer.layer() if timer is not None else nullcontext()


def _empty_cache(device) -> None:
    try:
        import torch as _t
        if str(getattr(device, "type", device)) == "cuda" and \
                _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        pass

# Keys the masked kernels index on ``folded`` (biases may be None / absent).
# The ``*_xpad_tilde`` (masked input pad ``T N_in``) + ``*_cpad_tilde``
# (compensation ``T W N_out``) are present only when the package was built with
# the Linear-boundary additive pad; absent -> mask-only (kernels use None).
FOLDED_LAYER_KEYS = (
    "wq_tilde", "wk_tilde", "wv_tilde", "wo_tilde",
    "bq_tilde", "bk_tilde", "bv_tilde", "bo_tilde",
    "wgate_tilde", "wup_tilde", "bgate_tilde", "bup_tilde",
    "wdown_tilde", "bdown_tilde",
    # Linear-boundary additive pad (optional)
    "wq_xpad_tilde", "wk_xpad_tilde", "wv_xpad_tilde", "wo_xpad_tilde",
    "wgate_xpad_tilde", "wup_xpad_tilde", "wdown_xpad_tilde",
    "wq_cpad_tilde", "wk_cpad_tilde", "wv_cpad_tilde", "wo_cpad_tilde",
    "wgate_cpad_tilde", "wup_cpad_tilde", "wdown_cpad_tilde",
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


def load_folded_head_dict(package_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load the full folded head shard (``w_lm_tilde`` + optional pad tensors)."""
    from pllo.deployment.folded_package import list_package_shards, load_shard
    for p in list_package_shards(Path(package_dir)):
        if p.stem == "head":
            return load_shard(p)
    raise FileNotFoundError(f"no head shard in {package_dir}")


def load_folded_head(package_dir: str | Path) -> torch.Tensor:
    """Load the folded final-norm + LM-head operator (``w_lm_tilde``)."""
    return load_folded_head_dict(package_dir)["w_lm_tilde"]


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


def _rmsnorm(x: torch.Tensor, eps: float, runner: Any = None) -> torch.Tensor:
    """RMSNorm core, dispatched through the nonlinear ``runner`` when given.

    ``runner=None`` keeps the historical ``rmsnorm_core`` path exactly (design A
    artifacts unaffected); a runner routes through the selected design and records
    the trust/accelerator accounting."""
    return runner.rmsnorm_core(x, eps) if runner is not None \
        else rmsnorm_core(x, eps)


def build_resident_folded_layers(package_dir, num_exec_layers: int, *,
                                 device: Any, dtype: Any,
                                 lora_package_dir=None) -> list[dict[str, Any]]:
    """Load + fold + move ALL ``num_exec_layers`` layers to ``device`` ONCE.

    Returns a list of GPU-resident folded layer dicts so a decode loop can reuse
    them across steps WITHOUT re-reading the shards from disk or re-copying to the
    device. These are the SAME public folded operators the worker already loads per
    step -- only the *lifetime* changes (cached vs reloaded). No mask secrets: the
    down projection is pre-folded and only ``*_tilde`` operators are materialised.
    ``lora_package_dir`` (optional) merges a folded-LoRA package per layer -- still
    only the folded/obfuscated contribution, never the raw adapter."""
    layers: list[dict[str, Any]] = []
    for ell in range(num_exec_layers):
        lt = _maybe_merge_lora(load_folded_layer(package_dir, ell),
                               lora_package_dir, ell)
        layers.append(build_folded_layer_dict(lt, device=device, dtype=dtype))
        del lt
    return layers


def load_resident_head(package_dir, *, device: Any, dtype: Any) -> torch.Tensor:
    """Load the folded final-norm+LM-head operator ONCE and move it to device
    (bare ``w_lm_tilde`` tensor; back-compatible). Use
    :func:`load_resident_head_dict` to also keep the optional head pad tensors."""
    return load_folded_head(package_dir).to(device=device, dtype=dtype)


def load_resident_head_dict(package_dir, *, device: Any, dtype: Any
                            ) -> dict[str, torch.Tensor]:
    """Load the full folded head shard ONCE (``w_lm_tilde`` + optional Linear-
    boundary pad tensors) and move it to device. Returns a dict so the head pad
    survives into the resident decode path."""
    head = load_folded_head_dict(package_dir)
    return {k: v.to(device=device, dtype=dtype) for k, v in head.items()}


def _head_nbytes(head: Any) -> int:
    if head is None:
        return 0
    if isinstance(head, dict):
        return int(sum(v.numel() * v.element_size() for v in head.values()
                       if v is not None))
    return int(head.numel()) * int(head.element_size())


def folded_layers_nbytes(layers: list[dict[str, Any]],
                         head: Any = None) -> int:
    """Total bytes of the resident folded operators (for memory accounting).
    ``head`` may be a bare ``w_lm_tilde`` tensor or the full head dict."""
    total = 0
    for folded in layers:
        for v in folded.values():
            if v is not None:
                total += int(v.numel()) * int(v.element_size())
    return total + _head_nbytes(head)


def apply_folded_layer_prefill(x_tilde: torch.Tensor,
                               layer_tensors: dict[str, torch.Tensor] | None,
                               config: Any, cos: torch.Tensor, sin: torch.Tensor,
                               eps: float, runner: Any = None,
                               timer: Any = None,
                               folded: dict[str, Any] | None = None
                               ) -> dict[str, Any]:
    """Masked prefill of ONE folded layer (no masks).

    ``x_tilde`` is the masked hidden ``[B, T, H]``; ``config`` + ``cos``/``sin``
    are the public per-layer block config + RoPE caches. Supply EITHER
    ``layer_tensors`` (built + moved to device here) OR a pre-built resident
    ``folded`` dict (reused as-is -- no disk load, no H2D copy). ``runner``
    (optional) selects the nonlinear design; ``timer`` (optional) accumulates
    attention/MLP/nonlinear sub-totals. Returns ``{"y_tilde": ..., "cache": ...}``."""
    if folded is None:
        folded = build_folded_layer_dict(layer_tensors, device=x_tilde.device,
                                         dtype=x_tilde.dtype)
    with _region(timer, "nonlinear"):
        r1 = _rmsnorm(x_tilde, eps, runner)
    with _region(timer, "attention"):
        a = _masked_attention(r1, folded, config, cos, sin, causal_offset=0,
                              runner=runner)
    x1 = x_tilde + a["out"]
    with _region(timer, "nonlinear"):
        r2 = _rmsnorm(x1, eps, runner)
    with _region(timer, "mlp"):
        mlp = _masked_mlp(r2, folded, runner)      # uses pre-folded wdown_tilde
    y_tilde = x1 + mlp["out"]
    return {"y_tilde": y_tilde,
            "cache": {"key_rope_tilde": a["key_rope_full"],
                      "value_tilde": a["value_full"]}}


def apply_folded_layer_decode(x_next_tilde: torch.Tensor,
                              layer_tensors: dict[str, torch.Tensor] | None,
                              cache: dict, position: int, config: Any,
                              cos: torch.Tensor, sin: torch.Tensor,
                              eps: float, runner: Any = None,
                              timer: Any = None,
                              folded: dict[str, Any] | None = None
                              ) -> dict[str, Any]:
    """Masked one-token decode of ONE folded layer (no masks), using the per-layer
    masked KV ``cache`` + absolute ``position``. Supply EITHER ``layer_tensors``
    (built here) OR a pre-built resident ``folded`` dict (reused as-is). ``runner``
    (optional) selects the design; ``timer`` (optional) accumulates sub-totals."""
    if folded is None:
        folded = build_folded_layer_dict(layer_tensors,
                                         device=x_next_tilde.device,
                                         dtype=x_next_tilde.dtype)
    pid = torch.tensor([position], device=x_next_tilde.device)
    with _region(timer, "nonlinear"):
        r1 = _rmsnorm(x_next_tilde, eps, runner)
    with _region(timer, "attention"):
        a = _masked_attention(r1, folded, config, cos, sin, causal_offset=None,
                              position_ids=pid,
                              past_key_rope=cache["key_rope_tilde"],
                              past_value=cache["value_tilde"], runner=runner)
    x1 = x_next_tilde + a["out"]
    with _region(timer, "nonlinear"):
        r2 = _rmsnorm(x1, eps, runner)
    with _region(timer, "mlp"):
        mlp = _masked_mlp(r2, folded, runner)
    y_tilde = x1 + mlp["out"]
    return {"y_tilde": y_tilde,
            "cache": {"key_rope_tilde": a["key_rope_full"],
                      "value_tilde": a["value_full"]}}


def _maybe_merge_lora(lt: dict, lora_package_dir, ell: int) -> dict:
    """If a folded-LoRA package is provided, merge its layer-``ell`` operators
    into the base folded layer dict (``W_tilde += a_tilde @ b_tilde``). The worker
    never receives raw A/B or masks -- only the folded ``*_tilde`` operators."""
    if lora_package_dir is None:
        return lt
    from pllo.deployment.lora_folded_package import (
        load_folded_lora_layer,
        merge_folded_lora,
    )
    try:
        fl = load_folded_lora_layer(lora_package_dir, ell)
    except FileNotFoundError:
        return lt                                    # this layer has no LoRA
    return merge_folded_lora(lt, fl)


def apply_folded_prefill(h_tilde: torch.Tensor, package_dir, num_exec_layers: int,
                         config: Any, cos: torch.Tensor, sin: torch.Tensor,
                         eps: float, *, layer_configs=None,
                         empty_cache: bool = True,
                         lora_package_dir=None, runner: Any = None,
                         timer: Any = None, resident_layers=None
                         ) -> dict[str, Any]:
    """Stream the first ``num_exec_layers`` folded layers over masked ``h_tilde``
    (no masks on the worker). Returns the masked hidden + per-layer masked KV.

    Default: one shard is loaded/folded/moved-to-device at a time (one resident).
    When ``resident_layers`` is given (a list of pre-built folded dicts, e.g. from
    :func:`build_resident_folded_layers`), the layers are reused as-is -- NO disk
    load, NO ``build_folded_layer_dict``, NO H2D copy per step. ``lora_package_dir``
    merges a folded-LoRA package per layer (ignored when resident_layers already
    baked it in). ``timer`` (optional) records per-layer + sub-stage times."""
    h = h_tilde
    kv: list[dict[str, Any]] = []
    for ell in range(num_exec_layers):
        cfg = layer_configs[ell] if layer_configs is not None else config
        if resident_layers is not None:
            with _layer_ctx(timer):
                out = apply_folded_layer_prefill(h, None, cfg, cos, sin, eps,
                                                 runner=runner, timer=timer,
                                                 folded=resident_layers[ell])
        else:
            lt = _maybe_merge_lora(load_folded_layer(package_dir, ell),
                                   lora_package_dir, ell)
            with _layer_ctx(timer):
                out = apply_folded_layer_prefill(h, lt, cfg, cos, sin, eps,
                                                 runner=runner, timer=timer)
            del lt
            if empty_cache:
                _empty_cache(h.device)
        h = out["y_tilde"]
        kv.append(out["cache"])
    return {"y_tilde": h, "kv": kv}


def apply_folded_decode(x_next_tilde: torch.Tensor, package_dir, kv: list,
                        position: int, num_exec_layers: int, config: Any,
                        cos: torch.Tensor, sin: torch.Tensor, eps: float, *,
                        layer_configs=None, empty_cache: bool = True,
                        lora_package_dir=None, runner: Any = None,
                        timer: Any = None, resident_layers=None
                        ) -> dict[str, Any]:
    """Stream a one-token masked decode over ``num_exec_layers`` folded layers,
    threading the per-layer masked KV. Returns the masked hidden + updated KV.

    When ``resident_layers`` is given the pre-built folded dicts are reused as-is
    (NO disk load / build / H2D copy per step -- the weight-resident decode path).
    ``lora_package_dir`` merges a folded-LoRA package per layer in the non-resident
    path. ``timer`` (optional) records per-layer + sub-stage times."""
    h = x_next_tilde
    new_kv: list[dict[str, Any]] = []
    for ell in range(num_exec_layers):
        cfg = layer_configs[ell] if layer_configs is not None else config
        if resident_layers is not None:
            with _layer_ctx(timer):
                out = apply_folded_layer_decode(h, None, kv[ell], position, cfg,
                                                cos, sin, eps, runner=runner,
                                                timer=timer,
                                                folded=resident_layers[ell])
        else:
            lt = _maybe_merge_lora(load_folded_layer(package_dir, ell),
                                   lora_package_dir, ell)
            with _layer_ctx(timer):
                out = apply_folded_layer_decode(h, lt, kv[ell], position, cfg,
                                                cos, sin, eps, runner=runner,
                                                timer=timer)
            del lt
            if empty_cache:
                _empty_cache(h.device)
        h = out["y_tilde"]
        new_kv.append(out["cache"])
    return {"y_tilde": h, "kv": new_kv}


def apply_folded_head(h_tilde: torch.Tensor, package_dir, eps: float,
                      runner: Any = None, timer: Any = None,
                      folded_head: Any = None) -> torch.Tensor:
    """Masked logits from the folded final-norm+LM-head operator:
    ``rmsnorm_core(h_tilde) @ w_lm_tilde`` (no masks; vocab mask is pre-folded).
    Supply ``folded_head`` to reuse a resident head (no shard load / H2D copy);
    it may be a bare ``w_lm_tilde`` tensor OR the full head dict (which carries the
    optional Linear-boundary pad ``w_lm_xpad_tilde`` / ``w_lm_cpad_tilde``).
    ``runner`` routes the final RMSNorm through the design; ``timer`` records the
    LM-head time (excludes the shard load)."""
    xpad = cpad = None
    if folded_head is None:
        head = load_folded_head_dict(package_dir)
        w = head["w_lm_tilde"].to(h_tilde.device, h_tilde.dtype)
        if head.get("w_lm_xpad_tilde") is not None:
            xpad = head["w_lm_xpad_tilde"].to(h_tilde.device, h_tilde.dtype)
            cpad = head["w_lm_cpad_tilde"].to(h_tilde.device, h_tilde.dtype)
    elif isinstance(folded_head, dict):
        w = folded_head["w_lm_tilde"]
        xpad = folded_head.get("w_lm_xpad_tilde")
        cpad = folded_head.get("w_lm_cpad_tilde")
    else:
        w = folded_head
    with _region(timer, "lm_head"):
        return _linear(_rmsnorm(h_tilde, eps, runner), w, None, xpad, cpad)
