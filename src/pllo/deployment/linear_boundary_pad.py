"""Linear-boundary additive input padding for the folded Qwen production path.

Design (paper-facing invariant):

  For a Linear ``Y = X W + b`` (row-vector convention ``y = x @ W``) the GPU's
  matmul operand is the ADDITIVELY PADDED masked input::

      X_tilde = (X - T) N_in

  and the folded operator + compensation make the output return to the EXISTING
  compatible multiplicative masked basis ``Y N_out`` (NOT a persistent residual
  pad)::

      W_tilde = N_in^{-1} W N_out
      b_tilde = b N_out
      C_pad   = T W N_out
      Y_tilde = X_tilde W_tilde + b_tilde + C_pad
              = (X - T) N_in N_in^{-1} W N_out + b N_out + T W N_out
              = X W N_out + b N_out = (X W + b) N_out = Y N_out.

This module provides two equivalent factorizations:

* :func:`fold_linear_with_input_pad` -- the literal ``(W_tilde, b_tilde, C_pad)``
  fold from raw ``W``, ``N_in^{-1}``, ``N_out`` and the raw pad ``T`` (used by the
  algebraic correctness test).
* :func:`masked_input_pad_and_compensation` -- the PRODUCTION route, which samples
  the pad directly in the masked input basis ``xpad = T N_in`` and computes
  ``C_pad = xpad @ W_tilde`` (= ``T W N_out`` exactly, reusing the already-folded
  ``W_tilde``). This needs neither ``N_in`` nor ``N_out`` at the call site, so it
  augments an existing folded layer/head dict in place.

GPU-visibility: only the folded/composed offsets ``xpad`` (= ``T N_in``) and
``C_pad`` (= ``T W N_out``) cross to the GPU -- never the raw pad ``T``, the raw
masks ``N_in``/``N_out``, or any plaintext state. The pad is boundary-LOCAL: it
enters only the Linear matmul operand and is compensated before the output is
consumed by the next (compatible) GPU-side operation. It never enters RMSNorm,
RoPE, softmax, or SiLU/SwiGLU, and is never persisted into the residual stream.

All compensation is PRECOMPUTED at trusted setup (one vector-matrix product per
module, reusing ``W_tilde``); the runtime cost is a fused broadcast subtract +
add (``online_extra_matmul_for_pad == 0``).
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "LINEAR_PAD_WEIGHT_KEYS",
    "HEAD_PAD_WEIGHT_KEY",
    "ALL_PAD_MODULES",
    "xpad_key",
    "cpad_key",
    "fold_linear_with_input_pad",
    "masked_input_pad_and_compensation",
    "add_input_pads_to_folded_layer",
    "add_input_pad_to_folded_head",
    "layer_pad_coverage",
    "module_name_for_weight_key",
    "linear_boundary_pad_report_fields",
    "default_linear_boundary_pad_report_fields",
]

# Folded weight keys for the per-layer Linear families (row-vector ``y = x @ W``).
LINEAR_PAD_WEIGHT_KEYS = (
    "wq_tilde", "wk_tilde", "wv_tilde", "wo_tilde",
    "wgate_tilde", "wup_tilde", "wdown_tilde",
)
HEAD_PAD_WEIGHT_KEY = "w_lm_tilde"

# Human-facing module names (the report's per-module coverage uses these).
_WEIGHT_TO_MODULE = {
    "wq_tilde": "q_proj", "wk_tilde": "k_proj", "wv_tilde": "v_proj",
    "wo_tilde": "o_proj", "wgate_tilde": "gate_proj", "wup_tilde": "up_proj",
    "wdown_tilde": "down_proj", "w_lm_tilde": "lm_head",
}
ALL_PAD_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj", "lm_head")


def module_name_for_weight_key(weight_key: str) -> str:
    return _WEIGHT_TO_MODULE.get(weight_key, weight_key)


def xpad_key(weight_key: str) -> str:
    """Masked-input pad tensor name for a folded weight key (``...xpad_tilde``).

    ``xpad`` is the composed offset ``T N_in`` (NOT a raw pad/mask), so the name
    carries the ``_tilde`` (folded/composed) marker and avoids every forbidden
    package substring."""
    return weight_key[:-len("_tilde")] + "_xpad_tilde"


def cpad_key(weight_key: str) -> str:
    """Compensation tensor name for a folded weight key (``...cpad_tilde``).

    ``cpad`` is the composed offset ``T W N_out`` (``= xpad @ W_tilde``)."""
    return weight_key[:-len("_tilde")] + "_cpad_tilde"


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def fold_linear_with_input_pad(
    weight: torch.Tensor, bias: torch.Tensor | None,
    n_in_inv: torch.Tensor, n_out: torch.Tensor, t_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Literal Linear-boundary pad fold (row-vector ``y = x @ W``).

    Returns ``(w_tilde, b_tilde, c_pad)``::

        w_tilde = N_in^{-1} W N_out
        b_tilde = b N_out                 (None if ``bias`` is None)
        c_pad   = T W N_out               (T = ``t_in``)

    so ``(x - T) N_in @ w_tilde + b_tilde + c_pad == (x W + b) N_out``. Raw
    ``T``/``N_in``/``N_out`` stay trusted; only ``w_tilde``/``b_tilde``/``c_pad``
    are GPU-visible."""
    w_tilde = n_in_inv @ weight @ n_out
    b_tilde = None if bias is None else bias @ n_out
    c_pad = t_in @ weight @ n_out
    return w_tilde, b_tilde, c_pad


def masked_input_pad_and_compensation(
    w_tilde: torch.Tensor, *, generator: torch.Generator | None = None,
    scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a masked-basis input pad and its compensation for one folded Linear.

    ``w_tilde`` is the already-folded ``[in, out]`` operator. Samples a broadcast
    (per-input-channel) masked pad ``xpad`` (= ``T N_in``) of shape ``[in]`` and
    returns ``(xpad, c_pad)`` with ``c_pad = xpad @ w_tilde`` (= ``T W N_out``),
    shape ``[out]``. Both are composed offsets safe to expose to the GPU; the raw
    ``T``/``N_in``/``N_out`` are never formed here. ``scale`` controls the pad
    magnitude (the output is mathematically invariant to it; it only changes the
    obfuscated matmul-operand view)."""
    in_features = int(w_tilde.shape[0])
    dtype, device = w_tilde.dtype, w_tilde.device
    if generator is not None and generator.device != torch.device(device):
        # torch requires the generator on the same device as the sampled tensor;
        # sample on the generator's device then move.
        xpad = (scale * torch.randn(in_features, generator=generator,
                                    dtype=torch.float32,
                                    device=generator.device)).to(device=device,
                                                                 dtype=dtype)
    else:
        xpad = (scale * torch.randn(in_features, generator=generator,
                                    dtype=torch.float32, device=device)).to(dtype)
    c_pad = xpad @ w_tilde
    return xpad, c_pad


# ---------------------------------------------------------------------------
# In-place augmentation of an exported folded layer / head dict
# ---------------------------------------------------------------------------


def _generator_for(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def add_input_pads_to_folded_layer(
    folded: dict[str, torch.Tensor], *, seed: int, scale: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Augment a per-layer folded tensor dict with input pad + compensation for
    every present Linear family. Independent pad per (seed, module). Returns the
    same dict (mutated) with ``<w>_xpad_tilde`` / ``<w>_cpad_tilde`` added."""
    for i, wkey in enumerate(LINEAR_PAD_WEIGHT_KEYS):
        w = folded.get(wkey)
        if w is None:
            continue
        g = _generator_for(seed + 1009 * (i + 1), w.device)
        xpad, cpad = masked_input_pad_and_compensation(w, generator=g,
                                                       scale=scale)
        folded[xpad_key(wkey)] = xpad
        folded[cpad_key(wkey)] = cpad
    return folded


def add_input_pad_to_folded_head(
    head: dict[str, torch.Tensor], *, seed: int, scale: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Augment a folded head dict (``w_lm_tilde``) with input pad + compensation."""
    w = head.get(HEAD_PAD_WEIGHT_KEY)
    if w is not None:
        g = _generator_for(seed + 7919, w.device)
        xpad, cpad = masked_input_pad_and_compensation(w, generator=g,
                                                       scale=scale)
        head[xpad_key(HEAD_PAD_WEIGHT_KEY)] = xpad
        head[cpad_key(HEAD_PAD_WEIGHT_KEY)] = cpad
    return head


# ---------------------------------------------------------------------------
# Coverage + report fields
# ---------------------------------------------------------------------------


def layer_pad_coverage(layer_tensor_names: Any,
                       head_tensor_names: Any = ()) -> dict[str, bool]:
    """Per-module pad coverage from the set of tensor names present in a layer
    shard (+ optional head shard). A module is covered iff both its ``xpad`` and
    ``cpad`` tensors are present."""
    names = set(str(n) for n in layer_tensor_names) | set(
        str(n) for n in head_tensor_names)
    cov: dict[str, bool] = {}
    for wkey in LINEAR_PAD_WEIGHT_KEYS:
        cov[module_name_for_weight_key(wkey)] = (
            xpad_key(wkey) in names and cpad_key(wkey) in names)
    cov["lm_head"] = (xpad_key(HEAD_PAD_WEIGHT_KEY) in names
                      and cpad_key(HEAD_PAD_WEIGHT_KEY) in names)
    return cov


def default_linear_boundary_pad_report_fields() -> dict[str, Any]:
    """Report fields for a folded package built WITHOUT the Linear-boundary pad
    (mask-only). All booleans reflect 'pad disabled'."""
    cov = {m: False for m in ALL_PAD_MODULES}
    return {
        "qwen_production_path_uses_linear_input_pad": False,
        "linear_boundary_pad_enabled": False,
        "linear_input_form": "X N_in",
        "linear_output_form": "Y N_out",
        "linear_pad_compensation_formula": "C_pad = T W N_out",
        "pad_scope": "linear_boundary_local",
        "persistent_residual_additive_pad": False,
        "nonlinear_masking_mode": "compatible_right_multiply_or_permutation",
        "pad_enters_rmsnorm_core": False,
        "pad_enters_rope_core": False,
        "pad_enters_softmax": False,
        "pad_enters_swiglu_core": False,
        "intermediate_tee_boundary_calls_per_layer": 0,
        "semantic_input_boundary_calls": 1,
        "semantic_final_logits_boundary_calls": 1,
        "raw_pad_visible_to_gpu": False,
        "raw_mask_visible_to_gpu": False,
        "c_pad_visible_to_gpu": False,
        "online_extra_matmul_for_pad": 0,
        "c_pad_materialization": "precomputed",
        "c_pad_runtime_cost": "none",
        "gpu_visible_pad_artifacts": [],
        "linear_pad_coverage": cov,
    }


def linear_boundary_pad_report_fields(
    *, enabled: bool, coverage: dict[str, bool] | None = None,
    scale: float | None = None,
) -> dict[str, Any]:
    """Full Linear-boundary pad audit fields for a build/probe/worker report.

    When ``enabled`` is False this is :func:`default_linear_boundary_pad_report_fields`.
    When True, ``coverage`` (per-module bool dict) drives ``linear_pad_coverage``
    + ``linear_pad_all_modules_covered``; the invariant booleans assert the
    paper-facing model (pad is boundary-local, not in any nonlinear core, not
    persisted in the residual, no intermediate TEE calls)."""
    fields = default_linear_boundary_pad_report_fields()
    if not enabled:
        return fields
    cov = dict(coverage or {m: True for m in ALL_PAD_MODULES})
    all_covered = all(cov.get(m, False) for m in ALL_PAD_MODULES)
    fields.update({
        "qwen_production_path_uses_linear_input_pad": True,
        "linear_boundary_pad_enabled": True,
        "linear_input_form": "(X - T) N_in",
        "linear_output_form": "Y N_out",
        "raw_pad_visible_to_gpu": False,
        "raw_mask_visible_to_gpu": False,
        "c_pad_visible_to_gpu": True,
        "online_extra_matmul_for_pad": 0,
        "c_pad_materialization": "precomputed",
        "c_pad_runtime_cost": "fused_add_or_slice_add",
        "gpu_visible_pad_artifacts": [
            "xpad_tilde (= T N_in, masked input pad)",
            "cpad_tilde (= T W N_out, compensation)",
        ],
        "linear_pad_coverage": cov,
        "linear_pad_all_modules_covered": all_covered,
    })
    if scale is not None:
        fields["linear_pad_scale"] = float(scale)
    return fields
