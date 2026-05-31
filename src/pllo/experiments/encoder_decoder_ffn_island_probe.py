"""Stage 5.3c — encoder-decoder FFN compatible-island probe.

Probes T5 / BART-style FFNs under ``nonlinear_mode ∈ {"trusted",
"compatible_islands"}``. Reuses the ``encoder_decoder`` candidate registry
so the probe runs against the same model loader as the Stage 6.2
cross-attention probe.

This is a **probe-level** integration:

* No full T5 / BART wrapper, no model.forward replacement.
* No LM head / encoder-decoder generation integration.
* The cross-attention mathematical invariants (Stage 6.2) are **not**
  modified.
* The probe verifies the operator-compatible FFN island recovers the
  plain FFN output: ``Y_tilde = Y N_out``, with
  ``online_extra_matmul_count = 0`` and pad compensation kept at the
  Linear boundary (``"linear_boundary_only"``).

Module auto-detection:

* ``wi`` + ``wo``       → unscaled FFN (T5 ``DenseReluDense`` /
  ``T5DenseActDense``); activation read from
  ``config.feed_forward_proj``.
* ``wi_0`` + ``wi_1`` + ``wo`` → gated FFN (T5 ``DenseGatedActDense``);
  activation read from ``config.feed_forward_proj`` (drops the
  ``"gated-"`` prefix). Supports SiLU via ``run_swiglu_mlp_island``;
  other gated activations are emitted as ``status="unsupported"`` with an
  explicit reason — they are not silently skipped.
* ``fc1`` + ``fc2``    → BART encoder FFN; activation read from
  ``config.activation_function``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pllo.architectures import (
    DEFAULT_ARCHITECTURE_MODELS,
    load_for_architecture,
)
from pllo.experiments.report_utils import compare
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    normalize_nonlinear_mode,
)
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.model_zoo.base import torch_dtype_from_string
from pllo.ops.compatible_masks import generate_permutation
from pllo.ops.nonlinear_islands import (
    get_activation,
    run_gelu_mlp_island,
    run_swiglu_mlp_island,
)


@dataclass
class EncoderDecoderFFNIslandProbeConfig:
    model_id: str | None = None
    batch_size: int = 2
    seq_len: int = 8
    use_pad: bool = True
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


def _extract_linear(module) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = module.weight.detach().clone().T.contiguous()
    bias = None if module.bias is None else module.bias.detach().clone()
    return weight, bias


def _t5_activation_type(config) -> tuple[str, bool]:
    """Return ``(activation_type, is_gated)`` from T5 config."""
    raw = getattr(config, "feed_forward_proj", "relu") or "relu"
    raw = raw.lower()
    if raw.startswith("gated-"):
        return raw.split("-", 1)[1], True
    return raw, bool(getattr(config, "is_gated_act", False))


def _bart_activation_type(config) -> str:
    raw = getattr(config, "activation_function", "gelu") or "gelu"
    raw = raw.lower()
    if raw in {"gelu_new", "gelu_pytorch_tanh"}:
        return "gelu"
    if raw == "swish":
        return "silu"
    return raw


def _discover_ffn(model) -> dict[str, Any]:
    """Return a descriptor of the encoder FFN to probe.

    Keys (always present):
      * ``ffn_type``     — ``"t5_dense_relu_dense"`` | ``"t5_gated"`` |
                           ``"bart_fc1_fc2"`` | ``"unknown"``
      * ``status``       — ``"loaded"`` | ``"unsupported"``
      * ``reason``       — only present when ``status=="unsupported"``.
    When ``status=="loaded"`` additional keys carry the extracted weights
    and the activation type.
    """
    # T5 encoder.block[0].layer[-1].DenseReluDense (wi/wo or wi_0/wi_1/wo)
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "block") and len(encoder.block) > 0:
        first_block = encoder.block[0]
        if hasattr(first_block, "layer") and len(first_block.layer) > 0:
            ff_wrapper = first_block.layer[-1]
            dense = getattr(ff_wrapper, "DenseReluDense", None)
            if dense is not None:
                activation_type, is_gated = _t5_activation_type(model.config)
                if hasattr(dense, "wi_0") and hasattr(dense, "wi_1") and hasattr(dense, "wo"):
                    return {
                        "status": "loaded",
                        "ffn_type": "t5_gated",
                        "activation_type": activation_type,
                        "is_gated": True,
                        "wi_up": dense.wi_0,   # ungated branch
                        "wi_gate": dense.wi_1,  # gated branch (activation here)
                        "wo": dense.wo,
                    }
                if hasattr(dense, "wi") and hasattr(dense, "wo"):
                    return {
                        "status": "loaded",
                        "ffn_type": "t5_dense_relu_dense",
                        "activation_type": activation_type,
                        "is_gated": False,
                        "wi": dense.wi,
                        "wo": dense.wo,
                    }
    # BART encoder.layers[0].fc1 / fc2
    inner = getattr(model, "model", model)
    encoder = getattr(inner, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers") and len(encoder.layers) > 0:
        first_layer = encoder.layers[0]
        if hasattr(first_layer, "fc1") and hasattr(first_layer, "fc2"):
            activation_type = _bart_activation_type(model.config)
            return {
                "status": "loaded",
                "ffn_type": "bart_fc1_fc2",
                "activation_type": activation_type,
                "is_gated": False,
                "wi": first_layer.fc1,
                "wo": first_layer.fc2,
            }
    return {
        "status": "unsupported",
        "ffn_type": "unknown",
        "reason": (
            f"No T5 DenseReluDense / DenseGatedActDense or BART fc1/fc2 found"
            f" on model class {type(model).__name__}"
        ),
    }


def _security_caveats() -> list[str]:
    return [
        "Compatible mask families are weaker than unrestricted dense masks.",
        "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
        "Probe-level T5/BART FFN integration only — not a full T5 or BART wrapper.",
        "LM head / encoder-decoder generation are not integrated.",
        "Cross-attention probe invariants are not modified.",
        "This is not a real TEE measurement.",
    ]


def _gated_activation_supported(activation_type: str) -> bool:
    return activation_type == "silu"


def _plain_unscaled_ffn(
    x: torch.Tensor,
    w_in: torch.Tensor,
    b_in: torch.Tensor | None,
    w_out: torch.Tensor,
    b_out: torch.Tensor | None,
    activation_type: str,
) -> torch.Tensor:
    act = get_activation(activation_type)
    z = x @ w_in
    if b_in is not None:
        z = z + b_in
    a = act(z)
    y = a @ w_out
    if b_out is not None:
        y = y + b_out
    return y


def _plain_gated_silu_ffn(
    x: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
) -> torch.Tensor:
    up = x @ w_up
    if b_up is not None:
        up = up + b_up
    gate = x @ w_gate
    if b_gate is not None:
        gate = gate + b_gate
    h = up * F.silu(gate)
    y = h @ w_down
    if b_down is not None:
        y = y + b_down
    return y


def run_encoder_decoder_ffn_island_probe(
    config: EncoderDecoderFFNIslandProbeConfig,
) -> dict[str, Any]:
    """Run one T5 / BART FFN compatible-island probe cell."""
    nonlinear_mode = normalize_nonlinear_mode(config.nonlinear_mode)
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = _atol_rtol(dtype)

    candidates = (
        (config.model_id,)
        if config.model_id is not None
        else DEFAULT_ARCHITECTURE_MODELS["encoder_decoder"]
    )
    try:
        model_id, model = load_for_architecture(
            "encoder_decoder", candidates=candidates
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "config": asdict(config),
            "status": "skipped",
            "candidates_tried": list(candidates),
            "reason": f"{type(exc).__name__}: {exc}",
            "ffn_metrics": {},
            "nonlinear_mode": nonlinear_mode,
        }
    model.eval()

    descriptor = _discover_ffn(model)
    hidden_size = int(getattr(model.config, "d_model", None) or getattr(model.config, "hidden_size"))

    if descriptor["status"] == "unsupported":
        return {
            "config": asdict(config),
            "status": "unsupported",
            "model_id": model_id,
            "model_class": type(model).__name__,
            "candidates_tried": list(candidates),
            "ffn_type": "unknown",
            "reason": descriptor["reason"],
            "nonlinear_mode": nonlinear_mode,
            "ffn_metrics": {},
        }

    ffn_type = descriptor["ffn_type"]
    activation_type = descriptor["activation_type"]
    is_gated = bool(descriptor.get("is_gated", False))

    if is_gated and not _gated_activation_supported(activation_type):
        return {
            "config": asdict(config),
            "status": "unsupported",
            "model_id": model_id,
            "model_class": type(model).__name__,
            "candidates_tried": list(candidates),
            "ffn_type": ffn_type,
            "activation_type": activation_type,
            "is_gated": True,
            "reason": (
                f"gated-{activation_type} FFN island not yet supported; Stage 5.2a"
                f" run_swiglu_mlp_island only covers SiLU. Skipping with explicit"
                f" limitation."
            ),
            "nonlinear_mode": nonlinear_mode,
            "ffn_metrics": {},
        }

    base_payload = {
        "config": asdict(config),
        "status": "loaded",
        "model_id": model_id,
        "model_class": type(model).__name__,
        "candidates_tried": list(candidates),
        "hidden_size": hidden_size,
        "ffn_type": ffn_type,
        "activation_type": activation_type,
        "is_gated": is_gated,
        "nonlinear_mode": nonlinear_mode,
        "lm_head_not_modified": True,
        "encoder_decoder_generation_not_modified": True,
        "cross_attention_probe_not_modified": True,
        "online_extra_matmul_count": 0,
    }

    if not is_gated:
        W_in, b_in = _extract_linear(descriptor["wi"])
        W_out, b_out = _extract_linear(descriptor["wo"])
        W_in = W_in.to(dtype=dtype, device=device)
        W_out = W_out.to(dtype=dtype, device=device)
        b_in = None if b_in is None else b_in.to(dtype=dtype, device=device)
        b_out = None if b_out is None else b_out.to(dtype=dtype, device=device)
        intermediate_size = int(W_in.shape[1])

        hidden_states = torch.randn(
            config.batch_size, config.seq_len, hidden_size, dtype=dtype, device=device
        )
        x_flat = hidden_states.reshape(-1, hidden_size)
        Y_plain_flat = _plain_unscaled_ffn(
            x_flat, W_in, b_in, W_out, b_out, activation_type
        )
        Y_plain = Y_plain_flat.reshape(config.batch_size, config.seq_len, hidden_size)

        base_payload["intermediate_size"] = intermediate_size

        if nonlinear_mode == "trusted":
            return {
                **base_payload,
                "nonlinear_mode_active": False,
                "permutation_dim": None,
                "pad_placement": "n/a",
                "uses_fresh_permutation": True,
                "uses_paired_permutation": False,
                "security_profile": "n/a",
                "security_caveats": [],
                "ffn_metrics": {
                    "max_abs_error": 0.0,
                    "relative_l2_error": 0.0,
                    "cosine_similarity": 1.0,
                    "allclose": True,
                },
                "tilde_invariant_metrics": {},
            }

        N_in, N_in_inv = generate_invertible_matrix(hidden_size, dtype, device)
        perm_data = generate_permutation(intermediate_size, dtype=dtype, device=device)
        perm = perm_data["perm"]
        N_out, N_out_inv = generate_invertible_matrix(hidden_size, dtype, device)
        pad_in = None
        if config.use_pad:
            pad_in = generate_pad(tuple(x_flat.shape), dtype=dtype, device=device, scale=1.0)

        island = run_gelu_mlp_island(
            x=x_flat,
            w1=W_in,
            b1=b_in,
            w2=W_out,
            b2=b_out,
            n_in=N_in,
            n_in_inv=N_in_inv,
            permutation=perm,
            n_out=N_out,
            activation_type=activation_type,
            pad_in=pad_in,
        )
        Y_tilde_flat = island["y_tilde"]
        expected_Y_tilde = island["expected_y_tilde"]
        Y_recovered_flat = Y_tilde_flat @ N_out_inv
        Y_recovered = Y_recovered_flat.reshape(
            config.batch_size, config.seq_len, hidden_size
        )
        correctness_metrics = compare(Y_plain, Y_recovered, atol=atol, rtol=rtol)
        tilde_metrics = compare(expected_Y_tilde, Y_tilde_flat, atol=atol, rtol=rtol)

        return {
            **base_payload,
            "nonlinear_mode_active": True,
            "permutation_dim": int(intermediate_size),
            "pad_placement": "linear_boundary_only" if config.use_pad else "n/a",
            "uses_fresh_permutation": True,
            "uses_paired_permutation": False,
            "security_profile": "proxy-evaluated, not formal",
            "security_caveats": _security_caveats(),
            "ffn_metrics": correctness_metrics,
            "tilde_invariant_metrics": tilde_metrics,
        }

    # ---- gated SiLU FFN ----
    W_up, b_up = _extract_linear(descriptor["wi_up"])
    W_gate, b_gate = _extract_linear(descriptor["wi_gate"])
    W_down, b_down = _extract_linear(descriptor["wo"])
    W_up = W_up.to(dtype=dtype, device=device)
    W_gate = W_gate.to(dtype=dtype, device=device)
    W_down = W_down.to(dtype=dtype, device=device)
    b_up = None if b_up is None else b_up.to(dtype=dtype, device=device)
    b_gate = None if b_gate is None else b_gate.to(dtype=dtype, device=device)
    b_down = None if b_down is None else b_down.to(dtype=dtype, device=device)
    intermediate_size = int(W_up.shape[1])

    hidden_states = torch.randn(
        config.batch_size, config.seq_len, hidden_size, dtype=dtype, device=device
    )
    x_flat = hidden_states.reshape(-1, hidden_size)
    Y_plain_flat = _plain_gated_silu_ffn(
        x_flat, W_up, b_up, W_gate, b_gate, W_down, b_down
    )
    Y_plain = Y_plain_flat.reshape(config.batch_size, config.seq_len, hidden_size)
    base_payload["intermediate_size"] = intermediate_size

    if nonlinear_mode == "trusted":
        return {
            **base_payload,
            "nonlinear_mode_active": False,
            "permutation_dim": None,
            "pad_placement": "n/a",
            "uses_fresh_permutation": True,
            "uses_paired_permutation": True,
            "security_profile": "n/a",
            "security_caveats": [],
            "ffn_metrics": {
                "max_abs_error": 0.0,
                "relative_l2_error": 0.0,
                "cosine_similarity": 1.0,
                "allclose": True,
            },
            "tilde_invariant_metrics": {},
        }

    N_in, N_in_inv = generate_invertible_matrix(hidden_size, dtype, device)
    perm_data = generate_permutation(intermediate_size, dtype=dtype, device=device)
    perm = perm_data["perm"]
    N_out, N_out_inv = generate_invertible_matrix(hidden_size, dtype, device)
    pad_in = None
    if config.use_pad:
        pad_in = generate_pad(tuple(x_flat.shape), dtype=dtype, device=device, scale=1.0)

    island = run_swiglu_mlp_island(
        x=x_flat,
        w_up=W_up,
        b_up=b_up,
        w_gate=W_gate,
        b_gate=b_gate,
        w_down=W_down,
        b_down=b_down,
        n_in=N_in,
        n_in_inv=N_in_inv,
        permutation=perm,
        n_out=N_out,
        pad_in=pad_in,
    )
    Y_tilde_flat = island["y_tilde"]
    expected_Y_tilde = island["expected_y_tilde"]
    Y_recovered_flat = Y_tilde_flat @ N_out_inv
    Y_recovered = Y_recovered_flat.reshape(
        config.batch_size, config.seq_len, hidden_size
    )
    correctness_metrics = compare(Y_plain, Y_recovered, atol=atol, rtol=rtol)
    tilde_metrics = compare(expected_Y_tilde, Y_tilde_flat, atol=atol, rtol=rtol)

    return {
        **base_payload,
        "nonlinear_mode_active": True,
        "permutation_dim": int(intermediate_size),
        "pad_placement": "linear_boundary_only" if config.use_pad else "n/a",
        "uses_fresh_permutation": True,
        "uses_paired_permutation": True,
        "security_profile": "proxy-evaluated, not formal",
        "security_caveats": _security_caveats(),
        "ffn_metrics": correctness_metrics,
        "tilde_invariant_metrics": tilde_metrics,
    }


__all__ = [
    "EncoderDecoderFFNIslandProbeConfig",
    "run_encoder_decoder_ffn_island_probe",
]
