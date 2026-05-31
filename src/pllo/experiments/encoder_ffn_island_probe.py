"""Stage 5.3c — encoder-only FFN compatible-island probe.

Probes the BERT-style FFN ``intermediate.dense → activation → output.dense``
under ``nonlinear_mode ∈ {"trusted", "compatible_islands"}``. Reuses the
``encoder_only`` candidate registry so the probe runs against the same model
loader as the Stage 6.1 attention probe.

This is a **probe-level** integration:

* No full BERT wrapper, no BertModel.forward replacement.
* No MLM head / pooler / classifier integration.
* LayerNorm remains a trusted shortcut and is **not** modified.
* The probe only verifies that the operator-compatible GELU MLP island
  (Stage 5.2a ``run_gelu_mlp_island``) recovers the plain FFN output:
  ``Y_tilde = Y N_out``, with ``online_extra_matmul_count = 0`` and pad
  compensation kept at the Linear boundary (``"linear_boundary_only"``).
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
from pllo.ops.nonlinear_islands import get_activation, run_gelu_mlp_island


_BERT_ACTIVATION_NAMES = {
    "GELUActivation": "gelu",
    "NewGELUActivation": "gelu",
    "GELU": "gelu",
    "PytorchGELUTanh": "gelu",
    "ReLU": "relu",
    "SiLUActivation": "silu",
    "SiLU": "silu",
    "Swish": "silu",
}


@dataclass
class EncoderFFNIslandProbeConfig:
    model_id: str | None = None
    batch_size: int = 2
    seq_len: int = 8
    use_pad: bool = True
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42


def _first_encoder_layer(model):
    base = getattr(model, "bert", None)
    if base is None:
        base = model
    encoder = getattr(base, "encoder", None)
    if encoder is None or not hasattr(encoder, "layer"):
        raise RuntimeError(
            f"Could not find encoder.layer on model class {type(model).__name__}"
        )
    return base.encoder.layer[0]


def _bert_ffn_components(model) -> tuple[Any, Any, str]:
    """Return ``(intermediate_dense, output_dense, activation_type)``."""
    layer = _first_encoder_layer(model)
    intermediate_dense = layer.intermediate.dense
    output_dense = layer.output.dense
    act_module = layer.intermediate.intermediate_act_fn
    act_name = type(act_module).__name__
    activation_type = _BERT_ACTIVATION_NAMES.get(act_name)
    if activation_type is None:
        cfg_activation = getattr(model.config, "hidden_act", None)
        if isinstance(cfg_activation, str):
            normalized = cfg_activation.lower()
            if normalized in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
                activation_type = "gelu"
            elif normalized in {"relu",}:
                activation_type = "relu"
            elif normalized in {"silu", "swish"}:
                activation_type = "silu"
    if activation_type is None:
        raise RuntimeError(
            f"Unsupported BERT activation {act_name!r}; expected gelu / relu / silu."
        )
    return intermediate_dense, output_dense, activation_type


def _extract_linear(module) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = module.weight.detach().clone().T.contiguous()
    bias = None if module.bias is None else module.bias.detach().clone()
    return weight, bias


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


def _security_caveats() -> list[str]:
    return [
        "Compatible mask families are weaker than unrestricted dense masks.",
        "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
        "Probe-level BERT FFN integration only — not a full BERT wrapper.",
        "MLM head / pooler / classifier are not integrated.",
        "LayerNorm remains trusted.",
        "This is not a real TEE measurement.",
    ]


def _plain_ffn(
    x: torch.Tensor,
    w_int: torch.Tensor,
    b_int: torch.Tensor | None,
    w_out: torch.Tensor,
    b_out: torch.Tensor | None,
    activation_type: str,
) -> torch.Tensor:
    act = get_activation(activation_type)
    z = x @ w_int
    if b_int is not None:
        z = z + b_int
    a = act(z)
    y = a @ w_out
    if b_out is not None:
        y = y + b_out
    return y


def run_encoder_ffn_island_probe(
    config: EncoderFFNIslandProbeConfig,
) -> dict[str, Any]:
    """Run one BERT FFN compatible-island probe cell."""
    nonlinear_mode = normalize_nonlinear_mode(config.nonlinear_mode)
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    atol, rtol = _atol_rtol(dtype)

    candidates = (
        (config.model_id,)
        if config.model_id is not None
        else DEFAULT_ARCHITECTURE_MODELS["encoder_only"]
    )
    try:
        model_id, model = load_for_architecture(
            "encoder_only", candidates=candidates
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

    intermediate_dense, output_dense, activation_type = _bert_ffn_components(model)
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(intermediate_dense.out_features)

    W_int, b_int = _extract_linear(intermediate_dense)
    W_out, b_out = _extract_linear(output_dense)
    W_int = W_int.to(dtype=dtype, device=device)
    W_out = W_out.to(dtype=dtype, device=device)
    b_int = None if b_int is None else b_int.to(dtype=dtype, device=device)
    b_out = None if b_out is None else b_out.to(dtype=dtype, device=device)

    hidden_states = torch.randn(
        config.batch_size, config.seq_len, hidden_size, dtype=dtype, device=device
    )
    x_flat = hidden_states.reshape(-1, hidden_size)

    Y_plain_flat = _plain_ffn(x_flat, W_int, b_int, W_out, b_out, activation_type)
    Y_plain = Y_plain_flat.reshape(config.batch_size, config.seq_len, hidden_size)

    base_payload = {
        "config": asdict(config),
        "status": "loaded",
        "model_id": model_id,
        "model_class": type(model).__name__,
        "candidates_tried": list(candidates),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "activation_type": activation_type,
        "nonlinear_mode": nonlinear_mode,
        "layernorm_remains_trusted": True,
        "mlm_head_not_modified": True,
        "pooler_not_modified": True,
        "classifier_not_modified": True,
        "online_extra_matmul_count": 0,
    }

    if nonlinear_mode == "trusted":
        # Default trusted mode: probe records that the FFN island is inactive
        # and the existing Stage 6.1 trusted shortcut semantics apply.
        return {
            **base_payload,
            "nonlinear_mode_active": False,
            "permutation_dim": None,
            "pad_placement": "n/a",
            "uses_fresh_permutation": True,
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

    # ---- nonlinear_mode == "compatible_islands" ----
    N_in, N_in_inv = generate_invertible_matrix(hidden_size, dtype, device)
    perm_data = generate_permutation(intermediate_size, dtype=dtype, device=device)
    perm = perm_data["perm"]
    N_out, N_out_inv = generate_invertible_matrix(hidden_size, dtype, device)

    pad_in = None
    if config.use_pad:
        pad_in = generate_pad(tuple(x_flat.shape), dtype=dtype, device=device, scale=1.0)

    island = run_gelu_mlp_island(
        x=x_flat,
        w1=W_int,
        b1=b_int,
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
    Y_recovered = Y_recovered_flat.reshape(config.batch_size, config.seq_len, hidden_size)

    correctness_metrics = compare(Y_plain, Y_recovered, atol=atol, rtol=rtol)
    tilde_metrics = compare(expected_Y_tilde, Y_tilde_flat, atol=atol, rtol=rtol)

    return {
        **base_payload,
        "nonlinear_mode_active": True,
        "permutation_dim": int(intermediate_size),
        "pad_placement": "linear_boundary_only" if config.use_pad else "n/a",
        "uses_fresh_permutation": True,
        "security_profile": "proxy-evaluated, not formal",
        "security_caveats": _security_caveats(),
        "ffn_metrics": correctness_metrics,
        "tilde_invariant_metrics": tilde_metrics,
    }


__all__ = [
    "EncoderFFNIslandProbeConfig",
    "run_encoder_ffn_island_probe",
]
