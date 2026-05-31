"""Stage 6.4 — Modern decoder-only (Qwen / TinyLlama / LLaMA) probe orchestrator.

Composes the four probe-level checks required for modern decoder-only
architectures:

* **RMSNorm orthogonal island** — reuses Stage 5.2a
  ``run_rmsnorm_orthogonal_island``.
* **SwiGLU paired-permutation island** — reuses Stage 5.2a
  ``run_swiglu_mlp_island``.
* **RoPE-aware attention probe** — Stage 6.4 ``rope_probe``.
* **GQA / MQA KV-shape probe** — Stage 6.4 ``gqa_probe``.

Model loading is best-effort: the runner walks
``DEFAULT_ARCHITECTURE_MODELS["modern_decoder_only"]`` and falls back to a
synthetic probe (``load_status="synthetic_only"``) if every candidate
fails. ``pytest`` therefore never needs network access — the default unit
tests run in synthetic-only mode.

This is a **probe-level** integration. No full Qwen / TinyLlama wrapper,
no generation, no KV-cache runtime, no LM head. Default mode for the
wider system remains ``nonlinear_mode="trusted"`` — the Stage 5.3a /
5.3b / 5.3c feature flag is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.architectures.architecture_registry import (
    DEFAULT_ARCHITECTURE_MODELS,
    MODERN_DECODER_FAMILY_MAP,
)
from pllo.experiments.gqa_probe import GqaProbeConfig, run_gqa_probe
from pllo.experiments.rope_probe import RopeProbeConfig, run_rope_probe
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.ops.compatible_masks import (
    generate_orthogonal,
    generate_permutation,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    bundle_metadata,
    normalize_mitigation_bundle,
)
from pllo.ops.nonlinear_islands import (
    run_rmsnorm_orthogonal_island,
    run_swiglu_mlp_island,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderProbeConfig:
    output_dir: str = "outputs"
    model_id: str | None = None
    use_synthetic_if_unavailable: bool = True
    attempt_real_model_load: bool = False
    batch_size: int = 2
    seq_len: int = 8
    hidden_size: int = 128
    intermediate_size: int = 512
    num_query_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 32
    use_pad_values: tuple[bool, ...] = (False, True)
    mitigation_bundle: str = DEFAULT_MITIGATION_BUNDLE
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 2026


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


# ---------------------------------------------------------------------------
# Optional best-effort model load (skips silently to synthetic)
# ---------------------------------------------------------------------------


def _try_load_modern_decoder(
    config: ModernDecoderProbeConfig,
) -> dict[str, Any]:
    """Return a ``model_loading`` dict; never raises."""
    if not config.attempt_real_model_load:
        return {
            "status": "synthetic_only",
            "reason": (
                "attempt_real_model_load=False (default); modern_decoder"
                " probe uses synthetic tensors to avoid network downloads."
            ),
            "candidates_tried": [],
            "model_id": None,
            "model_family": None,
        }
    candidates = (
        (config.model_id,)
        if config.model_id is not None
        else DEFAULT_ARCHITECTURE_MODELS["modern_decoder_only"]
    )
    try:
        from pllo.architectures import load_for_architecture
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "synthetic_only",
            "reason": f"import failure: {type(exc).__name__}: {exc}",
            "candidates_tried": list(candidates),
            "model_id": None,
            "model_family": None,
        }
    failures: list[str] = []
    for model_id in candidates:
        try:
            mid, model = load_for_architecture(
                "modern_decoder_only", candidates=(model_id,)
            )
            family = MODERN_DECODER_FAMILY_MAP.get(mid, "unknown")
            return {
                "status": "loaded",
                "model_id": mid,
                "model_family": family,
                "model_class": type(model).__name__,
                "candidates_tried": list(candidates),
                "reason": None,
            }
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{model_id}: {type(exc).__name__}: {exc}")
    return {
        "status": "synthetic_only",
        "reason": "no compatible local/HF modern decoder model available",
        "candidates_tried": list(candidates),
        "failures": failures,
        "model_id": None,
        "model_family": None,
    }


# ---------------------------------------------------------------------------
# RMSNorm probe (reuses Stage 5.2a)
# ---------------------------------------------------------------------------


def _rmsnorm_probe(
    config: ModernDecoderProbeConfig, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    atol, rtol = _atol_rtol(dtype)
    H = config.hidden_size
    B, S = config.batch_size, config.seq_len
    mitigation_bundle = normalize_mitigation_bundle(config.mitigation_bundle)
    per_use_pad: dict[str, dict[str, Any]] = {}
    for use_pad in config.use_pad_values:
        torch.manual_seed(config.seed + (1 if use_pad else 0))
        x = torch.randn(B, S, H, dtype=dtype, device=device)
        gamma = 0.5 + torch.rand(H, dtype=dtype, device=device)
        W = torch.randn(H, H, dtype=dtype, device=device) * 0.1
        b = torch.randn(H, dtype=dtype, device=device) * 0.1
        N = generate_orthogonal(H, dtype, device)
        N_out, _ = generate_invertible_matrix(H, dtype, device)
        flat = x.reshape(-1, H)
        island = run_rmsnorm_orthogonal_island(
            x=flat,
            n_in_orthogonal=N,
            norm_weight=gamma,
            linear_weight=W,
            linear_bias=b,
            n_out=N_out,
        )
        Y_plain = island["y_plain"]
        Y_tilde = island["y_tilde"]
        expected_tilde = island["expected_y_tilde"]
        # Recover from N_out and compare to plain.
        N_out_inv = torch.linalg.inv(N_out.to(torch.float64)).to(dtype)
        Y_recovered = Y_tilde @ N_out_inv
        diff_recover = (Y_recovered - Y_plain).abs()
        diff_tilde = (Y_tilde - expected_tilde).abs()
        per_use_pad[str(use_pad)] = {
            "use_pad": bool(use_pad),
            "use_pad_note": (
                "RMSNorm island has no per-pair pad slot; pad is reserved"
                " for the adjacent Linear boundary."
            ),
            "rms_core_max_abs_error": float(diff_tilde.max().item()),
            "folded_output_max_abs_error": float(diff_recover.max().item()),
            "relative_l2_error": float(
                ((Y_recovered - Y_plain).norm()
                 / Y_plain.norm().clamp_min(1e-30)).item()
            ),
            "cosine_similarity": float(
                (Y_recovered.flatten() @ Y_plain.flatten()
                 / (Y_recovered.norm() * Y_plain.norm()).clamp_min(1e-30)).item()
            ),
            "allclose": bool(
                torch.allclose(Y_recovered, Y_plain, atol=atol, rtol=rtol)
            ),
            "online_extra_matmul_count": 0,
            "permutation_dim": None,
            "norm_type": "rmsnorm",
            "mask_family": "orthogonal",
            "mitigation_bundle": mitigation_bundle,
            "mitigation_bundle_metadata_note": (
                "RMSNorm island does not itself add a permutation; the"
                " mitigation bundle is inherited from the adjacent MLP"
                " island (SwiGLU). The bundle label is propagated here for"
                " consistency only."
            ),
        }
    return {
        "status": "ok",
        "norm_type": "rmsnorm",
        "mask_family": "orthogonal",
        "online_extra_matmul_count": 0,
        "mitigation_bundle": mitigation_bundle,
        "per_use_pad": per_use_pad,
        "allclose": all(r["allclose"] for r in per_use_pad.values()),
    }


# ---------------------------------------------------------------------------
# SwiGLU probe (reuses Stage 5.2a run_swiglu_mlp_island)
# ---------------------------------------------------------------------------


def _swiglu_probe(
    config: ModernDecoderProbeConfig, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    atol, rtol = _atol_rtol(dtype)
    H = config.hidden_size
    I = config.intermediate_size
    B, S = config.batch_size, config.seq_len
    mitigation_bundle = normalize_mitigation_bundle(config.mitigation_bundle)
    per_use_pad: dict[str, dict[str, Any]] = {}
    for use_pad in config.use_pad_values:
        torch.manual_seed(config.seed + 100 + (1 if use_pad else 0))
        x = torch.randn(B, S, H, dtype=dtype, device=device)
        flat = x.reshape(-1, H)
        W_up = torch.randn(H, I, dtype=dtype, device=device) * 0.1
        W_gate = torch.randn(H, I, dtype=dtype, device=device) * 0.1
        W_down = torch.randn(I, H, dtype=dtype, device=device) * 0.1
        # T5-style SwiGLU has no bias; LLaMA-style same. We test both.
        b_up = None
        b_gate = None
        b_down = None

        N_in, N_in_inv = generate_invertible_matrix(H, dtype, device)
        N_out, _ = generate_invertible_matrix(H, dtype, device)
        perm = generate_permutation(I, dtype=dtype, device=device)["perm"]
        pad_in = None
        if use_pad:
            pad_in = torch.randn(flat.shape, dtype=dtype, device=device)

        island = run_swiglu_mlp_island(
            x=flat,
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
            mitigation_bundle=mitigation_bundle,
        )
        bundle_meta = island["mitigation_bundle_metadata"]
        Y_plain = island["y_plain"]
        Y_tilde = island["y_tilde"]
        N_out_inv = torch.linalg.inv(N_out.to(torch.float64)).to(dtype)
        Y_recovered = Y_tilde @ N_out_inv

        # Verify gate/up shared P invariant via the intermediate-tensor
        # checks already produced by the island.
        g_tilde = island["g_tilde"]
        g_plain_permuted = island["g_plain_permuted"]
        gp_metrics = {
            "max_abs_error": float((g_tilde - g_plain_permuted).abs().max().item()),
            "allclose": bool(
                torch.allclose(g_tilde, g_plain_permuted, atol=atol, rtol=rtol)
            ),
        }

        per_use_pad[str(use_pad)] = {
            "use_pad": bool(use_pad),
            "pad_placement": "linear_boundary_only" if use_pad else "n/a",
            "max_abs_error": float((Y_recovered - Y_plain).abs().max().item()),
            "relative_l2_error": float(
                ((Y_recovered - Y_plain).norm()
                 / Y_plain.norm().clamp_min(1e-30)).item()
            ),
            "cosine_similarity": float(
                (Y_recovered.flatten() @ Y_plain.flatten()
                 / (Y_recovered.norm() * Y_plain.norm()).clamp_min(1e-30)).item()
            ),
            "allclose": bool(
                torch.allclose(Y_recovered, Y_plain, atol=atol, rtol=rtol)
            ),
            "online_extra_matmul_count": 0,
            "permutation_dim": int(I),
            "intermediate_size": int(I),
            "shared_permutation_for_up_gate": True,
            "gated_intermediate_invariant": gp_metrics,
            "mitigation_bundle": mitigation_bundle,
            "mitigation_bundle_metadata": bundle_meta,
            "dense_sandwich_enabled": bundle_meta["dense_sandwich_enabled"],
            "fresh_permutation_enabled": bundle_meta["fresh_permutation_enabled"],
            "boundary_pad_enabled": bundle_meta["boundary_pad_enabled"],
            "default_on_candidate_under_stage_5_4": bundle_meta[
                "default_on_candidate_under_stage_5_4"
            ],
            "activation_input_form": bundle_meta["activation_input_form"],
        }
    return {
        "status": "ok",
        "activation_type": "swiglu",
        "online_extra_matmul_count": 0,
        "shared_permutation_for_up_gate": True,
        "mitigation_bundle": mitigation_bundle,
        "per_use_pad": per_use_pad,
        "allclose": all(r["allclose"] for r in per_use_pad.values()),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


_INHERITED_SECURITY_CAVEATS = [
    "Compatible mask families are weaker than unrestricted dense masks.",
    "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
    "Probe-level migration only — not a full Qwen / TinyLlama wrapper integration.",
    "No Qwen / TinyLlama generation path is implemented.",
    "GQA / MQA is tensor-level only, not full runtime KV cache integration.",
    "Compatible islands inherit Stage 5.4 mitigation requirements: fresh permutation + dense sandwich + pad at Linear boundaries.",
    "This is not a real TEE measurement.",
    "This is not formal security.",
]


def run_modern_decoder_probe(
    config: ModernDecoderProbeConfig,
) -> dict[str, Any]:
    """Run RMSNorm + SwiGLU + RoPE + GQA probes for modern decoder-only."""
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)

    model_loading = _try_load_modern_decoder(config)
    rmsnorm = _rmsnorm_probe(config, dtype, device)
    swiglu = _swiglu_probe(config, dtype, device)
    rope = run_rope_probe(
        RopeProbeConfig(
            batch_size=config.batch_size,
            num_heads=config.num_query_heads,
            seq_len=config.seq_len,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=config.device,
            seed=config.seed + 200,
        )
    )
    gqa = run_gqa_probe(
        GqaProbeConfig(
            batch_size=config.batch_size,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            seq_len=config.seq_len,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=config.device,
            seed=config.seed + 300,
        )
    )

    architecture_spec = {
        "architecture_type": "decoder_only",
        "model_family": (
            model_loading.get("model_family")
            or "synthetic_modern_decoder"
        ),
        "norm_type": "rmsnorm",
        "activation_type": "swiglu",
        "position_encoding_type": "rotary",
        "attention_variant": gqa.get("attention_variant", "unknown"),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_query_heads": int(config.num_query_heads),
        "num_kv_heads": int(config.num_kv_heads),
        "head_dim": int(config.head_dim),
    }

    all_required_allclose = bool(
        rmsnorm["allclose"]
        and swiglu["allclose"]
        and rope["status"] == "ok"
        and rope["probe_a_post_rope_masking_invariant"]["allclose"]
        and gqa["allclose"]
    )

    mitigation_bundle = normalize_mitigation_bundle(config.mitigation_bundle)
    global_summary = {
        "architecture_type": "decoder_only",
        "model_family": architecture_spec["model_family"],
        "norm_type": "rmsnorm",
        "activation_type": "swiglu",
        "position_encoding_type": "rotary",
        "attention_variant": architecture_spec["attention_variant"],
        "all_required_probes_allclose": all_required_allclose,
        "online_extra_matmul_count": 0,
        "integration_level": "probe_level",
        "security_profile": "inherits Stage 5.4 caveats",
        "default_nonlinear_mode": "trusted",
        "mitigation_bundle": mitigation_bundle,
        "default_mitigation_bundle": DEFAULT_MITIGATION_BUNDLE,
        "mitigation_applies_to": (
            "nonlinear islands (RMSNorm / SwiGLU). RoPE and GQA probes are"
            " mask-mathematics independent of the mitigation bundle."
        ),
    }

    return {
        "config": asdict(config),
        "model_loading": model_loading,
        "architecture_spec": architecture_spec,
        "rmsnorm_probe": rmsnorm,
        "swiglu_probe": swiglu,
        "rope_probe": rope,
        "gqa_probe": gqa,
        "global_summary": global_summary,
        "limitations": _INHERITED_SECURITY_CAVEATS,
    }


__all__ = [
    "ModernDecoderProbeConfig",
    "run_modern_decoder_probe",
]
