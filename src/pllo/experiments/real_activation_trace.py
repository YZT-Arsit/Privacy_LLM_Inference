"""Stage 5.5 — real-activation trace collection.

Drives the Stage 6.4b modern decoder block wrapper across many sessions
(each call uses fresh per-session masks/permutations) and collects pairs
of ``(plain_tensor, attacker_visible_tensor)`` at the key
boundaries that a Stage 5.4-style adaptive attacker would observe in a
deployed system. The output is consumed by
:mod:`pllo.experiments.real_activation_attacker`.

Scope:

* **Block-level only.** Reuses Stage 6.4b's
  ``ObfuscatedModernDecoderBlockWrapper.forward_with_traces``.
* **Real or synthetic block.** ``attempt_real_model_load=False`` (default)
  drives a synthetic LLaMA-shape block so pytest never hits the network.
  ``attempt_real_model_load=True`` tries the registered
  ``modern_decoder_only`` candidates and silently falls back to synthetic
  on any failure.
* **No tokenizer / embedding required.** Block input is sampled directly
  in hidden-state space; the limitation is honestly reported.
* **No tensor leakage in JSON.** ``traces`` is materialised in memory as
  ``torch.Tensor`` for the in-process attacker; the JSON-safe
  ``trace_summary`` carries only shapes, sample counts, fingerprints and
  scalar statistics — never the raw tensor.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.architectures.architecture_registry import (
    DEFAULT_ARCHITECTURE_MODELS,
    MODERN_DECODER_FAMILY_MAP,
)
from pllo.experiments.modern_decoder_block_probe import (
    _real_spec_and_weights,
    _synthetic_spec_and_weights,
    _try_load_real_block,
    ModernDecoderBlockProbeConfig,
    ModernDecoderLoadConfig,
)
from pllo.hf_wrappers.modern_decoder_block_wrapper import (
    ModernDecoderBlockWeights,
    ObfuscatedModernDecoderBlockWrapper,
)
from pllo.model_zoo.modern_decoder_spec import (
    ModernDecoderBlockSpec,
    spec_to_dict,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
    normalize_mitigation_bundle,
)


# Per the Stage 5.5 design the attacker observes a fixed set of tensors at
# Linear / SwiGLU / attention boundaries. Order matters only for output
# stability.
DEFAULT_TARGET_TENSORS: tuple[str, ...] = (
    "boundary_input",
    "q",
    "k",
    "v",
    "gate",
    "up",
    "swiglu_intermediate",
    "post_island",
    "final",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RealActivationTraceConfig:
    """Configuration for one trace-collection run.

    ``num_samples`` is the *target* count of token-row samples in the
    flattened dataset; the runner picks a session count by dividing it by
    ``batch_size × seq_len``. A "session" is one call to the wrapper, so
    masks/permutations refresh between sessions just as they would in
    deployment.
    """

    model_id: str | None = None
    attempt_real_model_load: bool = False
    allow_synthetic_fallback: bool = True
    local_files_only: bool = False
    output_dir: str = "outputs"
    seed: int = 2026
    num_samples: int = 512
    batch_size: int = 2
    seq_len: int = 8
    # Synthetic-block shape (ignored when a real model is loaded).
    synthetic_hidden_size: int = 64
    synthetic_intermediate_size: int = 128
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 16
    synthetic_rope_base: float = 10000.0
    use_pad: bool = True
    nonlinear_mode: str = "compatible_islands"
    mitigation_bundle: str = "fresh_perm_plus_sandwich_plus_pad"
    target_tensors: tuple[str, ...] = DEFAULT_TARGET_TENSORS
    dtype: str = "float32"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_block(
    config: RealActivationTraceConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[ModernDecoderBlockSpec, ModernDecoderBlockWeights, dict[str, Any], str]:
    """Return ``(spec, weights, model_loading, source)``.

    Reuses :mod:`pllo.experiments.modern_decoder_block_probe`'s loading
    helpers so the synthetic-fallback policy stays consistent with Stage
    6.4b.
    """
    load = ModernDecoderLoadConfig(
        model_id=config.model_id,
        attempt_real_model_load=config.attempt_real_model_load,
        allow_synthetic_fallback=config.allow_synthetic_fallback,
        device=config.device,
        dtype=config.dtype,
        local_files_only=config.local_files_only,
    )
    load_record = _try_load_real_block(load)
    real_pair = None
    if load_record["load_status"] == "loaded":
        model = load_record.pop("_model_obj", None)
        real_pair = _real_spec_and_weights(
            model, load_record["resolved_model_id"], dtype, device
        )
        if real_pair is None:
            load_record["load_status"] = (
                "synthetic_only"
                if config.allow_synthetic_fallback else "skipped"
            )
            load_record["load_error"] = (
                "loaded model but inspection / extraction failed"
            )
            load_record["fallback_used"] = bool(config.allow_synthetic_fallback)
    load_record.pop("_model_obj", None)
    if real_pair is None:
        # Synthetic fallback via the probe helper (gives us a matching spec).
        probe_cfg = ModernDecoderBlockProbeConfig(
            load=load,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            synthetic_hidden_size=config.synthetic_hidden_size,
            synthetic_intermediate_size=config.synthetic_intermediate_size,
            synthetic_num_attention_heads=config.synthetic_num_attention_heads,
            synthetic_num_key_value_heads=config.synthetic_num_key_value_heads,
            synthetic_head_dim=config.synthetic_head_dim,
            synthetic_rope_base=config.synthetic_rope_base,
            seed=config.seed,
        )
        spec, weights = _synthetic_spec_and_weights(probe_cfg, dtype, device)
        source = "synthetic_block"
    else:
        spec, weights = real_pair
        source = _source_label_for_model_id(load_record["resolved_model_id"])
    return spec, weights, load_record, source


def _source_label_for_model_id(model_id: str | None) -> str:
    """Compact source label used in trace summaries and reports."""
    if model_id is None:
        return "synthetic_block"
    family = MODERN_DECODER_FAMILY_MAP.get(model_id)
    if model_id == "hf-internal-testing/tiny-random-LlamaForCausalLM":
        return "tiny_random_llama_block"
    if family == "qwen_like":
        return "qwen_like_block"
    if family == "tinyllama":
        return "tinyllama_block"
    if family == "llama_like":
        return "llama_like_block"
    return "real_model_block"


# ---------------------------------------------------------------------------
# Flattening rules for each (visible, plain) tensor pair
# ---------------------------------------------------------------------------


def _flatten_pair(
    name: str, plain: torch.Tensor, visible: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Reshape a (plain, visible) pair into a 2D dataset.

    For per-head tensors (``q`` / ``k`` / ``v``) we flatten heads into the
    sample axis so the attacker sees one ``head_dim``-wide row per
    (batch, head, token). For per-token tensors (``gate`` / ``up`` /
    ``swiglu_intermediate`` / ``post_island`` / ``boundary_input`` /
    ``final``) we flatten batch × seq into the sample axis.
    """
    if plain.ndim == 4:                       # [B, H, S, D]
        B, H, S, D = plain.shape
        plain_flat = plain.permute(0, 2, 1, 3).reshape(B * S * H, D)
        visible_flat = visible.permute(0, 2, 1, 3).reshape(B * S * H, D)
    elif plain.ndim == 3:                     # [B, S, F]
        B, S, F = plain.shape
        plain_flat = plain.reshape(B * S, F)
        visible_flat = visible.reshape(B * S, F)
    elif plain.ndim == 2:                     # already flat
        plain_flat = plain
        visible_flat = visible
    else:
        raise ValueError(
            f"unexpected ndim {plain.ndim} for tensor {name!r}"
        )
    return {"plain": plain_flat, "visible": visible_flat}


# ---------------------------------------------------------------------------
# Fingerprinting (SHA-256 over float32 bytes; never the raw tensor)
# ---------------------------------------------------------------------------


def _fingerprint(t: torch.Tensor) -> str:
    buf = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def _tensor_statistics(t: torch.Tensor) -> dict[str, float]:
    """Scalar-only statistics, safe to publish to JSON / Markdown.

    Never publishes the raw tensor or per-element values.
    """
    flat = t.detach().to(torch.float32).reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "abs_max": float(flat.abs().max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "fingerprint_sha256_prefix": _fingerprint(t),
    }


# ---------------------------------------------------------------------------
# Collection entry point
# ---------------------------------------------------------------------------


def collect_real_activation_traces(
    config: RealActivationTraceConfig,
) -> dict[str, Any]:
    """Run the block wrapper across multiple sessions and return traces.

    Returned dict shape:

    .. code-block:: text

       {
         "config":         <RealActivationTraceConfig dict>,
         "model_loading":  <_try_load_real_block dict>,
         "source":         "synthetic_block" | "tiny_random_llama_block" | ...,
         "block_spec":     <ModernDecoderBlockSpec dict>,
         "traces":         {tensor_name: {"plain": Tensor, "visible": Tensor}},
         "trace_summary":  <JSON-safe summary>,
         "metadata":       <run-level dict: bundle, use_pad, etc.>,
       }

    ``traces`` is the in-memory dataset for downstream attackers. The
    JSON-safe ``trace_summary`` is the only field that should be written
    to disk.
    """
    if config.batch_size * config.seq_len <= 0:
        raise ValueError("batch_size * seq_len must be positive")
    bundle = normalize_mitigation_bundle(config.mitigation_bundle)
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)

    spec, weights, load_record, source = _resolve_block(config, dtype, device)
    tokens_per_session = config.batch_size * config.seq_len
    num_sessions = max(1, (config.num_samples + tokens_per_session - 1) // tokens_per_session)
    H = spec.hidden_size

    wrapper = ObfuscatedModernDecoderBlockWrapper(
        weights,
        dtype=dtype,
        device=device,
        use_pad=config.use_pad,
        nonlinear_mode=config.nonlinear_mode,
        mitigation_bundle=bundle,
    )

    # Accumulate per-tensor (plain, visible) chunks across sessions.
    accum: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"plain": [], "visible": []} for name in config.target_tensors
    }
    final_allclose: list[bool] = []
    final_max_abs_err: list[float] = []

    gen = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    for sess in range(num_sessions):
        x = torch.randn(
            config.batch_size, config.seq_len, H,
            generator=gen, dtype=torch.float32,
        ).to(dtype=dtype, device=device)
        y_recovered, report, traces = wrapper.forward_with_traces(x)
        final_allclose.append(bool(report["allclose"]))
        final_max_abs_err.append(float(report["max_abs_error"]))
        for name in config.target_tensors:
            plain_key = f"{name}_plain"
            visible_key = f"{name}_visible"
            if plain_key not in traces or visible_key not in traces:
                continue
            pair = _flatten_pair(name, traces[plain_key], traces[visible_key])
            accum[name]["plain"].append(pair["plain"])
            accum[name]["visible"].append(pair["visible"])

    # Stitch sessions into [N, D] datasets, truncated to num_samples for
    # determinism across (batch_size, seq_len) sweeps.
    stitched: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for name in config.target_tensors:
        if not accum[name]["plain"]:
            continue
        plain_t = torch.cat(accum[name]["plain"], dim=0)
        visible_t = torch.cat(accum[name]["visible"], dim=0)
        N = min(plain_t.shape[0], visible_t.shape[0])
        n_keep = min(N, max(1, config.num_samples * (N // max(1, num_sessions * tokens_per_session) or 1)))
        # The fairest cap is num_samples token-rows; for q/k/v the per-call
        # row count is num_heads × tokens_per_session, so we cap separately.
        n_keep = min(N, max(config.num_samples, 16))
        plain_t = plain_t[:n_keep]
        visible_t = visible_t[:n_keep]
        stitched[name] = {"plain": plain_t, "visible": visible_t}
        summary[name] = {
            "tensor_name": name,
            "num_samples": int(plain_t.shape[0]),
            "feature_dim": int(plain_t.shape[-1]),
            "plain_shape": list(plain_t.shape),
            "visible_shape": list(visible_t.shape),
            "source": source,
            "mitigation_bundle": bundle,
            "use_pad": bool(config.use_pad),
            "plain_statistics": _tensor_statistics(plain_t),
            "visible_statistics": _tensor_statistics(visible_t),
        }

    metadata = {
        "mitigation_bundle": bundle,
        "mitigation_bundle_metadata": bundle_metadata(
            bundle, use_pad=config.use_pad, online_extra_matmul_count=0
        ),
        "use_pad": bool(config.use_pad),
        "nonlinear_mode": config.nonlinear_mode,
        "num_sessions": int(num_sessions),
        "tokens_per_session": int(tokens_per_session),
        "all_sessions_allclose": bool(all(final_allclose)),
        "final_max_abs_error_max": float(max(final_max_abs_err))
        if final_max_abs_err else 0.0,
        "final_max_abs_error_mean": float(
            sum(final_max_abs_err) / max(1, len(final_max_abs_err))
        ),
    }

    return {
        "config": asdict(config),
        "model_loading": load_record,
        "source": source,
        "block_spec": spec_to_dict(spec),
        "traces": stitched,
        "trace_summary": summary,
        "metadata": metadata,
    }


__all__ = [
    "DEFAULT_TARGET_TENSORS",
    "RealActivationTraceConfig",
    "collect_real_activation_traces",
]
