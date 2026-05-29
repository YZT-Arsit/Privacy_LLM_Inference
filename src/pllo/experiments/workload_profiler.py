"""Stage 5.0.1 workload profiler — calibrated TEE/GPU cost-model for paper experiments.

Cost categories
---------------

Every method reports four explicit slices of workload:

* ``preprocessing_trusted_ops`` / ``preprocessing_transfer_bytes`` — one-off
  trusted-side work to materialise transformed weights, generate static
  masks, etc. Amortised over many sessions, so excluded from headline
  online latency.
* ``online_boundary_calls`` — synchronous trusted ↔ untrusted round trips
  during one full greedy generation. Counted with explicit per-method
  formulas (no per-Python-call inflation). Internal trusted-side bookkeeping
  such as mask-state creation or pad compensation generation is **not** a
  boundary call — it is online trusted compute.
* ``online_trusted_compute_ops`` — FLOPs performed inside the trusted side
  during inference (LayerNorm / GELU under trusted shortcut, mask state
  generation, pad compensation, sampling, logits recovery).
* ``online_gpu_ops`` — FLOPs performed on the GPU during inference (masked
  linear matmuls, attention scores, value aggregation, LM head matmul).

Methods
-------

* ``plain_hf_gpu`` — no protection; measured.
* ``tslp_trusted_nonlinear_baseline`` — every LayerNorm and GELU crosses TEE;
  projected.
* ``ours_current`` — Stage 4.9 wrapper as implemented; measured.
* ``ours_ideal_gpu_nonlinear`` — hypothetical where LN / GELU also run in
  the masked GPU domain; projected upper bound.
* ``amulet_style_reference`` — input mask + GPU pipeline + output recovery
  pattern; projected reference, **not** a re-implementation of Amulet.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from pllo.experiments.experiment_registry import (
    DEFAULT_COST_MODEL,
    INTERACTION_CATEGORIES,
    METHOD_BY_NAME,
    MODULE_CATEGORIES,
    CostModel,
    WORKLOAD_METHODS,
    WorkloadMethod,
)
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper
from pllo.model_zoo import ExternalModelConfig, get_model_loader, torch_dtype_from_string


# ---------------------------------------------------------------------------
# Config containers
# ---------------------------------------------------------------------------


@dataclass
class WorkloadProfileConfig:
    model_id: str = "sshleifer/tiny-gpt2"
    batch_size: int = 2
    prompt_len: int = 8
    max_new_tokens: int = 4
    use_pad: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 42
    warmup: int = 2
    repeat: int = 5
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COST_MODEL)


@dataclass
class ModuleCounts:
    """Per-module workload slice for one full generation under one method."""

    online_gpu_ops: int = 0
    online_trusted_compute_ops: int = 0
    online_boundary_calls: int = 0
    online_trusted_transfer_bytes: int = 0
    location: str = "gpu"  # "gpu", "tee", "mixed"


@dataclass
class InteractionCounts:
    """Per-interaction-type slice (e.g., 'trusted_layernorm') across one generation."""

    online_boundary_calls: int = 0
    online_trusted_transfer_bytes: int = 0
    online_trusted_compute_ops: int = 0
    notes: str = ""


_BYTES_PER_DTYPE = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float64": 8,
}


def _dtype_bytes(dtype: str) -> int:
    return _BYTES_PER_DTYPE.get(dtype, 4)


# ---------------------------------------------------------------------------
# Forward-graph descriptor
# ---------------------------------------------------------------------------


def _forward_specs(config: WorkloadProfileConfig) -> list[tuple[int, int, int]]:
    """Return ``[(tokens_in_forward, query_len, key_len)]`` for one full generation.

    Greedy generation runs one prefill (``prompt_len`` tokens) plus
    ``max_new_tokens - 1`` decode steps (the first new token comes from
    the prefill output).
    """
    forwards = [(config.prompt_len, config.prompt_len, config.prompt_len)]
    for i in range(max(config.max_new_tokens - 1, 0)):
        forwards.append((1, 1, config.prompt_len + i + 1))
    return forwards


def _model_constants(model_cfg, config: WorkloadProfileConfig) -> dict[str, int]:
    hidden = model_cfg.n_embd
    inter = 4 * hidden
    layers = model_cfg.n_layer
    heads = model_cfg.n_head
    head_dim = hidden // heads
    vocab = model_cfg.vocab_size
    return {
        "hidden": hidden,
        "inter": inter,
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "dtype_bytes": _dtype_bytes(config.dtype),
    }


# ---------------------------------------------------------------------------
# Per-forward formulas (explicit, documented, no Python call counting)
# ---------------------------------------------------------------------------


def _per_forward_boundary_calls(method: WorkloadMethod, layers: int) -> int:
    """Return number of trusted ↔ untrusted round trips per forward call.

    Formulas (documented for the paper):

    * ``plain_hf_gpu``               : 0   (no boundary)
    * ``tslp_trusted_nonlinear``     : 3L + 2  (LN_1 + LN_2 + GELU per layer + ln_f + LM head)
    * ``ours_current``               : 4L + 1  (4 obfuscated linears per layer + LM head)
    * ``ours_ideal_gpu_nonlinear``   : 1      (single fused GPU pipeline round trip)
    * ``amulet_style_reference``     : 1      (single fused GPU pipeline round trip)
    """
    if not method.linear_obfuscated and not method.layernorm_in_tee and not method.activation_in_tee:
        # plain
        return 0
    if method.fuses_gpu_pipeline and method.linear_obfuscated and not method.layernorm_in_tee:
        # ours_ideal / amulet — single fused GPU pipeline
        return 1
    # tslp: linear on GPU + non-linear in TEE → one crossing per non-linear
    if not method.linear_obfuscated and (method.layernorm_in_tee or method.activation_in_tee):
        crossings = 0
        if method.layernorm_in_tee:
            crossings += 2 * layers + 1  # ln_1, ln_2 per layer + ln_f
        if method.activation_in_tee:
            crossings += layers
        if method.lm_head_recovered_in_tee:
            crossings += 1
        return crossings
    # ours_current: 4 obfuscated linears per layer + LM head matmul round trip
    crossings = 4 * layers
    if method.lm_head_recovered_in_tee:
        crossings += 1
    return crossings


def _per_forward_trusted_transfer_bytes(
    method: WorkloadMethod,
    consts: dict[str, int],
    tokens: int,
) -> int:
    """Return bytes transferred across the boundary during one forward call."""
    hidden = consts["hidden"]
    inter = consts["inter"]
    layers = consts["layers"]
    vocab = consts["vocab"]
    db = consts["dtype_bytes"]
    if method.name == "plain_hf_gpu":
        return 0
    if method.name == "tslp_trusted_nonlinear_baseline":
        # Per layer: 2 * (LN in + out) + 1 * (GELU in + out)
        per_layer = 2 * 2 * hidden + 2 * inter
        # ln_f input/output 2*hidden, LM head logits recovery vocab
        return tokens * db * (layers * per_layer + 2 * hidden + vocab)
    if method.name == "ours_current":
        # Per layer (4 obfuscated linears, each transfers input + output):
        # c_attn   : hidden in, 3*hidden out → 4*hidden
        # c_proj   : hidden in, hidden out   → 2*hidden
        # c_fc     : hidden in, inter out    → hidden + inter
        # c_proj_mlp: inter in, hidden out   → inter + hidden
        per_layer = 4 * hidden + 2 * hidden + (hidden + inter) + (inter + hidden)
        # LM head: hidden in + vocab logits out (vocab mask recovery in trusted)
        return tokens * db * (layers * per_layer + hidden + vocab)
    if method.name in ("ours_ideal_gpu_nonlinear", "amulet_style_reference"):
        # Input mask (hidden) + LM head logits recovery (vocab).
        return tokens * db * (hidden + vocab)
    return 0


def _per_forward_trusted_compute_ops(
    method: WorkloadMethod,
    consts: dict[str, int],
    tokens: int,
    q_len: int,
    k_len: int,
) -> int:
    """Return trusted-side FLOPs performed online during one forward call."""
    hidden = consts["hidden"]
    inter = consts["inter"]
    layers = consts["layers"]
    heads = consts["heads"]
    head_dim = consts["head_dim"]
    vocab = consts["vocab"]
    if method.name == "plain_hf_gpu":
        return 0

    # LayerNorm in trusted (or in TEE for TSLP / ours_current).
    ln_ops_per_pos = 8 * hidden
    ln_total = 0
    if method.layernorm_in_tee:
        ln_total = tokens * (2 * layers + 1) * ln_ops_per_pos

    # GELU in trusted side.
    gelu_total = 0
    if method.activation_in_tee:
        gelu_total = tokens * layers * 8 * inter

    # LM head recovery (vocab mask) — element-wise multiply on the vocab dim.
    lm_head_recovery = tokens * vocab if method.lm_head_recovered_in_tee else 0

    # For ours_current the trusted side also performs attention scores,
    # softmax, value aggregation, mask-state creation, and (optionally) pad
    # compensation. These are real trusted-side compute, but they are *not*
    # boundary crossings.
    if method.name == "ours_current":
        # Per layer attention: scores (q_len*k_len*head_dim) + AV (q_len*k_len*head_dim)
        #   = 2 * tokens(=q_len in our forward defn) * k_len * hidden FLOPs
        # We already have q_len == tokens for the prefill case; for decode q_len == 1.
        attn_compute = 2 * layers * heads * q_len * k_len * head_dim * 2  # *2 for scores+AV
        # Mask state creation per linear ~ d_in*d_out (cheap matmul to invert),
        # bounded above by hidden^2 + inter*hidden + ...
        mask_state_ops = layers * (3 * hidden * hidden + 3 * inter * hidden)
        # Pad compensation ops per linear (T W N_out) ~ tokens*in*out per linear.
        pad_compensation_ops = 0
        if method.uses_pad:
            pad_compensation_ops = tokens * layers * (
                hidden * (3 * hidden)  # c_attn
                + hidden * hidden      # c_proj
                + hidden * inter       # c_fc
                + inter * hidden       # c_proj_mlp
            )
        return ln_total + gelu_total + attn_compute + mask_state_ops + pad_compensation_ops + lm_head_recovery
    # ours_ideal / amulet: only LM head recovery in trusted.
    return ln_total + gelu_total + lm_head_recovery


def _per_forward_gpu_ops(
    method: WorkloadMethod,
    consts: dict[str, int],
    tokens: int,
    q_len: int,
    k_len: int,
) -> int:
    """Return GPU-side FLOPs performed during one forward call."""
    hidden = consts["hidden"]
    inter = consts["inter"]
    layers = consts["layers"]
    heads = consts["heads"]
    head_dim = consts["head_dim"]
    vocab = consts["vocab"]
    # Linear matmuls per layer per position.
    linear_per_pos = (
        2 * hidden * 3 * hidden  # c_attn
        + 2 * hidden * hidden    # c_proj
        + 2 * hidden * inter     # c_fc
        + 2 * inter * hidden     # c_proj_mlp
    )
    # Attention scores + AV (computed on GPU for ours_ideal / amulet / plain,
    # and on trusted side for ours_current — but we still count attention
    # FLOPs as GPU ops for ours_current too because q_tilde @ k_tilde^T is
    # algebraically GPU-side matmul work even if our wrapper currently runs
    # it inside the same Python process).
    attn_per_layer = 2 * 2 * heads * q_len * k_len * head_dim  # scores + AV
    # LM head matmul.
    lm_head = 2 * hidden * vocab * tokens
    # LN + GELU GPU ops only count for methods that run them on GPU.
    extra_nonlinear = 0
    if not method.layernorm_in_tee:
        extra_nonlinear += tokens * (2 * layers + 1) * 8 * hidden
    if not method.activation_in_tee:
        extra_nonlinear += tokens * layers * 8 * inter
    return tokens * layers * linear_per_pos + layers * attn_per_layer + lm_head + extra_nonlinear


def _preprocessing(method: WorkloadMethod, consts: dict[str, int]) -> dict[str, int]:
    """Per-session preprocessing trusted cost (amortised)."""
    hidden = consts["hidden"]
    inter = consts["inter"]
    layers = consts["layers"]
    vocab = consts["vocab"]
    db = consts["dtype_bytes"]
    if not method.linear_obfuscated:
        return {"trusted_ops": 0, "transfer_bytes": 0}
    # W_tilde = N_in_inv @ W @ N_out for each linear. Cost ~ 2 * d_in^2 * d_out + 2 * d_in * d_out^2.
    per_layer_ops = (
        2 * hidden * hidden * (3 * hidden) + 2 * hidden * (3 * hidden) * (3 * hidden)  # c_attn
        + 2 * hidden * hidden * hidden + 2 * hidden * hidden * hidden                  # c_proj
        + 2 * hidden * hidden * inter + 2 * hidden * inter * inter                     # c_fc
        + 2 * inter * inter * hidden + 2 * inter * hidden * hidden                     # c_proj_mlp
    )
    # LM head vocab mask transform is diagonal-elementwise — O(hidden * vocab).
    per_layer_ops += 2 * hidden * vocab
    weight_bytes_per_layer = (
        hidden * 3 * hidden
        + hidden * hidden
        + hidden * inter
        + inter * hidden
    ) * db
    total_transfer_bytes = layers * weight_bytes_per_layer + hidden * vocab * db
    return {
        "trusted_ops": layers * per_layer_ops,
        "transfer_bytes": total_transfer_bytes,
    }


# ---------------------------------------------------------------------------
# Module breakdown (per-module slice of the same totals)
# ---------------------------------------------------------------------------


def _module_breakdown(
    method: WorkloadMethod,
    consts: dict[str, int],
    forwards: list[tuple[int, int, int]],
    batch_size: int,
) -> dict[str, ModuleCounts]:
    """Attribute the per-method workload to specific module categories."""
    hidden = consts["hidden"]
    inter = consts["inter"]
    layers = consts["layers"]
    vocab = consts["vocab"]
    heads = consts["heads"]
    head_dim = consts["head_dim"]
    db = consts["dtype_bytes"]
    total_tokens = sum(t for t, _q, _k in forwards)
    lm_head_tokens = total_tokens  # we apply LM head to every forward's positions

    modules = {name: ModuleCounts() for name in MODULE_CATEGORIES}

    # Embedding: trivial GPU lookup.
    modules["embedding"].online_gpu_ops = total_tokens * hidden * batch_size
    modules["embedding"].location = "gpu"

    # Linear matmuls — all methods do these on GPU.
    modules["attention_qkv"].online_gpu_ops = (
        layers * total_tokens * 2 * hidden * 3 * hidden * batch_size
    )
    modules["attention_output"].online_gpu_ops = (
        layers * total_tokens * 2 * hidden * hidden * batch_size
    )
    modules["mlp_fc"].online_gpu_ops = (
        layers * total_tokens * 2 * hidden * inter * batch_size
    )
    modules["mlp_proj"].online_gpu_ops = (
        layers * total_tokens * 2 * inter * hidden * batch_size
    )

    # Attention score + AV FLOPs aggregated over forwards.
    attn_flops = 0
    for tokens, q_len, k_len in forwards:
        attn_flops += 2 * 2 * heads * q_len * k_len * head_dim * batch_size
    modules["attention_score"].online_gpu_ops = layers * attn_flops

    # KV cache update (cheap memcpy on GPU).
    modules["kv_cache_update"].online_gpu_ops = (
        layers * (sum(t for t, _q, _k in forwards)) * 2 * hidden * batch_size
    )

    # LayerNorm: where it runs depends on method.
    ln_total = total_tokens * (2 * layers + 1) * 8 * hidden * batch_size
    if method.layernorm_in_tee:
        modules["layernorm"].online_trusted_compute_ops = ln_total
        modules["layernorm"].online_boundary_calls = (
            len(forwards) * (2 * layers + 1)
            if method.name == "tslp_trusted_nonlinear_baseline"
            else 0
        )
        modules["layernorm"].online_trusted_transfer_bytes = (
            sum(t for t, _q, _k in forwards) * (2 * layers + 1) * 2 * hidden * db * batch_size
            if method.name == "tslp_trusted_nonlinear_baseline"
            else 0
        )
        modules["layernorm"].location = "tee"
    else:
        modules["layernorm"].online_gpu_ops = ln_total
        modules["layernorm"].location = "gpu"

    # GELU: same pattern.
    gelu_total = total_tokens * layers * 8 * inter * batch_size
    if method.activation_in_tee:
        modules["activation"].online_trusted_compute_ops = gelu_total
        modules["activation"].online_boundary_calls = (
            len(forwards) * layers
            if method.name == "tslp_trusted_nonlinear_baseline"
            else 0
        )
        modules["activation"].online_trusted_transfer_bytes = (
            sum(t for t, _q, _k in forwards) * layers * 2 * inter * db * batch_size
            if method.name == "tslp_trusted_nonlinear_baseline"
            else 0
        )
        modules["activation"].location = "tee"
    else:
        modules["activation"].online_gpu_ops = gelu_total
        modules["activation"].location = "gpu"

    # LM head: matmul on GPU, optional recovery in TEE.
    modules["lm_head"].online_gpu_ops = 2 * hidden * vocab * lm_head_tokens * batch_size
    if method.lm_head_recovered_in_tee:
        modules["lm_head"].online_trusted_compute_ops = lm_head_tokens * vocab * batch_size
        modules["lm_head"].online_boundary_calls = len(forwards)
        modules["lm_head"].online_trusted_transfer_bytes = (
            lm_head_tokens * vocab * db * batch_size
        )
        modules["lm_head"].location = "mixed"
    else:
        modules["lm_head"].location = "gpu"

    # Attribute the remaining "obfuscated-linear round trips" to the four
    # linear modules. These are the round trips that ours_current does for
    # every Conv1D matmul (NOT mask-state bookkeeping).
    if method.linear_obfuscated and not method.fuses_gpu_pipeline:
        per_forward_linear_calls = 4 * layers
        # Distribute equally across the four linear categories.
        per_module_calls = len(forwards) * layers
        modules["attention_qkv"].online_boundary_calls += per_module_calls
        modules["attention_output"].online_boundary_calls += per_module_calls
        modules["mlp_fc"].online_boundary_calls += per_module_calls
        modules["mlp_proj"].online_boundary_calls += per_module_calls
        # Attribute transfer bytes accordingly.
        modules["attention_qkv"].online_trusted_transfer_bytes += (
            sum(t for t, _q, _k in forwards) * layers * 4 * hidden * db * batch_size
        )
        modules["attention_output"].online_trusted_transfer_bytes += (
            sum(t for t, _q, _k in forwards) * layers * 2 * hidden * db * batch_size
        )
        modules["mlp_fc"].online_trusted_transfer_bytes += (
            sum(t for t, _q, _k in forwards) * layers * (hidden + inter) * db * batch_size
        )
        modules["mlp_proj"].online_trusted_transfer_bytes += (
            sum(t for t, _q, _k in forwards) * layers * (inter + hidden) * db * batch_size
        )

    return modules


# ---------------------------------------------------------------------------
# Interaction breakdown (per-interaction slice of the total)
# ---------------------------------------------------------------------------


def _interaction_breakdown(
    method: WorkloadMethod,
    consts: dict[str, int],
    forwards: list[tuple[int, int, int]],
    batch_size: int,
) -> dict[str, InteractionCounts]:
    """Slice the per-method workload by interaction type (not by module)."""
    hidden = consts["hidden"]
    inter = consts["inter"]
    layers = consts["layers"]
    vocab = consts["vocab"]
    db = consts["dtype_bytes"]
    total_tokens = sum(t for t, _q, _k in forwards)
    lm_head_tokens = total_tokens

    out: dict[str, InteractionCounts] = {
        name: InteractionCounts() for name in INTERACTION_CATEGORIES
    }

    # --- input_masking ---
    if method.name in ("ours_ideal_gpu_nonlinear", "amulet_style_reference"):
        out["input_masking"].online_boundary_calls = len(forwards)
        out["input_masking"].online_trusted_transfer_bytes = (
            total_tokens * hidden * db * batch_size
        )
        out["input_masking"].notes = "Trusted prepares masked hidden state, single OCALL per forward."
    elif method.name == "ours_current":
        # Folded into the per-linear round trips below — no separate input
        # masking call.
        out["input_masking"].notes = "Bundled into per-linear round trips below."
    else:
        out["input_masking"].notes = "Not applicable."

    # --- trusted_layernorm ---
    if method.layernorm_in_tee:
        ln_ops = total_tokens * (2 * layers + 1) * 8 * hidden * batch_size
        out["trusted_layernorm"].online_trusted_compute_ops = ln_ops
        if method.name == "tslp_trusted_nonlinear_baseline":
            out["trusted_layernorm"].online_boundary_calls = len(forwards) * (
                2 * layers + 1
            )
            out["trusted_layernorm"].online_trusted_transfer_bytes = (
                total_tokens * (2 * layers + 1) * 2 * hidden * db * batch_size
            )
            out["trusted_layernorm"].notes = (
                "Each LN call is an explicit ECALL into TEE."
            )
        else:
            out["trusted_layernorm"].notes = (
                "Trusted shortcut: LN computed in trusted side without a"
                " separate boundary crossing (trusted already holds the"
                " plaintext)."
            )

    # --- trusted_gelu ---
    if method.activation_in_tee:
        gelu_ops = total_tokens * layers * 8 * inter * batch_size
        out["trusted_gelu"].online_trusted_compute_ops = gelu_ops
        if method.name == "tslp_trusted_nonlinear_baseline":
            out["trusted_gelu"].online_boundary_calls = len(forwards) * layers
            out["trusted_gelu"].online_trusted_transfer_bytes = (
                total_tokens * layers * 2 * inter * db * batch_size
            )
            out["trusted_gelu"].notes = "Each GELU call is an explicit ECALL into TEE."
        else:
            out["trusted_gelu"].notes = "Trusted shortcut, no separate boundary crossing."

    # --- lm_head_recovery ---
    if method.lm_head_recovered_in_tee:
        out["lm_head_recovery"].online_boundary_calls = len(forwards)
        out["lm_head_recovery"].online_trusted_compute_ops = (
            lm_head_tokens * vocab * batch_size
        )
        out["lm_head_recovery"].online_trusted_transfer_bytes = (
            lm_head_tokens * vocab * db * batch_size
        )
        out["lm_head_recovery"].notes = (
            "Logits transferred to trusted side and multiplied by inverse"
            " vocab mask."
        )

    # --- sampling (greedy argmax) ---
    out["sampling"].online_trusted_compute_ops = (
        lm_head_tokens * vocab * batch_size if method.name != "plain_hf_gpu" else 0
    )
    out["sampling"].notes = "Greedy argmax in trusted side; no extra boundary call."

    # --- preprocessing_weight_obfuscation ---
    if method.linear_obfuscated:
        pre = _preprocessing(method, consts)
        out["preprocessing_weight_obfuscation"].online_trusted_compute_ops = 0
        out["preprocessing_weight_obfuscation"].notes = (
            "Amortised over many sessions; not counted in online latency."
        )

    return out


# ---------------------------------------------------------------------------
# Wall-time measurement / projection
# ---------------------------------------------------------------------------


def _measure_wall_time(
    callable_: Callable[[], Any], warmup: int, repeat: int
) -> dict[str, float]:
    for _ in range(warmup):
        callable_()
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        callable_()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "measured_wall_time_ms": float(statistics.mean(times)),
        "measured_wall_time_ms_median": float(statistics.median(times)),
        "measured_wall_time_ms_stdev": float(statistics.stdev(times)) if len(times) >= 2 else 0.0,
    }


def _project_wall_time_ms(
    online_aggregate: dict[str, int],
    gpu_flops_per_ms: float,
    cost_model: CostModel,
) -> float:
    """Project online wall time from op counts."""
    gpu_ms = online_aggregate["online_gpu_ops"] / max(gpu_flops_per_ms, 1e-12)
    tee_ms = online_aggregate["online_trusted_compute_ops"] / max(
        gpu_flops_per_ms / cost_model.tee_to_gpu_flops_ratio, 1e-12
    )
    bytes_ms = online_aggregate["online_trusted_transfer_bytes"] / max(
        cost_model.tee_bytes_per_ms, 1e-12
    )
    call_ms = online_aggregate["online_boundary_calls"] * cost_model.tee_call_overhead_ms
    return gpu_ms + tee_ms + bytes_ms + call_ms


# ---------------------------------------------------------------------------
# Per-method aggregation
# ---------------------------------------------------------------------------


def _aggregate_method_online(
    method: WorkloadMethod,
    consts: dict[str, int],
    forwards: list[tuple[int, int, int]],
    batch_size: int,
) -> dict[str, int]:
    online_boundary = 0
    online_transfer_bytes = 0
    online_trusted_compute = 0
    online_gpu = 0
    for tokens, q_len, k_len in forwards:
        online_boundary += _per_forward_boundary_calls(method, consts["layers"])
        online_transfer_bytes += _per_forward_trusted_transfer_bytes(method, consts, tokens)
        online_trusted_compute += _per_forward_trusted_compute_ops(
            method, consts, tokens, q_len, k_len
        )
        online_gpu += _per_forward_gpu_ops(method, consts, tokens, q_len, k_len)
    return {
        "online_boundary_calls": online_boundary,
        "online_trusted_compute_ops": online_trusted_compute * batch_size,
        "online_trusted_transfer_bytes": online_transfer_bytes * batch_size,
        "online_gpu_ops": online_gpu * batch_size,
    }


# ---------------------------------------------------------------------------
# Plain HF greedy timing harness
# ---------------------------------------------------------------------------


def _plain_hf_greedy(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    with torch.no_grad():
        prefill = model(input_ids, use_cache=True)
        next_token = prefill.logits[:, -1, :].argmax(dim=-1)
        past = prefill.past_key_values
        new_tokens = [next_token]
        for _ in range(max_new_tokens - 1):
            step = model(next_token.unsqueeze(-1), past_key_values=past, use_cache=True)
            past = step.past_key_values
            next_token = step.logits[:, -1, :].argmax(dim=-1)
            new_tokens.append(next_token)
    return torch.cat([input_ids, torch.stack(new_tokens, dim=1)], dim=1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_workload_profile(config: WorkloadProfileConfig) -> dict[str, Any]:
    """Run the calibrated workload profile and return the structured report dict."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)

    loader_cfg = ExternalModelConfig(
        source="huggingface",
        model_id=config.model_id,
        device=config.device,
        dtype=config.dtype,
    )
    _, model = get_model_loader("hf").load(loader_cfg)
    model.eval()
    model_cfg = model.config

    vocab_size = model_cfg.vocab_size
    prompt_ids = torch.randint(
        0, vocab_size, (config.batch_size, config.prompt_len), device=device
    )

    consts = _model_constants(model_cfg, config)
    forwards = _forward_specs(config)

    # ---- Online aggregates + preprocessing per method ----
    online_aggregates: dict[str, dict[str, int]] = {}
    preprocessing_aggregates: dict[str, dict[str, int]] = {}
    module_breakdowns: dict[str, dict[str, ModuleCounts]] = {}
    interaction_breakdowns: dict[str, dict[str, InteractionCounts]] = {}
    for method in WORKLOAD_METHODS:
        online_aggregates[method.name] = _aggregate_method_online(
            method, consts, forwards, config.batch_size
        )
        preprocessing_aggregates[method.name] = _preprocessing(method, consts)
        module_breakdowns[method.name] = _module_breakdown(
            method, consts, forwards, config.batch_size
        )
        interaction_breakdowns[method.name] = _interaction_breakdown(
            method, consts, forwards, config.batch_size
        )

    # ---- Wall-time measurement for implemented methods ----
    plain_timing = _measure_wall_time(
        lambda: _plain_hf_greedy(model, prompt_ids, config.max_new_tokens),
        config.warmup,
        config.repeat,
    )
    wrapper = ObfuscatedGPT2ModelWrapper(
        model=model, dtype=dtype, device=device, use_pad=config.use_pad
    )
    ours_timing = _measure_wall_time(
        lambda: wrapper.generate_greedy(prompt_ids, config.max_new_tokens),
        config.warmup,
        config.repeat,
    )

    plain_gpu_flops = online_aggregates["plain_hf_gpu"]["online_gpu_ops"]
    gpu_flops_per_ms = max(
        plain_gpu_flops / max(plain_timing["measured_wall_time_ms"], 1e-9), 1.0
    )

    # ---- Per-method records ----
    method_records: dict[str, dict[str, Any]] = {}
    for method in WORKLOAD_METHODS:
        agg = online_aggregates[method.name]
        pre = preprocessing_aggregates[method.name]
        record: dict[str, Any] = {
            "title": method.title,
            "summary": method.summary,
            "implemented": method.implemented,
            "implementation_note": method.implementation_note,
            "citation_caveat": method.citation_caveat,
            "online_boundary_calls": agg["online_boundary_calls"],
            "online_trusted_compute_ops": agg["online_trusted_compute_ops"],
            "online_trusted_transfer_bytes": agg["online_trusted_transfer_bytes"],
            "online_gpu_ops": agg["online_gpu_ops"],
            "preprocessing_trusted_ops": pre["trusted_ops"],
            "preprocessing_transfer_bytes": pre["transfer_bytes"],
            "boundary_calls_formula": _boundary_calls_formula(method, consts["layers"]),
        }
        projected = _project_wall_time_ms(agg, gpu_flops_per_ms, config.cost_model)
        record["projected_wall_time_ms"] = float(projected)
        if method.name == "plain_hf_gpu":
            record["measured_wall_time_ms"] = plain_timing["measured_wall_time_ms"]
            record["measured_wall_time_ms_median"] = plain_timing["measured_wall_time_ms_median"]
            record["measured_wall_time_ms_stdev"] = plain_timing["measured_wall_time_ms_stdev"]
            record["wall_time_source"] = "measured"
        elif method.name == "ours_current":
            record["measured_wall_time_ms"] = ours_timing["measured_wall_time_ms"]
            record["measured_wall_time_ms_median"] = ours_timing["measured_wall_time_ms_median"]
            record["measured_wall_time_ms_stdev"] = ours_timing["measured_wall_time_ms_stdev"]
            record["wall_time_source"] = "measured"
        else:
            record["measured_wall_time_ms"] = None
            record["measured_wall_time_ms_median"] = None
            record["measured_wall_time_ms_stdev"] = None
            record["wall_time_source"] = "projected_from_op_counts"
        method_records[method.name] = record

    # ---- Module / interaction breakdowns as plain JSON ----
    module_json = {
        category: {
            method.name: asdict(module_breakdowns[method.name][category])
            for method in WORKLOAD_METHODS
        }
        for category in MODULE_CATEGORIES
    }
    interaction_json = {
        category: {
            method.name: asdict(interaction_breakdowns[method.name][category])
            for method in WORKLOAD_METHODS
        }
        for category in INTERACTION_CATEGORIES
    }

    # ---- Paper metrics ----
    tslp_agg = online_aggregates["tslp_trusted_nonlinear_baseline"]
    ours_agg = online_aggregates["ours_current"]

    def _reduction(tslp_value: int, ours_value: int) -> float:
        if tslp_value <= 0:
            return 0.0
        return (tslp_value - ours_value) / tslp_value

    gpu_offload_ratio = ours_agg["online_gpu_ops"] / max(
        ours_agg["online_gpu_ops"] + ours_agg["online_trusted_compute_ops"], 1
    )

    paper_metrics = {
        "boundary_call_reduction_vs_tslp": _reduction(
            tslp_agg["online_boundary_calls"], ours_agg["online_boundary_calls"]
        ),
        "trusted_transfer_reduction_vs_tslp": _reduction(
            tslp_agg["online_trusted_transfer_bytes"], ours_agg["online_trusted_transfer_bytes"]
        ),
        "online_trusted_compute_reduction_vs_tslp": _reduction(
            tslp_agg["online_trusted_compute_ops"], ours_agg["online_trusted_compute_ops"]
        ),
        "gpu_offload_ratio": float(gpu_offload_ratio),
        "preprocessing_amortized": True,
        "boundary_calls_per_forward": {
            method.name: _per_forward_boundary_calls(method, consts["layers"])
            for method in WORKLOAD_METHODS
        },
    }

    # ---- Interpretation ----
    main_online_bottleneck = max(
        MODULE_CATEGORIES,
        key=lambda cat: (
            module_breakdowns["ours_current"][cat].online_trusted_compute_ops
            + module_breakdowns["ours_current"][cat].online_trusted_transfer_bytes
        ),
    )
    # Next primitive: focus only on LN vs GELU among the trusted shortcuts.
    ln_tee = module_breakdowns["ours_current"]["layernorm"].online_trusted_compute_ops
    gelu_tee = module_breakdowns["ours_current"]["activation"].online_trusted_compute_ops
    next_primitive = "LayerNorm" if ln_tee >= gelu_tee else "GELU"

    return {
        "config": {
            **asdict(config),
            "cost_model": asdict(config.cost_model),
        },
        "calibration": {
            "gpu_flops_per_ms": gpu_flops_per_ms,
            "calibrated_from": "plain_hf_gpu",
        },
        "methods": method_records,
        "module_breakdown": module_json,
        "interaction_breakdown": interaction_json,
        "paper_metrics": paper_metrics,
        "interpretation": {
            "main_online_bottleneck": main_online_bottleneck,
            "next_primitive_to_replace": next_primitive,
            "cost_model_warning": "simulated cost model, not real SGX",
        },
        "limitations": [
            "Simulated TEE cost model — not real SGX wall-clock.",
            "Wall time for tslp_baseline, ours_ideal, and amulet_style_reference is projected, not measured.",
            "tiny-gpt2 (n_layer=2, n_embd=2, n_head=2) is far smaller than production GPT-2.",
            "FLOP / byte proxies use coarse constants; absolute numbers are illustrative.",
            "Qwen / Llama not yet covered — see Stage 5.4 roadmap.",
            "amulet_style_reference is a reference cost model, not a re-implementation of any published system.",
        ],
    }


def _boundary_calls_formula(method: WorkloadMethod, layers: int) -> str:
    """Return the per-forward boundary-call formula as a human-readable string."""
    if method.name == "plain_hf_gpu":
        return "0 (no boundary)"
    if method.name == "tslp_trusted_nonlinear_baseline":
        return f"3L + 2 = {3 * layers + 2} per forward (LN_1 + LN_2 + GELU per layer + ln_f + LM head)"
    if method.name == "ours_current":
        return f"4L + 1 = {4 * layers + 1} per forward (4 obfuscated linears per layer + LM head)"
    if method.name == "ours_ideal_gpu_nonlinear":
        return "1 per forward (single fused GPU pipeline round trip)"
    if method.name == "amulet_style_reference":
        return "1 per forward (single fused GPU pipeline round trip)"
    return "n/a"
