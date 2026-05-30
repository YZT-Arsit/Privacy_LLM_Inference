"""Registry of paper-grade experiment sweeps and method definitions (Stage 5.0).

Centralized constants for ``run_attention_experiments.py`` and
``run_workload_profile.py``. Keeping these here makes the sweep / method set
explicit and citable in the paper.

Stage 5.0.1: the cost-model registry now splits workload into four explicit
categories (preprocessing trusted cost, online boundary crossings, online
trusted compute, online GPU compute). See ``workload_profiler.py`` for the
formulas. ``has_implementation`` was renamed to ``implemented`` for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Attention sweep
# ---------------------------------------------------------------------------

ATTENTION_SWEEP: dict[str, tuple] = {
    "batch_size": (1, 2),
    "seq_len": (4, 8, 16),
    "decode_steps": (1, 2, 4),
    "use_pad": (True, False),
}


# ---------------------------------------------------------------------------
# Workload profile method definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadMethod:
    name: str
    title: str
    summary: str
    layernorm_in_tee: bool
    activation_in_tee: bool
    linear_obfuscated: bool
    uses_pad: bool
    lm_head_recovered_in_tee: bool
    fuses_gpu_pipeline: bool  # True ⇒ trusted only crosses for input prep + output recovery
    implemented: bool
    implementation_note: str
    citation_caveat: str
    # ---- Stage 5.2c extensions (optional fields, default-False for existing methods) ----
    uses_compatible_nonlinear_islands: bool = False
    uses_dense_sandwich: bool = False
    uses_fresh_permutation: bool = False
    online_extra_matmul_count: int = 0
    security_profile: str = "n/a"


WORKLOAD_METHODS: tuple[WorkloadMethod, ...] = (
    WorkloadMethod(
        name="plain_hf_gpu",
        title="Plain HuggingFace on GPU",
        summary="Plaintext HF GPT-2 forward / greedy decode. No protection.",
        layernorm_in_tee=False,
        activation_in_tee=False,
        linear_obfuscated=False,
        uses_pad=False,
        lm_head_recovered_in_tee=False,
        fuses_gpu_pipeline=True,  # entirely on GPU, no boundary crossings at all
        implemented=True,
        implementation_note="Hand-written HF greedy loop over plain model().",
        citation_caveat="No security; measured wall time is the GPU-only baseline.",
    ),
    WorkloadMethod(
        name="tslp_trusted_nonlinear_baseline",
        title="TSLP-style trusted non-linear baseline",
        summary=(
            "Linear / attention on GPU, every LayerNorm and GELU activation"
            " makes a TEE round-trip. Modeled after the trusted non-linear"
            " split common in shielded-inference literature."
        ),
        layernorm_in_tee=True,
        activation_in_tee=True,
        linear_obfuscated=False,
        uses_pad=False,
        lm_head_recovered_in_tee=True,
        fuses_gpu_pipeline=False,
        implemented=False,
        implementation_note=(
            "No real implementation in this repo. Wall time is projected from"
            " op counts using a documented cost model — not a measurement of"
            " any specific published system."
        ),
        citation_caveat=(
            "TSLP-style is used here as a generic non-linear-in-TEE baseline."
            " It is not a faithful re-implementation of any single published"
            " system. Adjust the cost model constants in WorkloadProfileConfig"
            " before drawing system-level conclusions."
        ),
    ),
    WorkloadMethod(
        name="ours_current",
        title="This work — current Stage 4.9 implementation",
        summary=(
            "Right-multiply mask + per-block Conv1D pad compensation,"
            " trusted LayerNorm / GELU shortcuts, diagonal vocab output mask"
            " on the LM head, internal ObfuscatedGPT2KVCache."
        ),
        layernorm_in_tee=True,
        activation_in_tee=True,
        linear_obfuscated=True,
        uses_pad=True,
        lm_head_recovered_in_tee=True,
        fuses_gpu_pipeline=False,  # trusted must call GPU per obfuscated linear
        implemented=True,
        implementation_note=(
            "Wall time is measured against the real ObfuscatedGPT2ModelWrapper"
            " generation path."
        ),
        citation_caveat=(
            "Trusted LayerNorm / GELU are engineering shortcuts; their TEE"
            " cost is included in proxies but their security model is"
            " unprotected non-linearity. See Stage 5.1 / 5.2 roadmap."
        ),
    ),
    WorkloadMethod(
        name="ours_ideal_gpu_nonlinear",
        title="This work — ideal: LN / GELU on GPU in masked domain",
        summary=(
            "Same wrapper but with LayerNorm and GELU executed inside the"
            " obfuscated GPU domain. The trusted side only crosses the"
            " boundary to prepare the masked input and to recover the LM head"
            " logits. Used as an upper bound, not a measured system."
        ),
        layernorm_in_tee=False,
        activation_in_tee=False,
        linear_obfuscated=True,
        uses_pad=True,
        lm_head_recovered_in_tee=True,
        fuses_gpu_pipeline=True,
        implemented=False,
        implementation_note=(
            "Hypothetical. Op counts come from the same model graph; LN / GELU"
            " FLOPs are reattributed from TEE to GPU. Wall time is projected,"
            " not measured."
        ),
        citation_caveat=(
            "Upper bound. Real obfuscated LN / GELU primitives are deferred to"
            " Stage 5.1 / 5.2 and may carry additional overhead this estimate"
            " does not capture."
        ),
    ),
    WorkloadMethod(
        name="ours_compatible_nonlinear_islands",
        title="This work — projected: operator-compatible nonlinear islands",
        summary=(
            "Modeled / projected method. RMSNorm core uses an orthogonal mask,"
            " LayerNorm core uses a mean-preserving orthogonal mask, GELU /"
            " ReLU / SiLU activations use permutation masks, and SwiGLU uses a"
            " paired permutation. Every mask transition is folded into adjacent"
            " Linear weights offline, so the masked forward executes with the"
            " same number of matmuls as the plaintext forward (Stage 5.2a"
            " verified ``online_extra_matmul_count = 0`` for every MLP island"
            " cell). Trusted shortcuts for LN and GELU are removed."
        ),
        layernorm_in_tee=False,
        activation_in_tee=False,
        linear_obfuscated=True,
        uses_pad=True,
        lm_head_recovered_in_tee=True,
        # Not a single fused pipeline — there is still per-layer dense-mask"
        # transition bookkeeping at the Linear boundaries between islands.
        fuses_gpu_pipeline=False,
        implemented=False,
        implementation_note=(
            "Projected, not measured. Stage 5.2a verified the correctness probe"
            " (28 cells, all_allclose=True, max_online_extra_matmul=0). Stage"
            " 5.2b validated the security proxy (fresh permutation + dense"
            " sandwich + pad at Linear boundaries are required mitigations)."
            " Not yet integrated into the GPT-2 / BERT / T5 wrappers — Stage"
            " 5.3 is the integration step."
        ),
        citation_caveat=(
            "Compatible mask families are weaker than unrestricted dense"
            " masks inside nonlinear islands. The Stage 5.2b security proxy"
            " quantified per-strategy linkability and permutation recovery,"
            " but the result is a naive-observer upper bound, NOT a formal"
            " security proof and NOT a real TEE measurement."
        ),
        uses_compatible_nonlinear_islands=True,
        uses_dense_sandwich=True,
        uses_fresh_permutation=True,
        online_extra_matmul_count=0,
        security_profile="proxy-evaluated, not formal",
    ),
    WorkloadMethod(
        name="amulet_style_reference",
        title="Amulet-style reference (cost model)",
        summary=(
            "Input masking + GPU obfuscated forward + output unmasking,"
            " modeled after the high-level pattern in Amulet-style systems."
            " Reference only, not a re-implementation."
        ),
        layernorm_in_tee=False,
        activation_in_tee=False,
        linear_obfuscated=True,
        uses_pad=True,
        lm_head_recovered_in_tee=True,
        fuses_gpu_pipeline=True,
        implemented=False,
        implementation_note=(
            "No implementation in this repo. Cost-model reference under the"
            " assumption that the entire obfuscated forward runs as a single"
            " GPU pipeline between trusted input masking and trusted output"
            " unmasking."
        ),
        citation_caveat=(
            "Amulet-style here means the abstract pattern of input mask + GPU"
            " forward + output recovery. Real Amulet systems may include"
            " primitives and overheads not captured by this proxy. Use with"
            " explicit attribution to assumptions."
        ),
    ),
)


METHOD_BY_NAME: dict[str, WorkloadMethod] = {m.name: m for m in WORKLOAD_METHODS}


# ---------------------------------------------------------------------------
# Cost-model constants (tunable, documented)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Tunable knobs for the workload cost model.

    The wall-time projection is intentionally simple:
    ``ms = gpu_ops / gpu_flops_per_ms + tee_ops / tee_flops_per_ms
          + tee_bytes / tee_bytes_per_ms + tee_calls * tee_call_overhead_ms``.

    ``gpu_flops_per_ms`` is calibrated at run time from the measured
    ``plain_hf_gpu`` wall-clock, so the relative ordering of methods is
    self-consistent within a single profile run.
    """

    tee_to_gpu_flops_ratio: float = 50.0  # TEE is ~50x slower per FLOP (conservative SGX estimate)
    tee_bytes_per_ms: float = 1024 * 1024 * 1024 / 1000  # 1 GB/s = 1 MB/ms
    tee_call_overhead_ms: float = 0.005  # 5 us per TEE round trip
    note: str = (
        "Cost-model constants are coarse defaults. They are not derived from"
        " hardware measurements of any specific TEE. See limitations section."
    )


DEFAULT_COST_MODEL = CostModel()


# ---------------------------------------------------------------------------
# Module-breakdown categories (used by both methods + profile output)
# ---------------------------------------------------------------------------


ModuleCategory = Literal[
    "embedding",
    "layernorm",
    "attention_qkv",
    "attention_score",
    "attention_output",
    "mlp_fc",
    "activation",
    "mlp_proj",
    "lm_head",
    "kv_cache_update",
]


MODULE_CATEGORIES: tuple[str, ...] = (
    "embedding",
    "layernorm",
    "attention_qkv",
    "attention_score",
    "attention_output",
    "mlp_fc",
    "activation",
    "mlp_proj",
    "lm_head",
    "kv_cache_update",
)


# ---------------------------------------------------------------------------
# Interaction breakdown (Stage 5.0.1)
# ---------------------------------------------------------------------------


INTERACTION_CATEGORIES: tuple[str, ...] = (
    "input_masking",
    "trusted_layernorm",
    "trusted_gelu",
    "lm_head_recovery",
    "sampling",
    "preprocessing_weight_obfuscation",
    # ---- Stage 5.2c additions for ``ours_compatible_nonlinear_islands`` ----
    "preprocessing_affine_folding",
    "preprocessing_permutation_absorption",
    "compatible_norm_core_gpu",
    "compatible_activation_island_gpu",
    "dense_sandwich_transition",
    "security_proxy_requirements",
)
