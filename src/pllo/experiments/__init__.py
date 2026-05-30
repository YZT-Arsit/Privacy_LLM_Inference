"""Paper-grade experiment harnesses (Stage 5.0)."""

from pllo.experiments.attention_probe import AttentionProbeConfig, run_attention_probe
from pllo.experiments.cross_architecture_summary import (
    ARCHITECTURE_SPECS,
    CrossArchitectureSummaryConfig,
    run_cross_architecture_summary,
)
from pllo.experiments.cross_attention_probe import (
    CrossAttentionProbeConfig,
    EncoderMemoryCache,
    run_cross_attention_probe,
)
from pllo.experiments.encoder_attention_probe import (
    EncoderAttentionProbeConfig,
    run_encoder_attention_probe,
)
from pllo.experiments.nonlinear_island_probe import (
    NonlinearIslandProbeConfig,
    run_nonlinear_island_experiments,
)
from pllo.experiments.norm_probe import (
    RMSNormOrthogonalProbeConfig,
    TrustedNormProbeConfig,
    run_rmsnorm_orthogonal_probe,
    run_trusted_norm_probe,
)
from pllo.experiments.experiment_registry import (
    ATTENTION_SWEEP,
    DEFAULT_COST_MODEL,
    INTERACTION_CATEGORIES,
    METHOD_BY_NAME,
    MODULE_CATEGORIES,
    WORKLOAD_METHODS,
    CostModel,
    WorkloadMethod,
)
from pllo.experiments.security_proxy import (
    GPU_VISIBLE_TENSORS,
    MASK_AUDIT_SPECS,
    SecurityProxyConfig,
    TRUSTED_ONLY_TENSORS,
    run_security_proxy_experiments,
)
from pllo.experiments.workload_profiler import (
    InteractionCounts,
    ModuleCounts,
    WorkloadProfileConfig,
    run_workload_profile,
)

__all__ = [
    "AttentionProbeConfig",
    "run_attention_probe",
    "EncoderAttentionProbeConfig",
    "run_encoder_attention_probe",
    "CrossAttentionProbeConfig",
    "EncoderMemoryCache",
    "run_cross_attention_probe",
    "ARCHITECTURE_SPECS",
    "CrossArchitectureSummaryConfig",
    "run_cross_architecture_summary",
    "SecurityProxyConfig",
    "run_security_proxy_experiments",
    "GPU_VISIBLE_TENSORS",
    "TRUSTED_ONLY_TENSORS",
    "MASK_AUDIT_SPECS",
    "TrustedNormProbeConfig",
    "run_trusted_norm_probe",
    "RMSNormOrthogonalProbeConfig",
    "run_rmsnorm_orthogonal_probe",
    "NonlinearIslandProbeConfig",
    "run_nonlinear_island_experiments",
    "WorkloadProfileConfig",
    "run_workload_profile",
    "ATTENTION_SWEEP",
    "WORKLOAD_METHODS",
    "METHOD_BY_NAME",
    "WorkloadMethod",
    "CostModel",
    "DEFAULT_COST_MODEL",
    "MODULE_CATEGORIES",
    "INTERACTION_CATEGORIES",
    "ModuleCounts",
    "InteractionCounts",
]
