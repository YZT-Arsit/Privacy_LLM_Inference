"""Paper-grade experiment harnesses (Stage 5.0)."""

from pllo.experiments.attention_probe import AttentionProbeConfig, run_attention_probe
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
from pllo.experiments.workload_profiler import (
    InteractionCounts,
    ModuleCounts,
    WorkloadProfileConfig,
    run_workload_profile,
)

__all__ = [
    "AttentionProbeConfig",
    "run_attention_probe",
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
