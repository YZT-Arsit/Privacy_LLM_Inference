"""Unified attack-evaluation framework.

Compares the security of plaintext GPU, STIP, ObfuscaTune-Qwen and our
amulet-style scheme under a common AttackResult schema + threat-model taxonomy.
Faithful reimplementations of eight attack families (see
``docs/attack_paper_audit.md``). Nothing here modifies the defenses.
"""

from __future__ import annotations

from .schema import (AttackResult, blocked_result, empty_metrics,
                     make_attacker_knowledge, make_input_info, make_runtime,
                     make_attack_config)
from . import (attacker_knowledge, metrics, qwen_hooks, registry, representations,
               result_io, defense_adapters, table_builder)
from .nn_embedding_inversion import run_nn_embedding_inversion
from .eia_optimization import run_eia_optimization
from .bre_bisr_attack import run_bre_bisr_attack, run_bre_backward_gradient_matching
from .kpa_known_plaintext import run_kpa_known_plaintext
from .permutation_multiset_attack import structural_leakage_probe, run_autoregressive_decode
from .arrowmatch_attack import run_arrowmatch_attack, load_external_arrowmatch
from .pia_prompt_inversion import run_pia_prompt_inversion
from .frequency_distribution_attack import run_frequency_distribution_attack

__all__ = [
    "AttackResult", "blocked_result", "empty_metrics", "make_attacker_knowledge",
    "make_input_info", "make_runtime", "make_attack_config",
    "attacker_knowledge", "metrics", "qwen_hooks", "registry", "representations",
    "result_io", "defense_adapters", "table_builder",
    "run_nn_embedding_inversion", "run_eia_optimization", "run_bre_bisr_attack",
    "run_bre_backward_gradient_matching", "run_kpa_known_plaintext",
    "structural_leakage_probe", "run_autoregressive_decode", "run_arrowmatch_attack",
    "load_external_arrowmatch", "run_pia_prompt_inversion",
    "run_frequency_distribution_attack",
]
