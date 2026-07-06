"""Unified AttackResult schema (v2) for the attack-evaluation framework.

Plain dataclasses (no heavy deps). One ``AttackResult`` per (attack, target
method, setting). The schema is explicit about *threat model*, *attacker
knowledge* and *implementation level* so the security table never conflates a
measured number, a worst-case (weight-leakage) number, a best-effort probe, and
a blocked (unimplementable) attack.

Honesty invariants (``validate``):
  * ``status="blocked"`` => an ``error``/``notes`` reason and no measured metric.
  * ``status="measured"`` => at least one non-null metric.
  * ``attacker_knowledge.has_model_weights`` => threat_model worst_case.
  * ``attacker_knowledge.has_known_plaintext_pairs`` => threat_model
    known_plaintext (or worst_case).
  * ``implementation_level`` in {full, best_effort, partial, blocked}.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --- controlled vocabularies -------------------------------------------------
ATTACK_FAMILIES = ("structural", "optimization", "cryptanalysis", "alignment",
                   "statistical", "adapter", "placeholder")
IMPLEMENTATION_LEVELS = ("full", "best_effort", "partial", "blocked")
STATUSES = ("measured", "failed", "blocked", "partial")
THREAT_MODELS = (
    "closed_model_no_weight_access",
    "public_model",
    "known_plaintext",
    "weight_leakage_worst_case",
    "split_inference",
)
TARGET_METHODS = (
    "plaintext_gpu",
    "stip_qwen",
    "obfuscatune_qwen_orthogonal",
    "obfuscatune_qwen_random",
    "ours_amulet_style",
    "ours_amulet_style_no_fresh_pad",
    "ours_amulet_style_weight_leakage_worst_case",
    # explicit ours variants (round: minimal-win analysis) — no longer the
    # ambiguous "ours_amulet_style"
    "ours_amulet_style_signed_perm",         # current mainline (global signed-perm)
    "ours_amulet_style_fresh_pad",           # candidate: fresh per-token orthogonal mask
    "ours_fresh_signed_perm",                # candidate: fresh signed-perm act + static perm fold
    "ours_non_isometric_variant",            # design candidate: fresh non-orthogonal well-cond
    "ours_amulet_style_kronecker",           # design_needed (Kronecker = nonlinear island)
    "ours_fresh_pad_kronecker",              # design_needed
    "toy",
)

_KNOWLEDGE_KEYS = (
    "has_algorithm", "has_secret_keys", "has_model_weights", "has_embedding_table",
    "has_known_plaintext_pairs", "has_intermediate_activations", "has_logits",
    "has_gradients", "has_labels_or_posteriors",
)
METRIC_KEYS = (
    "attack_success_rate", "one_minus_attack_success_rate",
    "token_recovery_top1", "token_recovery_top5", "token_recovery_top10",
    "token_recovery_top100", "sequence_exact_match", "mean_token_accuracy",
    "rouge_l_f1", "edit_distance", "cosine_similarity", "mse", "relative_l2_error",
    "matrix_recovery_error", "alignment_accuracy", "prompt_recovery_success",
    # structural / statistical extras
    "permutation_recovery_accuracy", "multiset_leakage_score",
    "norm_profile_match_accuracy", "distance_profile_match_accuracy",
    "frequency_rank_correlation",
    # KPA sample-complexity extras
    "heldout_deobfuscation_relative_l2_error", "num_known_pairs",
    # optimization-attack extras
    "loss_initial", "loss_final", "loss_reduction_ratio",
)
_INPUT_KEYS = ("num_samples", "batch_size", "seq_len", "hidden_size", "vocab_size",
               "dtype", "device")
_CONFIG_KEYS = ("num_steps", "num_restarts", "lr", "loss_type", "topk", "seed")
_RUNTIME_KEYS = ("wall_time_ms", "num_steps_run", "peak_memory_gb", "requires_gpu")


def empty_metrics() -> dict[str, Any]:
    return {k: None for k in METRIC_KEYS}


def make_attacker_knowledge(**kw: Any) -> dict[str, Any]:
    d = {k: False for k in _KNOWLEDGE_KEYS}
    d["has_algorithm"] = True                 # Kerckhoffs default
    d["has_intermediate_activations"] = True
    d.update({k: v for k, v in kw.items() if k in _KNOWLEDGE_KEYS})
    return d


def make_input_info(**kw: Any) -> dict[str, Any]:
    d = {k: None for k in _INPUT_KEYS}
    d.setdefault("device", "cpu")
    d.update({k: v for k, v in kw.items() if k in _INPUT_KEYS})
    return d


def make_attack_config(**kw: Any) -> dict[str, Any]:
    d = {k: None for k in _CONFIG_KEYS}
    d["topk"] = [1, 5, 10, 100]
    d.update({k: v for k, v in kw.items() if k in _CONFIG_KEYS})
    return d


def make_runtime(**kw: Any) -> dict[str, Any]:
    d = {k: None for k in _RUNTIME_KEYS}
    d["requires_gpu"] = False
    d.update({k: v for k, v in kw.items() if k in _RUNTIME_KEYS})
    return d


@dataclass
class AttackResult:
    attack_id: str
    attack_name: str
    attack_family: str
    target_method: str
    threat_model: str
    paper_source: str = ""
    implementation_level: str = "full"
    model_family: str = "toy"
    model_name_or_path: str | None = None
    task_type: str = "generation"
    attacker_knowledge: dict[str, Any] = field(default_factory=make_attacker_knowledge)
    input_info: dict[str, Any] = field(default_factory=make_input_info)
    attack_config: dict[str, Any] = field(default_factory=make_attack_config)
    metrics: dict[str, Any] = field(default_factory=empty_metrics)
    runtime: dict[str, Any] = field(default_factory=make_runtime)
    status: str = "measured"
    error: str | None = None
    notes: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        m = empty_metrics()
        m.update(d.get("metrics") or {})
        d["metrics"] = m
        k = make_attacker_knowledge()
        k.update(d.get("attacker_knowledge") or {})
        d["attacker_knowledge"] = k
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttackResult":
        known = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in known}
        r = cls(**kw)
        m = empty_metrics(); m.update(r.metrics or {}); r.metrics = m
        k = make_attacker_knowledge(); k.update(r.attacker_knowledge or {}); r.attacker_knowledge = k
        return r

    def has_any_metric(self) -> bool:
        return any(v is not None for v in (self.metrics or {}).values())

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.attack_family not in ATTACK_FAMILIES:
            errs.append(f"attack_family {self.attack_family!r} invalid")
        if self.implementation_level not in IMPLEMENTATION_LEVELS:
            errs.append(f"implementation_level {self.implementation_level!r} invalid")
        if self.status not in STATUSES:
            errs.append(f"status {self.status!r} invalid")
        if self.threat_model not in THREAT_MODELS:
            errs.append(f"threat_model {self.threat_model!r} invalid")
        if self.target_method not in TARGET_METHODS:
            errs.append(f"target_method {self.target_method!r} invalid")
        ak = self.attacker_knowledge or {}
        if self.status == "blocked":
            if not (self.error or self.notes):
                errs.append("blocked status requires an error/notes reason")
            if self.has_any_metric():
                errs.append("blocked status must not carry a measured metric")
        if self.status == "measured" and not self.has_any_metric():
            errs.append("measured status requires at least one non-null metric")
        if ak.get("has_model_weights") and self.threat_model not in (
                "weight_leakage_worst_case", "split_inference", "public_model"):
            errs.append("has_model_weights=True requires a white-box threat_model "
                        "(weight_leakage_worst_case / split_inference / public_model), "
                        "never closed_model_no_weight_access")
        if ak.get("has_known_plaintext_pairs") and self.threat_model not in (
                "known_plaintext", "weight_leakage_worst_case"):
            errs.append("has_known_plaintext_pairs=True requires threat_model=known_plaintext")
        return errs

    def validated(self) -> "AttackResult":
        errs = self.validate()
        if errs:
            raise ValueError(f"invalid AttackResult ({self.attack_id}/{self.target_method}): "
                             + "; ".join(errs))
        return self


def blocked_result(attack_id: str, attack_name: str, attack_family: str,
                   target_method: str, threat_model: str, reason: str,
                   *, paper_source: str = "", **kw) -> AttackResult:
    # a caller's base dict may carry implementation_level / status / etc.; the
    # blocked result overrides them, so drop any conflicting keys.
    for k in ("implementation_level", "status", "error", "notes"):
        kw.pop(k, None)
    kw.pop("paper_source", None)
    return AttackResult(
        attack_id=attack_id, attack_name=attack_name, attack_family=attack_family,
        target_method=target_method, threat_model=threat_model,
        paper_source=paper_source, implementation_level="blocked",
        status="blocked", error=reason, notes=reason, **kw)


__all__ = [
    "AttackResult", "blocked_result", "empty_metrics", "make_attacker_knowledge",
    "make_input_info", "make_attack_config", "make_runtime",
    "ATTACK_FAMILIES", "IMPLEMENTATION_LEVELS", "STATUSES", "THREAT_MODELS",
    "TARGET_METHODS", "METRIC_KEYS",
]
