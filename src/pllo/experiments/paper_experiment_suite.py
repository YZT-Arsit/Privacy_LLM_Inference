"""Stage 7.7 paper-ready experiment suite aggregator.

Combines the Stage 7.6e/f/g/h/i correctness + privacy experiments
with the new Stage 7.7a-g experiments into one paper-ready JSON +
Markdown report. The aggregator does NOT re-run the Stage 7.6
experiments; it loads their existing JSON outputs (if present) so
the suite stays under a minute. The Stage 7.7 experiments are run
directly because they are inexpensive.

The report is paper-safe by construction: every limitation section
states CPU local emulation only, no real TEE/GPU, no formal
cryptographic security, no hardware side-channel evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pllo.experiments.attention_privacy_modes import (
    AttentionPrivacyModesConfig,
    run_attention_privacy_modes,
)
from pllo.experiments.integrity_spotcheck import (
    IntegritySpotCheckConfig,
    run_integrity_spotcheck,
)
from pllo.experiments.lm_head_scalability import (
    LMHeadScalabilityConfig,
    run_lm_head_scalability,
)
from pllo.experiments.lora_protocol_integration import (
    LoRAProtocolConfig,
    run_lora_protocol_integration,
)
from pllo.experiments.multi_session_batching import (
    MultiSessionBatchingConfig,
    run_multi_session_batching,
)
from pllo.experiments.norm_granularity_low_interaction import (
    NormGranularityConfig,
    run_norm_granularity_low_interaction,
)
from pllo.experiments.paged_kv_abstraction import (
    PagedKVConfig,
    run_paged_kv_abstraction,
)
from pllo.experiments.paper_claims_audit_v2 import (
    run_paper_claims_audit_v2,
)
from pllo.experiments.paper_cost_model import (
    PaperCostModelConfig,
    run_paper_cost_model,
)


@dataclass(frozen=True)
class PaperExperimentSuiteConfig:
    outputs_dir: Optional[Path] = None  # default: <repo>/outputs


# ---------------------------------------------------------------------------
# Existing-stage loaders
# ---------------------------------------------------------------------------


_PREEXISTING_REPORTS = {
    "7.6e_modern_decoder_generation_correctness":
        "modern_decoder_generation_correctness.json",
    "7.6f_modern_decoder_low_interaction_correctness":
        "modern_decoder_low_interaction_correctness.json",
    "7.6g_modern_decoder_rope_safe_low_interaction":
        "modern_decoder_rope_safe_low_interaction.json",
}


def _load_existing(outputs_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for stage_key, fname in _PREEXISTING_REPORTS.items():
        p = outputs_dir / fname
        if p.exists():
            try:
                obj = json.loads(p.read_text())
                out[stage_key] = {
                    "loaded_from": str(p),
                    "status": obj.get("status", "ok"),
                    "summary_keys": sorted(list(obj.keys()))[:20],
                }
            except Exception as e:  # noqa: BLE001
                out[stage_key] = {
                    "loaded_from": str(p),
                    "status": "load_error",
                    "error": str(e),
                }
        else:
            out[stage_key] = {
                "loaded_from": str(p),
                "status": "missing",
            }
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_paper_experiment_suite(
    *, cfg: Optional[PaperExperimentSuiteConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = PaperExperimentSuiteConfig()
    repo_root = Path(__file__).resolve().parents[3]
    outputs_dir = cfg.outputs_dir or (repo_root / "outputs")

    # Stage 7.6e / 7.6f / 7.6g: load from existing JSON if present.
    pre = _load_existing(outputs_dir)

    # Stage 7.6h: run inline (fast).
    s_7_6h = run_norm_granularity_low_interaction(cfg=NormGranularityConfig())
    # Stage 7.6i.
    s_7_6i = run_attention_privacy_modes(cfg=AttentionPrivacyModesConfig())
    # Stage 7.7a-g.
    s_7_7a = run_lm_head_scalability(cfg=LMHeadScalabilityConfig())
    s_7_7b = run_lora_protocol_integration(cfg=LoRAProtocolConfig())
    s_7_7c = run_paged_kv_abstraction(cfg=PagedKVConfig())
    s_7_7d = run_multi_session_batching(cfg=MultiSessionBatchingConfig())
    s_7_7e = run_integrity_spotcheck(cfg=IntegritySpotCheckConfig())
    s_7_7f = run_paper_cost_model(cfg=PaperCostModelConfig())
    s_7_7g = run_paper_claims_audit_v2(outputs_dir=outputs_dir)

    stages: Dict[str, Any] = {
        **pre,
        "7.6h_norm_granularity_low_interaction": _stage_summary(s_7_6h),
        "7.6i_attention_privacy_modes": _stage_summary(s_7_6i),
        "7.7a_lm_head_scalability": _stage_summary(s_7_7a),
        "7.7b_lora_integration": _stage_summary(s_7_7b),
        "7.7c_paged_kv_abstraction": _stage_summary(s_7_7c),
        "7.7d_multi_session_batching": _stage_summary(s_7_7d),
        "7.7e_integrity_spotcheck": _stage_summary(s_7_7e),
        "7.7f_complexity_model": _stage_summary(s_7_7f),
        "7.7g_paper_claims_audit_v2": _stage_summary(s_7_7g),
    }

    # Paper claims table (from 7.7g).
    paper_claims_table = {
        c["id"]: {
            "status": c["status"],
            "evidence_artifact": c["evidence_artifact"],
            "remaining_blocker": c["remaining_blocker"],
        }
        for c in s_7_7g["claims"]
    }

    # Experiment matrix.
    experiment_matrix = [
        {"stage": "7.6e", "experiment": "modern_decoder_generation_correctness",
         "kind": "correctness", "loaded": pre.get(
             "7.6e_modern_decoder_generation_correctness",
             {"status": "missing"})["status"] in ("ok",)},
        {"stage": "7.6f", "experiment": "modern_decoder_low_interaction_correctness",
         "kind": "correctness", "loaded": pre.get(
             "7.6f_modern_decoder_low_interaction_correctness",
             {"status": "missing"})["status"] in ("ok",)},
        {"stage": "7.6g", "experiment": "modern_decoder_rope_safe_low_interaction",
         "kind": "privacy_leakage", "loaded": pre.get(
             "7.6g_modern_decoder_rope_safe_low_interaction",
             {"status": "missing"})["status"] in ("ok",)},
        {"stage": "7.6h", "experiment": "norm_granularity_low_interaction",
         "kind": "privacy_leakage", "loaded": True},
        {"stage": "7.6i", "experiment": "attention_privacy_modes",
         "kind": "privacy_leakage", "loaded": True},
        {"stage": "7.7a", "experiment": "lm_head_scalability",
         "kind": "scalability", "loaded": True},
        {"stage": "7.7b", "experiment": "lora_protocol_integration",
         "kind": "integration", "loaded": True},
        {"stage": "7.7c", "experiment": "paged_kv_abstraction",
         "kind": "infrastructure", "loaded": True},
        {"stage": "7.7d", "experiment": "multi_session_batching",
         "kind": "isolation", "loaded": True},
        {"stage": "7.7e", "experiment": "integrity_spotcheck",
         "kind": "active_adversary_proxy", "loaded": True},
        {"stage": "7.7f", "experiment": "paper_cost_model",
         "kind": "cost_model", "loaded": True},
        {"stage": "7.7g", "experiment": "paper_claims_audit_v2",
         "kind": "audit", "loaded": True},
    ]

    report: Dict[str, Any] = {
        "status": "ok",
        "stage": "7.7",
        "environment": {
            "device": "cpu",
            "dtype": "float64",
            "real_gpu": False,
            "real_tee": False,
            "network_required": False,
        },
        "stages": stages,
        "experiment_matrix": experiment_matrix,
        "paper_claims_table": paper_claims_table,
        "paper_claims_summary": s_7_7g["summary"],
        "supported_claims": [
            c["id"] for c in s_7_7g["claims"] if c["status"] == "supported"
        ],
        "proxy_supported_claims": [
            c["id"] for c in s_7_7g["claims"]
            if c["status"] == "proxy_supported"
        ],
        "unsupported_claims": [
            c["id"] for c in s_7_7g["claims"]
            if c["status"] == "unsupported"
        ],
        "remaining_blockers_before_real_gpu_tee": [
            "Real H100 CC / SGX / Gramine / Occlum / TEE platform.",
            "Real CUDA / FlashAttention / vLLM backend with fused "
            "confidential kernels.",
            "Real GPU paged-attention kernel + serving scheduler.",
            "Cryptographic verifiable-computation primitive.",
            "Hardware side-channel evaluation platform.",
            "LoRA training (backward) integration.",
        ],
        "limitations": [
            "CPU local emulation only.",
            "No real TEE / GPU deployment.",
            "No hardware side-channel evaluation.",
            "No formal cryptographic / semantic / differential-privacy "
            "security.",
            "Validates algebraic correctness, leakage accounting, and "
            "cost-model evidence only.",
        ],
        "paper_safe_summary_wording": (
            "We prepare a CPU local-emulation experiment suite that "
            "validates padded masked generation correctness, low-"
            "interaction operator-compatible execution, RoPE-safe "
            "pre-masking, norm-mask granularity, attention privacy "
            "modes, scalable LM-head alternatives, LoRA integration, "
            "paged KV abstraction, multi-session isolation, and "
            "probabilistic integrity spot-checking. These experiments "
            "establish algebraic correctness and leakage/cost "
            "accounting, but do not constitute real GPU/TEE "
            "deployment or formal cryptographic security."
        ),
        "unsafe_wording_to_avoid": [
            "real TEE/GPU performance",
            "formal cryptographic security",
            "semantic security",
            "full Qwen/LLaMA deployment",
            "attention maps hidden in exact low-interaction mode",
            "dense vocab mask is scalable",
            "active malicious accelerator fully handled",
            "hardware side channels evaluated",
        ],
    }
    return report


def _stage_summary(stage_report: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary so the aggregator JSON does not blow up."""
    return {
        "status": stage_report.get("status", "ok"),
        "stage": stage_report.get("stage"),
        "main_mode": stage_report.get("main_mode"),
        "limitations_excerpt": stage_report.get("limitations", [])[:3],
        "paper_safe_wording": stage_report.get(
            "paper_safe_wording"
        ) or stage_report.get("paper_safe_summary_wording"),
        "modes_evaluated": stage_report.get("modes_evaluated"),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Paper-Ready Experiment Suite")
    w()
    w(
        "_Stage 7.7: aggregate CPU local-emulation experiments into "
        "a paper-ready report. NO real GPU / TEE / formal cryptographic "
        "security._"
    )
    w()
    w("## Environment")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k, v in report["environment"].items():
        w(f"| {k} | {v} |")
    w()

    w("## Executive Summary")
    w()
    w(report["paper_safe_summary_wording"])
    w()

    w("## Experiment Matrix")
    w()
    w("| Stage | Experiment | Kind | Loaded |")
    w("|---|---|---|---|")
    for row in report["experiment_matrix"]:
        w(
            f"| {row['stage']} | `{row['experiment']}` | "
            f"{row['kind']} | {row['loaded']} |"
        )
    w()

    w("## Stage Status")
    w()
    w("| Stage Key | Status | Stage Id | Paper-Safe Wording |")
    w("|---|---|---|---|")
    for key, info in report["stages"].items():
        safe = info.get("paper_safe_wording") or ""
        if safe and len(safe) > 120:
            safe = safe[:117] + "..."
        w(
            f"| `{key}` | {info.get('status', 'n/a')} | "
            f"{info.get('stage', 'n/a')} | {safe} |"
        )
    w()

    s = report["paper_claims_summary"]
    w("## Paper Claims Summary")
    w()
    w("| Status | Count |")
    w("|---|---|")
    for k in ("supported", "proxy_supported", "cost_model_only", "unsupported"):
        w(f"| {k} | {s[k]} |")
    w()

    w("## Supported Claims")
    w()
    for cid in report["supported_claims"]:
        info = report["paper_claims_table"][cid]
        w(f"- `{cid}` — evidence: `{info['evidence_artifact']}`")
    w()
    w("## Proxy-Supported Claims")
    w()
    for cid in report["proxy_supported_claims"]:
        info = report["paper_claims_table"][cid]
        w(f"- `{cid}` — evidence: `{info['evidence_artifact']}`")
    w()
    w("## Unsupported Claims (NEVER write as supported)")
    w()
    for cid in report["unsupported_claims"]:
        info = report["paper_claims_table"][cid]
        w(f"- `{cid}` — blocker: {info['remaining_blocker']}")
    w()

    w("## Mode Comparison (from 7.6i and 7.7f)")
    w()
    w(
        "| Mode | exact | one_round_trip | attention_hidden | "
        "intermediate_tee_reentry |"
    )
    w("|---|---|---|---|---|")
    # Stage 7.6i comparison.
    w("| exact_visible_attention | True | True | False | False |")
    w("| trusted_softmax_attention | True | False | True | True |")
    w("| score_blinding_experimental | True | True | False | False |")
    w()

    w("## Cost / Complexity (from 7.7f)")
    w()
    w(
        "See [outputs/paper_cost_model.md](paper_cost_model.md) for the "
        "symbolic formulas, tiny-config counts, and LLaMA-7B-ish "
        "real-config estimates per protocol mode."
    )
    w()

    w("## Scalability Warnings")
    w()
    w("- Dense V x V LM-head mask is NOT feasible for real LLM vocab; ")
    w("  use permutation or block-diagonal (see 7.7a).")
    w("- Token-wise norm mask is O(s d^3) per call; sequence mode is ")
    w("  cheaper but exposes full Gram (see 7.6h / 7.7f).")
    w("- Trusted-softmax mode breaks one-round-trip property; adds ")
    w("  L extra TEE round trips per decode step (see 7.6i / 7.7f).")
    w()

    w("## Remaining Blockers Before Real GPU / TEE")
    w()
    for x in report["remaining_blockers_before_real_gpu_tee"]:
        w(f"- {x}")
    w()

    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Recommended Paper Wording")
    w()
    w(f"> {report['paper_safe_summary_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "paper_experiment_suite.json",
    md_filename: str = "paper_experiment_suite.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "PaperExperimentSuiteConfig",
    "render_markdown",
    "run_paper_experiment_suite",
    "write_reports",
]
