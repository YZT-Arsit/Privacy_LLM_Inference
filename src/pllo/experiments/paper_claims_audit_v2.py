"""Stage 7.7g -- Paper claims audit v2.

A table-of-claims that drives paper-safe wording. Each row has:
- claim text
- status in {supported, proxy_supported, cost_model_only, unsupported}
- evidence_artifact (path to the supporting JSON/MD)
- safe_wording
- unsafe_wording
- remaining_blocker_before_real_deployment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PaperClaimsAuditV2Config:
    outputs_dir: Optional[Path] = None  # used only to check artifact presence


CLAIMS: List[Dict[str, str]] = [
    {
        "id": "padded_full_generation_correctness",
        "claim": (
            "Padded boundary linears integrate into full modern-decoder "
            "generation; pads never enter RMSNorm / RoPE / SwiGLU / "
            "softmax cores."
        ),
        "status": "supported",
        "evidence_artifact":
            "outputs/modern_decoder_generation_correctness.json",
        "safe_wording": (
            "Padded full-generation correctness verified at float64 "
            "precision under CPU local emulation."
        ),
        "unsafe_wording": (
            "Padded generation gives cryptographic privacy."
        ),
        "remaining_blocker": "Real TEE/GPU deployment.",
    },
    {
        "id": "one_round_low_interaction_exact_mode",
        "claim": (
            "Low-interaction operator-compatible path achieves exact "
            "generation with one TEE-accelerator round trip per decode "
            "step under exact_visible_attention."
        ),
        "status": "supported",
        "evidence_artifact":
            "outputs/modern_decoder_low_interaction_correctness.json",
        "safe_wording": (
            "Exact low-interaction mode: greedy match 1.0, "
            "online_boundary_round_trips_per_decode_step = 1, "
            "intermediate_tee_reentry = false."
        ),
        "unsafe_wording": (
            "Exact low-interaction mode hides attention maps."
        ),
        "remaining_blocker": "Real TEE/GPU deployment.",
    },
    {
        "id": "rope_transient_plain_qk_eliminated",
        "claim": (
            "RoPE-safe pre-mask mode eliminates transient plain Q/K/V "
            "on the accelerator: qkv_projection_outputs_masked_directly "
            "= true."
        ),
        "status": "supported",
        "evidence_artifact":
            "outputs/modern_decoder_rope_safe_low_interaction.json",
        "safe_wording": (
            "Per-head block-diagonal RoPE-plane rotation masks commute "
            "with apply_rope and are folded into the qkv-projection, "
            "yielding masked Q_tilde / K_tilde / V_tilde directly."
        ),
        "unsafe_wording": (
            "Hides Q/K/V cryptographically."
        ),
        "remaining_blocker": "Real GPU fused-kernel implementation.",
    },
    {
        "id": "norm_full_gram_reduced_by_token_chunk_masks",
        "claim": (
            "Token/chunk norm-mask granularity disrupts the full-"
            "sequence Gram-matrix leakage that sequence-shared Q "
            "exhibits; row L2 norms remain preserved by RMSNorm "
            "correctness."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/norm_granularity_low_interaction.json",
        "safe_wording": (
            "Token-wise masking preserves per-row L2 norms (required "
            "by RMSNorm) and disrupts off-diagonal Gram; chunk(k) "
            "preserves within-chunk Gram and disrupts cross-chunk Gram."
        ),
        "unsafe_wording": (
            "Token-wise masking hides row norms."
        ),
        "remaining_blocker": "None for algebraic claim.",
    },
    {
        "id": "attention_maps_hidden_only_in_trusted_softmax_mode",
        "claim": (
            "Attention maps are hidden from the accelerator transcript "
            "ONLY in trusted_softmax_attention mode; exact_visible_"
            "attention exposes them by construction."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/attention_privacy_modes.json",
        "safe_wording": (
            "Attention hiding requires trusted/secure softmax or "
            "approximate attention. Exact low-interaction with "
            "accelerator-side softmax exposes the QK invariant."
        ),
        "unsafe_wording": (
            "Row-wise score shifts provide attention privacy."
        ),
        "remaining_blocker": "Real TEE with low boundary latency.",
    },
    {
        "id": "attention_maps_visible_in_exact_low_interaction_mode",
        "claim": (
            "exact_visible_attention mode preserves the QK invariant "
            "by construction; attention scores and probabilities are "
            "visible on the accelerator."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/attention_privacy_modes.json",
        "safe_wording": (
            "Exact low-interaction baseline trades attention-map "
            "privacy for one-round-trip exactness."
        ),
        "unsafe_wording": (
            "Exact low-interaction mode hides attention."
        ),
        "remaining_blocker": "None for algebraic claim.",
    },
    {
        "id": "scalable_lm_head_dense_mask_not_feasible",
        "claim": (
            "Dense V x V LM-head orthogonal mask is not feasible for "
            "real LLM vocab sizes; permutation and block-diagonal "
            "masks scale but disclose multiset/block leakage."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/lm_head_scalability.json",
        "safe_wording": (
            "Dense N_vocab not scalable to V >= 16k; permutation "
            "(O(V) storage) and block (O(V b) storage) are scalable "
            "alternatives with explicit leakage notes."
        ),
        "unsafe_wording": (
            "Dense vocab mask is scalable."
        ),
        "remaining_blocker": "Real serving runtime sample-loop integration.",
    },
    {
        "id": "lora_integration_supported_for_specified_sites",
        "claim": (
            "LoRA adapters integrate with the Stage 7.6h main protocol "
            "for q/k/v/o/up/gate/down_proj insertion sites; rank "
            "padding hides true rank but padded rank is observable."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/lora_protocol_integration.json",
        "safe_wording": (
            "Forward LoRA padded-boundary identity holds at every "
            "supported site at float64; A_tilde = M^{-1} A R, "
            "B_tilde = R^{-1} B N_out."
        ),
        "unsafe_wording": (
            "LoRA training is supported."
        ),
        "remaining_blocker": "LoRA backward / training not implemented.",
    },
    {
        "id": "paged_kv_invariant_supported_in_synthetic_abstraction",
        "claim": (
            "The per-(session, layer, head) masked KV invariant holds "
            "under a CPU synthetic paged cache with block-table "
            "remapping."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/paged_kv_abstraction.json",
        "safe_wording": (
            "Block-table indexing preserves K_tilde = K @ N_K and "
            "V_tilde = V @ N_V per session; cross-session block "
            "sharing disabled by default."
        ),
        "unsafe_wording": (
            "Paged cache is cryptographically isolated."
        ),
        "remaining_blocker": "Real GPU paged-attention kernel.",
    },
    {
        "id": "multi_session_mask_isolation_supported_in_cpu_simulation",
        "claim": (
            "Per-session masks (Q_l, N_K, N_V, N_vocab) are independent; "
            "same prompt under two sessions produces different masked "
            "boundary fingerprints; cross-session prefix sharing off "
            "by default."
        ),
        "status": "supported",
        "evidence_artifact": "outputs/multi_session_batching.json",
        "safe_wording": (
            "Per-session orthogonal masks are sampled independently; "
            "boundary fingerprints differ across sessions for the "
            "same prompt."
        ),
        "unsafe_wording": (
            "Continuous batching is cryptographically isolated."
        ),
        "remaining_blocker": "Real serving scheduler integration.",
    },
    {
        "id": "integrity_only_probabilistic_spot_check",
        "claim": (
            "Active-adversary integrity is supported ONLY as a "
            "probabilistic spot-check prototype; no verifiable "
            "computation."
        ),
        "status": "proxy_supported",
        "evidence_artifact": "outputs/integrity_spotcheck.json",
        "safe_wording": (
            "Detection rate scales with checked_fraction; no false "
            "alarms under correct execution; not a verifiable "
            "computation primitive."
        ),
        "unsafe_wording": (
            "Active malicious accelerator fully handled."
        ),
        "remaining_blocker": (
            "Cryptographic verifiable computation / authenticated "
            "dataflow."
        ),
    },
    {
        "id": "no_real_gpu_or_tee_wall_clock",
        "claim": (
            "No real GPU or TEE wall-clock latency / throughput is "
            "measured; all numbers are FLOP / byte estimates or "
            "CPU emulated counts."
        ),
        "status": "unsupported",
        "evidence_artifact": "outputs/paper_cost_model.json",
        "safe_wording": (
            "Complexity-model evidence only; no real wall-clock."
        ),
        "unsafe_wording": (
            "Measured real GPU/TEE performance."
        ),
        "remaining_blocker": "Actual hardware access (H100 CC / SGX).",
    },
    {
        "id": "no_formal_cryptographic_security",
        "claim": (
            "No formal cryptographic / semantic / differential-privacy "
            "security is claimed."
        ),
        "status": "unsupported",
        "evidence_artifact": "every-report-limitations-section",
        "safe_wording": (
            "Algebraic correctness + leakage / cost accounting only."
        ),
        "unsafe_wording": (
            "This is cryptographic security."
        ),
        "remaining_blocker": (
            "Cryptographic protocol design or formal proof out of "
            "scope for this project."
        ),
    },
    {
        "id": "no_full_qwen_or_llama_deployment_unless_real_wrapper",
        "claim": (
            "No full Qwen / LLaMA deployment unless a real wrapper "
            "exists; only synthetic tiny modern decoder is exercised."
        ),
        "status": "unsupported",
        "evidence_artifact": "outputs/paper_claims_audit_v2.json",
        "safe_wording": (
            "Tiny modern decoder used as paper-ready surrogate; "
            "scaling to LLaMA / Qwen requires the corresponding "
            "tokenizer + model loader + GPU kernels."
        ),
        "unsafe_wording": (
            "Qwen / LLaMA deployed in TEE-GPU split."
        ),
        "remaining_blocker": "Real model loader + real GPU.",
    },
    {
        "id": "no_hardware_side_channel_evaluation",
        "claim": (
            "Hardware side channels (timing, memory, power, RDMA) are "
            "NOT evaluated."
        ),
        "status": "unsupported",
        "evidence_artifact": "every-report-limitations-section",
        "safe_wording": (
            "Side-channel evaluation out of scope; would require real "
            "hardware platform and counter-measure design."
        ),
        "unsafe_wording": (
            "Side channels evaluated."
        ),
        "remaining_blocker": "Real hardware + side-channel platform.",
    },
]


def _classify_audit_result(claim: Dict[str, str], outputs_dir: Path) -> Dict[str, Any]:
    """For supported claims, verify the evidence artifact exists."""
    art = claim["evidence_artifact"]
    if art.startswith("outputs/"):
        path = outputs_dir / art[len("outputs/"):]
        exists = path.exists()
    else:
        path = None
        exists = True  # claims pointing at all-reports-section
    return {
        **claim,
        "evidence_artifact_exists": exists,
        "artifact_path": str(path) if path else None,
    }


def run_paper_claims_audit_v2(
    *, outputs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if outputs_dir is None:
        outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    audited = [_classify_audit_result(c, outputs_dir) for c in CLAIMS]
    return {
        "status": "ok",
        "stage": "7.7g",
        "main_mode": "paper_claims_audit_v2",
        "device": "cpu",
        "outputs_dir": str(outputs_dir),
        "claims": audited,
        "summary": {
            "supported": sum(1 for c in audited if c["status"] == "supported"),
            "proxy_supported":
                sum(1 for c in audited if c["status"] == "proxy_supported"),
            "cost_model_only":
                sum(1 for c in audited if c["status"] == "cost_model_only"),
            "unsupported":
                sum(1 for c in audited if c["status"] == "unsupported"),
        },
        "limitations": [
            "Audit table is paper-ready guidance, not a formal proof.",
            "Evidence-artifact existence is checked; semantic validity "
            "is not re-evaluated here.",
            "Unsupported claims must be NEVER written as supported in "
            "the paper.",
        ],
        "paper_safe_wording": (
            "All paper claims are classified into supported, "
            "proxy_supported, cost_model_only, or unsupported, with "
            "the corresponding safe and unsafe wording per claim."
        ),
        "unsafe_wording_to_avoid": [
            "Treating unsupported claims as supported.",
            "Citing missing-evidence artifacts as proof.",
            "Combining algebraic correctness with cryptographic "
            "security in the abstract.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Paper Claims Audit v2")
    w()
    w(
        "_Stage 7.7g: table of every paper-relevant claim, its "
        "support status, safe wording, unsafe wording, and remaining "
        "blocker before real deployment._"
    )
    w()
    s = report["summary"]
    w("## Summary")
    w()
    w("| Status | Count |")
    w("|---|---|")
    for k in ("supported", "proxy_supported", "cost_model_only", "unsupported"):
        w(f"| {k} | {s[k]} |")
    w()
    w("## Claims Table")
    w()
    w("| id | status | evidence | exists | remaining_blocker |")
    w("|---|---|---|---|---|")
    for c in report["claims"]:
        w(
            f"| `{c['id']}` | **{c['status']}** | "
            f"`{c['evidence_artifact']}` | "
            f"{c['evidence_artifact_exists']} | "
            f"{c['remaining_blocker']} |"
        )
    w()
    w("## Safe Wording Per Claim")
    w()
    for c in report["claims"]:
        w(f"### `{c['id']}`")
        w()
        w(f"- claim: {c['claim']}")
        w(f"- safe: {c['safe_wording']}")
        w(f"- unsafe: {c['unsafe_wording']}")
        w()
    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "paper_claims_audit_v2.json",
    md_filename: str = "paper_claims_audit_v2.md",
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
    "CLAIMS",
    "PaperClaimsAuditV2Config",
    "render_markdown",
    "run_paper_claims_audit_v2",
    "write_reports",
]
