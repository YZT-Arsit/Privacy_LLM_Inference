"""Stage 7.8d -- Decoder-only component coverage audit.

A paper-appendix-ready table that answers the reviewer question:
"What decoder-only components are covered by this protocol?"

For every component we list:
- status: supported / partially_supported / audit_only / unsupported
- reason
- required invariant if supported
- leakage surface
- remaining blocker
- artifact evidence (path or "n/a")

The audit dynamically reflects whether Stage 7.8a (sliding window) and
Stage 7.8b (precision) have produced outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Component table
# ---------------------------------------------------------------------------


# Each entry: status, reason, required_invariant, leakage_surface,
# remaining_blocker, artifact_evidence
A_COVERED: List[Dict[str, str]] = [
    {
        "component": "RMSNorm",
        "status": "supported",
        "reason": "operator-compatible orthogonal: RMSNormCore(H Q) = RMSNormCore(H) Q",
        "required_invariant": "Q orthogonal; gamma folded into next linear",
        "leakage_surface": "per-row L2 norms preserved (RMSNorm correctness)",
        "remaining_blocker": "none for algebraic claim",
        "artifact_evidence": "outputs/modern_decoder_low_interaction_correctness.json",
    },
    {
        "component": "SwiGLU",
        "status": "supported",
        "reason": "paired permutation P shared by gate / up branches",
        "required_invariant": "down_proj_compat = P^{-1} W_down @ Q_l",
        "leakage_surface": "permutation seed observable but uniform across batch",
        "remaining_blocker": "none for algebraic claim",
        "artifact_evidence": "outputs/modern_decoder_low_interaction_correctness.json",
    },
    {
        "component": "standard 1D RoPE",
        "status": "supported",
        "reason": "rotate-half RoPE commutes with RoPE-plane block-diagonal mask",
        "required_invariant": "B_K is block-diagonal 2D rotation in each RoPE pair",
        "leakage_surface": "RoPE-pair 2D norms preserved",
        "remaining_blocker": "none for algebraic claim",
        "artifact_evidence": "outputs/modern_decoder_rope_safe_low_interaction.json",
    },
    {
        "component": "GQA / MQA",
        "status": "supported",
        "reason": "N_Q[h] = N_K[h//group_size]^{-T} -> Q_tilde K_tilde^T = Q K^T",
        "required_invariant": "B_Q B_K^T = I per Q head",
        "leakage_surface": "score matrix visible in exact_visible_attention",
        "remaining_blocker": "none for algebraic claim",
        "artifact_evidence": "outputs/modern_decoder_rope_safe_low_interaction.json",
    },
    {
        "component": "causal attention",
        "status": "supported",
        "reason": "QK score invariant + causal mask in softmax",
        "required_invariant": "see attention privacy modes",
        "leakage_surface": "attention map in exact mode; hidden in trusted softmax",
        "remaining_blocker": "n/a",
        "artifact_evidence": "outputs/attention_privacy_modes.json",
    },
    {
        "component": "KV cache",
        "status": "supported",
        "reason": "K_tilde = K @ N_K, V_tilde = V @ N_V per session",
        "required_invariant": "per-session N_K, N_V orthogonal",
        "leakage_surface": "cache length visible",
        "remaining_blocker": "real serving runtime",
        "artifact_evidence": "outputs/modern_decoder_low_interaction_correctness.json",
    },
    {
        "component": "paged KV abstraction",
        "status": "supported",
        "reason": "block-table indexing preserves the masked KV invariant",
        "required_invariant": "per-(layer, head) mask shared across blocks of a session",
        "leakage_surface": "block-table observable; timing not evaluated",
        "remaining_blocker": "real GPU paged-attention kernel",
        "artifact_evidence": "outputs/paged_kv_abstraction.json",
    },
    {
        "component": "LM head",
        "status": "supported",
        "reason": "padded masked logits with trusted recovery",
        "required_invariant": "z_recovered = z_tilde @ N_vocab^{-1}",
        "leakage_surface": "dense N_vocab not scalable; see scalable LM head",
        "remaining_blocker": "n/a (see LM-head scalability)",
        "artifact_evidence": "outputs/lm_head_scalability.json",
    },
    {
        "component": "LoRA inference",
        "status": "supported",
        "reason": "padded LoRA boundary identity (Stage 7.7b)",
        "required_invariant": "A_tilde = M^{-1} A R, B_tilde = R^{-1} B N_out",
        "leakage_surface": "padded rank visible; true rank hidden by zero pad",
        "remaining_blocker": "LoRA training (backward)",
        "artifact_evidence": "outputs/lora_protocol_integration.json",
    },
    {
        "component": "generation processors inside TEE",
        "status": "supported",
        "reason": "main theorem (Stage 7.8c): logit processors are exact on recovered logits",
        "required_invariant": "z_recovered = z_plain at machine precision",
        "leakage_surface": "output length / stop timing observable unless padded",
        "remaining_blocker": "output-length side-channel hiding",
        "artifact_evidence": "outputs/generation_processor_coverage.json",
    },
]


B_PARTIAL: List[Dict[str, str]] = [
    {
        "component": "sliding window attention",
        "status": "supported" if (
            Path(__file__).resolve().parents[3]
            / "outputs" / "sliding_window_attention.json"
        ).exists() else "audit_only",
        "reason": "Stage 7.8a: KV window invariant + score invariant under window cut-off",
        "required_invariant": "K_tilde_window = K_plain_window @ N_K",
        "leakage_surface": "window-size policy is public; timing not evaluated",
        "remaining_blocker": "real FlashAttention / sliding-window CUDA kernel",
        "artifact_evidence": "outputs/sliding_window_attention.json",
    },
    {
        "component": "LayerNorm (non-LLaMA path)",
        "status": "audit_only",
        "reason": "theory mirrors RMSNorm-compatible orthogonal mask but is not the main Llama/Qwen path",
        "required_invariant": "mean and variance both invariant under orthogonal right-action only if mask is orthogonal AND row-mean-preserving",
        "leakage_surface": "row mean preserved",
        "remaining_blocker": "extra constraint that Q preserves mean direction; not exercised here",
        "artifact_evidence": "n/a",
    },
    {
        "component": "GELU MLP (non-SwiGLU path)",
        "status": "audit_only",
        "reason": "permutation island theory exists from earlier stages but not the main SwiGLU path",
        "required_invariant": "GELU is element-wise -> permutation-equivariant",
        "leakage_surface": "permutation seed observable",
        "remaining_blocker": "not exercised under the latest low-interaction wrapper",
        "artifact_evidence": "outputs/nonlinear_island_experiments.json",
    },
    {
        "component": "prefix cache (cross-session sharing)",
        "status": "audit_only",
        "reason": "private mode only; cross-session sharing flagged as leakage surface",
        "required_invariant": "n/a (off by default)",
        "leakage_surface": "if enabled, shared prefix K_tilde / V_tilde leaks across sessions",
        "remaining_blocker": "explicit public-prefix flag + threat-model declaration",
        "artifact_evidence": "outputs/paged_kv_abstraction.json",
    },
    {
        "component": "beam search",
        "status": "audit_only",
        "reason": "main theorem applies; not implemented end-to-end here",
        "required_invariant": "z_recovered = z_plain",
        "leakage_surface": "beam width / state observable on output channel",
        "remaining_blocker": "TEE-resident beam manager",
        "artifact_evidence": "outputs/generation_processor_coverage.json",
    },
    {
        "component": "quantization (fp16 / bf16 / int8 / int4)",
        "status": "partially_supported" if (
            Path(__file__).resolve().parents[3]
            / "outputs" / "precision_quantization_stability.json"
        ).exists() else "audit_only",
        "reason": "Stage 7.8b: simulated only; well-conditioned masks recommended for low precision",
        "required_invariant": "mask family well-conditioned (orthogonal / permutation)",
        "leakage_surface": "quantization scale per channel observable",
        "remaining_blocker": "real GPU fp16 / bf16 / int8 / int4 kernels",
        "artifact_evidence": "outputs/precision_quantization_stability.json",
    },
]


C_UNSUPPORTED: List[Dict[str, str]] = [
    {
        "component": "M-RoPE / multimodal positional encoding",
        "status": "unsupported",
        "reason": "M-RoPE mixes multiple position axes; one block-diagonal rotation may not commute across all axes",
        "required_invariant": "future work",
        "leakage_surface": "n/a",
        "remaining_blocker": "extend RoPE-plane analysis to multiple position dimensions",
        "artifact_evidence": "n/a",
    },
    {
        "component": "MoE router / expert dispatch",
        "status": "unsupported",
        "reason": "router output reveals expert selection; routing decisions cross the masked invariant",
        "required_invariant": "future work",
        "leakage_surface": "n/a",
        "remaining_blocker": "trusted routing or masked expert dispatch",
        "artifact_evidence": "n/a",
    },
    {
        "component": "Multi-Head Latent Attention",
        "status": "unsupported",
        "reason": "latent compression changes the (Q, K, V) algebra; not covered by current QK invariant",
        "required_invariant": "future work",
        "leakage_surface": "n/a",
        "remaining_blocker": "MLA-specific invariant derivation",
        "artifact_evidence": "n/a",
    },
    {
        "component": "speculative decoding",
        "status": "unsupported",
        "reason": "draft / target verification protocol crosses TEE boundary; not analysed",
        "required_invariant": "future work",
        "leakage_surface": "n/a",
        "remaining_blocker": "speculative-decode threat model + masked draft model",
        "artifact_evidence": "n/a",
    },
    {
        "component": "real vLLM / FlashAttention backend",
        "status": "unsupported",
        "reason": "CPU local emulation only; no real GPU kernel",
        "required_invariant": "n/a",
        "leakage_surface": "n/a",
        "remaining_blocker": "real GPU + serving runtime",
        "artifact_evidence": "n/a",
    },
    {
        "component": "real GPU / TEE hardware side channels",
        "status": "unsupported",
        "reason": "no real hardware platform available",
        "required_invariant": "n/a",
        "leakage_surface": "n/a",
        "remaining_blocker": "real hardware + side-channel platform",
        "artifact_evidence": "n/a",
    },
    {
        "component": "full active malicious security",
        "status": "unsupported",
        "reason": "Stage 7.7e is probabilistic spot-check only; no verifiable computation",
        "required_invariant": "n/a",
        "leakage_surface": "n/a",
        "remaining_blocker": "cryptographic verifiable computation",
        "artifact_evidence": "outputs/integrity_spotcheck.json",
    },
    {
        "component": "LoRA training (backward)",
        "status": "unsupported",
        "reason": "Stage 7.7b is forward only",
        "required_invariant": "n/a",
        "leakage_surface": "n/a",
        "remaining_blocker": "backward path through padded boundary; gradient masks",
        "artifact_evidence": "n/a",
    },
    {
        "component": "full Qwen / LLaMA deployment",
        "status": "unsupported",
        "reason": "synthetic tiny modern decoder only; no real model loader",
        "required_invariant": "n/a",
        "leakage_surface": "n/a",
        "remaining_blocker": "real HF / safetensors loader + GPU",
        "artifact_evidence": "n/a",
    },
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderComponentCoverageConfig:
    outputs_dir: Optional[Path] = None


def run_decoder_component_coverage_audit(
    *, outputs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    od = outputs_dir or (repo_root / "outputs")

    # Refresh the dynamic statuses based on outputs/.
    def _refresh(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for e in entries:
            new = dict(e)
            art = e["artifact_evidence"]
            if art != "n/a" and art.startswith("outputs/"):
                p = od / art[len("outputs/"):]
                new["artifact_exists"] = p.exists()
            else:
                new["artifact_exists"] = (art == "n/a")
            out.append(new)
        return out

    covered = _refresh(A_COVERED)
    partial = _refresh(B_PARTIAL)
    unsupp = _refresh(C_UNSUPPORTED)

    report = {
        "status": "ok",
        "stage": "7.8d",
        "main_mode": "decoder_component_coverage_audit",
        "device": "cpu",
        "outputs_dir": str(od),
        "covered_in_main_protocol": covered,
        "partially_covered_or_extension": partial,
        "not_covered_future_work": unsupp,
        "summary": {
            "covered": len(covered),
            "partial": len(partial),
            "unsupported": len(unsupp),
        },
        "limitations": [
            "CPU local emulation only.",
            "No real TEE / GPU deployment.",
            "No hardware side-channel evaluation.",
            "No formal cryptographic / semantic / differential-privacy "
            "security.",
            "No full Qwen / LLaMA deployment unless a real wrapper exists.",
        ],
        "paper_safe_wording": (
            "We provide a coverage table for common decoder-only "
            "components. Supported components carry algebraic "
            "evidence under CPU local emulation; partially supported "
            "components have audit-only or simulation-only evidence; "
            "unsupported components are listed as future work with "
            "explicit remaining blockers."
        ),
        "unsafe_wording_to_avoid": [
            "M-RoPE supported.",
            "MoE supported.",
            "Speculative decoding supported.",
            "Real quantized model deployment.",
            "Real vLLM serving support.",
            "Hardware side channels evaluated.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _row(e: Dict[str, str]) -> str:
    return (
        f"| `{e['component']}` | **{e['status']}** | {e['reason']} | "
        f"`{e['artifact_evidence']}` | "
        f"{e['remaining_blocker']} |"
    )


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Decoder-only Component Coverage Audit")
    w()
    w(
        "_Stage 7.8d: paper-appendix-ready table of which decoder-only "
        "components are covered by the masked low-interaction protocol._"
    )
    w()
    w("## Summary")
    w()
    s = report["summary"]
    w("| Category | Count |")
    w("|---|---|")
    w(f"| Covered in main protocol | {s['covered']} |")
    w(f"| Partially covered / extension | {s['partial']} |")
    w(f"| Not covered (future work) | {s['unsupported']} |")
    w()

    w("## A. Covered in Main Protocol")
    w()
    w("| Component | Status | Reason | Evidence | Remaining Blocker |")
    w("|---|---|---|---|---|")
    for e in report["covered_in_main_protocol"]:
        w(_row(e))
    w()

    w("## B. Partially Covered / Extension")
    w()
    w("| Component | Status | Reason | Evidence | Remaining Blocker |")
    w("|---|---|---|---|---|")
    for e in report["partially_covered_or_extension"]:
        w(_row(e))
    w()

    w("## C. Not Covered (Future Work)")
    w()
    w("| Component | Status | Reason | Evidence | Remaining Blocker |")
    w("|---|---|---|---|---|")
    for e in report["not_covered_future_work"]:
        w(_row(e))
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
    json_filename: str = "decoder_component_coverage_audit.json",
    md_filename: str = "decoder_component_coverage_audit.md",
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
    "DecoderComponentCoverageConfig",
    "render_markdown",
    "run_decoder_component_coverage_audit",
    "write_reports",
]
