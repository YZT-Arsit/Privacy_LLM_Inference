"""Stage 7.1 -- paper-ready artifact generator.

Reads the verified Stage 6.4.1 / 6.7 / 6.8 / 7.0 JSON reports and emits
structured paper artifacts: a method summary, correctness theorem drafts, a
complexity table, a leakage/boundary table, an ablation summary, a
safe/unsafe claim audit, limitations, and a combined report.

This stage adds NO masking functionality and makes NO security guarantee.
Forbidden ("unsafe") wording is confined to the explicitly-marked
claim-audit section (delimited by ``UNSAFE_CLAIMS_BEGIN/END`` markers); the
rest of every artifact uses cautious, hedged language. Missing source
reports degrade gracefully (the relevant section is emitted with status
``missing_source_report`` and recorded in metadata). No internet, no
transformers, no CUDA.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Markers around the only region allowed to contain forbidden phrasing.
UNSAFE_BEGIN = "<!-- UNSAFE_CLAIMS_BEGIN -->"
UNSAFE_END = "<!-- UNSAFE_CLAIMS_END -->"

# Forbidden phrases. Anywhere OUTSIDE a marked unsafe-claims block these must
# not appear in generated artifacts (enforced by the test suite).
UNSAFE_PHRASES: tuple[str, ...] = (
    "semantic security",
    "cryptographic security",
    "formal privacy guarantee",
    "all intermediate states fully hidden",
    "attention patterns are hidden",
    "equivalent to dense masks",
    "full logit privacy",
    "final output semantics are protected",
    "zero-overhead",
    "production-ready",
)

SAFE_CLAIMS: tuple[str, ...] = (
    "The implementation verifies exact masked/plain correctness for the "
    "synthetic full CausalLM skeleton.",
    "The design avoids intermediate TEE calls inside decoder blocks.",
    "The GPU receives masked embeddings instead of raw input ids.",
    "The GPU returns masked logits in the preferred boundary.",
    "The TEE recovers logits and performs sampling.",
    "RoPE-GQA correctness holds under pairwise complex-scaling masks.",
    "Multi-layer mask handoff is correct but may require an online handoff "
    "GEMM.",
    "Leakage surfaces are explicitly reported.",
)

# Each unsafe claim contains exactly one forbidden phrase from UNSAFE_PHRASES.
UNSAFE_CLAIMS: tuple[str, ...] = (
    "Claiming semantic security.",
    "Claiming cryptographic security.",
    "Claiming a formal privacy guarantee.",
    "Claiming all intermediate states fully hidden.",
    "Claiming attention patterns are hidden.",
    "Claiming RoPE masks are equivalent to dense masks.",
    "Claiming vocab permutation+scaling provides full logit privacy.",
    "Claiming final output semantics are protected.",
    "Claiming zero-overhead per-layer handoff.",
    "Claiming production-ready full model inference.",
)

REQUIRED_STATEMENT = (
    "These artifacts summarize verified correctness, cost, and leakage "
    "accounting for the masked CausalLM prototype. They do not constitute a "
    "semantic, cryptographic, or formal security proof."
)

# Cost / leakage table column orders.
_COST_COLS = [
    "variant", "implemented", "analytical_only", "gpu_flops_prefill",
    "gpu_flops_decode", "tee_flops_prefill", "tee_flops_decode",
    "transfer_bytes_prefill", "transfer_bytes_decode", "kv_cache_bytes",
    "boundary_calls", "handoff_gemm_flops", "lm_head_gpu_flops",
    "lm_head_tee_flops", "notes",
]
_LEAK_COLS = [
    "variant", "input_ids_visible_to_gpu",
    "plaintext_embedding_visible_to_gpu", "masked_embedding_visible_to_gpu",
    "plaintext_hidden_visible_to_gpu", "masked_hidden_visible_to_gpu",
    "attention_scores_visible_to_gpu", "attention_probs_visible_to_gpu",
    "plaintext_kv_cache_visible_to_gpu", "masked_kv_cache_visible_to_gpu",
    "plaintext_logits_visible_to_gpu", "masked_logits_visible_to_gpu",
    "sampled_token_ids_visible_to_gpu",
    "final_output_text_semantics_protected", "security_status", "caveats",
]
_VARIANT_ORDER = [
    "plain_synthetic", "masked_same_residual_mask",
    "masked_per_layer_residual_mask", "masked_per_layer_no_vocab_scaling",
    "masked_per_layer_with_vocab_scaling", "output_hidden_to_tee",
    "gpu_masked_lm_head",
]


@dataclass
class PaperArtifactConfig:
    output_dir: str = "outputs/paper_artifacts"
    include_math: bool = True
    include_tables: bool = True
    include_claim_audit: bool = True
    cost_report_path: str = "outputs/full_pipeline_cost_leakage.json"
    skeleton_report_path: str = "outputs/masked_causal_lm_skeleton_probe.json"
    boundary_report_path: str = "outputs/causal_lm_boundary_probe.json"
    rope_report_path: str = "outputs/rope_gqa_probe.json"
    strict_no_unsafe_claims: bool = True


def _load_json(path: str) -> tuple[dict[str, Any] | None, bool]:
    p = Path(path)
    if not p.is_file():
        return None, False
    try:
        return json.loads(p.read_text(encoding="utf-8")), True
    except (OSError, json.JSONDecodeError):
        return None, False


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def _method_summary() -> str:
    return """# Method Summary

## Threat model

- **Trusted TEE**: performs embedding lookup, input masking, logits
  recovery, and sampling.
- **Untrusted GPU**: executes the masked decoder stack on masked tensors.
- **Public base-model weights**: assumed known to the operator.
- **Protected**: user input ids, plaintext embeddings, plaintext
  intermediate hidden states, the plaintext KV cache, plaintext logits, and
  optional adapter states.
- **Non-goals**: TEE compromise; side channels; the semantics of output
  text once returned to the user; hiding the attention score pattern in the
  current design; and any formal-level guarantee.

## System boundary

1. The TEE performs the embedding lookup and input masking.
2. The GPU receives masked embeddings only.
3. The GPU executes the masked decoder stack.
4. The GPU returns masked logits, or masked hidden states, depending on the
   boundary variant.
5. The TEE recovers logits and samples the next token.

No intermediate TEE calls occur inside the decoder blocks.

## Operator-compatible masking

- Dense invertible masks in linear regions; pad only at linear boundaries.
- RMSNorm under orthogonal masks (its core preserves per-row norm).
- SwiGLU under a shared paired permutation.
- RoPE under pairwise complex-scaling masks (adjacent-pair convention).
- Vocab logits under permutation + positive diagonal scaling.

These mechanisms provide **operator-compatible leakage reduction** with
**verified correctness** and **explicit leakage accounting**; they reduce
direct exposure while preserving operator-specific invariants.

## Pipeline variants

- Shared residual mask (cheaper; weaker isolation).
- Per-layer residual masks (stronger isolation; adds a handoff GEMM).
- GPU masked LM head (preferred output boundary).
- Output hidden to TEE (GPU receives no logits; higher TEE compute).
"""


def _correctness_theorems() -> str:
    return r"""# Correctness Theorems (Drafts)

Row-vector convention throughout: activations are rows, ``x @ W``.

## Lemma 1 (Linear mask folding with optional pad)
**Assumptions.** ``N`` invertible, pad row ``T``, weight ``W``, optional
right mask ``M``. **Statement.** If ``X_tilde = (X - T) N`` and
``W_tilde = N^{-1} W M`` then ``X_tilde W_tilde + T W M = X W M``.
**Proof sketch.** ``(X-T)N N^{-1} W M = (X-T) W M``; add ``T W M``.
**Caveats.** Requires the constant ``T W M`` to be added on the trusted
side. **Verified by.** linear/LoRA stages and the embedding boundary.

## Lemma 2 (RMSNorm orthogonal-mask equivariance)
**Assumptions.** ``N`` orthogonal; ``eps`` applied to the mean of squares.
**Statement.** ``RMSNormCore(X N) = RMSNormCore(X) N``.
**Proof sketch.** Orthogonal ``N`` preserves each row's L2 norm, so the
per-row scale ``1/sqrt(mean(x^2)+eps)`` is unchanged; scaling commutes with
the right multiply. **Caveats.** Holds for orthogonal ``N`` only.
**Verified by.** Stage 6.5 / 6.6 (core invariant at machine precision).

## Lemma 3 (SwiGLU paired-permutation equivariance)
**Statement.** For a shared permutation ``P``, ``SwiGLU(A P, B P) =
SwiGLU(A, B) P``. **Proof sketch.** SiLU is elementwise and the
gate*up product is elementwise, so a shared column permutation commutes; the
down projection's matching row permutation cancels it. **Verified by.**
Stage 6.5 SwiGLU island.

## Lemma 4 (RoPE pairwise complex-scaling commutation)
**Assumptions.** Adjacent-pair RoPE; per-pair blocks
``M_i = [[a_i,-b_i],[b_i,a_i]]`` (no cross-pair mixing).
**Statement.** ``RoPE(X M) = RoPE(X) M``. **Proof sketch.** Each 2D block is
a scaled rotation; scaled rotations commute with the per-pair RoPE rotation
(SO(2) is abelian). **Caveats.** Requires the adjacent-pair convention and
strictly block-diagonal pair masks; weaker than dense masks. **Verified by.**
Stage 6.4 / 6.4.1.

## Lemma 5 (GQA attention score invariance)
**Statement.** For ``Q_tilde = Q M^{-T}`` and ``K_tilde = K M``,
``RoPE(Q_tilde) RoPE(K_tilde)^T = RoPE(Q) RoPE(K)^T``. **Proof sketch.**
``M^{-T} M^T = I`` after Lemma 4 moves RoPE through the masks. **GQA.** Each
query head uses the inverse-transpose of its mapped KV head's mask.
**Verified by.** Stage 6.4 / 6.5.

## Lemma 6 (V aggregation and output-projection recovery)
**Statement.** If ``V_tilde = V S`` and the output projection uses
``S^{-1} W_o N_out`` then ``Attn(Q,K,V_tilde) S^{-1} W_o N_out =
Attn(Q,K,V) W_o N_out``. **Proof sketch.** Attention weights are unchanged
(Lemma 5); ``S S^{-1} = I``. **Verified by.** Stage 6.5 / 6.6.

## Lemma 7 (Masked logits recovery)
**Statement.** If ``M_vocab = P D`` (permutation + positive diagonal) and
``L_tilde = L M_vocab`` then ``Recover(L_tilde) = L``. **Proof sketch.**
Divide by the diagonal, invert the permutation. **Caveats.** Recovery is
trusted-side; weaker than dense vocab masking. **Verified by.** Stage 6.7.

## Theorem 1 (Single decoder block correctness)
**Statement.** Under Lemmas 2-6, a masked decoder block on
``H_tilde = H N`` produces ``y_tilde = y_plain N`` (final residual aligned).
**Proof sketch.** Compose the lemmas through both residual adds; both
summands share the same mask. **Verified by.** Stage 6.5 / 6.6 (machine
precision).

## Theorem 2 (Multi-layer CausalLM skeleton correctness)
**Statement.** By induction over layers, ``H_l_tilde = H_l N_l`` for all
``l``, and ``Recover(L_tilde) = L_plain``, so the trusted greedy token equals
the plaintext greedy token. **Proof sketch.** Base case = embedding
boundary; step = Theorem 1 plus an orthogonal change-of-basis handoff
``N_l -> N_{l+1}``; output via Lemma 7. **Caveats.** The handoff is an online
GEMM on the residual stream (skip path is not offline-foldable).
**Verified by.** Stage 6.8.

## Theorem 3 (Bounded greedy decode correctness)
**Statement.** Given the same initial ids and deterministic greedy
sampling, the masked decode loop yields the same token ids as the plaintext
reference for the tested bounded horizon. **Proof sketch.** Per-step
Theorem 2 plus exact KV-cache append invariants. **Caveats.** Bounded
horizon; greedy only. **Verified by.** Stage 6.8 (token match rate 1.0).
"""


def _complexity_table(cost: dict[str, Any] | None) -> tuple[str, str]:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(_COST_COLS)
    md = ["# Complexity Table", "", "FLOPs counted as `2*M*N*K` per matmul.",
          ""]
    if cost is None:
        md.append("> status: **missing_source_report** "
                  "(`full_pipeline_cost_leakage.json` not found)")
        md.append("")
        md.append("Variants (no numbers available):")
        for v in _VARIANT_ORDER:
            md.append(f"- {v}")
            writer.writerow([v] + ["missing_source_report"] * (len(_COST_COLS) - 1))
    else:
        by = {c["variant"]: c for c in cost.get("cost_breakdown", [])}
        md.append("| " + " | ".join(_COST_COLS[:-1]) + " |")
        md.append("|" + "---|" * (len(_COST_COLS) - 1))
        for v in _VARIANT_ORDER:
            c = by.get(v)
            if c is None:
                continue
            row = [
                c["variant"], c["implemented"], c["analytical_only"],
                f"{c['gpu_flops_prefill']:.3e}", f"{c['gpu_flops_decode']:.3e}",
                f"{c['tee_flops_prefill']:.3e}", f"{c['tee_flops_decode']:.3e}",
                c["transfer_bytes_prefill"], c["transfer_bytes_decode"],
                c["kv_cache_bytes"], c["boundary_calls"],
                f"{c['handoff_gemm_flops']:.3e}",
                f"{c['lm_head_gpu_flops']:.3e}",
                f"{c['lm_head_tee_flops']:.3e}",
            ]
            md.append("| " + " | ".join(str(x) for x in row) + " |")
            writer.writerow(row + ["; ".join(c.get("notes", []))])
        md.append("")
    md += [
        "## Notes",
        "",
        "- Per-layer residual-mask handoff is not free.",
        "- The residual skip path prevents full offline folding of the "
        "handoff change-of-basis.",
        "- Shared-mask mode is the low-cost variant.",
        "- Per-layer-mask mode is the stronger isolation variant.",
        "- Output-hidden-to-TEE reduces GPU-visible logits but shifts the LM "
        "head cost into the TEE.",
        "",
    ]
    return "\n".join(md) + "\n", out.getvalue()


def _leakage_table(cost: dict[str, Any] | None) -> tuple[str, str]:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(_LEAK_COLS)
    md = ["# Leakage / Security-Boundary Table", ""]
    if cost is None:
        md.append("> status: **missing_source_report** "
                  "(`full_pipeline_cost_leakage.json` not found)")
        md.append("")
        for v in _VARIANT_ORDER:
            writer.writerow([v] + ["missing_source_report"] * (len(_LEAK_COLS) - 1))
    else:
        by = {s["variant"]: s for s in cost.get("leakage_surfaces", [])}
        head = [c for c in _LEAK_COLS if c != "caveats"]
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "---|" * len(head))
        for v in _VARIANT_ORDER:
            s = by.get(v)
            if s is None:
                continue
            row = [s.get(c) for c in head]
            md.append("| " + " | ".join(str(x) for x in row) + " |")
            writer.writerow([s.get(c) for c in _LEAK_COLS[:-1]]
                            + ["; ".join(s.get("caveats", []))])
        md.append("")
    md += [
        "## Explanation",
        "",
        "- This is leakage accounting, not a proof of any privacy property.",
        "- Operator-compatible masks preserve operator-specific invariants.",
        "- The attention score/probability pattern remains visible in the "
        "current design.",
        "- The RoPE pair partition remains visible.",
        "- Vocab permutation+scaling is weaker than dense vocab masking.",
        "",
    ]
    return "\n".join(md) + "\n", out.getvalue()


def _ablation_summary(cost: dict[str, Any] | None,
                      rope: dict[str, Any] | None) -> str:
    md = ["# Ablation Summary", ""]
    # pull leakage proxy numbers if available
    rope_line = ""
    src = None
    if cost is not None and cost.get("leakage_proxy", {}).get("rope_pair_norm"):
        src = cost["leakage_proxy"]["rope_pair_norm"]
    elif rope is not None:
        src = None
    if src is not None:
        rot = src.get("pairwise_rotation", {})
        cs = src.get("pairwise_complex_scaling", {})
        rope_line = (
            f" (cross-session pair-norm corr: rotation "
            f"{rot.get('cross_session_pair_norm_correlation', 'n/a')}, "
            f"complex-scaling "
            f"{cs.get('cross_session_pair_norm_correlation', 'n/a')})")
    md += [
        "## 1. Rotation vs complex-scaling RoPE mask",
        f"- Rotation preserves per-pair norm exactly.{rope_line}",
        "- Complex-scaling reduces direct pair-norm linkability.",
        "- Both preserve the RoPE pair partition (weaker than dense masks).",
        "",
        "## 2. Shared vs per-layer residual masks",
        "- Shared residual mask is cheaper (no handoff GEMM).",
        "- Per-layer residual masks add an online handoff GEMM.",
        "- Per-layer residual masks reduce stable cross-layer alignment.",
        "",
        "## 3. Plain logits vs masked logits",
        "- Plaintext logits are not acceptable under the preferred boundary.",
        "- Masked logits plus trusted-side recovery is the current default.",
        "",
        "## 4. GPU masked LM head vs output hidden to TEE",
        "- GPU masked LM head: lower TEE compute; the GPU sees masked logits.",
        "- Output hidden to TEE: the GPU sees no logits, but the TEE LM-head "
        "cost increases.",
        "",
        "## 5. Vocab permutation only vs permutation + scaling",
        "- Permutation hides the token-index alignment.",
        "- Positive diagonal scaling perturbs direct logit magnitude/rank "
        "visibility on the GPU side.",
        "- Trusted-side recovery restores exact logits before sampling.",
        "",
    ]
    return "\n".join(md) + "\n"


def _claim_audit_md() -> str:
    md = ["# Claim Audit", "", "## Safe claims", ""]
    md += [f"- {c}" for c in SAFE_CLAIMS]
    md += [
        "",
        "## Unsafe claims (forbidden wording -- listed only as an audit of "
        "phrasing that must not be used)",
        "",
        UNSAFE_BEGIN,
    ]
    md += [f"- {c}" for c in UNSAFE_CLAIMS]
    md += [UNSAFE_END, "", f"> {REQUIRED_STATEMENT}", ""]
    return "\n".join(md) + "\n"


def _claim_audit_obj(config: PaperArtifactConfig) -> dict[str, Any]:
    return {
        "stage": "7.1_paper_artifacts",
        "safe_claims": list(SAFE_CLAIMS),
        "unsafe_claims": list(UNSAFE_CLAIMS),
        "unsafe_phrases_tracked": list(UNSAFE_PHRASES),
        "strict_no_unsafe_claims": config.strict_no_unsafe_claims,
        "required_statement": REQUIRED_STATEMENT,
    }


def _limitations() -> str:
    items = [
        "Synthetic full skeleton, not a production full model.",
        "HF integration currently covers a single decoder layer only.",
        "No tokenizer / chat-template integration.",
        "No full real-checkpoint generation.",
        "Greedy decode only in the full skeleton.",
        "LM-head cost is not optimized.",
        "Per-layer handoff requires an online GEMM because of the residual "
        "skip path.",
        "Attention scores/probabilities remain visible to the GPU.",
        "KV-cache masks are reused within a generation session.",
        "The RoPE pair partition remains visible.",
        "Vocab permutation+scaling is weaker than dense vocab masking.",
        "No semantic-level, cryptographic-level, or formal-level security is "
        "claimed.",
        "Output text semantics are not protected once returned to the user.",
    ]
    return "# Limitations\n\n" + "\n".join(f"- {i}" for i in items) + "\n"


def _combined_report(parts: dict[str, str]) -> str:
    md = ["# Stage 7.1 -- Paper Artifacts (Combined Report)", "",
          f"> {REQUIRED_STATEMENT}", ""]
    order = [
        ("Method summary", "method_summary.md"),
        ("Correctness theorems", "correctness_theorems.md"),
        ("Complexity table", "complexity_table.md"),
        ("Leakage / boundary table", "leakage_boundary_table.md"),
        ("Ablation summary", "ablation_summary.md"),
        ("Claim audit", "claim_audit.md"),
        ("Limitations", "limitations.md"),
    ]
    for title, key in order:
        md.append(f"\n---\n\n## {title}\n")
        # demote the sub-document's top-level heading to avoid duplicate H1s
        body = "\n".join(
            ("#" + ln if ln.startswith("# ") else ln)
            for ln in parts[key].splitlines())
        md.append(body)
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def generate_paper_artifacts(config: PaperArtifactConfig) -> dict[str, Any]:
    cost, cost_found = _load_json(config.cost_report_path)
    skeleton, skeleton_found = _load_json(config.skeleton_report_path)
    boundary, boundary_found = _load_json(config.boundary_report_path)
    rope, rope_found = _load_json(config.rope_report_path)

    complexity_md, complexity_csv = _complexity_table(cost)
    leakage_md, leakage_csv = _leakage_table(cost)

    artifacts: dict[str, str] = {
        "method_summary.md": _method_summary(),
        "correctness_theorems.md": _correctness_theorems(),
        "complexity_table.md": complexity_md,
        "complexity_table.csv": complexity_csv,
        "leakage_boundary_table.md": leakage_md,
        "leakage_boundary_table.csv": leakage_csv,
        "ablation_summary.md": _ablation_summary(cost, rope),
        "claim_audit.md": _claim_audit_md(),
        "claim_audit.json": json.dumps(_claim_audit_obj(config), indent=2),
        "limitations.md": _limitations(),
    }
    # combined report references the markdown artifacts
    artifacts["stage_7_1_paper_artifacts.md"] = _combined_report(artifacts)

    return {
        "stage": "7.1_paper_artifacts",
        "artifacts": artifacts,
        "claim_audit": _claim_audit_obj(config),
        "required_statement": REQUIRED_STATEMENT,
        "metadata": {
            "source_reports": {
                "cost_report": "found" if cost_found else "missing_source_report",
                "skeleton_report":
                    "found" if skeleton_found else "missing_source_report",
                "boundary_report":
                    "found" if boundary_found else "missing_source_report",
                "rope_report": "found" if rope_found else "missing_source_report",
            },
            "missing_inputs": [
                name for name, found in (
                    ("cost_report", cost_found),
                    ("skeleton_report", skeleton_found),
                    ("boundary_report", boundary_found),
                    ("rope_report", rope_found))
                if not found
            ],
            "output_dir": config.output_dir,
        },
    }


def write_paper_artifacts(config: PaperArtifactConfig) -> dict[str, Any]:
    report = generate_paper_artifacts(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, content in report["artifacts"].items():
        (out_dir / name).write_text(content, encoding="utf-8")
        written.append(str(out_dir / name))
    report["written_files"] = written
    return report


__all__ = [
    "PaperArtifactConfig",
    "REQUIRED_STATEMENT",
    "SAFE_CLAIMS",
    "UNSAFE_BEGIN",
    "UNSAFE_CLAIMS",
    "UNSAFE_END",
    "UNSAFE_PHRASES",
    "generate_paper_artifacts",
    "write_paper_artifacts",
]
