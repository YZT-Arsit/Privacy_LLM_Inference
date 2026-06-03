"""Stage 7.6c reviewer-risk audit.

Read-only static audit: scans the LaTeX body and the key markdown reports,
emits a structured reviewer-risk-audit table (markdown / json / csv) plus a
revision plan and five specialised review files.

This module DOES NOT modify ``outputs/`` or ``paper_results/``. It only
writes new files under ``paper_draft/``. No new experiment is launched, no
baseline is added, no claim bucket is changed.

The wording scan is purely lexical: it looks for the dangerous-words list
the user enumerated in Stage 7.6c and tags each occurrence as
``safe`` / ``risky`` / ``unsafe`` based on the surrounding context. The
top-level risk catalogue, simulated reviewer questions, and revision plan
are encoded by hand in this module based on the Stage 7.6c reviewing of
the paper body.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
LATEX_DIR = REPO_ROOT / "paper_draft" / "latex" / "sections"
PAPER_DRAFT_DIR = REPO_ROOT / "paper_draft"
PAPER_RESULTS_DIR = REPO_ROOT / "paper_results"

LATEX_FILES = (
    "00_abstract.tex",
    "01_introduction.tex",
    "02_background.tex",
    "03_system_and_threat_model.tex",
    "04_design.tex",
    "05_correctness.tex",
    "06_security_analysis.tex",
    "07_evaluation.tex",
    "08_limitations.tex",
    "09_related_work.tex",
    "10_conclusion.tex",
    "a_notation.tex",
    "b_claims_mapping.tex",
)

# ---------------------------------------------------------------------------
# Risk catalogue.
#
# Each entry is the result of a Stage 7.6c human-style review of the paper
# body. Severity uses the four-level taxonomy:
#   low / medium / high / critical
# wording_fix_enough True means a textual revision is sufficient; no new
# experiment is required.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskItem:
    risk_id: str
    dimension: str
    severity: str
    location: str
    risky_wording_or_missing: str
    why_reviewer_may_object: str
    recommended_revision: str
    new_experiment_needed: bool
    wording_fix_enough: bool
    priority: str  # P0 / P1 / P2 / P3


# Dimension keys are kept stable so downstream summaries can group reliably.
DIM_NOVELTY = "novelty_risk"
DIM_THREAT = "threat_model_risk"
DIM_PRIORWORK = "prior_work_comparison_risk"
DIM_RUNTIME = "runtime_deployment_risk"
DIM_SECURITY = "security_claim_risk"
DIM_CORRECTNESS = "correctness_proof_risk"
DIM_EVAL = "evaluation_sufficiency_risk"
DIM_BASELINE = "baseline_fairness_risk"
DIM_WORDING = "wording_risk"


RISK_ITEMS: Tuple[RiskItem, ...] = (
    # ---------------- Novelty ----------------
    RiskItem(
        risk_id="NOV-01",
        dimension=DIM_NOVELTY,
        severity="high",
        location="sec:introduction (Our approach) + sec:related (Amulet-style and matrix obfuscation methods)",
        risky_wording_or_missing=(
            "The four ingredients in 'Our approach' read close to 'Amulet right-mask + LoRA'. "
            "Related work distinguishes Amulet but the contrast is one paragraph and uses "
            "'Amulet-style' phrasing that may be skimmed as 'we are an Amulet variant'."
        ),
        why_reviewer_may_object=(
            "A security reviewer who reads only the intro and related work may conclude the "
            "system is Amulet + LoRA + island wrappers, i.e. an engineering extension rather "
            "than a separate point on the design space."
        ),
        recommended_revision=(
            "Add a one-paragraph 'What is new vs. Amulet' explicit contrast in sec:related "
            "enumerating: (i) generation-compatible right-masking (Amulet KV-append "
            "counterexample, RQ13); (ii) operator-compatible nonlinear islands with proved "
            "invariance groups (Theorems 2-5); (iii) private LoRA backward + rank padding "
            "(Theorems 7-9); (iv) deployable trusted-controller / accelerator-backend split "
            "(sec:design:backend, RQ14). Cross-reference each item to its theorem and RQ."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="NOV-02",
        dimension=DIM_NOVELTY,
        severity="medium",
        location="sec:introduction (Why prior obfuscation does not cover generative LLMs)",
        risky_wording_or_missing=(
            "Cites 'Amulet-style matrix obfuscation' and 'secret-shared offload' together. "
            "The LoRA / KV-append / RoPE / GQA novelty list is good but the public-base-model "
            "vs. user-data distinction is implicit, not stated."
        ),
        why_reviewer_may_object=(
            "Reviewers from the cryptographic privacy community routinely ask 'why are you "
            "protecting the activations instead of the model?' If the public-base-model "
            "assumption is not stated up-front, the contribution looks confused."
        ),
        recommended_revision=(
            "Add one sentence to the introduction: 'Our threat model assumes the base-model "
            "weights are public; we protect the *user's* runtime data (prompt, hidden states, "
            "KV cache, LoRA adapter, gradients). Model weight extraction is out of scope.' "
            "Cross-reference sec:threat 'Allowed leakage'."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="NOV-03",
        dimension=DIM_NOVELTY,
        severity="medium",
        location="sec:design:backend + sec:eval:rq14",
        risky_wording_or_missing=(
            "RQ14 introduces the runtime API but does not explicitly position it as a "
            "*deployment-readiness primitive*, separate from the masking primitives. "
            "Reviewers may read it as plumbing rather than contribution."
        ),
        why_reviewer_may_object=(
            "Systems reviewers may say 'the backend split is engineering, not novelty'. The "
            "argument that the split makes future TEE/GPU swap possible without protocol "
            "changes should be made explicit, not implicit."
        ),
        recommended_revision=(
            "Add one paragraph at the end of sec:design:backend explaining that this "
            "boundary makes the protocol logic and the hardware substrate orthogonal -- the "
            "same TrustedController is reused across LocalCPUBackend and any future TEE / "
            "GPU backend without touching the masking code. Keep the 'not deployed' caveat."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P2",
    ),
    # ---------------- Threat model ----------------
    RiskItem(
        risk_id="THR-01",
        dimension=DIM_THREAT,
        severity="high",
        location="sec:threat (Allowed leakage) + sec:limitations item 9",
        risky_wording_or_missing=(
            "Sequence-length leakage is mentioned once ('unless a separate sequence-length "
            "pad is engaged (not implemented)'). Reviewer may underweight this and miss that "
            "we do not currently pad seq-length."
        ),
        why_reviewer_may_object=(
            "Sequence-length is a classical side-channel for prompt-content recovery in "
            "LLM serving. A reviewer who sees no explicit treatment will demand it."
        ),
        recommended_revision=(
            "Promote the seq-length leak into its own bullet in 'Allowed leakage' and "
            "reference sec:limitations item 9 explicitly. State that batch shape, "
            "per-layer tensor shape, and seq-len are visible by default."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="THR-02",
        dimension=DIM_THREAT,
        severity="high",
        location="sec:threat (Trusted-side controller) + sec:abstract + figures/captions",
        risky_wording_or_missing=(
            "Trusted-side controller is correctly described as 'TEE-like, locally emulated' "
            "in sec:threat, but the qualifier is not always co-located with 'trusted "
            "controller' in figures and captions."
        ),
        why_reviewer_may_object=(
            "A skim reviewer who only reads a figure caption may not notice the 'locally "
            "emulated' qualifier and conclude that a real TEE was deployed."
        ),
        recommended_revision=(
            "Audit every figure caption (fig:system-overview, fig:right-masked-decode, "
            "fig:dense-sandwich, fig:nonlinear-island, fig:lora-training, "
            "fig:measured-runtime-summary, fig:security-risk-matrix) to confirm each "
            "occurrence of 'trusted controller' is co-located with 'locally emulated' or "
            "an equivalent disclaimer."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="THR-03",
        dimension=DIM_THREAT,
        severity="medium",
        location="sec:threat (Adversary capabilities exercised in evaluation)",
        risky_wording_or_missing=(
            "Lists ridge / MLP / signature / Sinkhorn / linkability / DLG / spectral / "
            "ensemble / cost-model attackers. Does not say what *stronger* attacker is "
            "out of scope (e.g. learning-based inverters with substantially more data, "
            "white-box weight access)."
        ),
        why_reviewer_may_object=(
            "Reviewer will ask 'have you tried a stronger attacker?' We should pre-empt by "
            "naming the explicit stronger-attacker categories that we do not evaluate."
        ),
        recommended_revision=(
            "Add one sentence to 'Adversary capabilities' enumerating attackers we do not "
            "evaluate: large-scale learning-based inverters trained on millions of "
            "(masked, plain) pairs; white-box-weight inverters; hardware-side-channel "
            "attackers; multi-tenant trace correlation attackers."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="THR-04",
        dimension=DIM_THREAT,
        severity="medium",
        location="sec:threat (Out of scope) + sec:design:backend",
        risky_wording_or_missing=(
            "Out-of-scope correctly lists compromised TEE, HW side-channels, formal "
            "security, real-TEE wall-time, full fine-tune, framework integration, "
            "padded-rank hiding, outsourced loss/optimizer. The deployable-runtime "
            "boundary contract (Stage 7.5c) is not enumerated here."
        ),
        why_reviewer_may_object=(
            "A reviewer scanning the threat model only will not realise that the runtime "
            "boundary is interface-only; the qualifier currently lives only in "
            "sec:design:backend and sec:eval:rq14."
        ),
        recommended_revision=(
            "Add one out-of-scope bullet: 'Real-hardware deployment of the trusted "
            "controller / accelerator backend split. The runtime API is interface-ready "
            "(see sec:design:backend); only the LocalCPUBackend is implemented.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    # ---------------- Prior-work comparison ----------------
    RiskItem(
        risk_id="PWR-01",
        dimension=DIM_PRIORWORK,
        severity="high",
        location="sec:eval:rq13 (Cross-cutting observations final paragraph)",
        risky_wording_or_missing=(
            "'... highlights which primitives extend to generative LLMs (ours) and which "
            "do not (Slalom, DarKnight, Amulet static masking, CryptoNets arithmetic "
            "skeleton).' Reads as 'ours is better than them'."
        ),
        why_reviewer_may_object=(
            "Each baseline solves a different problem under a different threat model. "
            "Saying 'they do not extend to generative LLMs' invites the rebuttal 'because "
            "they were not designed for generative LLMs, and you do not provide their "
            "formal-security property in return.'"
        ),
        recommended_revision=(
            "Re-word as: 'The direct prior-work primitives we implement target different "
            "problems and threat models. Their primitive surface does not cover "
            "decoder-only generation, KV-cache append, or LoRA personalization, which is "
            "the surface our scheme targets; conversely, our scheme does not provide the "
            "formal cryptographic guarantees that CryptoNets / Gazelle / Delphi / SecureML "
            "/ MiniONN provide.' Drop the implicit 'ours wins' framing."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="PWR-02",
        dimension=DIM_PRIORWORK,
        severity="medium",
        location="sec:eval:rq13 (Setup paragraph)",
        risky_wording_or_missing=(
            "'We implement the primitive(s) of each named prior work directly from its "
            "paper formula and execute them under the same CPU local runtime API as our "
            "scheme.' Good. But the sentence is dense and could be misread as 'we "
            "reproduced these papers.'"
        ),
        why_reviewer_may_object=(
            "Reviewer may quote this sentence in isolation and claim we are overstating "
            "reproduction."
        ),
        recommended_revision=(
            "Add the disclaimer in the *same* sentence: 'We implement the named primitive "
            "of each prior work (NOT the full system) directly from its paper formula and "
            "execute that single primitive under the same CPU local runtime API as our "
            "scheme.' Then list which papers and which primitive."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="PWR-03",
        dimension=DIM_PRIORWORK,
        severity="medium",
        location="sec:eval:rq13 + sec:related (Privacy-preserving ML inference)",
        risky_wording_or_missing=(
            "Related work says 'They give formal-security guarantees but pay one to "
            "several orders of magnitude in overhead.' RQ13 reports local CPU runtime for "
            "ours and explicitly refuses runtime for cost-model rows. A reviewer may "
            "still try to compute an unfair ratio."
        ),
        why_reviewer_may_object=(
            "Reviewers can mis-attribute a comparison if both 'we are faster' and "
            "'they are formal' appear within a few pages of each other."
        ),
        recommended_revision=(
            "Add an explicit sentence to the RQ13 setup: 'Runtime numbers in the cost-model "
            "rows are deliberately left blank (NaN). Any ratio computed against an "
            "imputed runtime would be meaningless because the cryptographic protocols are "
            "not executed.' Mirror this in sec:related."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="PWR-04",
        dimension=DIM_PRIORWORK,
        severity="low",
        location="sec:eval:rq13 (Result) + tables/direct_prior_work_comparison.tex",
        risky_wording_or_missing=(
            "Result paragraph mentions Slalom's 'Freivalds-style randomised verification "
            "check' which sounds like a security claim; we should anchor it to integrity, "
            "not confidentiality."
        ),
        why_reviewer_may_object=(
            "A cryptographic reviewer who reads only the Result paragraph may flag this "
            "as confusing the integrity property with a privacy property."
        ),
        recommended_revision=(
            "Re-word as 'a Freivalds-style randomised integrity check (not a privacy "
            "primitive)'."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P2",
    ),
    # ---------------- Runtime deployment ----------------
    RiskItem(
        risk_id="RUN-01",
        dimension=DIM_RUNTIME,
        severity="high",
        location="sec:design:backend (last sentence) + sec:eval:rq14 (Result + Limitation)",
        risky_wording_or_missing=(
            "'the interface is backend-ready, not backend-deployed' / "
            "'tee_gpu_ready_interface=True means the protocol logic is backend-agnostic; "
            "it does NOT mean hardware isolation has been deployed.' Wording is correct, "
            "but the *name* 'tee_gpu_ready_interface' invites confusion."
        ),
        why_reviewer_may_object=(
            "A skim reviewer who reads only a table cell labelled 'tee_gpu_ready_interface "
            "= True' may conclude TEE / GPU deployment has happened."
        ),
        recommended_revision=(
            "Rename in *prose* to 'backend-agnostic interface flag' or 'interface-ready "
            "flag' in every sentence referencing it; keep the raw artifact key in the "
            "table caption with an explicit gloss: 'tee_gpu_ready_interface = True means "
            "the protocol logic is backend-agnostic; no TEE or GPU has been deployed in "
            "this artifact.' Optionally rename the column in the rendered table as a P1 "
            "follow-up."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="RUN-02",
        dimension=DIM_RUNTIME,
        severity="high",
        location="sec:eval:rq7 (Result) + tables/measured_runtime.tex + figures/measured_runtime_summary.png caption",
        risky_wording_or_missing=(
            "RQ7 reports 'Mean times: plain_synthetic_linear 0.002 ms, ..., "
            "multi_layer_lora_training_step 4.684 ms.' Numbers in ms invite the reading "
            "'these are real-system latencies'."
        ),
        why_reviewer_may_object=(
            "A reviewer pulled to the runtime table first may assume the numbers are "
            "comparable to production deployments."
        ),
        recommended_revision=(
            "Prepend every ms number with 'local-emulation ' in prose; ensure each "
            "runtime table / figure caption begins with 'Local-emulation runtime on tiny "
            "tiles; not real TEE wall-time, not GPU throughput.' Already partly present; "
            "audit for consistency."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="RUN-03",
        dimension=DIM_RUNTIME,
        severity="medium",
        location="sec:eval:summary (Cross-cutting observations)",
        risky_wording_or_missing=(
            "'so a future TEE or GPU deployment only needs to swap the backend object.' "
            "Reads as 'deployment is one engineering step away.'"
        ),
        why_reviewer_may_object=(
            "Reviewer may say 'this trivialises confidential-computing deployment; "
            "attestation, sealed memory, page-table side-channels are not just plumbing.'"
        ),
        recommended_revision=(
            "Soften to 'so the protocol logic does not block a future TEE or GPU "
            "deployment; we do not claim that the remaining engineering (attestation, "
            "sealed memory, side-channel hardening, multi-tenant isolation) is trivial.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="RUN-04",
        dimension=DIM_RUNTIME,
        severity="medium",
        location="docs/runtime_boundary.md sections 4-5",
        risky_wording_or_missing=(
            "The 'How to plug in a TEE backend' section reads like a how-to. Reviewers "
            "may read it as 'they shipped a TEE backend.'"
        ),
        why_reviewer_may_object=(
            "Same skim risk as RUN-01."
        ),
        recommended_revision=(
            "Prefix the section header with '(future work)' and add a banner sentence at "
            "the top: 'No TEE backend exists in this artifact; the snippet below is the "
            "contract a future deployment must satisfy.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    # ---------------- Security claim ----------------
    RiskItem(
        risk_id="SEC-01",
        dimension=DIM_SECURITY,
        severity="high",
        location="sec:security (Transcript obfuscation, Activation recovery and linkability)",
        risky_wording_or_missing=(
            "Repeated phrase 'Bounded near random chance in our evaluated setting.' / "
            "'keeps the worst-case attacker close to random chance'. The italicised "
            "wording 'Bounded' may be read as 'we prove a bound.'"
        ),
        why_reviewer_may_object=(
            "Cryptographic reviewer will say 'bounded' implies a proof; you only show "
            "empirical accuracy near 0.5."
        ),
        recommended_revision=(
            "Replace 'bounded' with 'the measured attacker accuracy stayed close to "
            "random chance' or 'did not deviate significantly from random chance in our "
            "tests' throughout sec:security."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="SEC-02",
        dimension=DIM_SECURITY,
        severity="medium",
        location="sec:security:adapter, sec:security:gradient",
        risky_wording_or_missing=(
            "Reports '\\Delta\\text{AUC} = +0.463' and '+0.478' but the baseline "
            "(fixed_masks_fixed_u) is mentioned only once. Reviewer may compute the "
            "post-mitigation AUC themselves and dispute the +0.463 phrasing if they "
            "cannot quickly find the baseline."
        ),
        why_reviewer_may_object=(
            "Numbers without their baseline are easy to challenge."
        ),
        recommended_revision=(
            "Add 'AUC reduced from 0.963 to 0.500 (\\Delta\\text{AUC} = +0.463 vs the "
            "fixed_masks_fixed_u baseline)' style absolute numbers, or reference the "
            "specific table cell in lora_training_summary."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="SEC-03",
        dimension=DIM_SECURITY,
        severity="medium",
        location="sec:security:rank + sec:security:boundary",
        risky_wording_or_missing=(
            "Claims boundary correctly says 'we do *not* claim ... padded LoRA rank is "
            "hidden'. But sec:security:rank uses 'hides the true rank' which is "
            "ambiguous: it could be read 'hides rank in general'."
        ),
        why_reviewer_may_object=(
            "Two adjacent sentences using 'hides' / 'is not hidden' with different "
            "subjects can confuse a reviewer in a hurry."
        ),
        recommended_revision=(
            "Be uniform: 'Rank padding removes the *true* rank r_true from tensor shape "
            "but leaves the *padded* rank r_pad visible.' Use 'removes from shape' rather "
            "than 'hides'."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="SEC-04",
        dimension=DIM_SECURITY,
        severity="low",
        location="sec:security:timing",
        risky_wording_or_missing=(
            "'Risk level: low.' for cost-model timing classifier. 'low' is the safe label, "
            "but a reviewer may still read this as 'timing is solved.'"
        ),
        why_reviewer_may_object=(
            "The cost-model proxy is not a side-channel evaluation; saying 'low' invites "
            "misreading."
        ),
        recommended_revision=(
            "Append 'under our cost-model proxy only; hardware timing side-channels are "
            "out of scope' explicitly at the end of sec:security:timing."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P2",
    ),
    # ---------------- Correctness proof ----------------
    RiskItem(
        risk_id="COR-01",
        dimension=DIM_CORRECTNESS,
        severity="medium",
        location="sec:correctness (Theorem 7 LoRA masked forward, Theorem 8 LoRA backward)",
        risky_wording_or_missing=(
            "Theorem 7 statement omits the row-vector convention 'X in R^{T x d_in} acts "
            "from the left'. Theorem 8 proof sketch is terse and may not convince a "
            "reviewer that the U^{-1} cancellation is total."
        ),
        why_reviewer_may_object=(
            "Backward-pass identities in masked space are a classical place for sign / "
            "transpose errors. Reviewer will want a full derivation."
        ),
        recommended_revision=(
            "Add a one-paragraph row-vector convention statement at the top of "
            "sec:correctness. Expand Theorem 8 proof to spell out the U^{-1} and "
            "N_out^{-1} substitutions explicitly. Cross-reference the artifact row "
            "lora_backward_experiments with max-grad-error ~ 1.3e-15."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="COR-02",
        dimension=DIM_CORRECTNESS,
        severity="medium",
        location="sec:correctness (Theorem 2 pointwise activation permutation island)",
        risky_wording_or_missing=(
            "Theorem 2 covers GELU/ReLU/SiLU pointwise. The 'dense mask does not commute' "
            "*counterexample* is mentioned in sec:design but not proved in sec:correctness."
        ),
        why_reviewer_may_object=(
            "A reviewer may ask 'why a permutation island and not a dense one?' The "
            "counterexample is the answer and should appear as a remark or lemma."
        ),
        recommended_revision=(
            "Add a 'Remark (non-commutation under dense masks)' after Theorem 2: for "
            "generic dense invertible N and pointwise phi, phi(XN) != phi(X) N. Cite an "
            "explicit two-line counterexample (phi = ReLU, N a 2x2 rotation)."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P2",
    ),
    RiskItem(
        risk_id="COR-03",
        dimension=DIM_CORRECTNESS,
        severity="low",
        location="sec:correctness (Theorem 6 attention and KV cache invariant)",
        risky_wording_or_missing=(
            "Theorem 6 covers Q K^T = Q\\tilde K\\tilde^T. The V mask absorption claim "
            "('absorbed by the trailing output projection') is in the design section but "
            "not in the theorem."
        ),
        why_reviewer_may_object=(
            "Reviewer who only reads the theorems may miss the V mask story."
        ),
        recommended_revision=(
            "Promote 'and the V mask N_V is absorbed into the trailing output "
            "projection' into Theorem 6's statement."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P2",
    ),
    RiskItem(
        risk_id="COR-04",
        dimension=DIM_CORRECTNESS,
        severity="low",
        location="sec:correctness (Theorem 9 rank padding factor-product equality)",
        risky_wording_or_missing=(
            "States two cases (A_dummy B_dummy = 0 vs tracked delta) but the proof sketch "
            "covers only case 1."
        ),
        why_reviewer_may_object=(
            "Reviewer may say 'where is the delta-folding proof?'"
        ),
        recommended_revision=(
            "Add a one-line proof sketch for the tracked-correction case: when "
            "A_dummy B_dummy = Delta, A_pad B_pad = A B + Delta; the trusted side "
            "subtracts Delta at recovery time, restoring A B."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P3",
    ),
    # ---------------- Evaluation sufficiency ----------------
    RiskItem(
        risk_id="EVA-01",
        dimension=DIM_EVAL,
        severity="high",
        location="sec:eval:rq1 / rq5 / rq8 / rq11 + sec:limitations item 5",
        risky_wording_or_missing=(
            "Every correctness / LoRA / toy-task / stability RQ uses 'synthetic' or "
            "'tiny-HF' configurations. The paper does name this as a limitation, but the "
            "evaluation section reads as 'we evaluated' rather than 'we evaluated on "
            "synthetic tiles only'."
        ),
        why_reviewer_may_object=(
            "Reviewer will say 'this is not a real LLM evaluation.' We cannot add new "
            "experiments under Stage 7.6c; we must surface the limitation in every RQ "
            "result paragraph rather than only in sec:limitations."
        ),
        recommended_revision=(
            "Append 'on synthetic / tiny-HF configurations only' to the Result line of "
            "RQ1, RQ5, RQ8, RQ11 so a reviewer skimming Result paragraphs sees the scope "
            "immediately, not only in the Limitation paragraph."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="EVA-02",
        dimension=DIM_EVAL,
        severity="medium",
        location="sec:eval:rq3 (Workload profile)",
        risky_wording_or_missing=(
            "RQ3 names 'tslp_trusted_nonlinear_baseline' and 'amulet_style_reference' as "
            "cost-model baselines but uses the word 'amulet_style' which is exactly the "
            "vague label the Stage 7.5c spec ruled out."
        ),
        why_reviewer_may_object=(
            "Stage 7.5c spec forbids 'Amulet-style' phrasing. The RQ3 cost-model table "
            "still uses the legacy label."
        ),
        recommended_revision=(
            "Rename 'amulet_style_reference' to 'amulet_static_phq_cost_model' in the "
            "narrative paragraph (the artifact key may stay for backward compatibility, "
            "but the prose should be precise)."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="EVA-03",
        dimension=DIM_EVAL,
        severity="medium",
        location="sec:eval:rq6 (Rank-inference and timing)",
        risky_wording_or_missing=(
            "Reports four 'high' risk rows (gradient rank inference, stronger-dummy "
            "spectral, stronger-dummy gradient) and one 'medium' (dummy-strategy "
            "classifier). The Result paragraph states the numbers but does not tie each "
            "'high' to a future-work item."
        ),
        why_reviewer_may_object=(
            "Reviewer who sees 'high' without a next-step plan may ask 'so why is this a "
            "contribution?'"
        ),
        recommended_revision=(
            "After each 'high' or 'medium' status add a sentence: 'This is reported as "
            "open; heterogeneous padded rank (sec:conclusion item c) is the proposed "
            "follow-up.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="EVA-04",
        dimension=DIM_EVAL,
        severity="low",
        location="sec:eval:rq2 + sec:limitations item 9 (sequence length leak)",
        risky_wording_or_missing=(
            "RQ2 covers KV cache append invariant but does not measure prompt-length "
            "leakage in the decode trace."
        ),
        why_reviewer_may_object=(
            "Reviewer may ask 'what about prompt-length recovery?' We must point to "
            "sec:limitations not promise an experiment we did not run."
        ),
        recommended_revision=(
            "Append 'we do not measure prompt-length or seq-length recovery; this is "
            "explicit in sec:limitations item 9' to RQ2 Limitation paragraph."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P2",
    ),
    # ---------------- Baseline fairness ----------------
    RiskItem(
        risk_id="BAS-01",
        dimension=DIM_BASELINE,
        severity="high",
        location="sec:eval:rq13 + sec:related",
        risky_wording_or_missing=(
            "Ours rows have full_system_reproduced=True; prior-work rows have False. "
            "Comparing 'we, full system' to 'prior work, primitive only' is unfair if "
            "not framed."
        ),
        why_reviewer_may_object=(
            "This is the single most likely reject reason from a systems reviewer: "
            "'unfair comparison.'"
        ),
        recommended_revision=(
            "Add an upfront paragraph to sec:eval:rq13: 'We compare *primitives*. For our "
            "scheme the primitive *is* the full generation path; for the prior-work rows "
            "the primitive is one of: delegated linear (Slalom), additive sharing "
            "(DarKnight), static PHQ (Amulet), polynomial activation skeleton "
            "(CryptoNets), or a cost-model row (Gazelle / Delphi / SecureML / MiniONN). "
            "Direct runtime / threat-model comparison across rows is not valid; the "
            "table reports each row in its own scope.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
    RiskItem(
        risk_id="BAS-02",
        dimension=DIM_BASELINE,
        severity="medium",
        location="sec:eval:rq13 (Result) + sec:related (FHE and MPC for private inference)",
        risky_wording_or_missing=(
            "Related work says 'Our approach is complementary, not a replacement.' Good. "
            "RQ13 result paragraph could echo this more loudly, especially next to the "
            "CryptoNets / Gazelle / Delphi / SecureML / MiniONN rows."
        ),
        why_reviewer_may_object=(
            "Reviewer may otherwise read the table as 'we beat HE/MPC.'"
        ),
        recommended_revision=(
            "Add one sentence at the end of RQ13 Result: 'Cryptographic baselines provide "
            "formal-security guarantees that our scheme does not provide; the comparison "
            "is *not* a security ranking.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    RiskItem(
        risk_id="BAS-03",
        dimension=DIM_BASELINE,
        severity="medium",
        location="sec:eval:rq9 (Baseline comparison)",
        risky_wording_or_missing=(
            "RQ9 compares ten *internal* variants. The word 'baseline' may be misread as "
            "'external system' if the reader is in a hurry."
        ),
        why_reviewer_may_object=(
            "Reviewer who jumps to RQ9 first may expect Slalom / Amulet rows and not find "
            "them; the external comparison is in RQ13."
        ),
        recommended_revision=(
            "Rename RQ9 prose 'mitigation-configuration comparison' or 'internal ablation "
            "baseline comparison' and add a one-line forward reference: 'External "
            "prior-work primitives are compared in RQ13.'"
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P1",
    ),
    # ---------------- Wording risk (cross-cutting) ----------------
    RiskItem(
        risk_id="WRD-01",
        dimension=DIM_WORDING,
        severity="high",
        location="all sections (automated wording scan, see unsafe_wording_review.md)",
        risky_wording_or_missing=(
            "Automated scan of the LaTeX body for dangerous words (secure, guarantee, "
            "private, protect, hide, outperform, SOTA, TEE-ready, GPU-ready, real-time, "
            "production, full system, reproduced)."
        ),
        why_reviewer_may_object=(
            "Any single occurrence outside a 'we do not claim' clause is a quotable "
            "overclaim."
        ),
        recommended_revision=(
            "See unsafe_wording_review.md for per-occurrence classification "
            "(safe / risky / unsafe) and per-occurrence revision suggestion."
        ),
        new_experiment_needed=False,
        wording_fix_enough=True,
        priority="P0",
    ),
)


# ---------------------------------------------------------------------------
# Simulated reviewer questions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewerQA:
    qid: str
    question: str
    where_answered: str
    answer_summary: str
    answer_status: str  # answered / partial / missing


REVIEWER_QAS: Tuple[ReviewerQA, ...] = (
    ReviewerQA(
        qid="Q1",
        question="Why is this not just Amulet with LoRA?",
        where_answered=(
            "sec:introduction (Why prior obfuscation does not cover generative LLMs); "
            "sec:related (Amulet-style and matrix obfuscation methods); "
            "sec:eval:rq13 (Amulet static PHQ + KV-append counterexample)"
        ),
        answer_summary=(
            "Three new ingredients: generation-compatible right masking (RQ13 KV-append "
            "counterexample shows Amulet's static left mask breaks), operator-compatible "
            "nonlinear islands with proved invariance groups (Theorems 2-5), and a "
            "private LoRA training path with masked backward (Theorems 7-9). The new "
            "deployable runtime API (sec:design:backend, RQ14) is a fourth ingredient."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q2",
        question="Why not use Slalom / DarKnight?",
        where_answered=(
            "sec:related (TEE-assisted ML inference); sec:eval:rq13 (Slalom delegated "
            "linear primitive, DarKnight k=2 sharing skeleton)"
        ),
        answer_summary=(
            "Slalom's delegated linear is implemented directly under our runtime API "
            "(RQ13) and is correctness-equivalent on the linear primitive but does not "
            "cover generation, KV append, nonlinear islands, or LoRA. DarKnight's k=2 "
            "additive sharing also covers only the linear primitive."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q3",
        question="Why not use HE/MPC?",
        where_answered=(
            "sec:related (FHE and MPC for private inference); sec:eval:rq13 "
            "(CryptoNets / Gazelle / Delphi / SecureML / MiniONN rows)"
        ),
        answer_summary=(
            "HE/MPC give formal-security guarantees but pay one to several orders of "
            "magnitude in overhead and restrict the operator set; ours sits at a "
            "different point on the spectrum (proxy-evaluated, near-GPU-native dense "
            "matmul). RQ13 records the cryptographic baselines as cost-model rows that "
            "deliberately do not produce a measured runtime."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q4",
        question="What exactly does the GPU see?",
        where_answered=(
            "sec:threat (Untrusted GPU paragraph + Allowed leakage); "
            "sec:design:linear, sec:design:right-mask"
        ),
        answer_summary=(
            "X-tilde, W-tilde, A-tilde, B-tilde, K-tilde, V-tilde, Y-tilde; shapes; "
            "dtypes; per-call timing; the public base-model weight W; padded LoRA rank "
            "r_pad; output tokens (by design)."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q5",
        question="What does the TEE do?",
        where_answered=(
            "sec:threat (Trusted-side controller); sec:design:runtime; "
            "sec:design:backend"
        ),
        answer_summary=(
            "Holds user prompt and private context; the LoRA adapter (A, B); optimizer "
            "state; loss closure; sampler; the mask sampler and pad sampler; performs "
            "mask transform and recovery; runs the LoRA backward / optimizer step; "
            "publishes a sanitised RuntimeTranscript."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q6",
        question="Is the TEE implemented?",
        where_answered=(
            "sec:threat (Trusted-side controller -- 'emulated as a local trusted "
            "runtime; it is not a real TEE'); sec:design:backend; sec:eval:rq14; "
            "sec:limitations item 2 and item 18-19; docs/runtime_boundary.md"
        ),
        answer_summary=(
            "No. The trusted controller is locally emulated. The deployable runtime API "
            "(TrustedController + AcceleratorBackend) ships exactly one backend, "
            "LocalCPUBackend. No real TEE, no real GPU, no attestation."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q7",
        question="Are the prior systems fully reproduced?",
        where_answered=(
            "sec:eval:rq13 (Setup + Limitation); "
            "tables/direct_prior_work_comparison.tex; sec:limitations items 16-17; "
            "appendix B (Stage 7.5c safe wordings)"
        ),
        answer_summary=(
            "No. Each prior-work row implements only the named primitive (delegated "
            "linear, k=2 additive sharing, static PHQ, polynomial-activation skeleton) "
            "or is cost-model-only (Gazelle, Delphi, SecureML, MiniONN). "
            "full_system_reproduced is False for every non-ours row. Arrow is recorded "
            "as missing_paper_formula."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q8",
        question="Are runtime numbers real GPU/TEE numbers?",
        where_answered=(
            "sec:eval:rq3, sec:eval:rq7, sec:eval:rq12, sec:eval:rq14, "
            "sec:limitations item 3; figures/measured_runtime_summary.png caption; "
            "paper_results/summary.md sections 4 and 13"
        ),
        answer_summary=(
            "No. Local-emulation latencies via time.perf_counter on small CPU tiles; "
            "wall_time_source is either projected_from_op_counts or "
            "measured_local_emulation. Not real TEE wall-time, not GPU throughput."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q9",
        question="What leaks through shape, length, rank, timing?",
        where_answered=(
            "sec:threat (Allowed leakage); sec:design:rank-pad; "
            "sec:security:rank, sec:security:timing; sec:limitations items 7, 9, 10"
        ),
        answer_summary=(
            "Per-layer tensor shapes are public; padded LoRA rank r_pad is visible; "
            "sequence length is not currently padded; cost-model timing is reported "
            "near random chance under proxy_equalized but real hardware-side-channel "
            "timing is out of scope."
        ),
        answer_status="partial",
    ),
    ReviewerQA(
        qid="Q10",
        question="Are proxy attacks enough?",
        where_answered=(
            "sec:security (every subsection labels itself proxy_supported); "
            "sec:security:boundary; sec:limitations items 1, 10; appendix B "
            "(unsupported claims U1)"
        ),
        answer_summary=(
            "Explicitly no in the formal sense: no cryptographic or semantic-security "
            "claim is made anywhere. The proxy results are reported with 'in our tested "
            "configurations' hedge and labelled needs_more_evaluation when a stronger "
            "attacker may succeed."
        ),
        answer_status="answered",
    ),
    ReviewerQA(
        qid="Q11",
        question="Does this work on real LLMs?",
        where_answered=(
            "sec:eval:rq1 (GPT-2 model wrapper + modern decoder wrapper); "
            "sec:limitations items 5, 8; sec:conclusion (next-step a, b)"
        ),
        answer_summary=(
            "GPT-2 (tiny) reproduces token-for-token; the modern decoder wrapper "
            "(RMSNorm + SwiGLU + RoPE + GQA) reproduces token-for-token in synthetic "
            "or tiny-HF configurations. Full Qwen / TinyLlama / LLaMA fine-tune is "
            "explicitly out of scope."
        ),
        answer_status="partial",
    ),
    ReviewerQA(
        qid="Q12",
        question="What claim remains if no formal security is provided?",
        where_answered=(
            "sec:introduction (What this paper is not); sec:security:boundary; "
            "sec:conclusion; appendix B (claims mapping)"
        ),
        answer_summary=(
            "The paper makes only correctness claims (S1-S8) and proxy-supported "
            "claims (P1-P5). The contribution is: a generation-compatible masked-"
            "execution boundary that *reproduces* the plain reference for the tested "
            "decoder-only LLMs and the synthetic LoRA training path, together with a "
            "deployable runtime API and an artifact-backed claim audit. No security "
            "theorem is claimed."
        ),
        answer_status="answered",
    ),
)


# ---------------------------------------------------------------------------
# Wording scan.
# ---------------------------------------------------------------------------


_WORD_RE = {
    "secure": re.compile(r"\bsecure(?:ly|ness)?\b", re.IGNORECASE),
    "guarantee": re.compile(r"\bguarantee(?:s|d|ing)?\b", re.IGNORECASE),
    "prove_security": re.compile(r"\b(prove|proves|proven)\b[^.]{0,40}\b(secure|security)\b", re.IGNORECASE),
    "private": re.compile(r"\bprivate(?:ly|ness)?\b", re.IGNORECASE),
    "protect": re.compile(r"\bprotect(?:s|ed|ing|ion)?\b", re.IGNORECASE),
    "hide": re.compile(r"\b(?:hide|hides|hidden|hiding)\b", re.IGNORECASE),
    "outperform": re.compile(r"\boutperform(?:s|ed|ing)?\b", re.IGNORECASE),
    "SOTA": re.compile(r"\bSOTA\b|\bstate[-\s]of[-\s]the[-\s]art\b", re.IGNORECASE),
    "direct_comparison": re.compile(r"\bdirect(?:ly)?[\s-]+comparab(?:le|ility)\b", re.IGNORECASE),
    "reproduced": re.compile(r"\breproduce(?:d|s|ing)?\b", re.IGNORECASE),
    "TEE_ready": re.compile(r"\bTEE[-\s]?ready\b", re.IGNORECASE),
    "GPU_ready": re.compile(r"\bGPU[-\s]?ready\b", re.IGNORECASE),
    "real_time": re.compile(r"\breal[-\s]?time\b", re.IGNORECASE),
    "production": re.compile(r"\bproduction(?:[-\s]scale|[-\s]grade)?\b", re.IGNORECASE),
    "full_system": re.compile(r"\bfull[-\s]?system\b", re.IGNORECASE),
}


# Phrases that, if present in the immediate context window of a dangerous word,
# downgrade the occurrence to ``safe``. Order matters: longest match wins.
_SAFE_CONTEXT_PATTERNS = (
    "we do not claim",
    "we do *not* claim",
    "we do not prove",
    "we do *not* prove",
    "is not a real",
    "is not real",
    "not a real",
    "not real",
    "no formal",
    "no real",
    "proxy",
    "needs_more_evaluation",
    "is not implemented",
    "not implemented",
    "out of scope",
    "we make no",
    "unsupported",
    "not formal",
    "do not claim",
    "does not claim",
    "we cannot",
    "is not deployed",
    "not deployed",
    "not a security",
    "we report",
    "proof sketch",
    "private (assumed)",
    "remain trusted",
    "remains trusted",
    "trusted-side",
    "ms (local",
    "local-emulation",
    "tested configurations",
    "in our tests",
    "in our tested",
    "we evaluate",
    "we instantiate",
    "is *not*",
    "is \\emph{not}",
    "phrases this paper never",
    "never uses positively",
    "not fully reproduced",
    "are not reproduced",
    "not reproduced",
    "NOT fully reproduced",
    "would require one of these",
    "never claim",
    "do not provide",
    "we do not produce",
    "we do not execute",
)


_LATEX_EMPH_RE = re.compile(r"\\emph\{([^{}]*)\}")
_LATEX_TEXTBF_RE = re.compile(r"\\textbf\{([^{}]*)\}")
_LATEX_TEXTTT_RE = re.compile(r"\\texttt\{([^{}]*)\}")


def _strip_latex_emphasis(text: str) -> str:
    """Reduce ``we do \\emph{not} claim`` to ``we do not claim`` for matching."""

    out = text
    for regex in (_LATEX_EMPH_RE, _LATEX_TEXTBF_RE, _LATEX_TEXTTT_RE):
        out = regex.sub(lambda m: m.group(1), out)
    return out


def _classify_wording(context: str, word_kind: str) -> str:
    """Return ``safe`` / ``risky`` / ``unsafe`` for a single occurrence."""

    lowered = _strip_latex_emphasis(context).lower()
    # Treat occurrences inside LaTeX double-quoted phrases (`` ... '') as
    # safe -- they are typically a list of forbidden phrases (e.g.
    # ``cryptographically secure'') rather than a positive claim.
    if "``" in context and "''" in context:
        return "safe"
    if any(safe in lowered for safe in _SAFE_CONTEXT_PATTERNS):
        return "safe"

    # Dangerous unsafe patterns no matter the context.
    bad_phrases = (
        "is secure",
        "provably secure",
        "cryptographically secure",
        "semantically secure",
        "fully reproduced",
        "outperforms",
        "we outperform",
        "deployment-ready",
        "production-ready",
        "real TEE wall-time",
        "real-tee wall-time",
        "real gpu throughput",
        "tee-ready",
        "gpu-ready",
        "full system reproduction",
    )
    if any(bp in lowered for bp in bad_phrases):
        return "unsafe"

    # Otherwise it is a risky occurrence -- requires a human eyeball.
    return "risky"


@dataclass(frozen=True)
class WordingHit:
    file: str
    line: int
    word_kind: str
    matched: str
    context: str
    classification: str


def scan_wording(latex_dir: Path = LATEX_DIR) -> List[WordingHit]:
    hits: List[WordingHit] = []
    for filename in LATEX_FILES:
        path = latex_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # The wording check section of b_claims_mapping.tex contains the
        # dangerous-word list itself; classify those literal matches as
        # ``safe`` since they are the *check*, not a claim.
        is_check_table = filename == "b_claims_mapping.tex"
        for line_no, line in enumerate(text.splitlines(), start=1):
            for kind, regex in _WORD_RE.items():
                for match in regex.finditer(line):
                    # Skip LaTeX command names like \prove, \section etc.
                    start = max(0, match.start() - 60)
                    end = min(len(line), match.end() + 60)
                    context = line[start:end]
                    if is_check_table:
                        classification = "safe"
                    else:
                        classification = _classify_wording(context, kind)
                    hits.append(
                        WordingHit(
                            file=str(path.relative_to(REPO_ROOT)),
                            line=line_no,
                            word_kind=kind,
                            matched=match.group(0),
                            context=context.strip(),
                            classification=classification,
                        )
                    )
    return hits


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------


def _render_risk_md(items: Iterable[RiskItem]) -> str:
    parts: List[str] = []
    for it in items:
        parts.append(f"- **Risk ID:** `{it.risk_id}`")
        parts.append(f"  - Severity: **{it.severity}**")
        parts.append(f"  - Dimension: `{it.dimension}`")
        parts.append(f"  - Location: {it.location}")
        parts.append(f"  - Risky wording or missing explanation: {it.risky_wording_or_missing}")
        parts.append(f"  - Why a reviewer may object: {it.why_reviewer_may_object}")
        parts.append(f"  - Recommended revision: {it.recommended_revision}")
        parts.append(
            "  - New experiment needed: "
            f"**{'yes' if it.new_experiment_needed else 'no'}** "
            f"; wording fix enough: **{'yes' if it.wording_fix_enough else 'no'}**"
        )
        parts.append(f"  - Priority: `{it.priority}`")
        parts.append("")
    return "\n".join(parts)


def _top_n_by_severity(items: Iterable[RiskItem], n: int = 10) -> List[RiskItem]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_items = sorted(items, key=lambda r: (order.get(r.severity, 9), r.risk_id))
    return sorted_items[:n]


def build_audit_markdown(
    items: Iterable[RiskItem],
    qas: Iterable[ReviewerQA],
    wording_counts: Dict[str, int],
    *,
    pytest_status: Optional[str] = None,
) -> str:
    items = list(items)
    qas = list(qas)
    severity_counts: Dict[str, int] = {}
    for it in items:
        severity_counts[it.severity] = severity_counts.get(it.severity, 0) + 1
    dim_counts: Dict[str, int] = {}
    for it in items:
        dim_counts[it.dimension] = dim_counts.get(it.dimension, 0) + 1

    out = io.StringIO()

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w("# Reviewer Risk Audit -- Stage 7.6c")
    w()
    w("_Static audit of paper_draft + paper_results. No experiment was run, "
      "no outputs/ or paper_results/ file was modified. Wording classification "
      "is lexical; risk-catalogue entries are encoded from the Stage 7.6c "
      "reading of the paper body._")
    w()

    w("## 1. Executive Summary")
    w()
    w(f"- Total risk items: **{len(items)}**.")
    w("- Severity distribution: "
      + ", ".join(f"`{k}`=**{v}**" for k, v in sorted(severity_counts.items())) + ".")
    w("- Dimension coverage: "
      + ", ".join(f"`{k}`={v}" for k, v in sorted(dim_counts.items())) + ".")
    w(f"- Simulated reviewer questions answered: "
      f"{sum(1 for q in qas if q.answer_status == 'answered')}/"
      f"{len(qas)} (partial: "
      f"{sum(1 for q in qas if q.answer_status == 'partial')}, "
      f"missing: {sum(1 for q in qas if q.answer_status == 'missing')}).")
    w(f"- Wording scan counts: "
      + ", ".join(f"`{k}`={v}" for k, v in sorted(wording_counts.items())) + ".")
    if pytest_status:
        w(f"- pytest status at audit time: {pytest_status}.")
    w()
    w("Bottom line: the paper body is heavily hedged. The remaining risks are "
      "wording overclaim, comparability framing (RQ13), local-emulation runtime "
      "framing (RQ7 / RQ12 / RQ14), and a few correctness-proof gaps. None of "
      "the P0 items requires a new experiment.")
    w()

    w("## 2. Top-10 Reviewer Risks")
    w()
    top10 = _top_n_by_severity(items, 10)
    for rank, it in enumerate(top10, start=1):
        w(f"### Rank {rank}: `{it.risk_id}` -- {it.severity} -- `{it.dimension}`")
        w()
        w(f"- Location: {it.location}")
        w(f"- Risky wording or missing explanation: {it.risky_wording_or_missing}")
        w(f"- Why a reviewer may object: {it.why_reviewer_may_object}")
        w(f"- Recommended revision: {it.recommended_revision}")
        w(f"- New experiment needed: **{'yes' if it.new_experiment_needed else 'no'}**; "
          f"wording fix enough: **{'yes' if it.wording_fix_enough else 'no'}**; "
          f"priority: `{it.priority}`")
        w()

    section_meta = (
        ("3. Novelty Risk", DIM_NOVELTY),
        ("4. Threat Model Risk", DIM_THREAT),
        ("5. Prior-Work Comparison Risk", DIM_PRIORWORK),
        ("6. Runtime Deployment Risk", DIM_RUNTIME),
        ("7. Security Claim Risk", DIM_SECURITY),
        ("8. Correctness Proof Risk", DIM_CORRECTNESS),
        ("9. Evaluation Sufficiency Risk", DIM_EVAL),
        ("10. Baseline Fairness Risk", DIM_BASELINE),
        ("11. Wording Risk", DIM_WORDING),
    )
    for title, dim in section_meta:
        w(f"## {title}")
        w()
        dim_items = [it for it in items if it.dimension == dim]
        if not dim_items:
            w("(no items recorded for this dimension)")
            w()
            continue
        w(_render_risk_md(dim_items))

    w("## 12. Simulated Reviewer Questions")
    w()
    for q in qas:
        w(f"### {q.qid}: {q.question}")
        w()
        w(f"- Answer status: **{q.answer_status}**")
        w(f"- Where answered: {q.where_answered}")
        w(f"- Answer summary: {q.answer_summary}")
        w()

    w("## 13. Revision Priority Plan")
    w()
    w("Detailed priority bands live in `paper_draft/revision_plan.md`. "
      "Counts:")
    prio_counts: Dict[str, int] = {}
    for it in items:
        prio_counts[it.priority] = prio_counts.get(it.priority, 0) + 1
    for prio in ("P0", "P1", "P2", "P3"):
        w(f"- `{prio}`: {prio_counts.get(prio, 0)}")
    w()
    w("No risk item requires a new experiment. Every P0 / P1 / P2 / P3 fix "
      "is a *wording or structural* change to the LaTeX body. Stage 7.6d "
      "(paper revision patch) will execute the P0 / P1 plan.")
    w()
    return out.getvalue()


def build_revision_plan_markdown(items: Iterable[RiskItem]) -> str:
    bands: Dict[str, List[RiskItem]] = {"P0": [], "P1": [], "P2": [], "P3": []}
    for it in items:
        bands.setdefault(it.priority, []).append(it)

    out = io.StringIO()

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w("# Revision Plan -- Stage 7.6c -> Stage 7.6d")
    w()
    w("_Priority bands for the paper revision patch. No new experiment is "
      "required at any band; every item is a wording or structural LaTeX edit._")
    w()
    band_titles = {
        "P0": "P0 -- must fix before submission",
        "P1": "P1 -- should fix before submission",
        "P2": "P2 -- nice to fix",
        "P3": "P3 -- appendix / artifact only",
    }
    for band, title in band_titles.items():
        w(f"## {title}")
        w()
        if not bands[band]:
            w("(no items)")
            w()
            continue
        for it in bands[band]:
            w(f"- `{it.risk_id}` ({it.severity}, `{it.dimension}`)")
            w(f"  - Location: {it.location}")
            w(f"  - Recommended revision: {it.recommended_revision}")
            w()
    return out.getvalue()


def build_topic_review_markdown(
    title: str,
    intro: str,
    items: Iterable[RiskItem],
    extra_sections: Iterable[Tuple[str, str]] = (),
) -> str:
    out = io.StringIO()

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w(f"# {title}")
    w()
    w(intro)
    w()
    items = list(items)
    if items:
        w("## Risk items")
        w()
        w(_render_risk_md(items))
    for section_title, body in extra_sections:
        w(f"## {section_title}")
        w()
        w(body)
        w()
    return out.getvalue()


def build_unsafe_wording_review(hits: Iterable[WordingHit]) -> str:
    hits = list(hits)
    counts: Dict[str, int] = {"safe": 0, "risky": 0, "unsafe": 0}
    by_word: Dict[str, Dict[str, int]] = {
        kind: {"safe": 0, "risky": 0, "unsafe": 0} for kind in _WORD_RE
    }
    for h in hits:
        counts[h.classification] = counts.get(h.classification, 0) + 1
        by_word.setdefault(h.word_kind, {"safe": 0, "risky": 0, "unsafe": 0})
        by_word[h.word_kind][h.classification] += 1

    out = io.StringIO()

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w("# Unsafe Wording Review")
    w()
    w("_Automated lexical scan of the LaTeX body. Each dangerous word "
      "occurrence is classified by the immediate context window:_")
    w("- `safe`  -- co-located with an explicit hedge "
      "(`we do not claim`, `not a real`, `proxy`, `local-emulation`, ...).")
    w("- `risky` -- not obviously safe; needs a human eyeball.")
    w("- `unsafe` -- contains a forbidden phrase (`is secure`, "
      "`provably secure`, `cryptographically secure`, `outperforms`, "
      "`real TEE wall-time`, `TEE-ready`, `GPU-ready`, `production-ready`, "
      "`full system reproduction`, ...).")
    w()
    w("Bound checks required by Stage 7.6c:")
    w("- `formal security`: any unsafe occurrence?")
    w("- `real TEE wall-time`: any unsafe occurrence?")
    w("- `full system reproduction`: any unsafe occurrence?")
    w()
    w(f"- Total occurrences: **{len(hits)}** "
      f"(safe={counts['safe']}, risky={counts['risky']}, "
      f"unsafe={counts['unsafe']}).")
    w()
    w("## By word kind")
    w()
    for word, cnts in sorted(by_word.items()):
        w(f"- `{word}`: safe={cnts['safe']}, "
          f"risky={cnts['risky']}, unsafe={cnts['unsafe']}.")
    w()

    for cls in ("unsafe", "risky", "safe"):
        cls_hits = [h for h in hits if h.classification == cls]
        w(f"## {cls.capitalize()} occurrences ({len(cls_hits)})")
        w()
        if not cls_hits:
            w("(none)")
            w()
            continue
        for h in cls_hits[:300]:
            w(f"- `{h.word_kind}` (`{h.matched}`) at "
              f"`{h.file}:{h.line}` -- {h.context!r}")
        if len(cls_hits) > 300:
            w(f"... ({len(cls_hits) - 300} more)")
        w()
    return out.getvalue()


def build_csv(items: Iterable[RiskItem]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "risk_id",
        "dimension",
        "severity",
        "priority",
        "location",
        "risky_wording_or_missing",
        "why_reviewer_may_object",
        "recommended_revision",
        "new_experiment_needed",
        "wording_fix_enough",
    ])
    for it in items:
        writer.writerow([
            it.risk_id,
            it.dimension,
            it.severity,
            it.priority,
            it.location,
            it.risky_wording_or_missing,
            it.why_reviewer_may_object,
            it.recommended_revision,
            "yes" if it.new_experiment_needed else "no",
            "yes" if it.wording_fix_enough else "no",
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Top-level runner.
# ---------------------------------------------------------------------------


@dataclass
class ReviewerRiskAuditResult:
    items: Tuple[RiskItem, ...]
    qas: Tuple[ReviewerQA, ...]
    wording_hits: Tuple[WordingHit, ...]
    output_paths: Dict[str, str]


def _topic_review_sections() -> Dict[str, Tuple[str, str, Tuple[str, ...]]]:
    """Map output filename -> (title, intro, dimensions)."""
    return {
        "novelty_positioning_review.md": (
            "Novelty Positioning Review",
            "Reviewer-risk pass focused on novelty positioning. Goal: ensure "
            "the four contributions (generation-compatible right masking, "
            "operator-compatible nonlinear islands, private LoRA training, "
            "deployable runtime API) are not dismissable as "
            "'Amulet plus LoRA' or as engineering plumbing.",
            (DIM_NOVELTY,),
        ),
        "threat_model_review.md": (
            "Threat Model Review",
            "Reviewer-risk pass focused on the system / threat model. Goal: "
            "ensure every leakage channel is named, every TEE / GPU / "
            "real-hardware boundary is co-located with its 'not deployed' "
            "qualifier, and that the deployable runtime API is enumerated "
            "in the out-of-scope list.",
            (DIM_THREAT,),
        ),
        "baseline_fairness_review.md": (
            "Baseline Fairness Review",
            "Reviewer-risk pass focused on the RQ13 prior-work comparison. "
            "Goal: ensure the comparison is framed as primitive-functional "
            "rather than as a security or runtime ranking, and that "
            "cryptographic baselines are explicitly complementary, not "
            "replaced.",
            (DIM_PRIORWORK, DIM_BASELINE),
        ),
        "evaluation_sufficiency_review.md": (
            "Evaluation Sufficiency Review",
            "Reviewer-risk pass focused on whether the evaluation is "
            "*sufficient* given the absence of full Qwen / TinyLlama / "
            "LLaMA fine-tunes and the synthetic-tile correctness path. "
            "Goal: ensure the synthetic-tile scope is restated in every "
            "RQ Result paragraph, not only in sec:limitations.",
            (DIM_EVAL,),
        ),
    }


def run_audit(write: bool = True) -> ReviewerRiskAuditResult:
    items = list(RISK_ITEMS)
    qas = list(REVIEWER_QAS)
    hits = scan_wording()

    wording_counts: Dict[str, int] = {}
    for h in hits:
        wording_counts[h.classification] = wording_counts.get(h.classification, 0) + 1

    audit_md = build_audit_markdown(items, qas, wording_counts)
    revision_md = build_revision_plan_markdown(items)
    unsafe_md = build_unsafe_wording_review(hits)

    output_paths: Dict[str, str] = {}

    if write:
        PAPER_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
        (PAPER_DRAFT_DIR / "reviewer_risk_audit.md").write_text(audit_md, encoding="utf-8")
        (PAPER_DRAFT_DIR / "revision_plan.md").write_text(revision_md, encoding="utf-8")
        (PAPER_DRAFT_DIR / "unsafe_wording_review.md").write_text(unsafe_md, encoding="utf-8")

        json_payload = {
            "stage": "7.6c",
            "items": [asdict(it) for it in items],
            "reviewer_qas": [asdict(q) for q in qas],
            "wording_summary": wording_counts,
            "wording_hits": [asdict(h) for h in hits],
            "outputs_modified": False,
            "paper_results_modified": False,
        }
        (PAPER_DRAFT_DIR / "reviewer_risk_audit.json").write_text(
            json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (PAPER_DRAFT_DIR / "reviewer_risk_audit.csv").write_text(
            build_csv(items), encoding="utf-8"
        )

        for filename, (title, intro, dims) in _topic_review_sections().items():
            subset = [it for it in items if it.dimension in dims]
            md = build_topic_review_markdown(title, intro, subset)
            (PAPER_DRAFT_DIR / filename).write_text(md, encoding="utf-8")

        output_paths = {
            "reviewer_risk_audit_md": str(PAPER_DRAFT_DIR / "reviewer_risk_audit.md"),
            "reviewer_risk_audit_json": str(PAPER_DRAFT_DIR / "reviewer_risk_audit.json"),
            "reviewer_risk_audit_csv": str(PAPER_DRAFT_DIR / "reviewer_risk_audit.csv"),
            "revision_plan_md": str(PAPER_DRAFT_DIR / "revision_plan.md"),
            "unsafe_wording_review_md": str(PAPER_DRAFT_DIR / "unsafe_wording_review.md"),
            "novelty_positioning_review_md": str(PAPER_DRAFT_DIR / "novelty_positioning_review.md"),
            "threat_model_review_md": str(PAPER_DRAFT_DIR / "threat_model_review.md"),
            "baseline_fairness_review_md": str(PAPER_DRAFT_DIR / "baseline_fairness_review.md"),
            "evaluation_sufficiency_review_md": str(PAPER_DRAFT_DIR / "evaluation_sufficiency_review.md"),
        }

    return ReviewerRiskAuditResult(
        items=tuple(items),
        qas=tuple(qas),
        wording_hits=tuple(hits),
        output_paths=output_paths,
    )


__all__ = [
    "RISK_ITEMS",
    "REVIEWER_QAS",
    "RiskItem",
    "ReviewerQA",
    "WordingHit",
    "ReviewerRiskAuditResult",
    "run_audit",
    "scan_wording",
    "build_audit_markdown",
    "build_revision_plan_markdown",
    "build_unsafe_wording_review",
    "build_csv",
    "build_topic_review_markdown",
]
