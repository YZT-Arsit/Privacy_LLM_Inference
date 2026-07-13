# Threat-Model Reconciliation Report

> **⛔ SUPERSEDED-IN-PART (2026-07-13):** this report classified the checkpoint-leak / public-weight material as a *worst-case ablation to keep, clearly labeled*. That stance is now superseded: **register 3 is fully OUT OF PAPER SCOPE (archived exploratory work)** and is removed from the paper's argument. See `out_of_scope_checkpoint_review.md` and `claim_ledger.md` (archived appendix). This document is retained as provenance for how the public→private framing was fixed.


**Purpose.** Resolve the public-vs-private base-weight conflict (`repository_evidence_audit.md` CF-1) now that the canonical deployment model is fixed. This report classifies every public-weight-implying statement in the repository by **origin** and by **whether it belongs to the threat model or only to the experimental implementation**, and recommends canonical wording. Per instruction, **no paper section is modified here** — this is an audit deliverable only.

**Date:** 2026-07-13. **Scope:** read-only classification. **Method:** repo-wide grep for public-weight language + direct reading of the pivotal threat-model, attack-definition, and evaluation-artifact files.

---

## 0. Canonical threat model (FIXED)

1. The foundation-model weights are **PRIVATE**; the model provider owns the pretrained checkpoint.
2. The attacker **never receives the plaintext checkpoint**. Only the masked weight `W̃ = N_in⁻¹ W N_out` ever crosses to the untrusted GPU/host.
3. The trusted component protects **both** the model parameters and the user's private assets (prompt, hidden states, KV cache, LoRA adapter, gradients, optimizer state).
4. Open-source checkpoints (Qwen, Llama) are used **only as experimental proxies** because proprietary production checkpoints are unavailable.

**Governing principle for this reconciliation:**
> Using an open-source model for experiments is an **evaluation-artifact** fact, NOT a threat-model assumption. A statement that the *experiment* loads a public Qwen checkpoint does **not** imply the *deployment* assumes public weights. These two must be kept in separate registers throughout the paper.

**Canonical sources already written** (use as the wording basis — do NOT re-derive):
- `paper_draft/system_and_threat_model.md` (already private-base: "base-model weights are proprietary/private … only the masked form `W̃` is ever dispatched").
- `paper_draft/threat_model_revision_patch.md` (provides an explicit **old→new phrase table**, reproduced in §3 below).
- `results/aaai_private_base/security/security_registry.yaml` (FROZEN private-base scenario: `from_scratch_private_base_model`; hard rule "attacker side never receives a public Qwen checkpoint").

---

## 1. Classification of every public-weight-implying statement

Origin codes: **[H]** historical draft · **[C]** implementation convenience · **[E]** evaluation artifact (incl. worst-case attacker ablation) · **[A]** abandoned assumption. Register: **TM** = purports to be part of the threat model · **EXP** = only about the experimental implementation.

| # | Location (quote / gist) | Origin | Register | Verdict & canonical action |
|---|---|---|---|---|
| S1 | `docs/paper_draft/00_paper_positioning.md`: "privacy-preserving inference for **public-weight** decoder-only LLMs. The public base weights are **not** the protection target." (also contribution **C1** "public-weight decoder-only LLMs") | **[H][A]** | TM | **Superseded.** This is the older `docs/paper_draft/` tree that adopted public-weight as the threat model, later abandoned by the top-level `paper_draft/` revision. Rewrite per §3; base weights ARE a protection target. |
| S2 | `docs/paper_draft/02_system_and_threat_model.md`: L7–8 "Holds the **public base model** weights … *weight values* are public. The base model's weights are **not** the protection target"; L16 "Base model weight values … are public"; L41 "**Public base model weights.** Not a protection target." | **[H][A]** | TM | **Superseded historical draft.** Direct contradiction of canonical. The newer `paper_draft/system_and_threat_model.md` already replaces it (private-base). Do not cite `docs/paper_draft/02` in the paper; treat `paper_draft/system_and_threat_model.md` as authoritative. |
| S3 | `docs/paper_draft/01_introduction.md` (public-weight framing consistent with S1/S2) | **[H][A]** | TM | **Superseded.** Same tree as S1/S2. Introduction was re-written private-base in `paper_draft/introduction.md`. |
| S4 | `results/attacks/THREAT_MODEL_AND_THEORY.md` §1: "architecture and *public* pretrained weights are known to everyone … folds the **exact public weights** (no fine-tuning) with secret masks." Drives the 98.5% real-Qwen-7B break + impossibility theorem. | **[E]** | TM (of the *ablation*) | **Reframe, do NOT delete.** This is a deliberately **worst-case attacker** analysis: "if the attacker held the plaintext checkpoint, structure-preserving masks cannot hide tokens." Under canonical private-base this is an **ablation / upper-bound**, not the main threat model. Report it as: *worst-case where the private checkpoint is assumed leaked*, motivating (a) that base-weight confidentiality is load-bearing and (b) the layer-0 TEE relocation. Explicitly label "public-weight ablation." |
| S5 | `docs/attack_definitions.md`: threat-model column `public_model / worst_case`, `public_model`; and L61 "`model_weights={public,obfuscated}` → **weight_leakage_worst_case, NOT the default threat model**." | **[E]** | EXP (attacker capability) | **Consistent — keep, relabel.** The artifact already calls the public-weight setting *worst-case, not default*. Under canonical, `public_model` attacks are the **worst-case row**; the **default** attacker gets only the transformed package. Keep the ablation; ensure paper labels the `public_model` column "worst-case (checkpoint-leak) ablation." |
| S6 | `results/attacks/THREAT_MODEL_AND_THEORY.md` L137–175: contrasts ObfuscaTune ("*proprietary/secret* model + private data — a **different** threat model") vs "ours = **open-weight** (protects user tokens)." | **[H]** | TM (positioning) | **Positioning FLIPS.** Under canonical private-base, OUR threat model now **matches** ObfuscaTune's dual-sided (weights+data hidden) family — ours is no longer "open-weight." Related-work must be rewritten: the axis of novelty is no longer "we protect user tokens while they protect weights," but generative-LLM + LoRA-training coverage, KV/RoPE/GQA handling, and the accelerator-side masked path. (See `citation_inventory.md` obfuscatune row.) |
| S7 | `docs/cross_machine_security_scope.md`: L58 "(`build_qwen7b_folded_package.py`, runs in **TDX**) reads the **public base** model"; L45–54 masked cross-machine decode. | **[C]** | EXP | **Not a threat-model leak.** The folding step reads the checkpoint **inside the trusted TDX domain** to build `W̃`; the untrusted worker only ever sees masked tables. "public base model" here = the open-source proxy checkpoint being folded trusted-side. Reword to "the trusted builder reads the (proprietary, here open-source-proxy) base model and emits only masked tables." |
| S8 | `results/real_qwen_tdx_audit/qwen05b/sgd_mode/summary.md`: "**Base weights plaintext on GPU** (final private-base threat model **NOT claimed**)." | **[C][E]** | EXP | **Honest evaluation caveat — preserve verbatim.** This real feasibility run kept base weights plaintext on the GPU for engineering simplicity; it self-declares that the private-base model is not exercised. This is exactly the artifact/threat-model separation: cite it as a **limitation of the real-hardware run**, not as a deployment assumption. Do NOT let it read as "we deploy with public weights." |
| S9 | `outputs/lora/*.md` (many): "LoRA adapter is **NEVER merged into the public base weight `W`**." | **[H]** | EXP (correctness boilerplate) | **Low-severity stale adjective.** The *never-merged* claim stands; "public" is a leftover modifier on `W`. Per revision-patch row (see §3), drop "public": "never merged into the base weight `W`." These are generated report artifacts; the paper should simply use the corrected phrasing. |
| S10 | `results/attacks/FINDINGS.md`, `public_weight_prompt_audit/*`, `ours_gram/`, `ours_pad/`, cross-gram "public" column | **[E]** | TM (of the ablation) | **Same as S4.** These are the executed public-weight worst-case attacks (real Qwen-7B). Keep as the worst-case/ablation evidence bundle; label public-weight; the *default* private-base results are the S1–S6 suite. |
| S11 | `docs/attack_evaluation_plan.md`, `docs/attack_result_schema.md`, `docs/STAGE_ARTIFACTS.md`, `README.md` (public-model references) | **[H]/[E]** | mixed | **Update pointers.** Planning/schema docs that still name a `public_model` default should mark it worst-case; README positioning line should adopt private-base. Low priority (not paper prose). |

**Statements already canonical (private-base) — no change, use as authority:**
`paper_draft/system_and_threat_model.md`, `paper_draft/introduction.md`, `paper_draft/limitations.md` (item 13), `paper_draft/threat_model_revision_patch.md`, `security_registry.yaml` (`from_scratch_private_base_model`; `attacker_does_not_know: [plaintext_W, original_checkpoint, masks_and_inverses]`), `results/aaai_private_base/claim_experiment_map.md`.

**Still-stale audits (public-weight wording, flagged superseded):** `paper_draft/reviewer_risk_audit.{md,csv,json}`, `paper_draft/novelty_positioning_review.md`, `paper_draft/threat_model_review.md`, `paper_draft/revision_plan.md`. Recommendation **NOV-02** ("add a 'weights are public' sentence") is **INVERTED** and must not be actioned.

---

## 2. Where public-weight belongs after reconciliation

| Register | What stays "public/open" | What becomes private |
|---|---|---|
| **Threat model (deployment)** | architecture, hyperparameters (`L,d,h,h_kv,d_h,V,intermediate_size`), KV-cache *length*, sliding-window size, output tokens/length, the fixed chat template, tensor shapes, `r_pad` | **base weights `W`** (only `W̃` crosses), prompt, hidden states, KV *values*, LoRA `A,B`, gradients, optimizer state |
| **Experimental implementation (proxy)** | the *open-source Qwen/Llama checkpoint used as a stand-in* for an unavailable proprietary model; the fact that a feasibility run kept base weights plaintext on the GPU (S8) | — (these are artifact facts, not deployment assumptions) |
| **Worst-case ablation** | the attacker is *granted* the plaintext checkpoint (S4/S5/S10) — used to prove the impossibility bound and motivate the TEE relocation | this is explicitly a stress test, labeled "public-weight ablation," NOT the default |

**The three registers must never be collapsed.** In particular: the 98.5% real-Qwen-7B recovery (S4/S10) is a *worst-case-ablation* result (attacker given the checkpoint), the S1–S6 suite is the *default private-base* result, and "we ran on Qwen-0.5B/7B" is an *experimental-proxy* fact.

---

## 3. Recommended canonical wording (from `threat_model_revision_patch.md`, endorsed)

Apply this substitution table when sections are eventually authored (do NOT apply yet):

| Old (public-weight) phrasing | Canonical (private-base) phrasing |
|---|---|
| "the base-model weights are public" / "public base weights" | "the base-model weights are **proprietary/private and are not revealed to the untrusted GPU/host**" |
| "we do not protect model weights" / "does not hide the base-model weights from the GPU" | "the protected assets are **private base weights, user fine-tuning data, LoRA adapter, gradients, KV cache, and hidden states**" |
| "model weight extraction is out of scope" | "base-weight confidentiality is a **protected asset** under a **mask-secrecy proxy** claim; a *stronger* weight-recovery attacker than the ones we evaluate is out of scope" |
| "the (public) base weight `W`" (LoRA never-merged clauses) | "the base weight `W`" (drop "public"; the never-merged claim stands) |
| "publicly computable from public `W`" (boundary-pad compensation) | "computed on the **trusted side** from the private `W`; only the masked-space term `T W N_out` crosses to the GPU" |
| "public-weight decoder-only LLM" (positioning / contribution C1) | "**private-weight** (provider-proprietary) decoder-only LLM served on an untrusted accelerator" |
| "ours = open-weight; ObfuscaTune = proprietary" (related work) | "ours and ObfuscaTune share a **dual-sided** (weights **and** data protected) model; we differ on generative-LLM + LoRA-training coverage, accelerator-side masked nonlinear handling, and KV/RoPE/GQA support" |

**Mandatory experimental-proxy disclaimer** (add once, in Evaluation / Setup, and cross-reference from the threat model):
> "Because production proprietary checkpoints are unavailable, all experiments use open-source checkpoints (Qwen2.5-0.5B/7B, etc.) **as proxies for a private provider model**. Their public availability is an artifact of our evaluation, not an assumption of the threat model; the deployment treats the base weights as private (only `W̃` crosses the boundary). Where a run keeps base weights plaintext on the GPU for engineering feasibility (e.g. the real-Qwen-0.5B TDX run), we say so explicitly and do **not** claim the private-base model for that run (see Limitations)."

---

## 4. Consequential downstream changes (for the eventual writing pass)

1. **C-THREAT-02 / CF-1 status → RESOLVED (private-base).** Update `claim_ledger.md` C-THREAT-02 and `repository_evidence_audit.md` CF-1 from "unresolved conflict" to "resolved: private-base canonical; public-weight = worst-case ablation + experimental proxy." (Recommended follow-up; not done here to respect the no-edit instruction — though these are audit artifacts, not paper sections.)
2. **Security section (`07_security_analysis.tex`).** Reframe C-SEC-PUBLIC-BREAK (98.5% break) as the **worst-case checkpoint-leak ablation**, and center the **default** private-base S1–S6 results + the layer-0 TEE relocation. The impossibility theorem (C-SEC-IMPOSS) becomes: "structure-preserving masks alone cannot substitute for keeping `W` private + the TEE relocation."
3. **Related work (`03_related_work.tex`).** Reposition vs ObfuscaTune/STIP as **same (dual-sided) threat family**, differentiating on coverage — NOT on "who protects weights." Removes the NOV-02 recommendation.
4. **Base-weight confidentiality is a PROXY claim.** Everywhere the paper protects `W`, it must say confidentiality "reduces to mask secrecy and is a proxy claim, not formal" (per `paper_draft/system_and_threat_model.md` L48). Do not upgrade to a formal/cryptographic weight-secrecy guarantee.
5. **Real-hardware caveat (S8).** The Qwen-0.5B TDX run kept base weights plaintext on the GPU — must be stated as a limitation of that run; it does not demonstrate the private-base data plane end-to-end.

---

## 5. Summary

- **Zero canonical statements need inventing** — the private-base wording already exists (`paper_draft/system_and_threat_model.md`, revision patch, `security_registry.yaml`).
- **No public-weight statement is silently rewritten.** Each is classified: S1–S3 (historical/abandoned drafts) are superseded; S4/S5/S10 (evaluation artifacts) are re-labeled *worst-case ablation*; S7/S8/S9 (implementation convenience) are experimental-proxy facts; S6 (positioning) flips the related-work axis.
- **The decisive separation** the paper must maintain: *private-base deployment* (threat model) ≠ *open-source proxy checkpoint* (evaluation artifact) ≠ *checkpoint-leak worst case* (ablation).
- **Next step (on your go-ahead):** author `04_problem_threat_model.tex` from the canonical sources, then propagate the §3 wording and the §4 consequences into Security and Related Work.
