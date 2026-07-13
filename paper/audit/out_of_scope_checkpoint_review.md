# Out-of-Scope Checkpoint-Leak Review

**Decision (2026-07-13):** the AAAI paper presents **one** threat model — canonical **private-base** (proprietary checkpoint never disclosed to the adversary; runtime execution is the protection target; open-source checkpoints are experimental proxies only). Under this model, **checkpoint-leak / public-weight / worst-case-disclosure experiments are OUT OF SCOPE** and must not participate in the paper's argument. The artifacts remain in the repository as **archived exploratory work**; this review disconnects them from the paper.

**Disposition codes:** **(A)** remove · **(B)** archive (keep recorded but move out of the paper claim hierarchy) · **(C)** rewrite (re-scope to canonical) · **(D)** keep only as repository documentation (not referenced by the paper).

**IN-SCOPE, do NOT touch (false positives to protect):** the *algebraic* impossibilities are unrelated to checkpoint disclosure and stay —
- **C-COR-06b** RMSNorm exact-covariance forces `TTᵀ=cI` ⇒ per-token norm leakage (`docs/rmsnorm_exact_norm_impossibility.md`);
- **C-OPT-ADAMW-GPU** GPU dense-masked AdamW non-commutation ("impossibility");
- `sections/06_correctness.tex` RMSNorm dense-mask non-commutation.
These are correctness/leakage-of-invariants facts, **not** checkpoint-disclosure results.

---

## 1. `paper/` — canonical workspace (edited by Task 2)

| # | File | Section / anchor | Why currently referenced | Why out of canonical scope | Disposition |
|---|---|---|---|---|---|
| P1 | `main.tex` | title (L49) + header comment (L2) | Title reads "…for **Public-Weight** LLM Inference…" | Canonical model is private-base | **(C)** retitle to "…for **Private (Proprietary) Foundation-Model** Inference and LoRA Adaptation" |
| P2 | `claim_ledger.md` | resolution banner, register-3 bullet (L17) | Defines register 3 as a "participating ablation" | Register 3 is now out of paper scope | **(C)** rewrite banner: register 3 = archived, not part of the paper |
| P3 | `claim_ledger.md` | **C-SEC-PUBLIC-BREAK** (98.5% break) | Was the register-3 worst-case ablation | Requires checkpoint disclosure | **(B)** move to ledger "ARCHIVED — OUT OF PAPER SCOPE" appendix; drop from security narrative & risk list |
| P4 | `claim_ledger.md` | **C-SEC-IMPOSS** (impossibility) | "masks can't hide tokens once weights are public" | Arises only after checkpoint disclosure | **(B)** archive to same appendix |
| P5 | `claim_ledger.md` | **C-SEC-TEE-K1** (layer-0 TEE relocation) | Defense motivated by the public-weight break | Its sole motivation is the out-of-scope break; not needed for private-base argument | **(B)** archive to same appendix |
| P6 | `claim_ledger.md` | **C-SEC-BLOCKED** (arrowmatch worst-case row) | Lists blocked worst-case attacks | "arrowmatch worst-case" is checkpoint/weight-leak | **(C)** drop the worst-case item; keep the "setup-blocked ≠ defeated" methodological note |
| P7 | `claim_ledger.md` | Highest-risk item 2; status-count note "only formal-security-adjacent result is impossibility" | Framed security around the break/impossibility | Out of scope | **(C)** rewrite: security centers default private-base S1–S6; remove impossibility line |
| P8 | `audit/repository_evidence_audit.md` | CF-1 register-3 bullet (L19); CF-2 (L24–26); §3 (L96); security evidence-path list (L128) | Register separation kept the break as register 3 | Register 3 out of scope | **(C)** rewrite CF-2 to "out-of-scope, archived"; qualify §3; keep evidence paths but label `results/attacks/*` archived |
| P9 | `sections/01_abstract.tex` | plan L8, L19 ("public base LLM weights") | Setting stated as public-weight | Canonical private-base | **(C)** rewrite plan to private-base |
| P10 | `sections/02_introduction.tex` | central Q L13; problem L20–23; **C1** L40; risk L73 | Framed the paper around public-weight LLMs | Canonical private-base | **(C)** rewrite framing + C1; drop "public weights don't remove private assets" motivation, replace with proprietary-model-on-untrusted-accelerator |
| P11 | `sections/04_problem_threat_model.tex` | STOP-CONDITION block L7–20; TODO L75 | Presented public-vs-private as unresolved | Resolved: private-base only | **(C)** rewrite to a single canonical private-base threat model; remove the conflict block |
| P12 | `sections/07_security_analysis.tex` | central-finding block L11–20; §7.4(a) public-base 7B L42–45; `\evid{C-SEC-PUBLIC-BREAK, C-SEC-IMPOSS}` | Security section was built on the break + TEE relocation | Out of scope | **(C)** rewrite: security = leakage characterization (in-scope invariants) + default private-base S1–S6 empirical resistance; remove break/impossibility/relocation |
| P13 | `sections/08_evaluation.tex` | RQ2 L23 "public-base 98.5% break"; RQ3 L27 `C-SEC-IMPOSS` | Evaluation outline reported the break | Out of scope | **(C)** RQ2 = default private-base attackers only; RQ3 drop impossibility |
| P14 | `sections/09_limitations.tex` | **U11** L24–25 (no privacy under public-base + 98.5% + impossibility + layer-0 TEE) | Limitation stated in public-base terms | Out of scope | **(C)** replace U11 with in-scope limitation (security is proxy-only under the default attacker; residual invariants visible but not exploitable without the checkpoint) |
| P15 | `sections/10_conclusion.tex` | L14 ("mechanism under public weights; layer-0 relocation") | Honest posture referenced the break | Out of scope | **(C)** rewrite to private-base posture |
| P16 | `appendices/attack_details.tex` | L20 layer-0 relocation ablation; L23 arrowmatch worst-case | Attack appendix listed worst-case items | Out of scope | **(C)** move worst-case/relocation items to an "Archived (out of scope)" note; keep default-attacker configs |
| P17 | `audit/threat_model_reconciliation.md` | whole doc (records public→private reconciliation) | Documents how the framing was fixed | Still valuable as provenance | **(D)** keep as documentation; add a top pointer that register 3 is now fully out of paper scope (this review supersedes its "reframe as ablation" stance) |

## 2. `paper_draft/` — legacy draft tree (NOT edited; superseded by `paper/`)

| # | File | Anchor | Why referenced | Why out of scope | Disposition |
|---|---|---|---|---|---|
| D1 | `paper_draft/security_analysis.md` | §51 "Public-base ablation (negative setting)" | Included the break as an in-paper ablation | Out of scope | **(B)** archived; the canonical `sections/07` will not carry a public-base ablation |
| D2 | `paper_draft/threat_model_revision_patch.md` | L21, L91–110 ("reported as an ablation", RQ13 public-weight) | Migration record that still keeps the ablation | Superseded by full removal | **(D)** repo documentation; note it stops one step short (ablation → now fully out) |
| D3 | `paper_draft/reviewer_risk_audit.md` | L113–114, L449 (public-base-model risk; "public base-model weight W" in leakage list) | Reviewer risks phrased in public-base terms | Superseded | **(B)** archived; canonical reviewer risks live in `paper/claim_ledger.md` |
| D4 | `paper_draft/novelty_positioning_review.md` | L21–22 | Novelty framed on public-base-model distinction | Superseded (positioning flipped) | **(B)** archived |
| D5 | `paper_draft/limitations.md` | item 13 | Already private-base, but explicitly cites the "public-weight cross-Gram… ablation" | Mostly in-scope; only the ablation clause is out | **(C)** when imported to `sections/09`, drop the trailing ablation clause |
| D6 | `paper_draft/introduction.md`, `system_and_threat_model.md` | private-base body | These are the CANONICAL sources | In scope | **keep** (authority for rewrite) |

## 3. `results/attacks/` and related — experiment artifacts (NEVER deleted; disconnected)

| # | Artifact | Why it exists | Why out of paper scope | Disposition |
|---|---|---|---|---|
| R1 | `results/attacks/FINDINGS.md`, `THREAT_MODEL_AND_THEORY.md`, `ours_gram/`, `ours_pad/`, `real_attention_fingerprint_qwen7b.json`, `task2_layer0_guardrail_result.json`, `*_qwen7b.json` | Public-weight break + impossibility + layer-0 relocation defense | All assume checkpoint disclosure | **(D)** keep as archived exploratory repository documentation; **remove from paper evidence lists** |
| R2 | `outputs/attacks/*` public-model rows (`nn_embedding_inversion`, `kpa` public columns, `arrowmatch` worst-case, `multiset_permutation_leakage` under public model) | Unified attack matrix incl. worst-case columns | `attack_definitions.md` labels these `public_model / worst_case` | **(D)** keep; the paper's evaluation (RQ2) cites only the **default-attacker** rows |
| R3 | `results/aaai_private_base/security/S1..S6/*` | Default private-base attacker suite | **IN SCOPE** — this is the canonical security evidence | **keep + promote** (C-SEC-S1..S6) |

## 4. `paper_results/` — generated tables (NOT regenerated here; deferred)

| # | Table | Issue | Disposition |
|---|---|---|---|
| T1 | `paper_results/markdown/security_claims_table.md` | Checked: contains T-ledger private-base claims only; **no** checkpoint/break rows | **keep as-is** (no out-of-scope content) |
| T2 | `paper_results/csv/direct_prior_work_comparison.*` (RQ13) | Contains public-weight prior-work framing per `threat_model_revision_patch.md` L110 | **(C)** regenerate without public-vs-private weight framing — **deferred** (would regenerate an artifact; not done now, flagged) |
| T3 | `paper_results/csv/security_proxy_summary.*` | Private-base proxy metrics; no checkpoint-break rows | **keep** |

---

## 5. Summary of dispositions
- **(A) remove:** none (nothing is deleted — artifacts are preserved per hard constraint).
- **(B) archive (recorded, out of paper hierarchy):** C-SEC-PUBLIC-BREAK, C-SEC-IMPOSS, C-SEC-TEE-K1 (paper/claim_ledger); D1/D3/D4 (paper_draft).
- **(C) rewrite (re-scope to canonical):** P1–P16 (paper/ ledger, audit, sections, appendix); D5 on import; T2 regenerate (deferred).
- **(D) keep as repository documentation:** P17 (reconciliation), D2, all of §3 `results/attacks/*` + worst-case `outputs/attacks/*` rows.

The single load-bearing consequence for the argument: **the security section loses its strongest (negative) result and the TEE-relocation defense**; it must stand on the default private-base leakage characterization + S1–S6 proxy resistance. See `claim_ledger.md` §"Paper claim hierarchy" and §6 below.

---

## 6. Reviewer audit — "if all checkpoint-leak material disappeared, would any contribution become unsupported?"

**Answer: NO contribution becomes fully unsupported. Exactly ONE (C3) must be REWRITTEN (re-scoped + strength downgraded).**

| Contribution | Support after pruning | Verdict |
|---|---|---|
| **C1** Masked-execution correctness (private decoder LLM) | C-COR-01..08, C-COR-GEN, C-COR-06b, C-THREAT-*, C-BND-* — all algebraic/experimental identities on masked execution, independent of whether `W` is public or private | **Intact.** No dependency on checkpoint-leak. |
| **C2** Masked LoRA + optimizer boundary | C-LORA-*, C-OPT-* — masked forward/backward/optimizer identities | **Intact.** Checkpoint-independent. |
| **C3** Leakage characterization + empirical resistance | C-SEC-LEAK, C-SEC-PIINV, C-SEC-S1..S6, C-SEC-UNIFIED (default rows), C-SEC-TIMING, C-NONCLAIM-06 | **Supported but MUST BE REWRITTEN.** Still has evidence (leakage characterization = algebraic+validated; S1–S6 = proxy on real Qwen-0.5B secret oracle). But it loses the strongest (negative) result + the concrete TEE-relocation defense; its **claimed strength drops to proxy-only**. Re-word: "we characterize residual leakage and empirically evaluate resistance under the DEFAULT private-base attacker; residual invariants remain visible but are not exploitable without the checkpoint; confidentiality is a mask-secrecy proxy, not formal or worst-case." |
| **C4** Cost + real-model validation | C-EFF-*, C-REAL-* | **Intact.** Checkpoint-independent. |

**New reviewer risks introduced by the pruning (must be managed in the write-up):**
1. **Security softness (highest).** The paper's most rigorous security artifact was the *negative* result (the break + impossibility); removing it leaves a **proxy-only** security claim whose real backstop is "the checkpoint stays private" (an assumption) + mask secrecy. Present honestly; do not imply a guarantee.
2. **Smaller-scale in-scope security evaluation.** The removed break was on real Qwen-7B; the in-scope S1–S6 is on Qwen-0.5B (secret oracle, CPU). A reviewer may note the in-scope security evaluation is smaller-scale. Consider (future work) running the default-attacker suite at larger scale.
3. **"Why obfuscate if security is proxy-only?"** Motivation must now lean on correctness + LoRA-training coverage + cost + the dual-sided (weights+data) coverage gap vs ObfuscaTune/STIP — NOT on a security guarantee.
4. **Base-weight confidentiality is a proxy claim** resting on mask secrecy; a weight-recovery attacker under the default model is out of scope (state as a limitation, not a defeated attack).
5. **Coherence check passes:** with C3 re-scoped, the paper presents ONE threat model; no section, figure, table, or claim depends on checkpoint disclosure.

**Conclusion:** the paper remains **logically complete** under the canonical private-base threat model. Only **C3 (security)** requires a rewrite — a re-scope to the default attacker + an explicit proxy-only calibration — not new experiments.
