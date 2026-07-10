# Private-Base Threat-Model Update

Repositions the paper from **public base weights** to **private base weights /
private fine-tuning**, and updates the nonlinear-leakage section accordingly.
Applied to the paper draft (LaTeX + Markdown); **not committed**. The full
old→new change list is in `paper_draft/threat_model_revision_patch.md`.

## The change

| | old (public-base) | new (private-base) |
|---|---|---|
| base weights | public; extraction out of scope | **proprietary/private; not revealed to the untrusted GPU/host** (only `W_tilde = N_in^{-1} W N_out` crosses) |
| protected assets | user runtime data only | **private base weights, user fine-tuning data, LoRA adapter, gradients, KV cache, hidden states** |
| weight confidentiality | n/a (conceded public) | **mask-secrecy proxy claim**, not formal/cryptographic |
| public-base attack results | (implicit main setting) | reported as an **ablation / negative setting** |

Under the private-base model the GPU never holds plaintext `W`, so it cannot map an
observed value multiset to a vocabulary item, solve the per-call channel permutation,
or fit a surrogate adapter — which is exactly why the recovery attacks fail here.

## Nonlinear leakage, re-scoped by regime

Evidence: `results/attacks/nonlinear_cross_gram_attack_audit/` (five-attacker panel,
mean over seeds 0/1/2). The islands are exact but **not value-hiding**:
`Z_tilde G_Z_tilde^T = Z G_Z^T`, the per-token value multiset, and `||G_Z||` leak
exactly.

| attack | private base (MAIN) | public base (ABLATION) |
|---|---|---|
| token reconstruction | **fail** — top-1 → frequency baseline (identity needs `W`; only token *linking* survives) | **high** — top-1 ≈ 1.0 via `sort(emb·W)` matching |
| target / private label | **fail** — top-1 ≈ base rate, AUC ≈ 0.62 marginal | high — recovered-token BoW, AUC ≈ 0.82 |
| future-token | **fail** — = bigram baseline | high — future rows recovered (single-forward) |
| adapter / ΔW extraction | **fail** — ΔW cosine ≈ 0 | high — cosine ≈ 1.0 (recovered global perm → lstsq ΔW) |
| membership inference | **medium — AUC ≈ 0.61** (gradient-norm; model-agnostic, survives private weights) | medium — same signal |

**Gram alone** (without the co-leaked value multiset) carries no vocabulary identity:
token top-1 stays at baseline. The danger is the **value multiset in combination with
plaintext weights**, not the Gram matrix in isolation.

## Classification (what the paper is now allowed to say)

- Under **private base weights**: the nonlinear cross-Gram / value multiset is
  **auxiliary structural leakage** — exact, non-zero, and **not eliminated**, but under
  the evaluated attackers it did **not** recover user tokens, targets, or a functional
  adapter. The one residual channel is **membership inference (medium)**.
- The **public-base** results are an **ablation** showing the same leak becomes **high**
  once the attacker holds plaintext `W`; this is what motivates the private-base model.

### Disallowed (kept out of the revised text)
- "No leakage exists." / "cross-Gram is harmless." — the leak is exact and measured;
  membership is a real medium channel.
- "Formal / cryptographic privacy proven." — weight confidentiality is a **mask-secrecy
  proxy claim**; the security section still disclaims formal indistinguishability.
- "Adapter protection proven." — adapter extraction was *run*; it fails only under
  private weights and **succeeds under public weights** (the ablation).

## Files touched (summary; details in the patch)

- **Threat model** (`03_system_and_threat_model.tex`, `system_and_threat_model.md`):
  Protected assets + Allowed leakage + summary table.
- **Intro / abstract / notation / design / related work** (LaTeX + MD): drop "public
  base weight" → "base weight"; base weights stated proprietary; boundary-pad
  compensation reworded to trusted-side/private.
- **Limitations** (`08_limitations.tex`, `limitations.md`): "weights assumed public" →
  "base-weight confidentiality is a mask-secrecy proxy claim".
- **Security analysis** (`06_security_analysis.tex`, `security_analysis.md`): new
  nonlinear-island structural-leakage subsection (private-main + public-ablation).
- **Figures** (`fig_system_overview.tex`, `fig_lora_training.tex`): "public `W`" →
  masked/private forms.
- **Not edited:** historical review artifacts (`reviewer_risk_audit.*`,
  `revision_plan.md`, `novelty_positioning_review.md`) still carry the old
  recommendation — flagged as superseded in the patch.

## Follow-ups (flagged, not done here)
Regenerate `security_proxy_summary` table + risk-matrix figure to add the nonlinear
rows; add a claims-mapping row for the private-base statement; verify the two TikZ
figures still compile; consider a one-line note in RQ13 that our model keeps `W`
private (unlike the public-weight obfuscation baselines).
