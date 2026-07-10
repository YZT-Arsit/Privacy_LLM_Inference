# Threat-Model Revision Patch — Public-Base → Private-Base Fine-Tuning

**Date:** 2026-07-10. **Status:** applied to the paper draft (both the LaTeX canonical
sections and the parallel Markdown draft). **Not committed.**

## Intent

Reposition the paper from a *public-base-weight* threat model to a
**private-base-weight fine-tuning** threat model:

- **Old:** "the base-model weights are public; we protect only the user's runtime
  data; model-weight extraction is out of scope."
- **New:** "the base-model weights are **proprietary/private and are not revealed to
  the untrusted GPU/host**; the protected assets are **private base weights, user
  fine-tuning data, LoRA adapter, gradients, KV cache, and hidden states**."

Only the *masked* weight `W_tilde = N_in^{-1} W N_out` ever crosses to the GPU; the
data plane already reflected this (`W_tilde`, not `W`, is what the GPU observes).
Under the new model, base-weight confidentiality reduces to **mask secrecy** and is a
**proxy claim, not a formal/cryptographic guarantee**. This is the setting under which
the public-weight cross-Gram / value-multiset recovery attacks (now reported as an
ablation) **do not succeed**, precisely because the attacker no longer holds plaintext
`W`.

## Phrase-level replacements (search targets → replacement)

| target phrase | replacement |
|---|---|
| "base-model weights are public" / "public base weights" | "base-model weights are proprietary/private and are not revealed to the untrusted GPU/host" |
| "we do not protect model weights" / "does not hide the base-model weights from the GPU" | "the protected assets are private base weights, user fine-tuning data, LoRA adapter, gradients, KV cache, and hidden states" |
| "model weight extraction is out of scope" | (removed) base-weight confidentiality is a **protected asset** under a mask-secrecy proxy claim; a *stronger* weight-recovery attacker than the ones we run is out of scope |
| "the (public) base weight `W`" (LoRA "never merged" clauses) | "the base weight `W`" (drop "public"; the never-merged claim stands) |
| "publicly computable from public `W`" (boundary-pad compensation) | "computed on the trusted side from the private `W` … only the masked-space term crosses to the GPU" |

## Files edited

### LaTeX (canonical build)
- `sections/03_system_and_threat_model.tex` — **Protected assets** now leads with the
  private-base-weight asset list and adds proprietary `W` (only `W_tilde` dispatched);
  **Allowed leakage** replaces "public base-model weights (assumed public)" with the
  *masked* `W_tilde` plus the mask-secrecy proxy caveat; summary table rows
  Holds/Protected/Allowed-leakage updated (`public W` → `masked W_tilde`; `W` added to
  Holds and Protected).
- `sections/01_introduction.tex` — "Even when the base model weights are public…" →
  "both the base model weights and the runtime data are private…"; "never merged into
  the public base weight" → "…the base weight".
- `sections/04_design.tex` — two boundary-pad "publicly computable from the public `W`"
  statements → "computed on the trusted side from the private `W`; only the masked-space
  term crosses to the GPU"; "never merged into the public base weight" → "…base weight".
- `sections/06_security_analysis.tex` — **new subsection**
  `sec:security:nonlinear` "Nonlinear-island structural leakage (cross-Gram and value
  multiset)" (see below).
- `sections/08_limitations.tex` — limitation "The base model weights are assumed
  public" fully rewritten to "Base-weight confidentiality is a mask-secrecy proxy claim,
  not a formal guarantee" (private-base framing + points to the ablation).
- `sections/09_related_work.tex`, `sections/00_abstract.tex` — "public base weight" →
  "base weight"; abstract adds one sentence that base weights are proprietary and only
  the masked form crosses the boundary.
- `sections/a_notation.tex` — `W` "the public base-model weight" → "the proprietary
  base-model weight (private; only the masked `W_tilde` is exposed to the GPU)".
- `figures/fig_system_overview.tex` — lateral node "public base weight `W`" →
  "masked base weight `W_tilde`"; comments updated (plaintext `W` never crosses to GPU).
- `figures/fig_lora_training.tex` — node "public `W`" (style `pub`) →
  "base `W` (private)" (style `trust`); caption "never merged into the public base
  weight `W`" → "…base weight `W`".

### Markdown draft (parallel)
- `system_and_threat_model.md` — Protected assets prepend private-base-weight asset
  list + proprietary `W`; Allowed leakage "public base-model weights (assumed public)"
  → masked `W_tilde` + proxy caveat.
- `introduction.md` — intro sentence and Contribution-3 "public base weight" updated.
- `design.md` — two boundary-pad "publicly computable from public `W`" statements →
  trusted-side/private framing; "never merged into the public base weight" → "…base
  weight".
- `security_analysis.md` — **new subsection 7.5b** (nonlinear structural leakage).
- `related_work.md`, `notation.md`, `limitations.md`, `figures.md` — "public base
  weight" / "public base-model weight" / "assumed public" updated as above.

### Not edited (historical review artifacts — flagged, superseded)
These contain the **old** recommended sentence ("Our threat model assumes the
base-model weights are public…") as review *recommendations*, not paper claims. They are
left as history but are now **superseded** and should not be acted on:
`reviewer_risk_audit.{md,json,csv}`, `revision_plan.md`, `novelty_positioning_review.md`.
Their recommendation (NOV-02) to add a "weights are public" sentence to the introduction
is now **inverted** by this patch. `threat_model_review.md` / `unsafe_wording_review.md`
also reference the old wording and should be re-run against the new threat model.

## New nonlinear-leakage subsection (both formats)

Added `Nonlinear-island structural leakage (cross-Gram and value multiset)`. Content
(backed by `results/attacks/nonlinear_cross_gram_attack_audit/`):

- The islands are exact but **not value-hiding**: the GPU sees the pre-activation up to
  a per-call channel permutation, so `Z_tilde G_Z_tilde^T = Z G_Z^T`, the per-token
  value multiset (sorted `Z_tilde` rows), and `||G_Z||` leak exactly.
- **Main setting (private base weights):** token reconstruction, target/label recovery,
  future-token prediction, and adapter/`ΔW` extraction **all fail** (token top-1 and
  future-token top-1 at the frequency baseline; `ΔW` cosine ≈ 0). **Membership inference
  AUC ≈ 0.61 is `medium` residual leakage** (model-agnostic gradient-norm signal, does
  not need `W`). The pure Gram matrix alone carries no vocabulary identity.
  Classification: **auxiliary structural leakage — exact, non-zero, NOT eliminated**, but
  did not recover user data/targets/adapter under the evaluated attackers.
- **Public-base ablation (negative setting):** the same leak is `high` — value-multiset
  matching against `sort(emb·W)` gives token recon top-1 ≈ 1.0, target AUC ≈ 0.82,
  future-token recovery, and full `ΔW` extraction cosine ≈ 1.0. Included only to motivate
  the private-base model.

## Downstream consistency items to revisit (not changed by this patch)

- **RQ13 / prior-work comparison** cites "Amulet static `P H Q`" etc. as public-weight
  obfuscation. Still valid as a *baseline description*; consider a sentence noting our
  model differs by keeping `W` private.
- **Security proxy summary table / risk matrix** (`tables/security_proxy_summary.tex`,
  `security_risk_matrix.png`) do not yet include the nonlinear cross-Gram rows; if the
  new subsection is kept, add its rows (reconstruction `needs_more_evaluation`,
  membership `medium`) and regenerate the figure.
- **Claims-mapping appendix** (`b_claims_mapping.tex`, `claims_mapping.md`) should get a
  row for the private-base threat-model statement and the nonlinear-leakage subsection.
- **`fig_system_overview` / `fig_lora_training`** are TikZ; verify they still compile and
  that the relabeled boxes read correctly against the legend.
