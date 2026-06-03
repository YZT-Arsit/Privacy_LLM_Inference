# Compile notes — Stage 7.6b LaTeX draft

This file tracks the items that still need a human pass before the draft becomes a submission. It is intentionally explicit so a co-author can scan it in two minutes.

## 1. TODO citations — verify or replace before submission

The following bib keys in `refs.bib` are flagged with `TODO` comments or have at least one unverified field (typically venue or exact title). They render and compile, but should be corrected before submission:

- `amulet2024todo` — Placeholder for the Amulet-style matrix-obfuscation paper. Author, title, venue, and year all need to be filled in once the exact reference is confirmed. Currently cited from [`sections/01_introduction.tex`](sections/01_introduction.tex) and [`sections/09_related_work.tex`](sections/09_related_work.tex).
- `kunkel2019tensorscone` — Cited as `arXiv:1902.04413`; venue (if any other than arXiv) needs verification.
- `gilad2016cryptonets` — Kept as a duplicate-key alias of `dowlin2016cryptonets`; pick one and remove the other once the project's citation key convention is finalised.

## 2. Figures still to hand-finish

The five TikZ schematics in [`figures/`](figures/) are functional but minimal. A graphics editor pass is recommended before submission to:

- Align panels and adjust whitespace.
- Replace plain rectangles with venue-style box shapes if the target conference has a house style.
- Replace dashed compensation arrows with a more readable annotation.
- Add a small icon set (lock for trusted side, GPU for the untrusted side, etc.) if desired.

Specific figures:

- `fig_system_overview.tex` (Figure 1) — the top-row item boxes are tightly packed; consider spreading them across two rows for clarity.
- `fig_right_masked_decode.tex` (Figure 2) — the right-hand annotation box could be visually attached to the cache rows by a brace.
- `fig_nonlinear_island.tex` (Figure 3) — three panels share a single caption; consider `\begin{subfigure}` if the target template prefers per-panel sub-captions.
- `fig_dense_sandwich.tex` (Figure 4) — the compensation arrow is dashed and originates from a floating box; an enclosing dashed rectangle for the trusted region would make the boundary explicit.
- `fig_lora_training.tex` (Figure 5) — dense; may benefit from being split into a forward subfigure and a backward subfigure when in the camera-ready phase.

The seven existing `.png` figures from the artifact pipeline (security risk matrix, measured runtime summary, etc.) are usable as-is at draft quality; for camera-ready they should be regenerated at a higher DPI or converted to PDF for vector rendering.

## 3. Tables that may overflow

The following tables come straight from the artifact pipeline (`paper_results/latex/`). Some have many columns (8–10) and may overflow at single-column conference widths:

- `tables/correctness_summary.tex` — 9 columns, 19 rows. Wrap with `\resizebox{\linewidth}{!}{...}` or push to landscape/appendix.
- `tables/security_proxy_summary.tex` — 10 columns, 14 rows. Same recommendation.
- `tables/workload_summary.tex` — 12 columns, 6 rows. May need `\resizebox` or horizontal abbreviation of column names.
- `tables/lora_training_summary.tex` — 13 columns, 11 rows. Same.
- `tables/measured_runtime.tex` — 14 columns, 7 rows. Same.
- `tables/paper_claims_audit.tex` — multi-row claim entries; consider moving to appendix `\input{tables/paper_claims_audit.tex}` only.

Recommendation: keep the human-readable headline tables in the body (Table 3 correctness, Table 4 workload, Table 5 security) and push the long ones to the appendix.

## 4. Unsafe wording to re-check by a co-author

[`unsafe_wording_check.md`](unsafe_wording_check.md) records every match for the unsafe-wording word list across the LaTeX sources. Every hit is currently inside a disclaimer, a claims-mapping table cell, or a Related Work comparison; no positive claim leaks. Before submission a human should re-scan this list to confirm no new positive use of these phrases has been introduced.

## 5. Target conference template choices

The placeholder is `\documentclass[10pt]{article}`. Once the target venue is decided, switch to one of:

- **IEEE S&P** — `\documentclass[conference]{IEEEtran}`. Use `IEEEtran.bst` for the bibliography.
- **USENIX Security** — usenix camera-ready style file; the conference provides a sample.
- **ACM CCS** — `\documentclass[sigconf]{acmart}`. Bibliography style is `ACM-Reference-Format`.
- **NDSS** — Internet Society template; uses a custom `\documentclass{ndss}`.

Each template has different rules for `\title`, `\author`, `\section` numbering, and table widths; the table-overflow note above interacts with this choice.

## 6. Stage-internal references to scrub

Section [`b_claims_mapping.tex`](sections/b_claims_mapping.tex) references `paper_results/markdown/paper_claims_audit.md`. This filename is internal to our project and should be replaced by an appendix reference in the camera-ready (e.g., ``the claims audit reproduced in this appendix''). Search for any other `paper_results/` and `outputs/` paths in the body before submission.

## 7. Plain-text dump checklist

Before submitting:

- [ ] Verify Figures 1--5 render with the chosen template's text width.
- [ ] Confirm every `\cite{}` resolves; replace `TODO:` bib entries.
- [ ] Confirm every `\Cref{}` resolves.
- [ ] Re-run the unsafe-wording grep (`unsafe_wording_check.md`).
- [ ] Re-confirm the artifact `.tex` tables still reflect the latest `paper_results/` output.
- [ ] Replace the placeholder author and affiliation.
- [ ] Check that the `\bibliographystyle` matches the venue.
