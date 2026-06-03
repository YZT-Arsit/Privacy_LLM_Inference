# Build instructions — `paper_draft/latex`

This directory contains the LaTeX source of the paper draft generated in Stage 7.6b.

## Compile

From this directory:

```
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

`cleveref` references resolve after the second `pdflatex` pass; bibliography keys after the `bibtex` pass; cross-references stabilise after the third pass.

Output: `main.pdf`.

## Required packages

The preamble in [`main.tex`](main.tex) loads:

- `amsmath`, `amssymb`, `amsthm` — equations and theorem environments.
- `booktabs`, `multirow`, `tabularx` — tables (the auto-generated ones in `tables/` use `\toprule/\midrule/\bottomrule`).
- `graphicx` — `.png` figures from `paper_results/figures/` (copied into `figures/`).
- `xcolor` with `table` option — coloured table cells if needed.
- `hyperref` (with `hidelinks`) and `cleveref` — clickable references and `\Cref`.
- `enumitem` — compact lists.
- `algorithm`, `algpseudocode` — pseudocode (reserved; not used in the current body).
- `tikz` and `subcaption` — Figures 1--5 (TikZ schematics in `figures/fig_*.tex`).
- `geometry` — 1 inch margins; replace with the target venue style when known.

All of these ship with a standard `texlive-full` (or MacTeX, MiKTeX) install.

## Directory layout

```
paper_draft/latex/
├── BUILD.md                  ← this file
├── compile_notes.md          ← TODOs, missing citations, hand-finish items
├── unsafe_wording_check.md   ← claim-discipline scan log
├── main.tex                  ← top-level document
├── macros.tex                ← shared math macros
├── refs.bib                  ← bibliography (≈30 entries, some TODOs)
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_system_and_threat_model.tex
│   ├── 04_design.tex
│   ├── 05_correctness.tex
│   ├── 06_security_analysis.tex
│   ├── 07_evaluation.tex
│   ├── 08_limitations.tex
│   ├── 09_related_work.tex
│   ├── 10_conclusion.tex
│   ├── a_notation.tex
│   └── b_claims_mapping.tex
├── tables/
│   ├── correctness_summary.tex       ← copied verbatim from paper_results/latex/
│   ├── security_proxy_summary.tex
│   ├── workload_summary.tex
│   ├── lora_training_summary.tex
│   ├── measured_runtime.tex
│   └── paper_claims_audit.tex
└── figures/
    ├── fig_system_overview.tex       ← TikZ schematic (Fig. 1)
    ├── fig_right_masked_decode.tex   ← TikZ schematic (Fig. 2)
    ├── fig_nonlinear_island.tex      ← TikZ schematic (Fig. 3)
    ├── fig_dense_sandwich.tex        ← TikZ schematic (Fig. 4)
    ├── fig_lora_training.tex         ← TikZ schematic (Fig. 5)
    ├── security_risk_matrix.png      ← copied from paper_results/figures/
    ├── measured_runtime_summary.png  ← copied from paper_results/figures/
    ├── correctness_error_summary.png
    ├── boundary_call_reduction.png
    ├── lora_training_errors.png
    ├── rank_inference_risk.png
    └── timing_proxy_before_after.png
```

## Re-generating tables and PNG figures

The `.tex` tables and `.png` figures in `tables/` and `figures/` are copies of artifacts produced by the Stage 7.5 paper-artifact pipeline. To refresh after a re-run of that pipeline:

```
python scripts/run_paper_results_all.py
cp paper_results/latex/*.tex   paper_draft/latex/tables/
cp paper_results/figures/*.png paper_draft/latex/figures/
```

## Switching to a conference template

The current `\documentclass[10pt]{article}` is a placeholder. To switch to IEEE S&P, USENIX Security, NDSS, or ACM CCS:

1. Replace the `\documentclass` line and remove `geometry`.
2. Replace `\maketitle` with the venue's title macros.
3. Re-check `cleveref` compatibility (most venues already load it).
4. Re-check `\section` numbering depth and `subsection` granularity for the target page budget.
5. Cross-check the bibliography style (`plain` may need to become `IEEEtran`, `usenix`, `acmart`, etc.).

See `compile_notes.md` for the running checklist of items to hand-finish before submission.
