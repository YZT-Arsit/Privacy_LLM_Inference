# Stage 7.1 — Paper Artifact Generator (with built-in claim audit)

This stage generates paper-ready artifacts from the verified Stage 6.4.1 /
6.7 / 6.8 / 7.0 reports. It adds **no** masking functionality and makes
**no** security guarantee. It includes a built-in **claim audit** /
limitations review so that forbidden ("unsafe wording to avoid") phrasing is
confined to an explicitly-marked section and never leaks into the method,
theorem, table, or ablation text. We do not claim any semantic-, formal-, or
cryptographic-level property; this document is an audit of phrasing and a
description of generated artifacts, out of scope for any security proof.

## Generated artifacts (`outputs/paper_artifacts/`)

| file | content |
|------|---------|
| `method_summary.md` | threat model, system boundary, operator-compatible masking, pipeline variants |
| `correctness_theorems.md` | Lemmas 1–7 + Theorems 1–3 (assumptions / statement / proof sketch / caveats / verifying stage) |
| `complexity_table.{md,csv}` | per-variant GPU/TEE FLOPs, transfer bytes, KV-cache bytes, boundary calls, handoff/LM-head FLOPs |
| `leakage_boundary_table.{md,csv}` | per-variant GPU-visibility flags + security status + caveats |
| `ablation_summary.md` | rotation vs complex-scaling, shared vs per-layer masks, plain vs masked logits, GPU LM-head vs hidden-to-TEE, perm vs perm+scale |
| `claim_audit.{md,json}` | safe claims + a marked list of forbidden phrasing to avoid |
| `limitations.md` | the full limitations list |
| `stage_7_1_paper_artifacts.md` | combined report |

## Design

- `PaperArtifactConfig` points at the four source JSON reports. A missing
  report degrades gracefully: the relevant section is emitted with status
  `missing_source_report` and recorded in `metadata.missing_inputs` — never a
  crash.
- Forbidden wording is centralized in `UNSAFE_PHRASES` and only ever appears
  inside `UNSAFE_CLAIMS_BEGIN` / `UNSAFE_CLAIMS_END` markers. The test suite
  strips those marked regions and asserts no forbidden phrase appears
  anywhere else in any generated markdown.
- Cautious wording throughout: "reduces direct exposure",
  "operator-compatible leakage reduction", "verified correctness", "explicit
  leakage accounting".

## Files

- `src/pllo/experiments/paper_artifact_generator.py`
- `scripts/generate_paper_artifacts.py`
- `tests/test_paper_artifact_generator.py` — 12 tests (no transformers).

## Required statement

These artifacts summarize verified correctness, cost, and leakage accounting
for the masked CausalLM prototype. They do not constitute a semantic,
cryptographic, or formal security proof.

## Next stage

**Stage 7.2** — convert artifacts into AAAI-style paper sections, or
**Stage 6.9** — real HF full-model local tiny-checkpoint integration.
