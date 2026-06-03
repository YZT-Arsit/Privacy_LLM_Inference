# Masked Execution Islands for Private Generative LLM Inference and Personalization

*Working title.* Alternative full title: *Generation-Compatible Privacy-Preserving LLM Inference and LoRA Personalization via Operator-Compatible Masked Execution.*

This is the Stage 7.6 Markdown draft of the paper body. Sections are written as stand-alone Markdown files under `paper_draft/` and stitched together below in the canonical paper order. The LaTeX conversion is **Stage 7.6b** (planned, not in this draft).

This draft consumes the Stage 7.5 paper-side artifacts under `paper_results/` *as they exist* (no new experiments, no re-classification of unsupported claims, no formal-security claims). The authoritative claim audit is `paper_results/markdown/paper_claims_audit.md`; the mirror inside this draft is `paper_draft/claims_mapping.md`.

## Section index

1. [Abstract](abstract.md)
2. [Introduction](introduction.md)
3. [Background and Motivation](background.md)
4. [System and Threat Model](system_and_threat_model.md)
5. [Design](design.md)
6. [Correctness Analysis](correctness.md)
7. [Security Analysis](security_analysis.md)
8. [Evaluation](evaluation.md)
9. [Limitations](limitations.md)
10. [Related Work](related_work.md)
11. [Conclusion](conclusion.md)

## Appendices and pinning documents

- [Notation](notation.md) — symbol glossary, including the list of phrases the body never uses.
- [Tables](tables.md) — canonical table list, source files, captions.
- [Figures](figures.md) — canonical figure list, generated paths, and `TODO: draw` items.
- [Claims mapping](claims_mapping.md) — every body sentence pinned to a `supported` / `proxy_supported` / `unsupported` audit row, with safe / unsafe wording.

## Reading order suggestion

A reader who wants to confirm the claim discipline before reading the body should read, in order:

1. `paper_draft/claims_mapping.md` (verbatim mirror of `paper_results/markdown/paper_claims_audit.md`).
2. `paper_draft/limitations.md` (what is explicitly out of scope).
3. `paper_draft/notation.md` (symbols + the list of phrases this draft never uses).
4. `paper_draft/abstract.md` and the rest of the body in order.

## What this draft is and is not

This draft is the **Markdown body** of the paper. It is *not*:

- a LaTeX submission (Stage 7.6b);
- a literature review with real citation keys (Stage 7.6b adds `\cite{}`);
- a new experiment (Stage 7.6 freezes the artifact set);
- a re-interpretation of any `unsupported` audit row.

The Markdown body, the tables in `paper_results/`, and the figures in `paper_results/figures/` together form a self-contained artifact for the paper writeup phase.

## Boundary contract carried over from Stage 7.5

Every sentence in this draft is consistent with the following Stage 7.5 boundary:

- No new obfuscation primitive.
- No new attack algorithm.
- No change to default inference / LoRA behavior.
- No re-classification of `unsupported` claims as `supported`.
- No formal / cryptographic / semantic security claim.
- No real TEE wall-time claim.
- No full Qwen / TinyLlama / LLaMA LoRA fine-tune claim.
- No PEFT / DeepSpeed / vLLM / FlashAttention compatibility claim.
- No claim that `padded_rank` is hidden.
- No claim that loss / optimizer is fully outsourced.
- No publication of raw tensors / private data / masks / adapters / gradients.
