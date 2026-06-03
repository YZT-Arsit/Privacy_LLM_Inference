# Unsafe Wording Check — LaTeX draft

This file records the result of a substring scan for unsafe positive wording across the LaTeX sources of the draft. Every hit is classified into one of three safe categories:

- **(D) Disclaimer** — the phrase appears inside a "we do not / make no / never claim" sentence.
- **(M) Mapping/Notation table cell** — the phrase appears in `sections/a_notation.tex` (banned-phrases list) or `sections/b_claims_mapping.tex` (unsafe-wording column).
- **(R) Related-Work comparison** — the phrase appears while contrasting our approach with cryptographic, formal-security, or framework-integration alternatives that we explicitly do not provide.
- **(B) Bibliography title** — the phrase is part of the title of a cited paper inside `refs.bib`; this is a real third-party title, not a claim about our own work.

Any hit that does not fit (D), (M), (R), or (B) is a bug and must be rewritten before submission.

## Scan command

```
grep -nEi 'provabl|cryptographic|semantic security|prevents all|hides padded|TEE-level guaranteed|fully outsourced loss|production-ready|hardware side-channel secure|guaranteed|real TEE wall-time' \
  main.tex macros.tex sections/*.tex refs.bib
```

## Hits and classifications

| File | Line | Phrase | Classification |
|---|---|---|---|
| sections/00_abstract.tex | 8 | "no formal, cryptographic, or semantic-security claim" | (D) |
| sections/01_introduction.tex | 56 | "not a formal-security paper… do not prove cryptographic indistinguishability, semantic security" | (D) |
| sections/03_system_and_threat_model.tex | 30 | "We make no formal indistinguishability or semantic-security claim" | (D) |
| sections/03_system_and_threat_model.tex | 35 | "Fully outsourced loss/optimizer" (out-of-scope list) | (D) |
| sections/05_correctness.tex | 4 | "We do not claim formal cryptographic security here" | (D) |
| sections/05_correctness.tex | 103 | "they do not claim cryptographic indistinguishability" | (D) |
| sections/06_security_analysis.tex | 4 | "No formal, cryptographic, semantic, or differential-privacy claim is made" | (D) |
| sections/06_security_analysis.tex | 41 | "we do not claim that rank padding hides the LoRA rank cryptographically, and we do not claim $\rpad$ is hidden" | (D) |
| sections/06_security_analysis.tex | 53 | "we do not claim that the system is secure; we do not claim semantic, cryptographic, or formal indistinguishability… real TEE wall-time" | (D) |
| sections/08_limitations.tex | 7 | "No formal/cryptographic/semantic security" | (D) |
| sections/09_related_work.tex | 8 | "give formal-security guarantees… our approach is not a cryptographic alternative" | (R) |
| sections/09_related_work.tex | 14 | "use cryptographic or obfuscation techniques… we trade formal-security guarantees for proxy-evaluated security" | (R) |
| sections/09_related_work.tex | 26 | "We report this as proxy_supported, not provably private" | (D) inside (R) |
| sections/09_related_work.tex | 32 | "no cryptographic/formal/semantic security claim; no real-TEE deployment" | (D) |
| sections/10_conclusion.tex | 8 | "no formal, cryptographic, or semantic security; no real TEE wall-time" | (D) |
| sections/a_notation.tex | 22 | banned-phrases list including "provably", "cryptographically secure", "semantically secure", "TEE-level secure", "prevents all leakage", "hides padded rank", "production wall-time on TEE", "full Qwen/TinyLlama/LLaMA fine-tuning", "fully outsourced loss/optimizer" | (M) |
| sections/b_claims_mapping.tex | 29 | unsafe-wording entry U1 ("Safe wording: We make no formal/cryptographic/semantic security claims.") | (M) |
| sections/b_claims_mapping.tex | 43 | wording checklist | (M) |
| refs.bib | 167 | section comment "Cryptographic private inference (FHE / MPC / hybrid)" | (B) |
| refs.bib | 194 | paper title "Delphi: A Cryptographic Inference Service for Neural Networks" | (B) |

## Phrase-by-phrase summary

- **"provably / cryptographically / semantic security"** — all hits are (D) disclaimers, (M) banned-list cells, or (R) Related-Work contrasts. No positive use.
- **"prevents all leakage"** — appears only in the banned-list (M) and unsafe-wording columns (M). No body use.
- **"hides padded rank"** — appears only in the banned-list (M) and one strong disclaimer in `sections/06_security_analysis.tex` line 41 (D). No positive use.
- **"real TEE wall-time"** — all hits are (D) disclaimers ("we do not claim real TEE wall-time", "not real TEE wall-time"). No positive use.
- **"fully outsourced loss/optimizer"** — appears only in (D) out-of-scope items and (M) banned-list cells. No positive use.
- **"production-ready"** — no hits.
- **"hardware side-channel secure"** — no positive hits; the closest disclaimer is "we do not evaluate hardware side-channels" (D) in `sections/06_security_analysis.tex` and `sections/08_limitations.tex`.
- **"guaranteed"** — only in (R) Related-Work contrasts ("formal-security guarantees" describing FHE/MPC line) and (D) disclaimers. No positive use of "our scheme is guaranteed".

## Verdict

All scanned hits are accounted for as (D), (M), (R), or (B). **No positive overclaim leaks into the body of the LaTeX draft.**

A human re-scan before submission is still recommended per [`compile_notes.md`](compile_notes.md) item 4. Re-running the scan command above with the same word list is the canonical check.
