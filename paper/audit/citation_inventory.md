# Citation Inventory

Tracks every external citation the paper needs, its verification status, and the intended use. **No entry enters `references.bib` until its metadata is verified from the original paper or an authoritative index.** Section skeletons use `\citeneeded{...}` (loud red placeholder) instead of `\cite{...}` until an entry is promoted here to **VERIFIED** and added to `references.bib`.

**Verification legend:** `LOCAL-PDF` = original PDF present in `papers/` and inspected · `STAGED` = well-known work reused from prior `refs.bib`, canonical metadata, but exact venue/pages still to confirm · `NEEDS` = cited but not locally verifiable (no PDF; must verify from authoritative index before use).

**Prior-draft warning:** `paper_draft/related_work.md` stated "citations below are placeholders; concrete references will be inserted in the LaTeX pass" — i.e. the entire prior bibliography is unverified. The fabricated `amulet2024todo` placeholder from `refs.bib` was deliberately NOT carried into `references.bib`.

---

## 1. Baseline / comparison papers with LOCAL PDF (verifiable) — highest priority

| Cite key | Proposition it supports | Target paper | Verified? | Comparison dimension | Missing fields to confirm |
|---|---|---|---|---|---|
| stip2024 | STIP is an adapted baseline (permutation obfuscation, model+data protected, prefill-oriented). | STIP: Secure Transformer Inference Protocol — Yuan, Zhang, Li (USTC), arXiv:2312.00025v2 (2024-05-08). | **LOCAL-PDF** (`papers/STIP-NDSS.pdf`) | adapted reproduction; leakage x-axis (ArrowMatch recovers its perm) | **Venue: filename/BASELINES.md say "NDSS 2026" but PDF is an arXiv preprint with placeholder ACM formatting — NDSS NOT confirmed in-PDF.** Pages n/a. |
| obfuscatune2025 | ObfuscaTune is an adapted (simulated-TEE) baseline; boundary-crossing cost comparison. | ObfuscaTune — Frikha, Walha, Mendes, Nakka, Jiang, Zhou (Huawei Munich), arXiv:2407.02960v2 (2025-01-14). | **LOCAL-PDF** (`papers/ObfuscaTune-AAAI.pdf`) | adapted reproduction; H800 cost model | **Venue "AAAI/PPAI-25 workshop" not stated in-PDF.** Confirm workshop vs main. |
| conjformer2026 | Direct-reproduction baseline on real Qwen-7B (equivariant transformer). | CONJFORMER / "Privacy from Symmetry", arXiv:2606.16461 (2026 per BASELINES.md). | **NEEDS** (no PDF; cited) | direct reproduction (real 7B) | Full author list, title, venue, exact arXiv id — verify. |
| arrow2025 | Arrow is an ATTACK (ArrowMatch recovers permutation obfuscation) — corrects RQ13 misclassification (CF-7). | "Game of Arrows: On the (In-)Security of Weight Obfuscation for On-Device TEE-Shielded LLM Partition," Wang et al. (PKU/ByteDance/UIUC), **USENIX Security 2025**. | **LOCAL-PDF** (`papers/Arrow.pdf`) | leakage x-axis | Confirm full author list + exact USENIX Sec 2025 pages. |
| permattack2025 | Permutation-based private inference is breakable (motivates threat model). | "An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs," Thomas, Zahran, Choi, Potti, Goldblum, Pal (Ritual/Stanford/Columbia), arXiv:2505.18332v1 (2025-05-23); BASELINES.md tags ICML'25. | **LOCAL-PDF** (`papers/Permutation.pdf`) | motivation / leakage | Confirm ICML'25 acceptance vs preprint. |
| pia2025 | Prompt-inversion attack against collaborative inference (motivation). | "Prompt Inversion Attack against Collaborative Inference of LLMs," Qu et al. (NUS/UC Berkeley/UFlorida/HKUST/NTU), arXiv:2503.09022v3 (2025-05-02). | **LOCAL-PDF** (`papers/PIA.pdf`) | attacker toolbox | Confirm venue. |
| bre2024 | Split-based private fine-tuning is vulnerable (motivates LoRA-gradient protection). | "Unveiling the Vulnerability of Private Fine-Tuning in Split-Based Frameworks … A Bidirectionally Enhanced Attack (BiSR)," Chen et al., **ACM CCS 2024**. | **LOCAL-PDF** (`papers/BRE.pdf`) | attacker toolbox (BRE/BiSR) | Confirm pages. |
| ednn2024 | Embedding-matrix inversion attack (motivates embedding-boundary handling). | "An Inversion Attack Against Obfuscated Embedding Matrix in Language Model Inference (EDNN)," Lin et al. (ByteDance), **EMNLP 2024**, pp. 2100–2104. | **LOCAL-PDF** (`papers/EDNN攻击.pdf`) | attacker toolbox (EIA) | Confirm authors. |

## 2. Foundations — STAGED in references.bib (confirm venue/pages before final)

| Cite key | Proposition | Verified? | Missing |
|---|---|---|---|
| vaswani2017attention | Transformer architecture. | STAGED | NeurIPS 2017 pages. |
| su2021roformer | RoPE positional embedding. | STAGED | Journal/venue (Neurocomputing 2024?) — confirm. |
| ainslie2023gqa | GQA. | STAGED | EMNLP 2023 pages. |
| zhang2019rmsnorm | RMSNorm. | STAGED | NeurIPS 2019 pages. |
| hu2022lora | LoRA. | STAGED | ICLR 2022 — confirm. |
| tramer2019slalom | Slalom (TEE linear offload) baseline lineage. | STAGED (also cite-only comparison) | ICLR 2019 — confirm. |

## 3. Cited but NOT locally verifiable — NEEDS verification before any use

| Cite key | Proposition | Comparison dimension | Missing (verify from authoritative index) |
|---|---|---|---|
| darknight2021 | TEE+GPU coded computation; primitive comparison. | primitive-level | **Venue conflict: RQ13 "USENIX Security 2021" vs BASELINES.md "MICRO 2021"** — resolve. Full metadata. |
| amulet | Matrix-obfuscation ancestor / RQ13 primitive / name of our own backend. | lineage + primitive | **No PDF; "arXiv 2512.07495" per BASELINES.md unverified.** Title/authors/venue. **Also a naming hazard (CF-7).** |
| cryptonets2016 | HE inference; arithmetic-skeleton comparison only. | analytical | ICML 2016 metadata. |
| juvekar2018gazelle | MPC/HE inference; cost-model-only. | cost-model-only | USENIX Sec 2018 metadata. |
| mishra2020delphi | Crypto inference; cost-model-only. | cost-model-only | USENIX Sec 2020 metadata. |
| mohassel2017secureml | MPC ML; cost-model-only. | cost-model-only | IEEE S&P 2017 metadata. |
| liu2017minionn | Oblivious NN; cost-model-only. | cost-model-only | CCS 2017 metadata. |
| bumblebee | Crypto Transformer inference upper-bound tier. | cite-only | **No PDF; NDSS 2025 per BASELINES.md** — verify title/authors. |
| volos2018graviton | GPU TEE. | related work | OSDI 2018 metadata. |
| zhu2019dlg / zhao2020idlg / yin2021gradinversion | Gradient leakage (motivation; S4 DLG uses this). | attacker toolbox | NeurIPS 2019 / CVPR 2021 metadata. |
| shokri2017membership | Membership inference (S5 MIA uses Shokri shadow-model). | attacker toolbox | IEEE S&P 2017 metadata. |
| abadi2016dpsgd | DP-SGD (contrast: we do NOT provide DP). | contrast | CCS 2016 metadata. |
| kwon2023vllm / dao2022flashattention / rasley2020deepspeed | Inference/training infra we do NOT integrate (U5). | non-integration note | confirm venues. |

## 4. Topics still needing a concrete citation (no key yet)

| Topic | Where needed | Note |
|---|---|---|
| KV-cache privacy | Related §3.4, threat model | No concrete reference located in repo — search required. |
| TEE-assisted NN inference survey | Related §3.1 | Beyond Slalom/Graviton, find a survey/anchor. |
| Protected / private LoRA fine-tuning | Related §3.5 | Distinguish from generic split learning. |
| Intel TDX / DCAP attestation | Method §5.9, App D | Cite official Intel TDX + DCAP QVL spec for the attestation claim (C-REAL-ATTEST). |
| SwiGLU / GLU variants | Background | `shazeer2020glu` in prior refs.bib — STAGE if used. |
| LayerNorm | Background | `ba2016layernorm` in prior refs.bib — STAGE if used. |

---

## Promotion checklist (before adding an entry to references.bib)
1. Confirm authors, exact title, venue, year (and pages if a proceedings) from the original PDF or an authoritative index (DBLP / ACL Anthology / venue proceedings). 2. Resolve the venue conflicts flagged above (STIP NDSS?, ObfuscaTune workshop?, DarKnight USENIX-vs-MICRO, Amulet arXiv id). 3. Replace the corresponding `\citeneeded{...}` in the section skeleton with `\cite{key}`. 4. Mark the row **VERIFIED** here with the resolved fields.
