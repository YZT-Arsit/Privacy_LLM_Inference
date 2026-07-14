# AAAI-27 Submission Architecture Audit

Audit date: 2026-07-14. Formatting authority: the locally supplied
`AuthorKit27/` directory; policy authority: the author's AAAI-27
submission instructions. The official `aaai2027.sty` and `aaai2027.bst` are
byte-identical to the copies already used by `paper/`.

## 1. Current Occupancy Before the Main/Supplement Split

The input PDF was `paper/build/main.pdf` (five pages after the preceding
compression pass). Bounding-box measurements use both AAAI columns and the
usable vertical text area; empty TODO text is reported as placeholder space,
not completed content.

| Material | PDF location | Actual occupancy | Completion status |
|---|---|---:|---|
| Abstract, Introduction, Related Work | page 1 before Threat Model | about 0.55 page | TODO placeholders |
| Problem and Threat Model | page 1, left to right column | 0.69 page | substantive |
| Method | page 1 right through page 2 right | 0.84 page | substantive |
| Correctness | page 2 right through page 3 left | 0.62 page | first compact draft |
| Security, Evaluation, Limitations, Conclusion | page 3 left | about 0.20 page | TODO placeholders |
| Technical appendices | page 3 right through page 5 | about 2.30 pages | mixed complete detail and TODO skeletons |

The five-page PDF is not evidence that a completed paper fits seven pages:
Abstract, Introduction, Related Work, Security, Evaluation, Limitations, and
Conclusion are not yet written.

## 2. Technical Material Previously Placed After the Body

The old `paper/main.tex` used `\appendix` and included detailed protocol,
operator construction, complete proofs, attack details, and reproduction
material in the main PDF. Under the authoritative rule, those blocks cannot
occupy post-page-seven pages. They are now detached from the main paper and
placed in `paper_supplement/`.

## 3. Material That Would Be Illegal After Page Seven

- Tensor partitions, transcript formalization, complete message schedules, and
  ownership/lifetime tables.
- Detailed linear compensation, RoPE/GQA/MQA, nonlinear, rank-padding, LoRA
  backward, gradient-recovery, and optimizer equations.
- Complete correctness proofs.
- Attack hyperparameters, extended leakage experiments, and extra results.
- Implementation correspondence and reproduction details.

If retained in `paper/main.tex`, every such block after technical page seven
would violate the rule even if labeled “appendix.” References are the only
permitted post-page-seven content.

## 4. Duplicated Content

- Party roles and tensor ownership appeared in Threat Model, Method overview,
  the figure, the main algorithm, the appendix partition, and the appendix
  lifecycle.
- Mask freshness and KV-session consistency appeared in Method prose, the
  appendix message sequence, the ownership table, and the lifecycle.
- Linear and LoRA compensation appeared in Method, operator construction, and
  proofs.
- Correctness conditions were repeated across Method, theorem statements, and
  full proofs.
- Reproduction lineage appeared in comments, claim ledgers, result summaries,
  and the former reproducibility appendix without a single archive manifest.

## 5. Essential Seven-Page Main-Paper Content

- Central private-model problem and exactly three administrative parties.
- TEE/GPU split inside the Cloud Service, protected assets, GPU transcript,
  admitted metadata, goals, and explicit security non-claims.
- One compact system figure and one lifecycle algorithm.
- Masked-linear equation, attention compatibility/session-cache condition, and
  LoRA factor transformation with the trusted/untrusted work split.
- Three principal correctness statements and finite-precision scope.
- Leakage characterization, evaluated-attacker boundary, and residual
  invariants in a combined Correctness/Security section.
- Headline correctness, security, efficiency, and comparison results with
  evidence-lineage labels.
- Limitations and conclusion.

## 6. Main-to-Supplement Migration Map

| Original block | Supplement destination | Main-paper summary | Claim IDs | Main self-contained? |
|---|---|---|---|---|
| Threat tensor/state partition and transcript | `sections/01_extended_system_runtime_protocol.tex`, Administrative Roles | three parties, protected assets, concise transcript | C-THREAT-01/02/03, C-BND-01 | yes |
| Preparation, messages, freshness | same file, Preparation/Messages | transformed operands; session-fixed KV masks | C-BND-01, C-COR-05 | yes |
| Ownership/lifetime table | same file, Table 1 | exact TEE/GPU division in prose/figure | C-THREAT-01/03 | yes |
| Full inference/adaptation lifecycle | same file, Complete Lifecycle | compact Algorithm 1 | C-BND-01, C-LORA-BWD | yes |
| Linear dimensions and compensation detail | `sections/03_operator_and_lora_constructions.tex` | central transformed/recovery equation | C-COR-01 | yes |
| RoPE/GQA/MQA/cache construction | same file | compatibility condition plus session rule | C-COR-02/03/04/05 | yes |
| Normalization/nonlinear boundaries | same file | trusted operators and compatible activations summarized | C-COR-06/07/08 | yes |
| Rank padding, LoRA backward, gradients, optimizer | same file | factor transform and trusted optimizer locus | C-LORA-FWD/BWD/RANK, C-OPT-* | yes |
| Full algebraic proofs | `sections/05_complete_correctness_proofs.tex` | three theorem statements and proof intuition | C-COR-01..05, C-LORA-FWD/BWD | yes |
| Attack configurations and extended results | `sections/06_extended_security_evaluation.tex` and `07_additional_experimental_results.tex` | headline attacks, leakage, and scope must remain in main Evaluation | C-SEC-* | conditional: main Evaluation still TODO |
| Implementation mapping | `sections/01_extended_system_runtime_protocol.tex`, Implementation Map | conceptual operator boundary remains in Method | C-BND-01, C-COR-* | yes |
| Reproduction detail | `sections/09_reproduction_details.tex` | evidence lineage must remain in Evaluation captions/setup | C-EFF-*, C-REAL-* | conditional: main Evaluation still TODO |

## 7. Code and Data Supplement Material

The Code and Data Supplement must contain or index only submission-time
artifacts: source code, preprocessing/analysis scripts, frozen configurations,
raw machine-readable results, seed and environment records, validation rules,
and licensing notes. Large model weights and datasets should be referenced by
stable public identifiers and licenses rather than redistributed without
permission. `submission_archive/code_data_manifest.md` is the controlling
inventory; rows with missing commands or environment records are explicitly
marked incomplete.

## 8. Reproducibility Checklist Inputs

The checklist needs evidence for theorem assumptions/proofs; dataset selection
and citations; preprocessing and experimental code; parameter search/final
settings; random seeds; hardware/software environment; metric definitions;
run counts; variability/statistical tests; and public-availability/licensing
status. The official form is copied unchanged except for response placeholders.
`paper_reproducibility/checklist_evidence_map.md` provides the evidence linkage.

## 9. Anonymous-Review Risks

- Legacy JSON/Markdown artifacts embed local absolute paths.
- Git history and remotes may expose identity and must not enter an archive.
- Cloud logs may expose hostnames, instance IDs, account identifiers, TDX host
  metadata, cache paths, and credentials.
- Repository URLs and comments may identify authors or institutions.
- PDF metadata must contain only the template version; no title/author/path.
- Self-citations and acknowledgments require a separate anonymity review.
- Supplement and checklist sources must use `Anonymous Submission` and no local
  paths in rendered text.

## 10. Projected Seven-Page Allocation

| Section | Target pages |
|---|---:|
| Abstract | 0.18 |
| Introduction | 0.70 |
| Related Work | 0.45 |
| Problem and Threat Model | 0.50 |
| Method | 1.15 |
| Combined Correctness and Security | 1.00 |
| Evaluation | 2.20 |
| Limitations and Conclusion | 0.42 |
| Float/layout buffer | 0.25 |
| **Projected total technical content** | **6.85** |

The architecture has a 0.15-page contingency below the hard limit. Compliance
must be remeasured from the completed PDF; this projection is not a claim that
the placeholder draft already satisfies the final page rule.

## 11. Post-Split Build and Architecture Gate

The stable post-split main PDF has three current pages. Threat Model begins on
page 1 at the left-column coordinate $y\approx386$ and Method begins on page 1
in the right column at $y\approx438$, giving Threat Model occupancy of about
**0.54 page**. Method ends when Correctness begins on page 2 in the right column
at $y\approx269$, giving Method occupancy of about **0.87 page**. References
begin on page 3 at $y\approx661$; current technical occupancy is approximately
**2.47 pages**, but most surrounding sections remain TODOs.

The architecture gate passes:

- main, Supplement, and official checklist latexmk exit codes are 0;
- no stable undefined references/citations, duplicate labels, or overfull boxes;
- no appendix or other technical content follows the main-paper bibliography;
- there is no page after page seven in the current main PDF, so the post-page-
  seven rule is satisfied; when the paper is completed, only references may
  occupy such pages;
- Threat Model is 0.54 page (limit 0.55) and Method is 0.87 page (limit 1.25);
- the completed-body projection is 6.85 pages including buffer; and
- no font, margin, spacing, page-break, or negative-space manipulation was used.

The empty bibliography warning is expected while Related Work has no promoted
citations; `paper/main.tex` nevertheless contains the required single
`\bibliography{references}` declaration and no explicit bibliography-style
command. This warning must disappear during the citation-writing pass.
