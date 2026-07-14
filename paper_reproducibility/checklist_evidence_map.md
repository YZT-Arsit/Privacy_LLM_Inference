# Reproducibility Checklist Evidence Map

This file explains every answer in the unmodified official checklist structure.
Evidence is submission-time only. Repository-relative links avoid embedding a
local username in the checklist PDF.

### Q 1.1

**partial.** Method Algorithm 1 and claims C-BND-01/C-COR-01 provide the
conceptual outline. Source: `paper/sections/05_method.tex`; implementation:
`src/pllo/`; archive status: [R1--R4](../submission_archive/code_data_manifest.md#current-main-paper-experimental-results).

### Q 1.2

**yes.** The [claim ledger](../paper/claim_ledger.md) distinguishes definition,
proof, measured, proxy, projected, and unsupported statuses; main Correctness
separates algebra from validation. No single script/artifact is applicable.

### Q 1.3

**no.** Related Work remains a TODO and verified pedagogical citations have not
yet been inserted. See [citation inventory](../paper/audit/citation_inventory.md).

### Q 2.1

**yes.** Claims C-COR-01--05 and C-LORA-FWD/BWD are algebraic contributions;
the statements are in `paper/sections/06_correctness.tex`, with proofs in the
Technical Supplement.

### Q 2.2

**partial.** Principal invertibility, ordering, mask-tying, and cache restrictions
are stated; the full assumption/state partition is in
`paper_supplement/sections/01_extended_system_runtime_protocol.tex`. Remaining
claim risks are tracked in the [ledger](../paper/claim_ledger.md).

### Q 2.3

**partial.** Three principal correctness claims are formal theorems. Leakage,
optimizer-impossibility, and RMSNorm limitations are not all formalized in the
main paper.

### Q 2.4

**partial.** Full proofs exist for C-COR-01--05 and C-LORA-FWD/BWD in
`paper_supplement/sections/05_complete_correctness_proofs.tex`; not every
analytical ledger claim has a complete submitted proof.

### Q 2.5

**yes.** Main Correctness gives cancellation intuition and explicit failure
conditions; complete algebra is in the supplement. Numerical scripts/artifacts
are indexed as [R1--R4](../submission_archive/code_data_manifest.md#current-main-paper-experimental-results).

### Q 2.6

**no.** Theoretical-tool citations are not yet present because Related Work and
the citation pass are incomplete. Environment record: `pyproject.toml`; archive
entry: not applicable.

### Q 2.7

**partial.** Linear, attention/cache, and LoRA identities have synthetic probes
([R1--R4](../submission_archive/code_data_manifest.md#current-main-paper-experimental-results));
not every theoretical restriction has an empirical test in a real deployment.

### Q 2.8

**no.** Some counterexample/negative-control code exists, but the Code and Data
Supplement is not packaged and several historical runners are missing. See R1--R3
and the archive readiness notes.

### Q 3.1

**yes.** Candidate Evaluation uses public benchmark datasets and prompt sets;
claims C-EFF-A10/C-REAL-QWEN identify their scoped evidence. The main Evaluation
is not yet authored.

### Q 3.2

**no.** Dataset-choice motivation is absent from the TODO Evaluation section.
Candidate evidence is indexed in [R7](../submission_archive/code_data_manifest.md#r7--a10-plus-tdx-per-batch-profile).

### Q 3.3

**NA.** No novel dataset is currently claimed. No script/raw artifact/environment
or archive row applies.

### Q 3.4

**NA.** No novel dataset is currently claimed, so no future-release claim is
made.

### Q 3.5

**no.** Dataset citations have not been inserted because Related Work and
Evaluation are TODOs. See `paper/references.bib` and the citation inventory.

### Q 3.6

**partial.** Candidate benchmark/model sources appear public, but licenses and
redistribution rights have not been verified for the archive. See R5/R7 and the
manifest licensing fields.

### Q 3.7

**NA.** No non-public research dataset is currently part of an authorized main
result. Private User data is a threat-model asset, not an experimental dataset.

### Q 4.1

**yes.** The paper contains computational experiments; claim families C-COR,
C-LORA, C-SEC, C-EFF, and C-REAL identify them. Scripts and raw outputs are
indexed in the Code/Data manifest; `pyproject.toml` is the base environment record.

### Q 4.2

**no.** The number/range of development hyperparameters and selection criteria
are not comprehensively recorded. The A10 profile documents one selection gate,
but this does not cover all experiments.

### Q 4.3

**partial.** Preprocessing scripts and prompt artifacts exist, but the separate
Code/Data archive is not packaged. R5/R7 identify known commands and missing
provenance; environment evidence is incomplete.

### Q 4.4

**partial.** Much of the source is present under `src/` and `scripts/`, while
R1--R3 document missing historical wrappers. The archive has only a manifest,
not a validated source bundle.

### Q 4.5

**no.** No repository code license is present, so public redistribution with a
research-use license is unsupported. The answer does not rely on a future-release
promise.

### Q 4.6

**partial.** Core modules contain implementation comments, but systematic
paper-step cross-references are incomplete. Claim mapping is in the ledger;
R1--R4 map scripts/artifacts.

### Q 4.7

**partial.** `results/aaai_private_base/seed_manifest.json` records primary,
mask, and restart seeds. Several older artifacts use different single seeds and
do not document all RNG/library determinism settings.

### Q 4.8

**partial.** Hardware is recorded for selected A10/H800/TDX artifacts and dtype
is generally embedded in JSON, but exact OS, CPU, memory, PyTorch, Transformers,
and driver versions are incomplete. Base dependency ranges are in `pyproject.toml`.

### Q 4.9

**partial.** Machine-readable artifacts define errors, match rates, latency, and
proxy metrics, and the claim ledger motivates their interpretation. Formal metric
definitions are not yet written in the TODO Evaluation section.

### Q 4.10

**no.** Some artifacts record repeats (for example R6 has five), while others
are single runs or omit repetition counts. The paper does not yet state a count
for every reported result.

### Q 4.11

**no.** R6 reports standard deviation, but most candidate main results lack
variation/confidence summaries. Single-step and single-profile results dominate.

### Q 4.12

**no.** No comprehensive statistical-significance testing is recorded for the
claimed improvements. No script/raw test artifact or archive row supports “yes.”

### Q 4.13

**partial.** Many JSON artifacts embed final configurations and R4 has complete
synthetic LoRA defaults, but the paper lacks a consolidated final-parameter table
and several hardware commands/environments are incomplete.
