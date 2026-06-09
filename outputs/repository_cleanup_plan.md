# Repository Cleanup Plan

Human-confirmation plan derived from
`outputs/repository_cleanup_audit.{json,csv,md}`. Generated after a
read-only audit of 697 tracked files. Nothing here is deleted blindly:
low-risk junk is removed by `scripts/clean_repository_low_risk.py`;
Markdown / outputs changes go only through
`outputs/repository_cleanup_allowlist.json` applied by
`scripts/apply_repository_cleanup_allowlist.py`.

## Audit headline

- Total tracked files: **697**
- Junk items (caches): **18**
- `keep_referenced`: 594 · `keep`: 76 · `keep_core_artifact`: 26 · `keep_uncertain`: 1
- Possibly unreferenced (needs review): **1**
- Possible duplicates: **1** (false positive — see below)

The repository is already well-referenced. The only real cleanup is
Python caches plus one stale orphan output.

## 1. Automatically deleted low-risk files

Removed by `scripts/clean_repository_low_risk.py --apply` (caches only;
never touches tracked source / docs / outputs content):

- `.pytest_cache/`
- 17 × `__pycache__/` directories under `src/`, `scripts/`, `tests/`

These are regenerated on the next test / import run. No `.pyc`, `.bak`,
`.DS_Store`, `*~`, or `tmp/` `debug/` `scratch/` files were present.

## 2. Suggested human deletion (Markdown)

**None.** Every Markdown file is referenced or is a paper / docs
document. No stale or duplicate Markdown was found.

## 3. Suggested archive (outputs)

Applied via the allowlist (`archive` list):

| file | reason |
|---|---|
| `outputs/gpt2_block_correctness_small.json` | Stale (2024-05-28), 975 B, **not referenced** by any script / test / README / docs, and superseded by the referenced `outputs/gpt2_block_correctness.json` (produced by `scripts/run_gpt2_block_correctness.py`). Archived to `outputs/archive/` rather than deleted, so it stays recoverable. |

## 4. Suggested docs merges

**None required.** The long stage log has already been moved from the
README to `docs/STAGE_ARTIFACTS.md`. Paper notes already live under
`docs/paper_draft/`; paper-side tables under `paper_results/`. No
duplicate docs were found.

## 5. Uncertain but suspicious files

The single `keep_uncertain` item is `outputs/gpt2_block_correctness_small.json`,
handled by archive above. There are no other uncertain files.

The audit flagged `paper_claims_audit` as a "versioned sibling" group,
but this is a **false positive**: only `paper_claims_audit_v2.{json,md}`
exist under `outputs/` (there is no non-v2 sibling there), and v2 is the
current, referenced artifact. No action.

## 6. Core files that must NOT be deleted

Protected by path and/or core-artifact rules (audit actions `keep`,
`keep_core_artifact`, `keep_referenced`):

- `README.md`, `pyproject.toml`
- `src/`, `tests/`, `scripts/`, `docs/`, `paper_results/`, `paper_draft/`
- Stage 5.7 / 5.8 / 7.6 core outputs:
  - `outputs/permutation_invariant_leakage.*`
  - `outputs/lookup_nonlinear_cost_proxy.*`
  - `outputs/masked_gradient_lora_training.*`
  - `outputs/masked_gradient_lora_security_proxy.*`
  - `outputs/lora_training_inference_lifecycle.*`
  - `outputs/stage_7_6_claims_consistency.*`
- Paper claims / limitations / summary artifacts under `outputs/` and `paper_results/`
- `docs/STAGE_ARTIFACTS.md`, `docs/PAPER_THEORY_OUTLINE.md`, `docs/PAPER_EVALUATION_MAP.md`

## Verification gate

After applying, the cleanup is validated by:

- `python -m pytest -q`
- the Stage 7.6 claims-consistency scanner
- smoke runs of `run_lookup_nonlinear_cost_proxy.py`,
  `run_masked_gradient_lora_training.py`,
  `run_permutation_invariant_leakage.py`,
  `audit_repository_cleanup.py`

Results are recorded in `outputs/repository_cleanup_final_report.{json,md}`.
