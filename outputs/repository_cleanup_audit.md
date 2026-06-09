# Repository Cleanup Audit

Read-only audit. This document recommends actions; it does not delete anything. Low-risk junk is removed by `scripts/clean_repository_low_risk.py`; Markdown / outputs deletions require an explicit allowlist applied by `scripts/apply_repository_cleanup_allowlist.py`.

## Summary

- total_files_tracked: **705**
- junk_items: **10**
- markdown_files: **124**
- outputs_files: **201**
- scripts_files: **80**
- tests_files: **120**
- docs_files: **16**
- possible_duplicates: **1**
- possibly_unreferenced: **1**

### Action counts

| action | count |
|---|---|
| `keep` | 75 |
| `keep_core_artifact` | 26 |
| `keep_referenced` | 603 |
| `keep_uncertain` | 1 |

## Junk (low-risk delete candidates)

10 items (caches / temp / editor backups). Removed by the low-risk cleaner.

- `src/pllo/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/architectures/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/backends/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/cache/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/experiments/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/hf_wrappers/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/masks/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/model_zoo/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/ops/__pycache__/` — low-risk junk (cache / temp / editor backup)
- `src/pllo/utils/__pycache__/` — low-risk junk (cache / temp / editor backup)

## Possible duplicates / versioned siblings

- `paper_claims_audit` — outputs/paper_claims_audit_v2.json, outputs/paper_claims_audit_v2.md (versioned sibling(s) present)

## Possibly unreferenced files (need human review)

| path | action | reason |
|---|---|---|
| `outputs/repository_cleanup_plan.md` | `keep_uncertain` | output not matched to core set or references |

## Action legend

- `keep` — protected path (README / pyproject / src / tests / scripts / docs / paper_*).
- `keep_core_artifact` — Stage 5.7 / 5.8 / 7.6 core output, claims, summary, or limitations artifact.
- `keep_referenced` — referenced by README / docs / scripts / tests / src.
- `keep_uncertain` — not obviously referenced; do NOT delete without human review.
- `move_to_docs` / `move_to_archive` — relocation suggestions (never auto-applied).
- `delete_candidate` — low-risk junk only.

