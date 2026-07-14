# Anonymous Code and Data Supplement Staging Area

This directory is the submission-time manifest and sanitization plan for the
separate AAAI-27 Code and Data Supplement. It is not yet a distributable archive:
the repository has no verified redistribution license, several historical
commands are missing, and legacy outputs contain identifying absolute paths.

- `code_data_manifest.md`: claim-to-artifact inventory and readiness status.
- `commands/`: sanitized, archive-relative command records.
- `evidence_map/`: checklist and paper-claim mappings.
- `sanitized_results/`: only compact results that pass the anonymization and
  licensing gates.

Do not include `.git/`, model caches, credentials, raw cloud logs, private
datasets, or machine-specific environment files in the submitted archive.
