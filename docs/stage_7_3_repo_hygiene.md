# Stage 7.3 — Repository Hygiene, Output-Size Guard & Evidence Manifest

This stage adds **no** masking functionality and makes **no** security claim.
It makes the repository safer, smaller, easier to reproduce, and easier to
present in a paper. It never deletes tracked files automatically, never runs
the full test suite, and never requires CUDA / transformers / internet / real
checkpoints.

## Components

| Component | Entry point | Behavior |
|---|---|---|
| Hygiene audit (read-only) | [run_repo_hygiene_audit](../src/pllo/experiments/repo_hygiene.py) / [scripts/run_repo_hygiene_audit.py](../scripts/run_repo_hygiene_audit.py) | classifies tracked vs untracked generated artifacts; recommends `.gitignore` entries; emits safe-cleanup commands for **untracked** only; lists tracked artifacts for **manual** decision. Deletes nothing. |
| `.gitignore` helper | `ensure_gitignore_entries` | idempotently appends missing patterns under a header; never duplicates, never removes. |
| Output-size guard | `check_output_sizes` / [scripts/check_output_sizes.py](../scripts/check_output_sizes.py) | warns > `warn_mb` (10), fails > `fail_mb` (100); excludes local model checkpoints; exit 1 on any failure. Deletes nothing. |
| Evidence manifest | `generate_evidence_manifest` / [scripts/generate_evidence_manifest.py](../scripts/generate_evidence_manifest.py) | compact paper-ready summary of Stages 6.4–7.6 → `outputs/evidence_manifest.{json,md}`. No tensor dumps. |
| Lightweight repro | [scripts/run_lightweight_repro.py](../scripts/run_lightweight_repro.py) | import check + 3 targeted test files + 1 tiny HF probe + size guard. No full suite, no full Stage 7.6 scanner, no paper artifacts. |

## Commands

```bash
pytest tests/test_repo_hygiene.py -q
python scripts/run_repo_hygiene_audit.py --output outputs/repo_hygiene_audit.json
python scripts/generate_evidence_manifest.py --output-dir outputs
python scripts/check_output_sizes.py --output-dir outputs --warn-mb 10 --fail-mb 100
python scripts/run_lightweight_repro.py
```

## Audit snapshot (this repo)

- git available: **True**
- tracked generated candidates: **508** (301 `__pycache__`/`.pyc`, 5
  `*.egg-info`, 202 `outputs/*.{json,md,csv}`) — flagged for **manual**
  decision, **not** deleted.
- untracked generated candidates: **6** (probe outputs / `.pytest_cache`).
- outputs total size: **~4.85 MB**; largest file ~1.79 MB; no failures.

### Manual decision for tracked artifacts

This repository tracks compiled caches, egg-info, and generated `outputs/`
reports. Stage 7.3 **never** removes them. To stop tracking without deleting
working copies (a manual, reviewable decision):

```bash
git rm -r --cached '**/__pycache__' '*.egg-info'
# review outputs/ individually before untracking
git commit -m "stop tracking generated artifacts"
```

The freshly-added `.gitignore` patterns prevent **future** generated files
from being tracked; they do not retroactively untrack existing files.

## `.gitignore` entries added

`__pycache__/`, `*.py[cod]`, `*.pyo`, `.pytest_cache/`, `.ruff_cache/`,
`.mypy_cache/`, `.coverage`, `htmlcov/`, `build/`, `dist/`, `*.egg-info/`,
`outputs/paper_artifacts/`, `outputs/paper_sections/`,
`outputs/*claims_consistency*.{json,csv,md}`, `outputs/*probe*.{json,md}`,
`outputs/*cost*.{json,md,csv}` (appended under
"`# Generated artifacts and local caches`"; idempotent).

## Global caveats (carried into the manifest)

- No semantic, cryptographic, or formal security is claimed.
- Attention scores / probabilities remain GPU-visible.
- Vocab permutation+scaling is weaker than dense vocab masking.
- Real production full-model inference is not validated (extracted-weight
  reference, tiny/random models, greedy decode).

## Files

- [src/pllo/experiments/repo_hygiene.py](../src/pllo/experiments/repo_hygiene.py)
- [scripts/run_repo_hygiene_audit.py](../scripts/run_repo_hygiene_audit.py),
  [scripts/check_output_sizes.py](../scripts/check_output_sizes.py),
  [scripts/generate_evidence_manifest.py](../scripts/generate_evidence_manifest.py),
  [scripts/run_lightweight_repro.py](../scripts/run_lightweight_repro.py)
- [tests/test_repo_hygiene.py](../tests/test_repo_hygiene.py) — 12 tests.

## Next stage

**Stage 7.4** — real local tiny-checkpoint smoke test (if a suitable local
model exists), or **Stage 8.0** — paper-ready evaluation-table consolidation.
