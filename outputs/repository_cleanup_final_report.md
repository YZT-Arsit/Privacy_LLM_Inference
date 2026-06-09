# Repository Cleanup Final Report

Audit-first cleanup of `Privacy_LLM_Inference`. No GitHub upload, no PR,
no protocol-math change, no default-config change. Markdown / outputs
changes went only through an explicit, reference-checked allowlist.

## 1. Files deleted (low-risk only)

18 Python cache directories (`.pytest_cache/` + 17 × `__pycache__/`
under `src/`, `scripts/`, `tests/`), via
`scripts/clean_repository_low_risk.py --apply`. These regenerate
automatically. No `.pyc`, `.bak`, `.DS_Store`, `*~`, or
`tmp/`/`debug/`/`scratch/` files existed.

No Markdown, source, test, script, or output **content** was deleted.

## 2. Files archived (outputs)

| from | to | reason |
|---|---|---|
| `outputs/gpt2_block_correctness_small.json` | `outputs/archive/outputs/gpt2_block_correctness_small.json` | Stale (2024-05-28), 975 B, **unreferenced**, superseded by the referenced `outputs/gpt2_block_correctness.json`. Archived (reversible), not deleted. |

Applied via `outputs/repository_cleanup_allowlist.json` →
`scripts/apply_repository_cleanup_allowlist.py --apply`. The apply
script confirmed the file exists, is not on a protected path, and has
no real references (cleanup-control documents are excluded from the
reference scan) before acting.

## 3. Files moved / docs merged

None. The long stage log was already moved to
`docs/STAGE_ARTIFACTS.md` in the prior README rewrite; paper notes are
already under `docs/paper_draft/`. No duplicate docs found.

## 4. Kept-uncertain files

- `outputs/repository_cleanup_plan.md` — the cleanup-control document
  itself; retained intentionally.

No source / test / script / core-output file is uncertain.

## 5. Verification

| check | result |
|---|---|
| `python -m pytest -q` | **1307 passed, 4 skipped, 0 failed** (785 s) |
| Stage 7.6 claims-consistency scanner | **112 files, 0 unsafe, passes=True** |
| smoke `run_lookup_nonlinear_cost_proxy.py` | `status=ok`, `formal_security_claim=False`, `cryptographic_lookup_implemented=False` |
| smoke `run_masked_gradient_lora_training.py` | `status=ok`, `adamw_status=explicitly_raised_as_designed`, `claims_passes=True` |
| smoke `run_permutation_invariant_leakage.py` | ok (18 tensors / bundle) |
| smoke `audit_repository_cleanup.py` | ok |
| README + docs relative-link check | **34 links checked, 0 broken** |

## 6. Invariants preserved

- Protocol math: **unchanged**.
- Default config: **unchanged** (`nonlinear_mode="trusted"`, `mitigation_bundle="fresh_perm_only"`, `inter_block_mask_mode="plain_boundary"`, `constant_time_decode_mode="off"`, `implemented=False`, `security_profile="proxy-evaluated, not formal"`).
- Core Stage 5.7 / 5.8 / 7.6 artifacts: **all present** (permutation_invariant_leakage, lookup_nonlinear_cost_proxy, masked_gradient_lora_training, masked_gradient_lora_security_proxy, lora_training_inference_lifecycle, stage_7_6_claims_consistency).
- No file referenced by README / docs / tests / scripts / src / paper_results was deleted.

## Cleanup tooling (reusable)

- `scripts/audit_repository_cleanup.py` — read-only audit → `outputs/repository_cleanup_audit.{json,csv,md}`.
- `scripts/clean_repository_low_risk.py` — caches/junk only; `--dry-run` default, `--apply` to act.
- `scripts/apply_repository_cleanup_allowlist.py` — reference-checked allowlist executor for Markdown/outputs; `--dry-run` default.
- `outputs/repository_cleanup_allowlist.json` — explicit, human-reviewed action list.
- `outputs/repository_cleanup_plan.md` — human-confirmation plan.
