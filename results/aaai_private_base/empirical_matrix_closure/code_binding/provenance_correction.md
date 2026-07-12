# Provenance correction (PHASE A)

## The defect in the prior binding
`code_baseline/experiment_code_binding.json` (v1) recorded
`worktree_diff_sha256 = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
That value is the SHA-256 of the **empty byte string** — `git diff HEAD` was empty at snapshot
time. But at that moment the executed implementation (modified `h800_d4_worker.py`, and untracked
`tdx_grad_correction_service.py`, `gate0_l10_orchestrator.py`, `full_matrix_equivalence.py`,
`build_correction_bundle.py`, `rank_mask_ablation.py`, `analyze_matrix.py`, `matrix_accounting.py`)
was **uncommitted**. An empty diff therefore bound *nothing* about the code that actually ran.
This is corrected here with a content-addressed **source bundle** over the real executed files
(`experiment_source_bundle.tar`, sha `5253952c…`) plus per-file hashes, independent of commit state.

## Precise commit audit (`36b2d6c → d4f44dd → a95b59f → 9773d9c`)
All four commits: **author == committer == `Zhaoting Yang <yangzhaoting21@gmail.com>`**, message
**empty (verified: `%s` is "")**, linear parent chain (`be0d660←36b2d6c←d4f44dd←a95b59f←9773d9c`).

| commit | git-time | files | contains Gate-0/0.5/matrix artifacts? |
|---|---|---|---|
| `36b2d6c` | 02:40 | 23 | yes — PHASE-6 worker/package/attestation/TDX-loss + dry-run results |
| `d4f44dd` | 03:24 | 62 | yes — D4 1/10-step + deployment-smoke results |
| `a95b59f` | 04:25 | 17 | yes — Gate 0.5 `gate05_o1/` + audit script |
| `9773d9c` | 07:32 | 106 | yes — full_lora_matrix + 7 new scripts (+11022 lines) |

### Exact mechanism (how the commits contain the agent's artifacts without the agent committing)
The agent wrote every file via its Write/Edit tools into the working tree, where they sat as
**untracked/modified**. An external actor operating under the user's git identity then ran
`git add` + `git commit` (empty message) **outside the agent's tool calls** — the agent issued
zero `git add`/`git commit`/`git rebase` invocations (auditable in the tool transcript). The
commit therefore records agent-authored file **content** under the user's **authorship and
committership**. "Not made by the agent" refers to the git operation; "contains the agent's
artifacts" refers to the file content — the mechanism connecting them is: *agent writes to
worktree → user commits the worktree*.

### Did any executed result depend on code not represented by HEAD?
- **At execution time (this session): YES.** The L10 1-step/10-step gates, the fp64 matrix, the
  rank-mask ablation, and the deployment run all executed against files that were uncommitted when
  they ran.
- **Now: NO.** Commit `9773d9c` captured those exact files; local content == HEAD, and the
  hardware-side copies (H800 worker, TDX correction + loss services) hash-match local byte-for-byte
  (`code_binding.json → remote_executed_code_sha256_16`, verified this run). The prior results are
  thus retrospectively bound by `9773d9c` + the source bundle.

## Binding rule for all new runs (PHASE C–J)
Each new run manifest embeds: `head_hash` (currently `9773d9c…`), `tracked_diff_sha256` (live),
and `experiment_source_bundle_sha256` (`5253952c…`). If new uncommitted edits are made for L11/L12
support, the live tracked-diff hash + a rebuilt bundle capture them. Prior results keep their
retrospective `9773d9c` binding. **Nothing is committed by the agent.**
