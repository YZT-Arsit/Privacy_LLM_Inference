# External-commit audit + code baseline (Section 0)

**Current HEAD:** `a95b59f6c1d96f2ad4950220d93078218b81c607`
**Worktree at baseline:** clean (0 modified / 0 untracked); `git diff HEAD` is empty →
`worktree_diff.sha256 = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
(the canonical SHA-256 of the empty byte string).
**`git commit` commands issued by this agent:** 0 (verified — this agent ran no commit/rebase/filter).

## What changed externally
The plan was written when HEAD was `d4f44dd`. Since then a **third** external commit
`a95b59f` landed (git-time `Sun Jul 12 04:25:48 2026 +0800`). All three commits below share
the same author identity and empty messages; they **persist this agent's own Gate-0 / Gate-0.5
generated artifacts** into git.

| commit | git-time | files | content |
|---|---|---|---|
| `36b2d6c` (parent `be0d660`) | 02:40:50 | 23 | PHASE-6 infra: `gate0_build_private_package.py`, `h800_unified_worker.py`, `h800_d4_worker.py`, `d4_trusted_verifier.py`, `tdx_trusted_loss_service.py`, `gate0_d4_attestation.py`, `gate0_d4_orchestrator.py` + dry-run/loader/attestation-manifest results |
| `d4f44dd` (parent `36b2d6c`) | 03:24:02 | 62 | D4 one-step + 10-step + deployment-smoke results, `h800_d4_deployment_smoke.py`, per-step local_msg envelopes, trajectory/perf/counters |
| `a95b59f` (parent `d4f44dd`) | 04:25:48 | 17 | Gate 0.5 deliverables: `gate05_o1_optimizer_audit.py` + `results/.../gate05_o1/*` (16 files) |

Author of all three: `Zhaoting Yang <yangzhaoting21@gmail.com>` (the user's own identity).

## Semantic-impact determination
- **Do these commits alter experiment semantics relative to the agent's outputs?** **No.** The
  committed files are byte-for-byte the artifacts this agent already produced and validated at
  Gate-0 / Gate-0.5. Committing them changed their git-tracking status, not their content or
  meaning. No source file was modified beyond what the agent wrote; no dataset, evaluation
  harness, or security-harness *semantics* were introduced by the act of committing.
- **Do they touch the private package / worker / optimizer / datasets / eval / security?** They
  *contain* the package builder, worker, TDX loss service, verifier/orchestrator, attestation,
  optimizer-audit, and deployment scripts — because those are the agent's deliverables now under
  version control. Categorised counts are in `external_commit_audit.json`. None of these files
  differ from the already-validated versions.
- **Reruns of invalidated prerequisites?** None required. Nothing semantically changed, so the
  Gate-0 D2/D3/package/D4 and Gate 0.5 results remain valid and are **not** rerun (also per the
  explicit do-not-rerun list).

## Actions taken
- Not reverted (per instruction).
- Not attributed to this agent's authorship (per instruction). Recorded factually as external
  commits whose *content* is the agent's prior deliverables.
- HEAD hash + empty-worktree-diff hash computed and frozen here.

## Binding policy for this matrix
Every `full_lora_matrix` run manifest embeds `head_hash = a95b59f…` plus a **live**
`worktree_diff_sha256` recomputed at run start. New uncommitted experiment code written during
the matrix (new scripts under `scripts/`, new results) will change the live worktree diff, so the
live hash each run records pins the exact code that produced it. The clean-tree baseline value
`e3b0c442…` is the reference. **Nothing is committed by this agent.**
