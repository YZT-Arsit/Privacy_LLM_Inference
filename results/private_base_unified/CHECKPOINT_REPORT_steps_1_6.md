# Private-base main line — first checkpoint report (steps 1–6)

Per the plan, work STOPS here for review before the first new H800 run. No H800/TDX
run was performed this stretch. All numbers below are CPU/fp64 or static-audit; none
is a real Qwen/GPU result.

## 1. Frozen private-base design
`design_spec.md` / `design_spec.json` / `threat_model.md` / `experiment_matrix.md`
(FROZEN v1.0). Encodes A–H: masked residual `H_tilde=H·N`; weights
`W_tilde=N_in^{-1}·W·N_out` (GPU sees only `W_tilde`); QKV `Q_tilde=RoPE(Q)·R`,
`K_tilde=RoPE(K)·R^{-T}`, `V_tilde=V·S` (score-preserving for any invertible R);
masked append-compatible KV cache; `paper_safe`/A_rightmul/0-trusted-nonlinear/
0-plaintext-hidden; O(V) monomial logits; attention-score exposure recorded; and
that non-orthogonal masks are a **candidate, not proven** defence. The archived
`rank_masked_sgd_50step` (plaintext base+activations) is frozen and excluded from any
private-base claim.

## 2. Plaintext-weight exposure audit — **VERDICT: FAIL** (as it must, today)
`code_alignment/matrix.{json,md}` (25 components × 14 fields),
`plaintext_exposure_inventory.md`, `missing_integration.md`.
- **Central finding:** the current **training** path loads a full plaintext HF
  checkpoint onto the untrusted GPU (`gate3_worker.py:147 load_model → .to("cuda")`,
  `:218 model(input_ids)`, `:219 out.logits.float()`), materialising plaintext
  embeddings, hidden states, and logits. Only LoRA (rank `U`) + a logit permutation
  are masked. The default in-process generation worker re-folds in the untrusted
  process too (`gpu_worker.py:205 → qwen_masked_session.py:165`).
- The **compliant** masked-base path exists only for **inference** (`folded_worker` +
  package backend load pre-folded `*_tilde` shards, name-screened, mask secrets
  TDX-only). Masked-base **training** is proven only as a CPU contract.
- 5 components have `training_support=no` (embedding, decoder-layer family, KV cache,
  prefill, decode). Attention-score exposure recorded as by-design.
- Consequence: **no private-base run may use the current plaintext-loading trainer.**
  Reaching PASS is integration (point the trainer at a TDX-built folded package +
  masked embeddings through the existing shared kernels), not new crypto.

## 3. Transformed-package design (packager + scanner + guard)
`src/pllo/deployment/private_base_package.py`; `private_package/*`; 10/10 tests.
- Packager writes **transformed `*_tilde` artifacts only**; fail-closed refusal of
  untransformed names, mask-secret names (`n_res/_inv/pi/perm/d_vocab/...`), and
  plaintext model-tensor hints. Manifest asserts `contains_plaintext_*` /
  `contains_mask_secrets = False` and records the 4 mask domains by name only.
- `scan_package_for_plaintext` independently flags plaintext shards, serialized
  masks, caches, temp/debug dumps (verified on planted positives; clean on a clean
  package).
- `assert_private_base_or_abort` (`private_base_required=True`) aborts if a plaintext
  checkpoint/model is on the GPU path. (Synthetic package here; the real package is
  built from the real checkpoint at run time, then scanned + hash-pinned.)

## 4. LoRA-aware folded-forward — implementation status
`src/pllo/experiments/unified_masked_training.py` (+ shared kernels
`masked_training_kernels.py`, monomial `monomial_logit_mask.py`). Extended to **all 7
projections** (q/k/v/o/gate/up/down), differentiable `W_eff_tilde = W_base_tilde +
A_tilde·B_tilde` with correct per-projection in/out mask bases; base term frozen, masks
constant. **CPU-validated only** — NOT yet wired to the real H800 Qwen worker (that is
the next engineering step, gated to after this checkpoint).

## 5. CPU full-block correctness (C1) — closes at fp64 epsilon
`code_alignment/c1_cpu_contract.json`; 14/14 tests. One synthetic Qwen block, 7-proj
LoRA: forward recovery 1.3e-15, residual mask-invariant 3e-16 (genuinely masked),
private CE diff 0.0, gradA/gradB recovered 1.9e-16, masked SGD 1.1e-16, next-step
logits 8.9e-16. Counters all 0: `plaintext_base_weight_materializations`,
`plaintext_embedding_materializations`, `plaintext_hidden_materializations`,
`nonlinear_trusted_calls`, `trusted_nonlinear_ops_count`, `packed_update`,
`trusted_optimizer_calls`, `silent_fallbacks`. Masked embedding lookup verified masked.
Mask spaces separated (feature / lora_rank / nonlinear_permutation / vocabulary).

## 6. O1 optimizer conclusions (A / B / C)
`o1_optimizer/*`; 5/5 tests. All three recover plaintext SGD (10-step max |Δloss|:
O1-A 1.8e-15, O1-B 2.7e-15, O1-C 0.0). The difference is leakage + invocations:
- **O1-A** orthogonal direct masked SGD (2 inv/step) — no Gram exposed; but orthogonal
  masks are what self-Gram alignment recovers → **weak** private-weight defence.
- **O1-B** non-orthogonal preconditioned SGD (2 inv/step) — needs the mask Gram
  `(PP^T)·g·(Q^TQ)`; naive masked SGD is WRONG for non-orthogonal (err 0.11→0.20).
- **O1-C** trusted optimizer, packed grads (3 inv/step) — no Gram on GPU.

## 7. Do the preconditioners leak the mask Gram? — **YES for O1-B**
The O1-B update multiplies the masked gradient by `PP^T` / `Q^TQ` **on the GPU** — the
exact mask Gram the non-orthogonal masking was meant to hide (strictly more than the
self-Gram attack extracts). Under the RMSNorm constraint the left feature-Gram is
trivial (`=I`) for q/k/v/gate/up (orthogonal residual input), but `S^TS` (o_proj) and
`U^TU` (rank) are exposed. **O1-B is NOT paper-safe.** O1-C avoids the leak (trusted
side) at +1 crossing. A custom backward is not assumed sufficient vs an offline
package attacker.

## 8. Proposed main deployable profile (pre-W1)
- **Mainline feasibility/correctness:** **O1-A** — orthogonal feature+rank masks,
  direct GPU masked SGD, 2 trusted inv/step, no Gram leak, `paper_safe`. Same
  invocation profile as the archived rank-SGD, now over a **masked base**.
- **Non-orthogonal reference:** **O1-C** — held for S1 comparison (3 inv/step).
- **Decisive caveat:** under `paper_safe` the residual mask is forced ORTHOGONAL by
  RMSNorm, so the residual weight chain `W_tilde=N^{-1}WN` is orthogonally masked
  regardless of optimizer. Whether that chain is recoverable is the **W1 go/no-go** —
  if W1 breaks it, private-base needs MORE LAYERS BEHIND THE TEE, not a different
  optimizer. Do not commit to a final profile before W1/S1/A1.

## 9. Unresolved blockers
1. **Provenance FAIL** — wire the trainer to the TDX-built folded package + masked
   embeddings (real integration) before any private-base run.
2. **OI-1 (RoPE ordering)** — realise Q/K masking with 0 plaintext `Q_rope`
   materialisation (RoPE-commuting output basis, or RoPE in the masked basis).
3. **RMSNorm ⇒ orthogonal residual mask** — non-orthogonal cannot protect the
   residual weight chain under `paper_safe`; strong base-weight security may require
   more TEE layers (W1 decides).
4. **O1-B Gram leak** — not paper-safe; needs trusted-side (O1-C) or an unproven
   custom backward.
5. **Attention scores exposed** by design (not claimed confidential).
6. **Git push blocker** — a 485 MB `.pt` blob is in 2 UNPUSHED commits
   (92b8610, c09a60a). `.gitignore` now covers `*.pt` and the files are untracked
   (staged, no commit), but the blob remains in those 2 commits and will fail the
   push until they are rewritten (e.g. `git filter-repo --strip-blobs-bigger-than
   90M`, or an interactive rebase dropping the file). Low-risk (unpushed) but rewrites
   history — NOT done unprompted.
7. Real packager must build from the actual checkpoint + pin its hash into
   `report_data`; the runtime guard must be wired into the real worker startup.

## 10. Files changed
**New source:** `src/pllo/deployment/private_base_package.py`,
`src/pllo/experiments/private_base_optimizer.py`. **Extended:**
`src/pllo/experiments/unified_masked_training.py` (7 projections + private-base
counters + mask-space record). **From the prior stage (part of paper_safe):**
`execution_profile.py`, `unified_training_config.py`, `masked_training_kernels.py`,
`monomial_logit_mask.py`. **Config:** `.gitignore` (large-artifact patterns).
**Reports:** `results/private_base_unified/**` (design, threat, experiment matrix,
code_alignment/{matrix,exposure,missing,c1}, o1_optimizer/*, private_package/*).
Nothing committed.

## 11. Tests (67 new/updated, all green)
`test_execution_profile.py` 16 · `test_monomial_logit_mask.py` 10 ·
`test_unified_masked_training.py` 14 · `test_unified_training_config.py` 12 ·
`test_private_base_optimizer.py` 5 · `test_private_base_package.py` 10.

## Explicit non-claims
- Private-base security is NOT claimed from "the package is transformed."
- Non-orthogonal-mask security is NOT claimed (pending W1/S1/A1).
- Attention-score confidentiality / zero-leakage / formal security are NOT claimed.
