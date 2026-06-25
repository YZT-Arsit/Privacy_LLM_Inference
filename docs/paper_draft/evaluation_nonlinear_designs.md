# Evaluation: the two nonlinear designs (E15 / E16)

The folded Qwen pipeline can evaluate the transformer nonlinear "islands"
(GELU/SiLU MLP, Softmax, LayerNorm/RMSNorm, and the trusted-softmax shortcut)
under **two designs**, and the advisor requires the *same* full experiment suite
to run under both so either can be chosen later with complete results.

## The two designs

* **`current`** (design A / baseline). The nonlinearity is evaluated **inside the
  trusted boundary** (unmask -> nonlinearity -> remask); the untrusted
  accelerator only ever sees masked/folded linear payloads. This is the design
  validated end-to-end (no-LoRA TDX-lite / TDX-attested decode). Its security
  status is `trusted_boundary` and the claim status is `established`.

* **`trusted_shortcut`** (design B / alternative). The bulk of the nonlinearity
  is **migrated onto the untrusted accelerator** via an Amulet-style lifted view,
  keeping only a small trusted reduction shortcut (softmax/norm denominators) in
  the boundary. Correctness is exact, but its security is **not formally
  claimed** (selector-leakage caveat, under discussion). Its security status is
  `not_formally_claimed`.

Every paper-facing report carries a canonical `nonlinear_backend` field plus a
design metadata hash that binds into the attestation runtime hash, so design A
evidence can never be silently reused for design B.

## Execution status (READ THIS BEFORE CLAIMING trusted_shortcut)

As of the 2026-06 wiring patch, **`trusted_shortcut` is now genuinely EXECUTED in
the real Qwen folded-package / worker / probe / E3 / E9 path — it is no longer a
metadata tag.** The folded worker dispatches its nonlinear ops through
`pllo.deployment.folded_nonlinear.FoldedNonlinearRunner`, which for design B uses
the Amulet-migrated backend (`pllo.nonlinear.amulet_backend`):

* **Which ops are migrated.** The **MLP activation** (SiLU in SwiGLU; GELU is also
  supported) is *lifted* onto the untrusted accelerator via a selector lift —
  this is the Amulet lifted island (`extra.location=untrusted_accelerator`,
  `lift_k>=2`, `trusted_calls=0`, `gpu_bytes>0`). **Softmax** and **RMSNorm** are
  migrated onto the accelerator keeping only a small **trusted reduction
  shortcut** (row-max / mean-square), consistent with the design's name.
* **Which ops stay trusted (design A).** Under `current` every nonlinear island
  is evaluated in the trusted boundary (`silu_reference` / `rmsnorm_core` /
  `torch.softmax`), numerically identical to the historical folded path — so
  existing `current` H800 artifacts are unaffected.
* **Correctness.** The selector lift gathers the valid (scale-1) column, so the
  activation output is exact; RMSNorm/softmax are the standard formulas. The
  package-backed decode under `trusted_shortcut` matches the trusted in-process
  reference token-for-token (`tests/test_trusted_shortcut_real_path_wiring.py`).
* **Security.** Still **not formally claimed** for `trusted_shortcut`
  (`security_claim_status=under_discussion`, selector-leak caveat). Correctness /
  performance / utility evidence is allowed; proven-security claims are refused by
  `final_submission_gate.py` regardless of execution.

Real execution evidence (do not fabricate — it is stamped from measured
`NonlinearOpResult` counters by the worker): a paper-facing `trusted_shortcut`
report carries `nonlinear_op_backend=amulet_migrated`, `amulet_lift_executed=true`,
`lifted_nonlinear_ops_count>0`, `lift_k>=2`, `lifted_gpu_bytes>0`,
`nonlinear_execution_status=lifted_on_accelerator`, plus `migrated_ops_by_type`
(silu/softmax/rmsnorm) and the trusted/accelerator byte split. The capability
stamp from `nonlinear_design_report_fields(...)` sets `amulet_lift_executed=false`
by default; an **execution-bearing** run OVERRIDES it with the worker's measured
evidence (probes call `backend.nonlinear_execution_evidence()`; the remote
path retrieves it from the worker `/health` endpoint).

Honesty guards still enforced (do not bypass):

* An **execution-bearing** `trusted_shortcut` report (decode / prefill / E3 / E9 /
  protocol demo) that lacks the lift evidence above is treated as **tag-only**
  (it ran the `current` path under a design tag) and is rejected by the claim
  validator and `final_submission_gate.py`
  (`trusted_shortcut_not_executed_in_real_path`), and **E15 refuses to compare or
  recommend** it. A non-execution report (a folded-package **build**, which only
  folds weights and never runs a nonlinearity) is design-independent and is NOT
  flagged.
* The GPU worker must be started with the matching `--nonlinear-backend` so it
  actually executes the lift; the attestation runtime hash binds the design, so
  design A evidence can never be replayed for design B.

**Rebuild implications.** The folded *package* tensor format is design-INDEPENDENT
(folding is identical regardless of the nonlinear design) and the design metadata
hash is unchanged, so **the folded packages do NOT need rebuilding**. However,
**any `trusted_shortcut` inference artifacts (decode / E3 / E9 / pairwise reports)
generated BEFORE this wiring are tag-only and MUST be regenerated** with a worker
started under `--nonlinear-backend trusted_shortcut` so the reports carry the real
lift counters. `current` artifacts are unaffected.

## E15 — five comparison tables

`scripts/run_e15_nonlinear_design_comparison.py` consolidates ALL per-design
report JSONs into five tables, each keyed by backend with `None` where a metric
is absent:

1. **Correctness** — `operator_allclose`, `logits_error`/`logits_mae`,
   `token_exact_match`, `task_utility_delta` (pairwise utility), and
   `lora_utility_delta` (E10).
2. **Security** — counts of `gpu_visible_plaintext_fields` and
   `leaked_secret_fields`, `worker_has_mask_secrets`, `worker_has_raw_lora`,
   the transcript scan (pass/fail), the negative tests (`all_passed`), and the
   attestation binding (`runtime_hash_bound`).
3. **Performance** — `prefill_latency`, `decode_latency`, `latency_per_token`,
   `trusted_bytes`, `gpu_bytes`, `boundary_calls`, `peak_gpu_memory`,
   `setup_build_time` (`folded_weight_generation_time_s`), `package_size`
   (`folded_weight_size_gb`), and `amortized_setup_cost` (from an E4 report).
4. **Deployment** — booleans `tdx_lite_supported`, `tdx_attested_supported`,
   `remote_h800_supported`, `lora_supported`, `public_benchmark_supported`,
   derived from the presence of qualifying reports per backend.
5. **Recommendation** — per-axis winners (correctness / security / latency /
   trusted_transfer) plus a final recommendation with a one-line rationale.

## No recommendation without complete evidence

This is the load-bearing honesty rule:

> A concrete recommendation is emitted for an axis only when **both** designs
> have **complete** evidence for that axis. If any required evidence is missing,
> `recommendation_status` is `insufficient_evidence` and `missing_evidence`
> enumerates exactly what is missing (`"<backend>: <what's missing>"`), and **no**
> `final_recommendation` is produced.

Additional conservatism for security: because `trusted_shortcut` has
`security_status = not_formally_claimed`, it is **never** recommended for
security over `current` unless its security evidence is both complete **and**
favorable. When the security evidence does not favor a formally-claimed design,
the final recommendation defaults to the latency winner (correctness being exact
for both designs).

## E16 — nonlinear ablation

`scripts/run_e16_nonlinear_ablation_report.py` isolates what changes when the
nonlinear design is swapped, producing seven ablation rows (design identity,
nonlinear boundary calls, trusted bytes due to nonlinear [trusted_bytes proxy],
latency overhead due to nonlinear [decode-latency delta], security difference,
package-size difference, LoRA compatibility difference) plus a numeric
`deltas_summary` (`trusted_shortcut - current`). It renders Markdown, CSV, and a
LaTeX tabular.

## Placeholder note

The canonical tables in the paper are **generated**, not hand-written: run
`scripts/run_e15_nonlinear_design_comparison.py` (and
`scripts/run_e16_nonlinear_ablation_report.py`) over the **real per-design runs**
of the full suite. Until both designs have complete, paper-ready, non-dry-run
evidence, the comparison will correctly report `insufficient_evidence` and emit
no recommendation.
