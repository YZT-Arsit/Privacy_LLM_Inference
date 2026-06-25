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

A code audit (2026-06) established that **`trusted_shortcut` is currently a
correctness/efficiency PROTOTYPE only — it is NOT wired into the real Qwen
folded-package / worker / probe / E3 / E9 path.** The real folded worker
(`src/pllo/deployment/folded_worker.py`) imports only the *current*
trusted-island primitives (`pllo.ops.nonlinear_islands`); it never invokes the
Amulet-style lifted backend (`pllo.nonlinear.amulet_backend` /
`pllo.ops.amulet_lifted_islands`). Selecting `--nonlinear-backend
trusted_shortcut` on the real path today only **tags** the design into report
metadata + the attestation runtime hash — **the lift does not execute**; the run
actually computes the `current` trusted-boundary nonlinearity.

The Amulet backend itself is real and exercised at the op level (the registry
maps `trusted_shortcut -> amulet_migrated`; `AmuletMigratedNonlinearBackend.gelu`
lifts onto the untrusted accelerator: `extra.location=untrusted_accelerator`,
`lift_k>=2`, `trusted_calls=0`, `gpu_bytes>0` — see
`scripts/run_nonlinear_backend_microbench.py` and
`tests/test_trusted_shortcut_execution.py`). It is the *integration into the real
Qwen pipeline* that is not yet done.

Consequences (enforced in code, do not bypass):

* `trusted_shortcut` is **non-paper-facing** until wired. `build_qwen7b_folded_package.py`,
  the folded probes, `run_e3_remote_decode_scaling.py`, and (under `--require-real`)
  `run_e9_task_utility_benchmark.py` **refuse** a real `trusted_shortcut` run
  (exit nonzero) unless `--allow-unwired-nonlinear` is given for an explicitly
  non-paper prototype, or `--dry-run`.
* A report has genuine execution evidence only if it carries
  `nonlinear_op_backend=amulet_migrated`, `amulet_lift_executed=true`,
  `lifted_nonlinear_ops_count>0`, `lift_k>=2`, `lifted_gpu_bytes>0`. The default
  report stamp sets `amulet_lift_executed=false` / `nonlinear_execution_status=
  tag_only_prototype_not_wired`; only a wired real path may override these from
  measured `NonlinearOpResult` counters.
* The claim validator and `final_submission_gate.py` fail any tag-only
  `trusted_shortcut` evidence with `trusted_shortcut_not_executed_in_real_path`,
  and **E15 refuses to compare or recommend** a tag-only `trusted_shortcut`
  (`recommendation_status=insufficient_evidence`).

**To make `trusted_shortcut` paper-facing**, wire
`op_backend_for_design(nonlinear_backend)` + `make_nonlinear_backend(op_backend)`
into the real worker/boundary nonlinear path, collect the `NonlinearOpResult`
counters into the report (`amulet_lift_executed` / `lifted_nonlinear_ops_count` /
`lift_k` / `lifted_gpu_bytes` / `trusted_calls` / `trusted_bytes` / `gpu_bytes`),
then flip `_REAL_PATH_EXECUTION["trusted_shortcut"]` in
`src/pllo/experiments/nonlinear_designs.py` to the executed status. **Any
`trusted_shortcut` H800 inference artifacts generated before that wiring are
tag-only and MUST be regenerated** (the folded *package* itself is design-
independent and need not be rebuilt; only the inference/decode/E3/E9 reports must
be re-run once the lift actually executes). `current` is unaffected.

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
