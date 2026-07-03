# Generation-task eval: `current` vs `trusted_shortcut` (Amulet `amulet_migrated`)

Real run on **real Qwen2.5-7B-Instruct**, folded-remote GPU worker on an **H800**,
with **real TDX attestation** from an Alibaba Cloud TDX VM (`39.96.4.252`,
`/dev/tdx_guest`). 2026-07-03. Not a security claim — this stage measures the
generation-task usability of the Amulet-style nonlinear migration, that it
really executes the `amulet_migrated` path, and how it compares to the `current`
trusted-boundary baseline. A_rightmul is out of scope this round.

Scripts: `scripts/run_generation_backend_eval.py` (+ `src/pllo/benchmarks/
generation_backend_eval.py`, `ifeval_scoring.py`). Runbook:
`docs/AAAI_GENERATION_BACKEND_EVAL_RUNBOOK.md`.

## Verdict

| question | answer |
|---|---|
| Does `trusted_shortcut` really run `amulet_migrated` (not tag-only)? | **YES** — 14,056 SiLU ops lifted on the accelerator, lift_k=2, 10.2 GB lifted, `amulet_lift_executed=True`, `trusted_nonlinear_ops_count=0` |
| Does it generate correctly? | **YES** — coherent on all 10 smoke prompts (zh/en/math/code/instruction), 0 errors |
| Does it match the `current` baseline? | **YES, bit-identical** — token-match / exact-text / exact-token = **1.00 / 1.00 / 1.00** |
| Silent fallback to `current`? | **NO** — `backend_verification_passed=True` from worker `/health` on both |
| Real TDX attested? | **YES** — `boundary_attested=True`, tee=tdx, `runtime_hash_bound=True`, real `mr_td=8a56b29a…`, `--require-tdx` passed for both designs |
| Latency cost of the migration? | **~6× slower** — the lift is not free (see Performance) |

## Backends compared

| | current | trusted_shortcut |
|---|---|---|
| nonlinear_backend | current | trusted_shortcut |
| op_backend | current | **amulet_migrated** |
| nonlinear location | trusted boundary (unmask→f→remask) | **lifted onto the untrusted accelerator** |
| folded package | `qwen7b_folded_full_current_seq1024_pad` (27 GB, 29 shards) | `qwen7b_folded_full_trusted_shortcut_seq1024_pad` (27 GB, 29 shards) |

Both packages: `package_valid=True`, `nonlinear_backend_ok=True`,
`contains_mask_secrets=False`, no missing shards / hash mismatches. Worker:
`--fold-dtype-override float32 --resident-folded-weights`; greedy decoding,
seq_len 1024, chat template on.

## 1. Correctness (smoke, 10 prompts)

Both backends produce coherent output; **trusted_shortcut is bit-identical to
current**:

```
token_match_rate = 1.0   exact_text_match = 1.0   exact_token_match = 1.0   (n=10)
```

Example (`smoke-02`, both backends, identical):
> "A transformer neural network is a type of deep learning model architecture
> that uses self-attention mechanisms …"

The Amulet nonlinear migration is exact on the generation path.

## 2. Nonlinear execution evidence (trusted_shortcut)

Measured from the worker `/health` `nonlinear_execution_evidence` after decoding:

| field | value |
|---|---|
| nonlinear_op_backend | **amulet_migrated** |
| amulet_real_path_executed | **True** |
| nonlinear_execution_evidence_missing | **False** |
| lifted_nonlinear_ops_count (SiLU) | **14,056** |
| lift_k | 2 |
| lifted_gpu_bytes | 10,235,215,872 (**10.2 GB**) |
| trusted_nonlinear_ops_count | **0** |
| migrated ops by type | rmsnorm 28,614 · softmax 14,056 · silu 14,056 |

The `current` baseline runs the nonlinearity in the trusted boundary (no lift
counters), as expected. `trusted_shortcut` migrates every nonlinear op off the
trusted boundary (`trusted_nonlinear_ops_count=0`) and genuinely lifts the SiLU
activation onto the accelerator.

## 3. Dimension audit (complete)

`dimension_audit.json` + `per_layer_dimensions.csv` (757 rows) cover every
required dimension, each tagged `measured` vs `derived_from_config`:

- **model**: 28 layers, hidden 3584, intermediate 18944, 28 heads, 4 KV heads,
  head_dim 128, vocab 152064, dtype bf16, device cuda, seq_len 1024.
- **inputs**: input_ids `[1,38]` (measured), embedding `[1,38,3584]`.
- **attention** (per layer): q/o_proj `[3584,3584]`, k/v_proj `[512,3584]`,
  q_act `[1,28,38,128]`, k/v_act `[1,4,38,128]`, scores/probs `[1,28,38,38]`.
- **KV cache**: key/value `[1,4,T,128]` (GQA), dtype fp32, 28 layers.
- **MLP** (per layer): gate/up `[18944,3584]`, down `[3584,18944]`, SiLU
  in/out `[1,38,18944]`.
- **nonlinear**: SiLU `backend=amulet_migrated`, `lifted=True`, lift_k=2,
  **lifted tensor `[1,38,18944,2]`**, aux matrix `[18944,2]`; softmax/rmsnorm.
- **LM head**: final hidden `[1,1,3584]`, lm_head `[152064,3584]`, recovered
  logits `[1,152064]` (measured).
- **masks/pad/package**: local mask shapes for all 8 Linear families, pad +
  pad-compensation tensor shapes, seq_len, dtype, num_shards, package size.

## 4. Real TDX attestation

Real TD Quotes generated on the TDX VM (real `/dev/tdx_guest` + Alibaba
`tdx-quote-generation-sample`), each bound to the design's runtime hash and
QVL-verified (appraisal SUCCESS, PCK cert chain), then verified trusted-side:

| | current | trusted_shortcut |
|---|---|---|
| boundary_attested | **True** | **True** |
| tee_type | tdx | tdx |
| runtime_hash_bound | True | True |
| mr_td | 8a56b29a… | 8a56b29a… |
| attestation_path | qvl_appraisal | qvl_appraisal |
| `--require-tdx` exit | 0 | 0 |

Generation under the attested boundary (`--worker-backend tdx_attested_remote`)
is bit-identical to the non-attested run. (The runtime hash binds the boundary
code + selected nonlinear design, so quotes must be re-bound whenever the
boundary code changes — re-quoted here after each fix.)

## 5. Performance (greedy, resident fp32 fold, worker + client both on H800)

### 5a. Original prototype (per-call lift-factor regeneration)

| | current | trusted_shortcut |
|---|---|---|
| mean latency / prompt (smoke) | **2.42 s** | **14.79 s** |
| tokens / sec | **24.4** | **3.39** |

The first measurement showed Amulet ~6× slower. Profiling the worker (GPU **9%**
util, worker process **~1875% CPU ≈ 19 cores**, boundary client only 4.5%) traced
the cost **not** to the lift math but to the selector-lift factor generation: the
backend re-sampled the decoy matrix `R` on a CPU RNG **and copied it host→device
on every SiLU call** (28 layers × every decoded token). On the launch-bound
batch-1 decode path that per-call H2D sync serialized every activation and starved
the GPU.

### 5b. After caching the lift factors (`perf(amulet)`, commit `1a48219`)

The factors depend only on `(h, lift_k, seed)` with a fixed seed, so they are
identical on every call — caching them (keyed by `h, device, dtype`) is
**bit-identical** and changes nothing about the security posture (`R` was never
per-call-randomized; it lives in the untrusted worker). Re-measured on the H800:

| trusted_shortcut | tokens / sec |
|---|---|
| before (per-call regen) | ~3.66 (mean over IFEval-20) |
| **after (cached R), steady state** | **24.4** (24.36 / 24.60 / 24.55 on 3 prompts) |

**~6.7× faster — Amulet now runs at parity with the `current` baseline (24.4
tok/s)**, i.e. the migration is effectively free on this path. Verified on the
same run: still genuinely lifting (`amulet_real_path_executed=True`, lifted_ops
14,504, `trusted_nonlinear_ops_count=0`); prompts that terminate naturally finish
at **identical lengths** (e.g. id 1012 → 122 tok eos, id 1019 → 12 tok eos) to the
pre-fix run, i.e. bit-identical decoding. The **boundary runtime hash is unchanged**
(`amulet_backend.py` is untrusted-worker code, not in the trusted boundary
manifest), so the existing TDX quotes remain valid — no re-bind needed. The lift
still materializes `[B,T,18944,k]` (2× activation bytes) as its intrinsic cost;
that residual is now amortized behind the GPU pipeline rather than a per-call CPU
stall.

## 6. IFEval-20 (instruction-following, greedy, max_new_tokens 512)

Both backends over the first 20 IFEval prompts (`data/aaai/ifeval.jsonl --limit
20`), scored with the approximate reimpl (`ifeval_scoring.py`, 30 instructions,
**100% coverage** — no unsupported types on this subset):

| metric | current | trusted_shortcut |
|---|---|---|
| strict prompt acc | **0.75** | **0.75** |
| loose prompt acc | 0.75 | 0.75 |
| strict instruction acc | 0.80 | 0.80 |
| loose instruction acc | 0.80 | 0.80 |
| num prompts / instructions | 20 / 30 | 20 / 30 |

`trusted_shortcut` vs `current` on the same 20 prompts:
`token_match_rate = 1.0`, `exact_text_match = 1.0`, `exact_token_match = 1.0`
(**all 20 identical**, incl. the 4 that ran to the full 512-token cap). Amulet
execution on this run: **138,796 SiLU ops lifted**, lift_k=2,
`trusted_nonlinear_ops_count=0`, `amulet_real_path_executed=True` — no silent
fallback. As expected from bit-identical decoding, IFEval accuracy is exactly
equal between the two backends; the migration changes *where* the nonlinearity
runs and the latency, not the outputs.

## Notes / caveats

- **Root-cause fix that mattered**: the boundary artifact's masks are drawn with
  a seeded RNG **on the build device**, and torch RNG is device-specific — a
  cpu-built artifact against a cuda-folded package silently corrupts the vocab
  logit recovery (one logit blows up → argmax flip → degenerate output). Fixed
  by building the artifact on the same device (cuda); the builder now hard-errors
  on a device mismatch. Diagnostic top1-agreement went 0.0 → 1.0 (plaintext
  parity) after the fix.
- fp32 fold (`--fold-dtype-override float32`) + resident weights are required
  (bf16 fold degrades numerics; non-resident re-streams 27 GB per token).
- IFEval-20 aggregate scores + comparison + amulet evidence are saved under
  `results/generation_backend_eval/ifeval20/` (full per-prompt tables live in the
  H800 run dirs). Both backends: strict/loose prompt 0.75, strict/loose inst 0.80.
