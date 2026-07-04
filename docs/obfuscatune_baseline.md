# ObfuscaTune baseline

A self-contained, **simulator-only** re-implementation of the obfuscation
scheme from *ObfuscaTune: Obfuscated Offsite Finetuning and Inference of
Proprietary LLMs on Private Datasets* (Frikha et al., arXiv:2407.02960,
PPAI-25 / AAAI-W), for a fair, apples-to-apples comparison against our
amulet-style / trusted-shortcut scheme.

It lives entirely under `src/pllo/baselines/obfuscatune/`, `scripts/obfuscatune/`
and `tests/obfuscatune/` and does **not** touch the amulet / trusted-shortcut
mainline.

## 1. Paper method summary

ObfuscaTune protects a *proprietary model* + *private data* against an
honest-but-curious cloud provider by combining a TEE with a linear obfuscation:

- The **low-parameter** layers (token/positional embeddings, LayerNorm, softmax,
  activation, LM head; ~5% of params) stay **inside the TEE**.
- The **high-parameter** attention/MLP linear layers are **obfuscated** with
  random matrices and executed **outside the TEE**:

  Input projections (Q/K/V, MLP first linear):

      X* = X Ra ,   W* = Ra^{-1} W     ⇒   X* W* = X W

  Output projections (attention output proj, MLP second linear):

      W* = W Rb ,   O* = H W* = H W Rb ,   O = O* Rb^{-1} = H W

- Because the masks cancel, the untrusted side sees the **TRUE plaintext**
  Q/K/V and the MLP intermediate. Security therefore rests on the model
  **weights being secret** — it is **not** an open-weight defense.
- Every **non-linearity** (LayerNorm, softmax, activation) runs **inside the
  TEE** on de-obfuscated values, so each block round-trips the TEE boundary
  multiple times.
- Numerical accuracy requires **low-condition-number** matrices. The paper uses
  **orthogonal** matrices (κ = 1, inverse = transpose, error-free). App. B gives
  a construction for a **prescribed** κ via `R = QA S QB^T`.

## 2. What we implement / what we don't

**Implemented (exact primitive, numerically tested):**
- Random-matrix generators: orthogonal (κ=1), naive Gaussian invertible
  (uncontrolled κ, `torch.linalg.inv`), and prescribed-κ (`QA S QB^T`, App. B).
- Obfuscated-linear primitive for both input and output projections, for
  `torch.nn.Linear` and HF GPT-2 `Conv1D` weight layouts.
- Simulated-TEE modules (linear / attention-projection / MLP) with per-phase
  boundary accounting (matmul counts, transfer bytes, condition numbers).
- A block-level and full-model HF GPT-2 wrapper (eval), reproducing GPT-2's
  exact attention/MLP math while routing the six linears through obfuscation.
- Four run modes: `unprotected`, `obfuscatune_orthogonal`, `obfuscatune_random`,
  `obfuscatune_cond_sweep`.
- The condition-number ablation (Table 2 numerical-error trend).
- A LoRA finetuning **smoke test** (loop runs, loss drops).
- The unified comparison schema (`metrics.comparison_row`).

**NOT implemented (out of scope, honestly labeled):**
- The full nanoGPT LoRA-finetuning pipeline + `lm-eval-harness` on
  WebQs/OBQA/PIQA/SciQ (`full_system_reproduced = False`).
- A real TEE — the TEE/outside split is a **simulator** with counters.
- Any empirical attack evaluation — security is captured as structural
  declarations + proxy metrics only.

## 3. File structure

```
src/pllo/baselines/obfuscatune/
  __init__.py              # re-exports (legacy: ObfuscaTune/ObfuscaTuneConfig/matrix_with_condition_number)
  config.py                # ObfuscaTuneConfig + method-name helpers
  random_matrices.py       # orthogonal / gaussian_invertible / matrix_with_condition_number / condition_number
  linear_obfuscation.py    # obfuscate_input_right / weight_left / weight_output_right / deobfuscate_output_right + Linear/Conv1D adapters
  modules.py               # ObfuscaTuneLinearSimulator / AttentionProjectionSimulator / MLPSimulator (+ PhaseMetrics)
  metrics.py               # correctness / cost / numerical proxies + comparison_row schema
  hf_gpt2_obfuscatune.py   # tiny-random GPT-2 + block/full obfuscated forward + run_mode
  protocol.py              # ObfuscaTune(BaselineProtocol) registry entry (legacy API preserved)

scripts/obfuscatune/
  _common.py                       # shared arg parsing / model building / IO
  run_correctness_check.py         # unprotected vs orthogonal vs random
  run_condition_number_sweep.py    # κ ∈ {1,8,32,128,160} + random (Table 2 trend)
  run_latency_profile.py           # wall-time mean/std/p50/p95 (CPU simulator proxy)
  run_lora_smoke.py                # LoRA loop smoke (torch backend default; optional peft)
  summarize_obfuscatune_results.py # summary .md + .csv + comparison rows

tests/obfuscatune/
  test_random_matrices.py
  test_linear_obfuscation.py
  test_hf_gpt2_obfuscatune_correctness.py
  test_condition_number_effect.py
```

All model loading is **offline**: either a tiny random `GPT2Config` (no
download) or a local path with `local_files_only=True`.

## 4. Running the local correctness check

```bash
python scripts/obfuscatune/run_correctness_check.py --dry-run          # print planned config
python scripts/obfuscatune/run_correctness_check.py --tiny-random-config
python scripts/obfuscatune/run_correctness_check.py \
    --model-name-or-path /path/to/local/gpt2 --seq-len 32 --dtype float32
```

Writes `outputs/obfuscatune/correctness/correctness_<dtype>_L<layers>.json`.
Expected: `obfuscatune_orthogonal` max-abs logit error ~1e-7 (fp32) / ~1e-16
(fp64) with argmax match 1.0; `obfuscatune_random` larger but finite.

## 5. Running the condition-number sweep

```bash
python scripts/obfuscatune/run_condition_number_sweep.py --tiny-random-config
python scripts/obfuscatune/run_condition_number_sweep.py \
    --kappas 1 8 32 128 160 --dtype float32
```

Writes `.json` / `.csv` / `.md` under `outputs/obfuscatune/condition_sweep/`.
Reproduces the Table 2 **trend** (error grows monotonically with κ; naive random
worst) as a numerical-error curve, not downstream QA accuracy.

## 6. Running the latency profile

```bash
python scripts/obfuscatune/run_latency_profile.py --tiny-random-config \
    --num-runs 20 --warmup 3
```

Writes `outputs/obfuscatune/latency/latency_<dtype>_L<layers>.json`. This is a
**CPU-simulator wall-time proxy** (obfuscation matmuls + boundary bookkeeping),
NOT a real-TEE deployment measurement — do not quote it as a TEE latency number.

### LoRA smoke + summary

```bash
python scripts/obfuscatune/run_lora_smoke.py --tiny-random-config --steps 40
python scripts/obfuscatune/run_lora_smoke.py --backend peft   # optional; errors clearly if peft absent
python scripts/obfuscatune/summarize_obfuscatune_results.py
```

## 7. Output field reference

**correctness JSON** (`results[]`): `model_name`, `dtype`, `seq_len`,
`batch_size`, `num_layers`, `mode`, `max_abs_error_vs_unprotected`,
`mean_abs_error_vs_unprotected`, `relative_l2_error`,
`logits_argmax_match_rate`, `nan_or_inf_count`, `condition_number_stats`
(`mean`,`max`), `boundary_calls`, `tee_transfer_bytes`,
`exposed_plaintext_tensors`, `wall_time_ms`.

**condition_sweep JSON** (`results[]`): `condition_number`, `matrix_type`,
`max_abs_error`, `mean_abs_error`, `relative_l2_error`,
`logits_argmax_match_rate`, `nan_or_inf_count`, `condition_number_measured_mean`.

**latency JSON** (`results[]`): `mode`, `wall_time_ms_{mean,std,p50,p95}`,
`slowdown_vs_unprotected`, `num_runs`.

**lora_smoke JSON**: `config` (incl. `lora_backend`, `rank`, `alpha`, `dropout`,
`steps`, `n_trainable_params`), `initial_loss`, `final_loss`, `loss_decreased`,
`losses[]`, `structural_note`.

## 8. Comparing with the amulet-style scheme

Use `metrics.comparison_row(...)` to emit a row of the unified schema
(`method`, `correctness`, `security_proxy`, `cost`, `numerical`). Method labels:
`unprotected`, `amulet_style`, `trusted_shortcut`, `obfuscatune_orthogonal`,
`obfuscatune_random`, `obfuscatune_cond_{8,32,128}`.

**Keep the `security_proxy.notes` field.** ObfuscaTune and our scheme have
**different threat models** and their security claims must not be conflated:

| | ObfuscaTune | Ours (amulet-style / A_rightmul) |
|---|---|---|
| Protects | proprietary **model weights** + private data | user **input** / LoRA / KV cache / logits |
| Base model | **secret** (weights obfuscated) | **public** |
| Q/K/V outside TEE | **plaintext (exposed)** | masked |
| Non-linearities | **inside TEE** (per-block crossings) | on untrusted GPU (single entry/exit) |
| TEE auth | assumed | attested (TDX) |

Only the **correctness** and **cost** columns are directly comparable across the
two; the security columns describe different guarantees.

## 9. Current limitations

- **Simulated TEE only** — not a real TEE; the split is bookkeeping.
- Correctness is validated on **GPT-2 / tiny-random GPT-2** (offline). Larger
  models are supported via `--model-name-or-path` but untested here.
- **LoRA is a smoke test** — verifies the loop runs and the loss drops; it does
  **not** reproduce WebQs/OBQA/PIQA/SciQ, and it trains in plaintext (the
  obfuscation is validated on the inference path).
- Security is **structural annotation + proxy metrics** only; no attack eval.
- **No network access / no auto-install**: offline model loading; the `peft`
  LoRA backend is an optional import that errors clearly when absent.
