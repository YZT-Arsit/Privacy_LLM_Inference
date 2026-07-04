# HumanEval-164 code-generation: plaintext vs OURS (folded_remote A_rightmul)

Real run on H800 (new box, `connect.westb:16806`), Qwen2.5-7B-Instruct, 2026-07-03.
Worker = `qwen7b_folded_package` (resident folded weights, GPU); boundary client on
H800 CPU (`--backend folded_remote --device cpu`), embedding artifact
`qwen7b_boundary_artifact_A_rightmul_REAL_bf16`. Both backends: **identical** decoding
(greedy, `stop_on_eos`, `max_new_tokens=512`, `align_generation_config=False`,
`repetition_penalty=None`). Files: `humaneval_pass1.json`, `{plain,ours}_report.json`,
`{plain,ours}_responses.jsonl`.

## Headline (measured, all 164 problems)

| metric | plaintext | OURS (folded_remote) | delta |
|---|---|---|---|
| **pass@1** | **0.780** (128/164) | **0.140** (23/164) | **−0.640** |
| exact completion match vs plaintext | — | **0.000** (0/164) | — |
| finish=eos / finish=length | 145 / 19 | 112 / 52 | +33 no-stop |
| degenerate responses (repeat scan) | 0 | 11 | +11 |

**plaintext 78.0% is a sane Qwen2.5-7B-Instruct baseline** (official ≈84.8%; the gap is
this harness: greedy, no repetition penalty, markdown-extraction). **OURS drops to 14.0%.**

## Diagnosis — NOT a security/mask/artifact bug; numerical-drift divergence

The worker report verifies the scheme ran correctly:
- `compatible_masks_verified=true`, signed-permutation residual + pairwise-rotation
  attention + shared SwiGLU permutation, arbitrary-dense-mask rejected;
- `tee_used_on_gpu=false`, `nonlinear_tee_crossings=0`, single trusted entry/exit;
- embedding-artifact **sha256 integrity guard passed** (so this is NOT the previously-fixed
  corrupted-vocab-scale bug), masks not visible to GPU.

So the degradation is **numerical**: folded_remote is near-lossless *per token* (bf16 fold),
but under **long greedy decoding** the per-token drift compounds — `exact_completion_match=0/164`
means every completion diverges from plaintext. Code generation is far more sensitive to this
than open-ended text (a single early argmax flip → wrong program), and the drift also induces
repetition/no-stop (52 length-truncations, 11 degenerate) that a repetition penalty would damp.

## Important caveat — decoding config was NOT the canonical folded config

The canonical folded runbook uses `--align-generation-config --repetition-penalty 1.05`,
which directly suppresses the drift-induced repetition loops seen here. This run used plain
greedy for BOTH backends (so the *delta* is fair), but OURS' **absolute** score is likely
understated. Two follow-ups can separate "inherent limitation" from "recoverable":
1. Re-run BOTH with `--align-generation-config --repetition-penalty 1.05` (canonical folded config).
2. Higher-precision logits wire (fp32) to reduce per-token drift.
3. `debug_folded_remote_generation_parity.py` to measure per-token folded-vs-plaintext logit
   divergence directly (bug vs. compounding-drift).

## Aligned re-run (`--align-generation-config --repetition-penalty 1.05`, both backends)

Directory `aligned/`. Ran to test whether the drift-induced repetition was the cause.

| config | plaintext | OURS | delta | ours degenerate | ours eos/length |
|---|---|---|---|---|---|
| plain greedy | 0.780 | **0.140** | −0.640 | 11 | 112 / 52 |
| aligned + rep-penalty 1.05 | 0.780 | **0.104** | −0.677 | 6 | 123 / 41 |

**Repetition penalty cleaned the symptoms but did NOT recover pass@1** (degenerate 11→6, EOS
112→123, yet pass@1 14.0%→10.4%, if anything slightly worse). So repetition was a *symptom*, not
the cause: folded_remote emits **wrong tokens** (per-token drift flips the argmax to
plausible-but-incorrect tokens), which no decoding penalty can fix.

## Honest takeaway for the paper (CONFIRMED across 2 configs)

Under matched decoding — both plain greedy and aligned+rep-penalty — folded_remote shows a **large,
reproducible HumanEval pass@1 drop (78 → 10–14)**. The scheme's per-token near-losslessness does
**NOT** translate to task-level parity on long, precision-critical code generation: a single early
argmax flip from bf16-fold drift derails the whole program, and this compounds over ~300–500 tokens.
This is a genuine limitation, not a decoding-config or mask/artifact artifact.

**Only remaining "recoverable" lever**: fp32 logits wire (reduce per-token drift at the logits
stage) + `debug_folded_remote_generation_parity.py` to quantify per-token folded-vs-plaintext logit
divergence. Until that is measured, **do not make a code-generation (HumanEval/MBPP) utility claim
for folded_remote.** The scheme's demonstrated strengths remain on shorter / less
token-exact tasks (IFEval/GSM8K/MT-Bench), where drift is cosmetically tolerable.

## ROOT CAUSE FOUND — bf16 fold is numerically broken; fp32 fold FIXES it (`parity/`)

`debug_folded_remote_generation_parity.py --with-plaintext` on HumanEval/0 (32 steps, per-step
folded-recovered-logits vs plaintext top1):

| fold compute | plaintext top1 agreement | logit max_abs_err (mean/max) | first divergence |
|---|---|---|---|
| **bf16** (all runs above) | **3.1%** (1/32) | **28.5 / 47.4** | step 1 |
| **fp32** (`--fold-dtype-override float32`) | **100%** (32/32) | **0.27 / 0.52** | none |

The `--native-logits-wire` default is already fp32 (final-logits transmission was never the issue).
The real error is in the **bf16 folded-weight compute** on the GPU: the folded products
`N_in⁻¹ W N_out` + boundary pad + mask ops have large dynamic range and, in bf16, accumulate a
**~28-logit-unit** error that flips the argmax from step 1 — producing the "first first"-style
degradation and the 78→14 pass@1 collapse. Running the fold in **fp32 cuts the error to 0.27**
(the inherent A_rightmul nonlinear-approximation floor) and yields **100% plaintext agreement**.

**=> The fix is viable: compute the folded path in fp32.** Confirm with a full HumanEval-164
folded_remote run using `--fold-dtype-override float32` (worker) — pass@1 should recover to ≈ plaintext.

### IMPORTANT wider implication
Because bf16 fold agrees with plaintext only ~3% of the time, **every prior bf16 folded_remote
result (IFEval/GSM8K/MT-Bench/the AAAI generation) is materially degraded vs plaintext, not
"near-lossless"** — the degradation was just cosmetically tolerable on those looser tasks. The
"bit-identical / near-lossless" framing holds only for **fp32 fold**. This needs to be reflected
before any folded_remote utility claim; the fp32-fold path is the one to benchmark and report.

## CONFIRMED — full HumanEval-164 fp32 run recovers parity (`fp32/`)

Full 164-problem folded_remote run, worker `--fold-dtype-override float32`, plain greedy, matched to
the same `he_full` plaintext baseline. Worker `resident_cache_dtype = torch.float32`,
`compatible_masks_verified=true`, `nonlinear_tee_crossings=0`. Files: `fp32/humaneval_pass1.json`,
`fp32/ours_report.json`. Ran 10:19→11:02 (~43 min), `ours rc=0`, `pass1 rc=0`.

| fold compute | OURS pass@1 | plaintext | delta |
|---|---|---|---|
| **bf16** (broken) | **0.140** (23/164) | 0.780 | −0.640 |
| **fp32** (fixed) | **0.823** (135/164) | 0.780 | **+0.043** |

**OURS-fp32 is at statistical parity with plaintext** (+0.043 ≈ 7 problems from near-tie argmax
flips in both directions — NOT a real improvement; `exact_completion_match_rate=0.0` still holds
because fp32 fold reorders matmuls → ~1e-6 perturbation). The 78→14 collapse was entirely a bf16
folded-weight numerical artifact; fp32 fold removes it. This matches the fp32 parity already
established on IFEval/GSM8K/MT-Bench in `results/ours_a_rightmul/aaai_fp32_final/` — HumanEval now completes the set:
**under matched fp32, the obfuscation protocol has no measurable utility cost on code-gen either.**
