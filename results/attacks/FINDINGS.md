# Adversary probe — findings (in progress)

Reference defense being compared: **CONJFORMER** (Yukhimchuk et al., arXiv 2606.16461,
Jun 2026) — orthogonally-equivariant transformer for SPLIT INFERENCE (client rotates
embeddings by a secret orthogonal U; ConjFormer = scalar-RMSNorm + blockwise weight
conjugation makes the body exactly O(d)-equivariant). Their attack suite: embedding/model
inversion (§3.1), norm-based invariant (§3.2), two-sided ALS-Procrustes (§3.3), Gram-based
weight alignment (§3.4). **Correction (2026-07-02, from reading the actual PDF):** CONJFORMER's
defense is the *rotation*, not fine-tuning — they state embedding inversion "remains highly
effective even after fine-tuning" and use fine-tuning only for (a) utility recovery (+0.4%
perplexity) and (b) removing the embedding *norm* structure for the norm attack (App. D.4).
Crucially, **page 4 explicitly lists attention logits (and norms, pairwise distances,
repeated-token patterns) as orthogonal invariants that "remain visible to the server," a
limitation they did NOT evaluate.** So our A3 attention-fingerprint result does NOT refute
them — it QUANTIFIES their acknowledged, unevaluated leak, at 7B (they used GPT-2 / Llama-1B).
Also: **Lin et al. EDNN** (EMNLP 2024) — element-wise differential NN + the
"fixed-point nonexistence" security requirement for embedding obfuscation.

## Setup correction (important)

Our A_rightmul folded masks are **orthogonal and structured** (signed permutation for
residual/input, pairwise rotation for attention), *required* by the compatible-mask
conditions — NOT general-invertible. So the earlier hope "non-orthogonal N_in defeats
Procrustes/norm" does **not** hold as stated. The real picture:

- We do **not** fine-tune → folded weights use the **exact public W** (`W̃ = N_in⁻¹ W N_out`,
  `W = W_public`). CONJFORMER instead relies on fine-tuning to make `W̃ ≠ W` so alignment
  degrades. So our exact-W setting is a priori *easier* for weight alignment — must test.
- We add a **linear-boundary additive pad**: GPU input operand `X̃ = (X − T) N_in`.

## Synthetic probe results (scripts/attacks/synthetic_alignment_probe.py)

d=64, vocab=2000, 5 families sharing the input mask. Numbers = token-recovery % (top1/top10).

| attack | no pad | pad 0.1 | pad 0.5 |
|---|---|---|---|
| A0 raw NN (no defense, ceiling) | 100 / 100 | — | — |
| A1 norm-based | **100** / — | 1.95 / — | 0.39 / — |
| A2 shared-mask alignment (no pad-knowledge) | ~0 / 1–8 | ~0 / 1–8 | ~0 / 1–6 |

random baseline ≈ 0.05% top1, 0.5% top10.

### Solid, defensible conclusions
1. **RETRACTED (2026-07-02): the additive pad does NOT defeat the norm attack.**
   Rigorous data-flow proof (folded_worker.py / embedding_artifact.py): the GPU's actual
   layer-0 input is `h_tilde = X @ N_0` where `N_0` is the ORTHOGONAL residual mask and the
   input is PAD-FREE (the linear-boundary pad is applied only inside `_linear` when forming the
   q/k/v operand — it never enters the residual stream / RMSNorm input). Orthogonal `N_0`
   preserves row norms, so `‖h_tilde_row‖ = ‖X_row‖` EXACTLY → the true per-token embedding
   norm leaks to the GPU regardless of the pad. Synthetic confirms: norm attack on the real
   `h_tilde` = 100% top-1 at every pad scale (0.0/0.1/0.5); the pad only perturbs the *padded
   Linear operand* `(X−T)N_in`, which is NOT the surface that carries the embedding norm.
   → The norm attack is a REAL, currently-unmitigated vulnerability, and WORSE for us than for
   CONJFORMER: they flatten embedding-norm structure by fine-tuning the embedding table
   (their App. D.4); we do not fine-tune, so that mitigation is unavailable. (Synthetic uses
   distinctive per-token norms so it reads 100%; on real Qwen top-1 will be lower because many
   tokens share norms, but the exact structural leak `‖h_tilde‖=‖X‖` holds.)
   Open mitigation options: residual-stream additive pad (needs compensation through every
   nonlinear island — defeats the boundary-local design), pre-mask embedding normalization
   (changes semantics), non-orthogonal residual mask (breaks A_rightmul correctness), or honest
   disclosure as a limitation.
2. **CONJFORMER's two-sided weight-alignment attack is structurally unidentifiable against us.**
   Because each Linear family uses an **independent** output mask `N_out^i` (vs CONJFORMER's
   single **global** right factor `U`), the per-family objective `‖W̃_i − A W_i B_i‖` is **zero
   for every orthogonal A** (A·W_i and W̃_i always share singular values ⇒ some orthogonal B_i
   fits exactly). The attack that works for CONJFORMER (aggregate the shared U across blocks via
   Gram) has no shared right factor to aggregate here.

### Open question (NOT yet resolved — do not claim "safe")
3. The only structure left is the **shared input mask** `N_in` (common to q/k/v/gate/up), attackable
   via the **left Gram** `W̃_i W̃_iᵀ = N_in⁻¹ (W_i W_iᵀ) N_in⁻ᵀ` (right mask cancels, `W_i` known).
   Our Riemannian solver drives the Gram residual to ~0 but the recovered A does **not** invert the
   mask (token recovery near-random). This is ambiguous between (a) genuine non-identifiability
   (joint orthogonal stabilizer of `{W_iW_iᵀ}` larger than ±I / discards phase) and (b) our solver
   hitting a spurious residual-0 minimum. Must be settled by a clean identifiability argument +
   validation on the **real Qwen-7B weights** before any security claim.

## A3 — Attention-logit fingerprint attack: FULL BREAK (most severe finding)

scripts/attacks/attention_fingerprint_probe.py. Result: **100% top-1 token recovery** with as
few as **12** (layer,head) self-attention logits; `mask_invariance_maxerr = 0.0` and
`pad_invariance_maxerr = 0.0` (exact).

Why it works and why it is structural:
- A_rightmul certifies `attention_qk_scores_preserved` (needed for correctness) ⇒ the GPU softmax
  runs over the TRUE attention logits `Q̂K̂ᵀ = QKᵀ`. The single-position self-attention logit per
  (layer,head) is a deterministic, mask-INVARIANT fingerprint `f(token) ∈ R^{B·H}`.
- The attacker recomputes `f` for the whole vocab from PUBLIC weights (mask-invariance makes the
  observed and table `f` match), then NN-matches. It can obtain the observed `f` by running
  single-token offline inference on the masked embeddings (the model runs on its GPU).
- **This bypasses mask secrecy entirely** — no mask/pad recovery is needed; a mask-invariant
  quantity directly identifies the token.
- The linear-boundary pad is compensated (C_pad) before attention ⇒ does not perturb the logit.
- We do NOT fine-tune ⇒ the fingerprint table is EXACT. CONJFORMER's only mitigation (App. D.5:
  fine-tuning shifts `f` off the table) is unavailable to us ⇒ strictly worse than CONJFORMER.

Real Qwen2.5-7B has ≈28×28 = 784 such features → massively over-determined; expect a full break.
This is the core tension of running nonlinearities on masked state with structure-preserving
masks: preserving attention scores = exposing a token fingerprint.

**REAL-WEIGHT CONFIRMED (Qwen2.5-7B, scripts/attacks/real_attention_fingerprint.py):**
784-dim fingerprint, 20000 candidate tokens, with a realistic fp32-table/bf16-query precision gap:
**top-1 = 98.5%, top-10 = 98.6%, top-100 = 99.0%**. (The <100% is bf16 precision flipping a few
near-collisions; result at results/attacks/real_attention_fingerprint_qwen7b.json.) The
attention-fingerprint attack is a confirmed full break of token privacy on the real model.

## A3b — Fingerprint confirmed on CONJFORMER's OWN model families (rebuttal-ready)

scripts/attacks/conjformer_model_fingerprint.py. CONJFORMER's conjugation preserves attention
logits (their Fig. 2: Q̂K̂ᵀ=QKᵀ), so the fingerprint is unchanged by their rotation; only open
question = are their smaller models' logits distinctive. Result (10000 candidates, table=fp32,
query=bf16 round-trip, random top-1 = 0.01%):

| model (their family) | arch | fp dim | top-1 |
|---|---|---|---|
| **GPT-2** (124M, LayerNorm + learned pos + fused QKV) | gpt2 | 144 | **100.0%** |
| Qwen2.5-7B (RMSNorm+RoPE+GQA = Llama family, at scale) | llama | 784 | 98.5% |
| **Llama-3.2-1B-Instruct** (their exact 2nd model) | llama | 512 | **99.82%** |

GPT-2 = full 100% recovery (even more distinctive than 7B — LayerNorm's learned per-dim gain
makes logits more token-specific). So the attention-logit fingerprint is a full token-recovery
break on CONJFORMER's own models. Kept as a SMALL confirming point / rebuttal material, not a
headline: its role is to justify that the layer-0 TEE relocation is a real, needed defense.
(CONJFORMER only reported GPT-2 / Llama-1B and small-scale PubMed; we run 7B + large benchmarks.)

## A4 — Can fine-tuning defeat A3 without destroying the model? NO (fingerprint tracks utility)

scripts/attacks/private_weight_defense{,_b,_c}.py + fp_utility_pareto.py, real Qwen2.5-7B-Instruct
(RTX 5090), 8-10k candidate tokens, 784-dim fingerprint. Attacker's table FROZEN to the pristine
PUBLIC base; deployed model fine-tuned (observed fingerprints = deployed model's, mask-invariant).

**IMPORTANT framing correction:** this does NOT refute CONJFORMER. CONJFORMER never claimed
fine-tuning defeats the attention fingerprint — page 4 lists attention logits as an unmitigated
visible invariant. A4 QUANTIFIES that acknowledged gap and shows it is fine-tuning-robust.

**Pareto (fp_utility_pareto.py, full q/k/v/o fine-tune, 200 steps, held-out perplexity):**

| attn lr | fingerprint top-1 | held-out ppl | train loss |
|---|---|---|---|
| 0 (base) | 98.6% | 1.694 | — |
| 2e-5 (utility IMPROVES) | **98.5%** | **1.303** | 0.032 |
| 1e-4 | 66.9% | 2.73 | 0.107 |
| 5e-4 | 0.76% | 76.9 (broken) | 4.03 |
| 2e-3 | **0.025%** (≈random) | 387 (garbage) | 5.30 |

random baseline ≈ 0.01-0.012% top-1. **The fingerprint only reaches random once perplexity
has exploded (387).** At the single point where utility is preserved/improved (lr 2e-5,
ppl 1.30) the fingerprint is fully intact (98.5%). Earlier LoRA runs agree: rank-32/600-step
overfit (train loss 0.03) → 92%; common-mode removal → 94.9% (not a shared-offset artifact).

**Conclusion:** fine-tuning CANNOT remove the attention-logit fingerprint without destroying
the model. **Structural reason:** the fingerprint depends on the RMSNorm-normalized embedding
*direction* (RMSNorm erases scale), and its vector norm is huge (~9337); any utility-preserving
fine-tune is a ~1% perturbation that preserves cosine-NN identity. A working LM must keep token
representations separable — that separability *is* the fingerprint. The leak is fundamental to
running attention softmax over correctness-preserving masked state; the only real fix is a TEE
that hides the leaking reductions (task 2), not fine-tuning.

Notes: (1) embedding grads are sparse (only tokens seen in training move), and RMSNorm erases
embedding norm anyway, so the fingerprint is driven by the dense q/k/v/o — the correct knob,
which the Pareto sweeps. (2) The probe fingerprints ISOLATED single tokens (worst case, all 28
layers context-free). In real multi-token deployment only **layer 0**'s diagonal is context-free
per-token (validated in A5), motivating the layer-0 TEE-relocation guardrail.

## A5 / Task-2 — Layer-0 TEE-relocation guardrail: validated on REAL prompts

scripts/attacks/task2_layer0_guardrail.py, real Qwen2.5-7B-Instruct, 64 real gsm8k
prompts, 10000-token candidate pool. Attacker matches per-position attention-diagonal
fingerprints against an isolated-token public table. Per-position top-1 recovery:

| visible layers | pos 0 | pos 1-4 | pos 5-15 | pos 16+ | all |
|---|---|---|---|---|---|
| **layer 0 alone** (what a real attacker uses) | **100.0** | — | — | — | **100.0** |
| all 28 layers, naive 784-dim cosine | 100.0 | 8.2 | 4.69 | 6.16 | 7.65 |
| **GUARDRAIL: drop layer 0** (layers 1-27) | 57.81 | 0.39 | 0.0 | 0.08 | **1.09** |
| drop layers 0 AND 1 | 37.5 | 0.39 | 0.0 | 0.08 | 0.73 |

Per-layer single-layer recovery: **layer0 = 100%**, every other layer 0.24-5.07%
(layer2 best at 5.07). random ≈ 0.01%.

**Findings:**
1. On real multi-token prompts the entire per-token leak is concentrated in **layer 0**:
   its diagonal self-logit is a context-free function of token_p → 100% recovery at every
   position. Layers 1-27 are contextualized → individually ≤5%, useless per-token.
2. The guardrail (relocate layer-0 RMSNorm+attention+residual into the TEE at the *existing*
   single handoff — no extra round-trip) removes exactly layer 0 + the raw masked embedding
   from the GPU. Per-token recovery collapses **100% → 1.09%**, and for well-contextualized
   positions (pos ≥ 16) to **0.08% ≈ random**.
3. Residual leak = the FIRST token(s): position 0 has no context at any layer, so it is still
   57.8% recoverable from layers 1-27 (37.5% if layer 1 is also relocated). Fully closing
   position 0 needs relocating one more layer or handling the first token specially.

**A5b / Task-2b — pos0 residual + chat-template framing (task2b_pos0_and_chat.py).**
Chat-formatted prompts (Qwen template, 19-token system+role prefix), 64 prompts, 10k candidates:
- **pos0 is context-free at EVERY layer**: per-layer pos0 recovery = 100% at layers 0,4,8,…,24.
  Relocating the first k layers (k=1,2,3,4,6) leaves **pos0 = 100%** — you cannot push position 0
  to random by relocating early layers, because it attends only to itself at all depths.
- **But the residual leak is the public template, not user data.** Dropping just layer 0 (k=1):
  **USER-CONTENT recovery = 0.05%** (random = 0.01%), template-prefix 5.35%, pos0 100%. Relocating
  more layers (k=2..6) gives **0.05% user-content — no additional benefit**. The residual pos0 /
  prefix leakage lives entirely inside the 19-token fixed chat template (system prompt + role
  headers), which is public boilerplate.
- **Cost result:** k=1 (a single attention block in the TEE, no extra round-trip) is optimal;
  "first token via TEE" is unnecessary (token 0 = public template) and anyway its K/V must stay
  GPU-visible for later tokens to attend, so it cannot be fully hidden without more relocation.

**Exactness:** relocating layer-0 attention into the TEE is the *identical* arithmetic
(same RMSNorm / QK softmax / O-proj / residual, same dtype) executed in a different location
→ output is bit-identical by construction; no approximation. **Cost:** one of 28 attention
blocks computed in the TEE per forward (≈3.6% of attention FLOPs for prefill; one cheap
attention step per token at decode), with **no extra TEE round-trip** (the single handoff
just moves deeper). Accounting change only: layer-0's softmax+RMSNorm become trusted ops
(nonlinear_trusted_calls: 0 → 2 for layer 0); all other layers stay zero-crossing.

## A6 — CONJFORMER's attack suite run against OUR scheme (honest self-audit)

scripts/attacks/conjformer_attack_suite_on_ours.py, real Qwen2.5-7B, 8000 tokens, adversary
holds PUBLIC weights (= our deployed weights, since we fold EXACT public W, no fine-tune).
Random baseline top-1 = 0.0125%.

| CONJFORMER attack | our scheme | reading |
|---|---|---|
| A1a embedding inversion, naive cosine NN | 0.06% | signed-perm defeats naive cosine (like their rotation) |
| **A1b embedding inversion, permutation-invariant** (sort \|coords\|) | **98.59%** | **BREAKS us — real weakness** |
| A2 norm attack | **98.45%** | leaks (orthogonal preserves norm; we don't fine-tune to flatten it) |
| A3 Gram mask-recovery (naive diagonal match) | 0.03% (failed) | inconclusive: Qwen weight-Gram diagonal only ~22% distinct → naive matching fails; stronger graph-matching not attempted |

**Honest findings (this is a disadvantage vs CONJFORMER, must disclose):**
1. **Our signed-permutation residual mask is weaker than CONJFORMER's dense rotation for embedding
   inversion.** A signed permutation preserves each token's coordinate-value MULTISET, so sorting
   `|coords|` of the masked residual `X@N0` matches the public embedding table and recovers **98.6%**
   of tokens — *without recovering the mask at all*. A dense orthogonal U mixes coordinates and
   destroys this multiset, so CONJFORMER does not leak here. This is the concrete cost of our choice
   of STRUCTURED masks (needed to keep nonlinearities exact on the GPU with zero crossings).
2. **Norm leaks (98.45%)** because we fold exact public weights (no fine-tuning). CONJFORMER
   fine-tunes the embedding to flatten the norm structure (their App. D.4) — a mitigation we lack.
3. **Weight alignment:** per-family independent output masks make the continuous two-sided
   Procrustes unidentifiable (shared with CONJFORMER's block-specific O_b). The structure-aware
   Gram-diagonal shortcut did NOT trivially recover the signed permutation on real Qwen weights
   (diagonal not distinct enough); a stronger graph-matching attack is untested. Not the binding
   attack anyway, since A1b recovers tokens directly without the mask.

**Why this does NOT sink the paper — it sharpens the thesis:** A1b/A2 attack the RAW masked residual
embedding `X@N0` that the GPU sees at layer-0 input. The **layer-0 TEE-relocation guardrail (A5)
removes exactly that** — after relocation the GPU's first-visible state is post-attention-0 `x1@N0`,
which is contextualized (not a clean per-token multiset), so A1b/A2 lose their table-match on user
tokens (same mechanism that took the fingerprint 100%→0.05%). So: our masks alone are WEAKER than
CONJFORMER's dense rotation (structured-mask cost), but only the TEE boundary provides real token
privacy for EITHER scheme — masks (structured or dense) are not the security boundary. **CONFIRMED (task2c_inversion_norm_guardrail.py): the layer-0 guardrail collapses A1b + A2 on
user content, exactly like it did the fingerprint.** Real chat prompts, 8000 candidates, per
hidden_states[ℓ] (ℓ=0 = raw embedding, HIDDEN by guardrail; ℓ≥1 = GPU-visible after guardrail),
USER-CONTENT top-1 (random 0.0125%):

| hidden_states[ℓ] | A1b inversion (USER) | A2 norm (USER) |
|---|---|---|
| [0] embedding (guardrail HIDES) | 100% | 100% |
| [1] layer-0 output (GPU sees) | 0.0% | 1.56% |
| [2] | 0.0% | 0.0% |
| [4] | 1.76% | 2.44% |
| [8],[16] | 0.0% | 0.0% |
| [27] | 0.31% | 1.56% |

pos0 inversion stays 100% at all depths (context-free first token) but pos0 = fixed public chat
template. → **Unified result: one guardrail (relocate layer 0 into TEE) simultaneously defeats the
attention fingerprint (A3), sorted-abs embedding inversion (A1b), and norm attack (A2) on user
tokens**, because all three read the layer-0 GPU-visible state. Masks (structured or dense) are not
the security boundary; the TEE boundary is. Our structured-mask weakness (A6) is real but moot once
the guardrail is in place. Residual leak = public template prefix only.

## A7 — ObfuscaTune (arXiv:2407.02960) as a comparison baseline (对照组)

ObfuscaTune obfuscates the high-parameter attention+MLP linears OUTSIDE a TEE with `X*=X Ra`,
`W*=Ra⁻¹ W` (so `X* W* = X W`) and runs every non-linearity (RMSNorm/softmax/SiLU) INSIDE the TEE.
It protects a *proprietary* (secret) model + private data — the opposite trust model to ours (open
public weights, all non-linearity on the untrusted GPU, single entry/exit). Baseline implemented in
`src/pllo/baselines/obfuscatune.py` (+7 unit tests); real-model eval `scripts/attacks/obfuscatune_real_eval.py`,
latency `scripts/attacks/obfuscatune_latency.py`. Results in `results/attacks/obfuscatune_*.json`.

**A7a Utility — orthogonal (κ=1) is lossless; error grows with condition number** (GPT-2, fp32,
their own model, 48 linears obfuscated, 204-token passage). κ=1: next-token top-1 agreement **1.000**,
perplexity **75.525 = base 75.525**, logit MSE 2.7e-9. Sweeping κ ∈ {1,4,16,64,256,1024}: logit MSE
rises monotonically 2.7e-9 → **4.0e-7** and max logit err 4.9e-4 → 3.1e-3, reproducing their Table 2
trend (why they mandate orthogonal matrices). At fp32 top-1 agreement stays 1.000 through κ=1024 —
the numerical error only bites at reduced precision / deeper compounding, but the *trend* is confirmed.
Takeaway: at κ=1 ObfuscaTune is utility-lossless, **same as ours** → the real comparison axis is
COST and SECURITY MODEL, not utility.

**A7b Security — the untrusted device sees the TRUE Q/K/V** (exposure, κ=1, fp32 reference vs fp32
obfuscated on the same input to isolate the mask residual from bf16 quantization):

| model | tensor | rel. err (obf vs true linear) | true value exposed? |
|---|---|---|---|
| GPT-2 | c_attn (Q/K/V) | 1.0e-6 | **yes** |
| Qwen2.5-7B | q_proj | 3.7e-7 | **yes** |
| Qwen2.5-7B | k_proj | 8.9e-8 | **yes** |
| Qwen2.5-7B | v_proj | 4.9e-7 | **yes** |

Because `Ra Ra⁻¹ = I`, the projections cancel and the accelerator reconstructs the exact plaintext
Q/K/V. ObfuscaTune's confidentiality therefore rests entirely on the **weights being secret**; in the
open-weight threat model (ours) it provides **zero input privacy** — an attacker with the public
weights reads tokens directly off the exposed Q/K/V (no fingerprint NN-table needed). Contrast ours:
the GPU sees only masked `q̂/k̂` (`Q̂K̂ᵀ = QKᵀ` but `Q̂ ≠ Q`), and the k=1 guardrail closes even the
fingerprint channel.

**A7c Latency — the "all non-linearity in TEE" tax** (H800 PCIe, Qwen2.5-7B dims × 28 layers, bf16;
`obfuscatune_latency_h800.json`). Two TEE regimes, both schemes measured on the same box:

| scheme (TEE type) | prefill S=512 | prefill S=1024 | decode / tok | TEE crossings |
|---|---|---|---|---|
| plaintext floor | 32.1 ms (1.00×) | 75.5 ms (1.00×) | 11.49 ms (1.00×) | 0 |
| **ours** (confidential-GPU TEE) | 32.2 ms (**1.00×**) | 75.5 ms (**1.00×**) | 11.48 ms (**1.00×**) | 2 |
| **ours** (CPU enclave) | 77.2 ms (2.40×) | 229.5 ms (3.04×) | 29.0 ms (2.52×) | 2 |
| ObfuscaTune (confidential-GPU TEE) | 34.4 ms (1.07×) | 79.6 ms (1.05×) | 12.54 ms (1.09×) | 224 |
| ObfuscaTune (CPU enclave) | **2637 ms (82×)** | **8626 ms (114×)** | **135.5 ms (11.8×)** | 224 |

Reading: with a **confidential GPU** both are cheap (ObfuscaTune 1.05–1.09×, ours ~1.00×) — but
ObfuscaTune still exposes true Q/K/V and needs secret weights. On a **plain CPU TDX enclave**
ObfuscaTune's 224 boundary round-trips make it **unusable (82–114× prefill, 11.8× decode)**, ≈35×
slower than ours; our zero-nonlinear-crossing design + k=1 guardrail stays a bounded 2.4–3.0× prefill
/ 2.5× decode (one-time layer-0 attention block, not per-block). The 224-vs-2 crossing gap is the
whole point of running non-linearities on the untrusted GPU.

*Note on the CPU-enclave column:* those H800-host-CPU numbers are a proxy (host-CPU-dependent — a
weaker 5090-box CPU makes both far larger); the robust conclusion is the **order of magnitude** (224
crossings ≈ 2 orders, 2 crossings do not). The **confidential-GPU** column is GPU-bound and
hardware-meaningful; cross-checked on RTX 5090 (plaintext 66.6 ms, ours_gpu 63.1 ms ≈1.00×,
obfuscatune_gpu 72.0 ms ≈1.08× at S=512), matching H800.

**A7d Definitive TDX-enclave TEE-compute cost** (real Intel TDX guest, `tdx_guest`=True in
`/proc/cpuinfo`, 4 vCPUs, torch 2.8+cpu, fp32, Qwen-7B dims; `scripts/attacks/tdx_nonlinear_cost.py`,
`tdx_nonlinear_cost.json`). This measures the *in-enclave nonlinear compute* each scheme puts in the
TEE per forward — the component the host-CPU proxy could not pin down — including TDX memory-encryption
overhead:

| in-enclave nonlinear compute | prefill S=512 | prefill S=1024 | decode / tok |
|---|---|---|---|
| per-op: RMSNorm / softmax-attn / SiLU | 1.15 / 16.67 / 5.65 ms | 2.05 / 76.65 / 13.65 ms | — |
| **ObfuscaTune** (all nonlinear, ×28 layers) | **689.1 ms** | **2643.3 ms** | 24.09 ms |
| **Ours** (k=1 layer-0 attention only) | 17.8 ms | 78.7 ms | 0.82 ms |
| ratio (Obf / ours) | **38.7×** | **33.6×** | **29.4×** |

The softmax attention (O(S²)) dominates the enclave cost. Decisively: ObfuscaTune's enclave nonlinear
compute *alone* — 689 ms (S=512) / 2643 ms (S=1024) — **exceeds the entire plaintext forward** (32 /
75 ms on H800) by 9–35×, before any GPU-linear or network-transfer cost is added, because all 28
layers' non-linearities are serialised through a small 4-vCPU trust domain. Ours adds only 17.8 / 78.7
ms of enclave compute for the k=1 guardrail (~30–39× less). This is the hardware-anchored version of
the CPU-enclave column and confirms: **running every non-linearity in a real TDX enclave is the
dominant cost; keeping them on the untrusted GPU (ours) avoids it.**

## A8 — CryptoGen (arXiv:2602.08798) as a comparison baseline (对照组 #2)

CryptoGen is a **pure-crypto** hybrid HE+MPC system for secure autoregressive generation with an
encrypted, reusable KV cache (its contribution: attention O(L²)→O(L) vs BOLT). NO TEE; client-server,
semi-honest, **dual-sided** (client data AND model weights hidden); linear layers in HE (BFV CT×PT),
every non-linearity in interactive MPC (EzPC). It is the pure-crypto opposite of our TEE-anchored,
open-weight, GPU-nonlinear scheme. Baseline in `src/pllo/baselines/cryptogen.py` (+7 unit tests):
faithfully reimplements + verifies (on plaintext-packed slots, err ~1e-15) their core algorithms —
outer/inner/diagonal packings, diagonal CT×PT matvec, ARCC inner-inner/inner-outer attention with the
O(log d) folding-sum, and slot-aware encrypted KV-cache concatenation (512 tokens → 4 ciphertexts vs
512 naive) — plus a cost model reproducing their Tables I/II/IV.

**A8a Real HE latency** (`scripts/attacks/cryptogen_he_latency.py`, Microsoft SEAL via TenSEAL 0.3.16,
BFV n=8192, RTX-5090 box CPU; `cryptogen_he_latency.json`). We measured the **real** ciphertext-op
wall-clock (not just cited it):

| real BFV op (n=8192) | measured |
|---|---|
| CT×PT multiply | 0.74 ms |
| CT×CT multiply | 16.9 ms |
| folding-sum reduction (`.sum`, log-depth Galois rot) | 75.5 ms |
| inner-product dot | 87.6 ms |
| single Galois rotation (derived) | 7.55 ms |

Composed GPT-2 decoder block (d1=768, ffn=3072) using CryptoGen's **diagonal** encoding (d ct-pt mults
+ d rotations per matvec): **63–104 s/block** (L=64→512) — the same order as their Table IV
(9.6–65.55 s); the naive inner-product HE matvec is a 612–652 s/block upper bound. A **real** 64-output
CT×PT matvec ran end-to-end in 5.45 s (85 ms/output ≈ measured dot 87.6 ms), validating the composition.

**Headline (real-measured).** A GPT-2 token = 12 blocks ≈ **13–21 minutes** under real BFV (paper's own
≈ 787 s/token at L=512). A **single** homomorphic folding-sum reduction (75 ms) alone **exceeds our
entire per-token latency on Qwen-7B** (11.5 ms, H800) by ~6.5×; one CT×CT multiply (16.9 ms) also
exceeds it. So CryptoGen is ~10⁴–10⁵× slower than ours **and** was only ever demonstrated on GPT-2 —
pure-crypto dual-sided privacy is real but does not scale to 7B. (MPC non-linear cost via EzPC not
executed here; per their Table IV it is a minority of block time, dominated by the CT×PT/CT×CT costs
measured above.)

## Not yet implemented
- **Attention-fingerprint attack** (CONJFORMER §3.5): the one leak we likely SHARE — if
  `Q̂K̂ᵀ = QKᵀ` (pairwise-rotation correctness), the GPU sees true attention logits. We don't
  fine-tune, so the "fine-tuning shifts fingerprints" mitigation is unavailable to us. This is a
  priority and a likely genuine exposure.
- **EDNN** (Lin et al.): our orthogonal `N_in` is not the glide-reflection `I − 2/d·E`, so the
  neighbor-difference invariant does not hold; also worth a formal "fixed-point nonexistence" check.
- **Real-weight validation** on the deployed Qwen-7B folded package (adversary = GPU operator,
  runs on H800 which has both public Qwen weights and the folded package).
