# CONJFORMER (arXiv:2606.16461) real reproduction on Qwen2.5-7B-Instruct

Baseline comparison for the AAAI paper. CONJFORMER makes the transformer exactly
O(d)-equivariant (scalar-gain RMSNorm + blockwise orthogonal weight conjugation)
so the server runs the whole forward pass in a rotated basis. We reproduced it
faithfully on the real Qwen2.5-7B-Instruct (H800, `connect.westb:19948`,
2026-07-03) and measured generation quality, latency/throughput, and fine-tuning
time. Code: `src/pllo/baselines/conjformer.py`, `scripts/attacks/conjformer_real_eval.py`,
`tests/test_baselines_conjformer.py` (6 tests, all pass).

## Reproduction is numerically correct (equivariance verified)

- **RoPE compatibility (the crux on Qwen)**: HF applies RoPE as `rotate_half`
  (pairs dim i with i+d_h/2). The Q/K rotation O_b must commute with that; we build
  O_b from 2x2 SO(2) blocks on those exact planes -> attention logits preserved
  after RoPE (Q̂K̂ᵀ=QKᵀ) to <1e-9. GQA handled (KV-head rotation reused across the
  query-head group); V rotation R_b is full per-head orthogonal; SwiGLU uses a
  permutation P_b.
- **End-to-end**: server-on-rotated-embeddings == retrofit(scalar-RMSNorm)-on-plain,
  **logit top-1 agreement = 1.0000** on the real 7B (hidden_rel_err 4.5e-2 is just
  bf16 round-off in a 3584-dim rotation; fp32 gives ~1e-7). So **the obfuscation
  itself adds ZERO utility cost** — any gap vs plaintext is purely the required
  scalar-RMSNorm architecture change.

## Headline results (Qwen2.5-7B-Instruct)

| metric | plaintext (unmodified) | CONJFORMER | our A_rightmul (fp32) |
|---|---|---|---|
| **generation quality** (IFEval prompts, greedy) | coherent, follows instructions | **gibberish** | coherent (parity) |
| readable-ASCII ratio (8 prompts, fp32) | **0.995** | **0.083** | ~0.995 |
| IFEval prompt-strict acc | 66.17% | **≈0 (no fine-tune)** | 68.76% |
| decode throughput (bf16) | 62.7 tok/s | **61.5 tok/s (1.02x)** | ~1/9x (TEE round-trip) |
| architecture change | none | per-channel→scalar RMSNorm | none |
| fine-tuning to work | none | **required** (~24 min/2000 iter, H800 lower bound) | none |

### 1. Generation quality — CONJFORMER's off-the-shelf instruct model is gibberish
Replacing Qwen2.5-7B-Instruct's per-channel RMSNorm gain with a single scalar
(CONJFORMER's one required architecture change), **without fine-tuning**, collapses
the model to garbage: `ᐈ相关负责//////OUCH...` on every prompt (readable-ASCII 0.083
vs plaintext 0.995; identical in fp32 and bf16). The trained per-channel norm gains
carry essential learned scaling; a scalar cannot represent them. This is NOT a
degenerate-repeat loop (max same-token run ≈1.4) — it is coherent-looking noise, so
naive degeneration metrics miss it; the readable-ASCII ratio and the raw samples
(`qwen7b_generate_fp32_samples.json`) are the evidence. This is *consistent with the
paper's own retrofit curves* (Fig. 6: retrofit starts at very high PPL and only
recovers after fine-tuning) — the paper never serves a no-fine-tune scalar-RMSNorm
model. **To make CONJFORMER usable you must fine-tune; to recover instruction-
following on a 7B instruct model you'd need the original instruct recipe/data, which
the client does not have.**

### 2. Latency / throughput — CONJFORMER inference is essentially free
Decode: plaintext 62.7 tok/s vs CONJFORMER 61.5 tok/s, **overhead 1.02x**. The
server runs a normal Qwen2 forward (same FLOPs); the client adds only a d×d
rotate/de-rotate per step and no TEE round-trip. **This is where CONJFORMER beats
ours**: our folded_remote pays a TEE-boundary round-trip + fp32 logits wire (~9x per
token). Honest contrast: CONJFORMER is cheaper at inference *because* it pushes the
cost into a mandatory offline retraining and gives up serving an off-the-shelf model.

### 3. Fine-tuning time — CONJFORMER needs it, ours needs none
Retrofit fine-tuning throughput on the 7B: **5.7k tok/s** (fwd+bwd, SGD + gradient
checkpointing; full-param AdamW's 56GB fp32 states need multi-GPU — the paper uses
4×A100). Extrapolated retrofit (paper D.3 ~2000 iters): **~24 min** compute lower
bound on one H800; a real quality-recovering fine-tune (esp. instruct-preserving) is
substantially larger. **Ours requires ZERO fine-tuning** — it serves the unmodified
Qwen2.5-7B-Instruct at fp32 parity (IFEval 68.76 / GSM8K 90.52 / HumanEval 82.3 /
MT-Bench 6.76).

## Takeaway for the paper

CONJFORMER and ours make different trades. CONJFORMER's rotation is exact and its
inference is ~free (1.02x), but it **requires an architecture change (scalar RMSNorm)
that breaks an off-the-shelf instruct model until it is re-trained**, and it never
demonstrates task accuracy (only perplexity, up to Llama-3.2-1B). Ours keeps the
unmodified instruct model, needs no fine-tuning, and shows measured task parity —
at the cost of per-token TEE-boundary latency. The clean one-line contrast:
**CONJFORMER pays offline (retraining + can't serve a stock instruct model); ours
pays online (latency). Both preserve the model function; only ours is validated on
downstream task accuracy.**

## Recovery confirmation — fine-tuning DOES fix the break, but only the LM part

To confirm the scalar-RMSNorm collapse is recoverable (paper Fig. 6 / Table 2), we
retrofitted **Qwen2.5-1.5B-Instruct** (same Qwen2 arch, full-param AdamW fits one
H800) and fine-tuned it on wikitext-2-raw-v1 (the paper's own retrofit recipe — a
plain LM corpus), tracking held-out val PPL + generation. `qwen1.5b_recover.json`.

| stage | val PPL (wikitext-2) | generation |
|---|---|---|
| original per-channel instruct | **10.08** | fluent **and follows the instruction** |
| retrofit scalar-RMSNorm, no fine-tune | **1.6e15** (collapsed) | blank / broken |
| + fine-tune 100 steps (~3 min) | 35.2 | becoming fluent |
| + fine-tune 300 steps (~9 min) | 24.3 | fluent |
| + fine-tune 800 steps (~24 min) | **27.0** | fluent, **but ignores the instruction** |

Two honest findings:
1. **The break is real and recoverable**: scalar-RMSNorm sends PPL from 10 → 1.6e15
   (blank/gibberish); fine-tuning pulls it back to a *functional* LM within ~100
   steps (PPL 35, fluent text). Confirms CONJFORMER's mechanism works — at a real
   wall-cost (**~24 min / 800 steps on a 1.5B**; a 7B and the paper's full recipe
   cost more). The recovered model still conjugates to an **exactly equivariant**
   server (top-1 = 1.0) — recovery doesn't break the obfuscation.
2. **Plain-corpus fine-tuning restores language modeling but NOT instruction
   following, and does not reach the original PPL.** It plateaus at PPL ~24–27
   (≈2.6× the original 10.08) and the step-800 sample is fluent English that
   **ignores the prompt** ("You are tasked with performing administrative
   functions…" for a "summarize this Wikipedia page" instruction). The scalar-norm
   also has genuinely reduced capacity. **Recovering the *instruct* behaviour would
   require the original instruction-tuning data/recipe, which the client does not
   have** — exactly the cost we flagged. The paper only ever reports LM perplexity
   (never task/instruction accuracy), consistent with this.

Net: CONJFORMER's break is fixable (LM-wise, ~24 min+ of fine-tuning), but on an
off-the-shelf **instruct** model a client cannot cheaply get the instruction-
following back. **Ours serves the unmodified instruct model at parity with zero
fine-tuning.**

### Capacity floor — LM recovery plateaus ~2.9x baseline and then overfits

A longer, scheduled run (cosine LR 3e-4, warmup 100, 2000 steps, ~59 min) does NOT
close the gap — it exposes a capacity floor + overfitting (`qwen1.5b_recover_cosine.json`):

| step | val PPL | train loss |
|---|---|---|
| 250 | 33.0 | 3.59 |
| 1250 | **28.8 (best)** | 1.74 |
| 2000 | **56.6 (overfit)** | 0.20 |

Train loss falls to 0.20 while val PPL bottoms at **28.8** (step 1250, ≈2.9× the
original 10.08) then rises to 56.6 — the scalar-RMSNorm retrofit **cannot reach the
original PPL** by fine-tuning on a modest corpus (wikitext-2, ~2M tokens); it plateaus
and overfits. The paper's near-baseline recovery (Table 2, within ~1%) therefore
relies on either **from-scratch equivariant pretraining** (Table 1 — retraining the
whole model) or **large in-domain fine-tuning**, both far more costly than the
"lightweight retrofit" framing suggests. Recovery is real (1e15 → 28.8) but partial
and hits a capacity/data wall.

### Instruction-following recovery via self-distillation — only partial, still off-topic

The realistic recovery for an instruct model: the client holds the public original
instruct model, so it can self-distill — generate (prompt→response) pairs from the
original and SFT the retrofit model to imitate them (response-masked loss). We ran
this on 300 IFEval-prompt pairs, 4 epochs, held-out 20 (`qwen1.5b_instruct_recover.json`).

| stage | distill PPL (held-out) | instruction relevance* | generation |
|---|---|---|---|
| original instruct | — | **0.238** | on-topic + formatted |
| retrofit, no fine-tune | 3.7e15 | **0.000** | blank / broken |
| self-distill SFT, step 100 (best) | 26.8 | 0.061 | fluent |
| self-distill SFT, step 300 (overfit) | 56.5 | **0.089** | fluent + bullet-formatted **but off-topic** |

\*relevance = fraction of prompt content-words echoed in the response (on-topic proxy).

Self-distillation (teacher-gen 12.5 min + SFT 4.8 min = **17.3 min** on a 1.5B)
recovers **fluency and generic instruct-style formatting** (bullets, sections) but
**only ~37% of the instruction-relevance** (0.089 vs 0.238) and the content is
off-topic/hallucinated — e.g. for "describe prehistoric megaliths, give two
responses separated by 6 asterisks" the recovered model emits fluent bullets about
"an ancient forest… themed zones… religious ceremonies", ignoring megaliths and the
format. It also overfits within 300 steps (PPL 27→56). Recovered model still
conjugates to an exact server (top-1 = 1.0). Matching the original instruct behaviour
would need far more distillation data + compute (approaching a full re-SFT) — the
cost climbs toward re-training, exactly what ours avoids.

## Bottom line on "can CONJFORMER do generation?"

The **generation mechanism** is fine (exact, ~1.02× latency). The obstacle is that
CONJFORMER requires a **client-trained** model (scalar-RMSNorm + fine-tuned), so an
**off-the-shelf instruct** model must have its instruction-following re-learned:
plain-corpus fine-tuning doesn't restore it at all; self-distillation restores only
~37% on a modest budget and overfits. This is a real, general limitation of the
retrofit path (it holds for any pretrained model; it's *worst* for aligned/instruct
models), not an artifact of picking Qwen. **Ours serves the unmodified instruct model
at measured task parity with zero training** — a different, and for cloud-served
strong models more practical, deployment point.

## Files
- `qwen7b_generate_fp32.json` / `_samples.json` — 8 IFEval prompts, fp32, quality
- `qwen7b_all_bf16.json` / `_samples.json` — verify + generate + latency + finetune, bf16
- `qwen1.5b_recover.json` / `_samples.json` — retrofit→wikitext fine-tune PPL curve (constant LR)
- `qwen1.5b_recover_cosine.json` / `_samples.json` — cosine LR 2000-step run (capacity floor + overfit)
- `qwen1.5b_instruct_recover.json` / `_samples.json` — self-distillation instruction-following recovery
- reproduction verified by `tests/test_baselines_conjformer.py`; scripts:
  `conjformer_real_eval.py`, `conjformer_finetune_recover.py`, `conjformer_instruct_recover.py`
