# Private-Model LoRA Training Audit — Scheme A validity

**Setting audited (NOT the old public-weight setting):** base model weights are
**private**; the attacker controls the untrusted GPU/host but does **not** know
plaintext base weights, plaintext activations, masks, or pads. Goal = private
model + private user data + private LoRA adapter. Qwen *generation* (inference)
path is assumed already validated; the question is whether **obfuscated LoRA
*finetuning*** (Scheme A: full / near-full masked-domain training) is valid.

Audit performed against the **current code**, not documentation. All numbers
below are measured (float64) on 2026-07-09. Reproduction scripts:
`private_model_attacks.py` (security) and the in-repo runners cited per section.

---

## TL;DR verdict

- There is **no full masked-domain LoRA finetuning** for real transformer
  operators in the code. The real-operator training path
  (`ops/lora.py` + `ops/lora_rank_padding.py` + `multilayer_lora_training.py`)
  **recovers plaintext at every operator boundary** and runs every nonlinearity,
  the loss, and the optimizer **trusted-side**. It outsources only the masked
  **linear GEMMs** to the untrusted GPU.
- Therefore the implemented scheme is **B** (TEE-controlled forward/backward with
  GPU-outsourced masked GEMM), with **trusted-side AdamW**.
- A **full masked-domain** trainer (**A1**) exists **only as a synthetic
  standalone `Y = X A B` linear toy** (`masked_gradient_lora*.py`): SGD/momentum
  exact, **AdamW explicitly refused**. It has **no** RMSNorm/RoPE/softmax/SwiGLU/
  attention and uses MSE, so it is not a real LoRA-finetuning scheme.
- **A2 / A3 for real operators: Not implemented.** No obfuscated LoRA training has
  ever been run on GPT-2, Qwen, or LLaMA. The only real-model LoRA training in the
  repo (`train_qwen7b_lora_dolly.py`) is **plaintext PEFT** ("never touches the
  protocol/folding path"); folding is inference-only.
- New private-model security finding: the **additive pad does not hide activation
  geometry** once the pad-compensation term is GPU-visible (it must be, for
  correctness). A GPU-only attacker strips the pad and recovers the full plaintext
  **activation Gram** (pairwise geometry + norms) with **zero known-plaintext**.

**Final classification: `B` (GEMM-outsourced masked linear + trusted-side
everything-else, incl. trusted AdamW). Scheme A for real models = Not yet
implemented.**

---

## 1. Forward/backward execution — recovers plaintext at every boundary

The real-operator path does **not** keep tensors masked through the network. Each
LoRA linear op masks → GEMM → **recovers plaintext**:

- Forward: [`run_masked_rank_padded_lora_linear`](../../../src/pllo/ops/lora_rank_padding.py) →
  `y = recover_masked_output(y_tilde, n_out_inv)` (plaintext `y` returned).
- Backward: [`run_masked_rank_padded_lora_backward`](../../../src/pllo/ops/lora_rank_padding.py) →
  `recover_lora_gradients(...)` (plaintext `grad_a`, `grad_b`, `grad_x`).
- Between ops ([`_forward`](../../../src/pllo/experiments/multilayer_lora_training.py)),
  the hidden state, attention, activation, residual, and logits are **plaintext**;
  base weights `W` are passed **in plaintext** into each op (the op folds them
  internally). Autograd (`loss_masked.backward()`) runs over the **plaintext**
  graph; only the per-op GEMM is masked.

**Answer:** full forward/backward is **NOT** in the masked domain. Plaintext
activations/gradients are recovered in the trusted boundary at each op. This is
GPU-outsourced masked GEMM (Scheme B), not masked-domain training.

## 2. Operator coverage (Qwen/LLaMA operators)

| Operator | Forward masked-domain? | Backward masked-domain? | Where it actually runs |
|---|---|---|---|
| Q/K/V/O projections (GEMM) | GEMM masked, **output recovered to plaintext** | GEMM masked, **grad recovered** | GPU GEMM; TEE mask/recover |
| gate/up/down projections (GEMM) | same | same | GPU GEMM; TEE mask/recover |
| LoRA A/B matmuls | masked GEMM, recovered | masked GEMM, recovered | GPU GEMM; TEE mask/recover |
| RMSNorm | ❌ | ❌ | not implemented (tiny model has none) |
| RoPE | ❌ | ❌ | not implemented |
| attention softmax | ❌ plaintext | ❌ plaintext | `simple_attention_proxy`, trusted-side |
| SwiGLU / SiLU | ❌ plaintext | ❌ plaintext (autograd) | `_silu`, trusted-side |
| residual add | ❌ plaintext | ❌ plaintext | trusted-side |
| LM head / CE loss | ❌ plaintext | ❌ plaintext | `H @ head.W`, MSE/CE trusted-side |

**No nonlinear operator has a masked-domain forward or backward.** Only linear
GEMMs are outsourced, and even those are recovered to plaintext at the boundary.

## 3. LoRA gradient correctness (real-op path, `multilayer_lora_training`, tiny synthetic transformer, seed 7, 4 steps)

| Metric | Value | Tolerance | Pass |
|---|---|---|---|
| `max_loss_diff` (masked vs plain) | 6.2e-15 | 1e-9 | ✅ |
| `max_forward_err` | 6.2e-14 | 1e-9 | ✅ |
| `max_grad_a_real_err` (recovered to plaintext coords) | 9.5e-16 | 1e-7 | ✅ |
| `max_grad_b_real_err` | 4.0e-15 | 1e-7 | ✅ |
| `max_update_a_err` / `max_update_b_err` | 1.3e-13 / 1.8e-14 | 1e-7 | ✅ |
| `max_dummy_contribution_norm` | 3.8e-15 | 1e-9 | ✅ (stays zero) |
| `allclose` | **True** | — | ✅ |

Parameters recovered back to plaintext coordinates match plaintext rank-`r` LoRA
to machine precision **per step**. Note: exactness is by construction — masks are
exactly invertible and plaintext is recovered each boundary, so "obfuscated" and
"plaintext" trajectories are the *same computation* up to fp round-off.
**Final adapter error / generation token-match after training: N/A** — no real
model was trained obfuscated (see §6).

Contrast — the synthetic A1 toy (`masked_gradient_lora_training`): SGD/momentum
recovery ≤ 2e-15, but its rank-padding **dummy cancellation breaks after step 1**
(`dummy_contribution_norm` → **0.28**, because its optimizer updates the padded
rank). The multilayer path avoids this by sizing the optimizer to `true_rank` and
re-sampling dummies fresh each step (`any_dummy_update_applied=False`).

## 4. Optimizer equivalence

- **SGD, momentum-SGD:** exact in both the real-op path (trusted-side) and the A1
  toy (masked-domain), ≤ ~1e-13.
- **AdamW:**
  - Real-op path (`multilayer_lora_training`, `optimizer='adamw'`): **exact**, and
    the report marks `optimizer_summary.location = "trusted"`. → **trusted-side
    AdamW, not masked-domain AdamW.**
  - A1 masked-domain toy with **dense** masks: **explicitly unsupported** —
    `masked_adamw_step_unsupported` raises `DenseMaskedAdamWUnsupported`
    (coordinate-wise `v ← β₂v+(1−β₂)g²` does not commute with a dense orthogonal
    mixer). Not faked, not approximated. ✅ honest.
  - **Signed-permutation masks + masked-domain AdamW: not wired into any training
    loop** (a `create_signed_permutation` helper exists but is unused by training).
    So masked-domain AdamW exactness is **untested / unclaimed**.

## 5. Security in the private-model setting (measured against the real Scheme-B op)

Public-weight forward oracle **disabled** (weights private). GPU-visible per op:
`x_tilde, w_tilde, a_tilde, b_tilde, pad_compensation C, y_tilde`. Masks are
**dense orthogonal** (QR), plus an additive per-row Gaussian **pad**, plus a
rank-space mixer `U`. Default policy is **fresh masks + fresh pad + fresh U per
step**.

| Attack | Condition | Result | Verdict |
|---|---|---|---|
| **Compensation pad-strip** (GPU-only, no known-plaintext) | pad on, square proj | recover `x·N_in` at relerr **1.7e-14** → activation-Gram off-diag corr **1.0** (blinded baseline 0.50) | **LEAK — pad gives no geometry protection** |
| **KPA / Procrustes on `N_in`** | pad off, `k=d_in` known plaintext activations | `N_in` relerr **2e-14** (exact) | dense mask KPA-breakable |
| **KPA on `N_in`** | pad on, `k=d_in` / `2·d_in` | relerr 0.90 / 0.60 (degraded) | pad degrades naive KPA, **but** strip-then-KPA → exact |
| **Multi-step adapter linkability** | **fresh** masks (default) | abs-cos **0.093** (noise floor) | linkage defeated |
| **Multi-step adapter linkability** | static masks | abs-cos **0.995** | full trajectory leak |
| Gradient inversion from masked transcript | — | recovers `x·N_in` at best (same as pad-strip); literal `x` needs `N_in` (⇒ known-plaintext KPA) | geometry leaks; coordinates gated on KPA |

Key points:
- **The additive pad does not hide activation geometry.** The compensation term is
  algebraically `C = (T·N_in) @ W_eff_tilde`, and the GPU knows `W_eff_tilde =
  w_tilde + scale·a_tilde·b_tilde`, so it inverts and subtracts to obtain `x·N_in`
  whose Gram equals `x·xᵀ`. This needs **no** known-plaintext and is **not** fixed
  by fresh masks (Gram is invariant under any orthogonal `N_in`). It leaks all
  pairwise inner products, norms, distances, duplicate/near-duplicate structure,
  and membership signal for the private training activations. (Square projections
  ⇒ exact; non-square up/down proj ⇒ least-squares, partial — untested here.)
- Because this construction is shared with the inference-side linear-boundary pad,
  the same geometry leak likely affects the *validated generation path* too — flag
  for separate review; this audit measured only the training op.
- **Fresh-mask default is doing the load-bearing work:** it defeats cross-step
  linkability and prevents KPA amortization across steps. Under (mis)configured
  static masks the entire adapter trajectory and inputs are recoverable.
- KPA requires the attacker to **know plaintext activations**, which under a fully
  private model (private embeddings/weights) is itself a strong assumption for
  hidden layers — so the *unconditional* concern is the compensation geometry leak,
  not KPA.

## 6. Real model status

| Level | Status |
|---|---|
| Synthetic (`Y=XAB`, MSE) | ✅ run — A1 masked-domain toy |
| Tiny synthetic transformer | ✅ run — Scheme B real-op path (`multilayer_lora_training`, `tiny_lora_transformer`; no RMSNorm/RoPE) |
| GPT-2 | ❌ never run (obfuscated training gated `available_not_run`) |
| Qwen / LLaMA (obfuscated training) | ❌ **not implemented, never run** |
| Qwen LoRA training that exists | plaintext PEFT (`train_qwen7b_lora_dolly.py`), folding is inference-only |

No obfuscated real-model LoRA training results exist on disk.

## 7. Final classification

**`B` — TEE-controlled forward/backward + GPU-outsourced masked GEMM, with
trusted-side AdamW.** The masked-domain full-training scheme (A1) exists only as a
synthetic linear toy; A2/A3 for real transformer operators are **Not yet
implemented**.

---

## 8. Allowed vs disallowed claims

### ✅ Allowed (supported by current code + measurements)
- "We outsource the private-model LoRA **linear GEMMs** (base + adapter, forward
  and backward) to an untrusted GPU under invertible masks; the GPU never sees
  plaintext base weights, activations, adapters, or gradients. Nonlinearities,
  loss, and the optimizer stay trusted-side."
- "Masked GEMM outsourcing preserves plaintext rank-`r` LoRA training to float64
  machine precision per step (loss, forward, recovered gradients, and parameter
  updates), for **SGD, momentum-SGD, and trusted-side AdamW**."
- "Rank padding hides the true LoRA rank from shape; in the real-op path the dummy
  slice never enters the optimizer and its contribution stays at machine zero."
- "Dense-mask **masked-domain** AdamW is not supported and we raise rather than
  approximate."
- "Fresh-per-step masks defeat cross-step adapter/gradient linkability
  (abs-cos ≈ 0.09 vs ≈ 0.99 static)."
- "Validated on synthetic + a tiny synthetic transformer."

### ❌ Disallowed (not supported / contradicted)
- ❌ "Full / near-full **obfuscated (masked-domain) LoRA finetuning**." — It is
  GEMM-outsourcing with plaintext recovery at every boundary (Scheme B).
- ❌ "Nonlinear operators (RMSNorm/RoPE/softmax/SwiGLU) are computed under
  obfuscation during training." — All plaintext trusted-side.
- ❌ "Masked-domain AdamW." — AdamW is **trusted-side**; dense masked-domain AdamW
  is refused; signed-perm masked AdamW is untested.
- ❌ "We trained Qwen/LLaMA LoRA privately / behind a TEE." — Never run; the real
  Qwen LoRA training is plaintext PEFT.
- ❌ "The additive pad hides user-data / activation geometry from the GPU." —
  **Refuted:** the GPU-visible compensation term strips the pad and recovers the
  full activation Gram (corr 1.0) with zero known-plaintext.
- ❌ "Private user-data confidentiality against the untrusted GPU." — Only
  **relational/geometry** privacy is broken (Gram leak); literal coordinates
  additionally require KPA. Do not claim data confidentiality without addressing
  the compensation leak.
- ❌ Any differential-privacy / membership guarantee — not measured.

### Required next steps before Scheme A can be claimed
1. Fix or re-architect the **pad-compensation** so it is not GPU-visible in a form
   that inverts to `x·N_in` (e.g. keep `C` trusted-side and add during recovery,
   or bind it so `W_eff_tilde` cannot be inverted out). Re-audit the generation
   path for the same leak.
2. If masked-domain training is the goal, implement masked forward/backward for the
   nonlinear operators (or a TEE-nonlinear-island) — currently none exist.
3. For masked-domain AdamW, either wire signed-permutation masks and test
   exactness, or state trusted-side AdamW explicitly.
4. Run at least one real model (GPT-2/Qwen) end-to-end before any real-model claim.
