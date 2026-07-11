# LOCAL GATE-0 REPORT — private-base + unified LoRA (real Qwen2.5-0.5B)

**Verdict: `LOCAL_GATE0_READY_FOR_D4`**

Local (CPU/fp64) Gate-0 chain complete on the real, byte-verified checkpoint. No H800
and no TDX run was performed. Work STOPS here for review before any H800 unified-worker
change (Phase 6+7 remain gated). Nothing committed.

---

## 0. Checkpoint — verified real
- `model.safetensors` SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
  **byte-identical** to the H800 source (this is a source↔destination byte-identity
  check, **not** a claim of an independently-published upstream canonical SHA-256).
- tokenizer.json `c0382117…`, config.json `479dcf0c…` pinned in the registry.
- Full load verified: 24 layers, hidden 896, 14 q / 2 kv heads, intermediate 4864,
  vocab 151936, eps 1e-6, rope_theta 1e6, **tie_word_embeddings=True** (embedding and
  LM head share storage and are numerically identical — one source E), 494 M params.

## 1. D3 — masked operator fwd/bwd contract — **PASS**
`scripts/gate0_d3_operator_contract.py` → `gate0_d3/`.
- **840** checks on **real captured activations** (real GSM8K prompt), **all 24 layers**,
  **0 failed**; **72 negative controls all fired** (independent-perm 19.8, dense-orth
  through SiLU 2.7, `RoPE(QR)≠RoPE(Q)R` 61.6). Positive ops at fp64 (RMSNorm
  equivariance fwd+bwd, SwiGLU shared-perm, RoPE paired-mask score preservation, LoRA
  fold on **all 7 targets** with grad/SGD recovery on real activations).
- Trust-domain-scoped counters: trusted diagnostic oracle plaintext materialisations
  NONZERO (legitimate); `untrusted_worker_counters_applicable=false` (no untrusted
  worker in local D3); `protected_operator_plaintext_shortcuts=0`,
  `nonlinear_trusted_calls=0`, `silent_fallbacks=0`.
- Tied-embedding audit written (storage_shared, numeric_equal, lm_head reconstructed
  from tied embed_tokens; two transformed views ⇒ D2 cross-view attack mandatory).

## 2. Private package — built, validated, immutable — **PASS**
`scripts/gate0_build_private_package.py` → `private_package/gpu_package/` (242
artifacts), root hash **`bfd578b809ef2313…`** (pinned in registry as the D2 target).
- **242 transformed `*_tilde` artifacts only.** Every base weight, embedding, LM-head,
  RMSNorm γ, and bias is **folded/masked**; no plaintext tensor and no mask secret is
  written. Names deliberately drop the plaintext `.weight`/`.bias` stem (fail-closed
  packager policy) and pass the independent plaintext-absence scan **clean (0 findings)**.
- **170 folds validated at fp64**, max error **5.97e-13**. RoPE-commutation
  **8.9e-16**, score-preservation **8.9e-15**.
- **OI1 resolved for the Q/K pre-RoPE bias:** a per-plane rotation `B` that provably
  commutes with RoPE masks q/k (weights + biases) without breaking score preservation
  (validated in-build, fail-closed); V/V-bias masked by `S`; SwiGLU by permutation `P`;
  vocab by a monomial permutation represented as an **index vector** (never a dense
  151936² matrix).
- Packager unit tests still green (10/10).

## 3. D2 — private-base weight-confidentiality identifiability — **PASS (claim scoped)**
`scripts/gate0_d2_identifiability.py` → `gate0_d2/`. Ran against the exact hash-pinned
package. **Positive controls first (3/3 pass):** Gram column-perm recovery with
reference (acc 1.0), spectral preservation (1e-14), tied-view anchor basis recovery
(5.8e-15). Attacks valid only where their control passed; 4 families recorded
`unavailable` (no verified impl — **not** counted as failures).

| Threat | Attacker has | Weight recon cosine | Functional model recovered |
|---|---|---|---|
| **T0** | transformed package only | **≈ 0.00** (12/12 weights) | **No** ✔ (required) |
| **T2** | + 1/8/32 anchor pairs | residual-basis 0.18 → 0.28 → 0.40 | No (underdetermined, <H=896) |
| **T3** | + plaintext weights (leakage) | **1.0000** (12/12); residual mask on down_proj to 1e-14 | **Yes** (expected worst case) |

- **T0 does NOT recover a functional model** and the tied views do not pin the residual
  basis (γ_f scaling breaks their co-diagonalisation). Empirical claim
  `private_base_weight_confidentiality_empirical` **holds under the private-weight (no
  plaintext reference) assumption.**
- **T3 fully recovers** — the masks are *not* claimed to protect an attacker who already
  holds the plaintext weights; this is the honest worst-case ablation and its success
  also validates the attacks are real.

### Documented residual leaks (NOT claimed hidden — surfaced honestly)
1. **Singular-value spectra of γ-free weights** (`down_proj`, `o_proj`) are preserved
   exactly under the orthogonal/permutation masks (T0 gap ~1e-14). γ-folded weights
   (q/k/v/gate/up) are perturbed (gap 1–30). Spectrum alone ≠ functional recovery, but
   this is a real structural leak and a candidate motivator for a future non-orthogonal
   mask on γ-free weights.
2. **Attention token–token scores exposed by design** (paired mask `NqNkᵀ=I`).
3. **Tied-E two-view cross-view surface** — fully invertible only with a reference/anchor
   (T3/T2), not at T0.
4. **Q/K bias planar-rotation pair-norms** preserved (the OI1 `B` mask is orthogonal;
   per-plane norms leak, values do not).

These are consistent with `base_weight_security.proven=false`; no cryptographic or
zero-leakage claim is made.

## 4. What is NOT yet done (correctly gated)
- **D4** (real H800 + real TDX, fresh attestation, one real GSM8K batch,
  untrusted-worker-required-zero counters) — **not started**; requires wiring the unified
  worker to load this package. **Stop point per the frozen chain.**
- POST_D4_10STEP, DEPLOY_SMOKE (gsm8k/sst2) — gated after D4.
- No commit performed.

## 5. Files produced this stretch
- `scripts/gate0_d3_operator_contract.py`, `scripts/gate0_build_private_package.py`,
  `scripts/gate0_d2_identifiability.py`
- `results/aaai_private_base/gate0_d3/*`, `private_package/*` (+ `gpu_package/` 242
  artifacts, hash pin), `gate0_d2/*`, `checkpoint/tied_embedding_audit.*`
- `results/aaai_private_base/experiment_registry.yaml` (checkpoint/tokenizer/config +
  package-root hashes pinned)

**READY to proceed to D4 on review.** Awaiting go-ahead before touching the H800 worker.
