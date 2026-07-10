# Nonlinear Cross-Gram — Real-Attack Audit

Experiment / audit only. numpy/scipy/sklearn, mean over seeds [0, 1, 2]. No production path modified; not committed.

**Question:** the permutation island leaks — exactly — `ZGZ^T`, `UGU^T`, the per-token value multiset (`Z_tilde`/`U_tilde` mod a global column permutation the GPU already applies), and the per-token gradient norm. Does that auxiliary structural leakage translate into real attacks? Measured below; the decisive axis is **public vs private base weights**.

## Headline results (mean over seeds)

| attack | public base weights | private base weights | verdict |
|---|---|---|---|
| 1 token reconstruction | top1 **1.0** (baseline 0.0391) | top1 0.0391 (id=baseline; linking F1 0.8154) | **HIGH under public weights** |
| 2 target/label prediction | top1 0.5667, AUC **0.8221** (base acc 0.4083) | top1 0.4583, AUC 0.6198 | public: MEDIUM, private: none |
| 3 membership inference | AUC **0.6063** (TPR@1%FPR 0.0731, gradnorm_threshold) | same (model-agnostic gradient-norm signal) | **MEDIUM, both regimes** |
| 4 adapter extraction (ΔW) | cosine **1.0**, KL 0.0, token-match 1.0 | cosine 0.0 (FAILURE) | public: HIGH, private: none |
| 5 future-token prediction | next-token top1 **1.0** (bigram baseline 0.0222) | = baseline | public: HIGH (single forward), private: none |

**Gram-ONLY attacker** (uses `ZGZ^T` but NOT the value multiset, public weights): token top1 0.0391 = baseline. The Gram matrix alone carries no vocab identity without a labeled reference — the danger is the **co-leaked value multiset**, not the Gram per se.

## Required conclusions

- **Token reconstruction?** YES under public weights (top1 1.0); NO under private weights (only token *linking* — equal tokens are detectable, not identifiable).
- **Target/label recovery?** YES under public weights (recovered-token bag-of-words, AUC 0.8221); under private weights top1 0.4583 does not beat the base rate 0.4083 (AUC 0.6198 is only marginally above 0.5) — not a usable recovery.
- **Membership above random?** YES — AUC 0.6063 via the exactly-leaked per-token gradient norm; this signal is model-agnostic and survives **private** weights.
- **Adapter extraction / functional clone?** YES under public weights (ΔW cosine 1.0, heldout KL 0.0); NO under private weights (cosine 0.0 — cannot recover inputs, solve the column permutation, or fit ΔW). Note: private `token_match` 1.0 is high only because this adapter is functionally small, NOT because extraction succeeded — `dW_cosine` is the real signal.
- **Future-token beyond baseline?** YES under public weights, but this is the *single-forward* setting where the future positions' activations are already visible (= token reconstruction on those rows). In genuine streaming decode only the current step is exposed; the meaningful future-prediction then reduces to running the recovered public model. Private weights = bigram baseline.

**Classification.** Under **private base weights** the nonlinear cross-Gram is *auxiliary structural leakage*: it did NOT recover user tokens, targets, or a functional adapter — except **membership inference (MEDIUM)**, which rides the model-agnostic gradient-norm signal. Under **public base weights** the co-leaked pre-activation value multiset is **HIGH**: exact token reconstruction, target recovery, and full adapter (ΔW) extraction. This matches the CP-uniqueness / pre-activation-exposure dichotomy from the prior audits.

## Allowed conclusion (per-regime)

- *Private weights:* "Nonlinear permutation islands leak the exact token×token activation–gradient Gram matrices and per-token gradient norms, but under the evaluated attackers this auxiliary structural leakage did not recover user data, target tokens, or functional adapters — with the exception of membership inference (medium), which the gradient-norm signal enables."
- *Public weights:* the same leak (specifically the co-leaked pre-activation value multiset) is HIGH: token/target/future-token recovery and ΔW adapter extraction all succeed.

## Disallowed conclusions (NOT claimed here)
- "No leakage exists." — the Gram/value/gradient-norm leaks are exact and measured.
- "Cross-Gram is harmless." — only stated per-regime, with attack numbers.
- "Formal privacy proven." — no formal claim; this is an empirical attacker panel.
- "Adapter protection proven." — adapter extraction was run; it SUCCEEDS under public weights and only fails under private weights.

## Threat-model scope

`attacker_sees_only_transcript = true` (masked activations `Z_tilde`/`U_tilde`, backward `GZ_tilde`, all mod a global column permutation). `public_base_weights` is the decisive toggle. Our paper's threat model is public base weights + protect user data — so the public-weights column is the operative one, and it shows the value-multiset leak is a genuine user-data / adapter-privacy risk, consistent with the paper reporting nonlinear-region leakage as proxy-only / `needs_more_evaluation` rather than `low`.
