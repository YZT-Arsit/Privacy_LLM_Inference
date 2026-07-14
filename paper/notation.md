# Notation and Dimensional Audit

Authoritative symbol table for the paper. Every symbol lists **dimensions**, **ownership** (public/private), and **side** (trusted vs accelerator-visible). Kept in sync with `macros.tex`. Consolidated from `paper_draft/notation.md`, `design.md`, `system_and_threat_model.md`, `PAPER_THEORY_OUTLINE.md`, `threat_model_revision_patch.md`.

**Row-vector convention** (activations are rows; right-action masks multiply on the right). Distinguish object types — these are NOT interchangeable: orthogonal matrix, permutation matrix, signed permutation, invertible dense matrix, one-time additive pad, cancellation padding, rank-space mixer, residual-space mask, attention-head-space mask.

**Equality operators:** `=` plaintext/trusted equality · `≟` (tested) numerically verified allclose in the stated config (give magnitude+dtype) · `≐` (alg) exact algebraic identity from construction. In `macros.tex`: `\eqtested`, `\eqexact`.

---

## 0. Entities, sets, and architecture constants (Problem/Threat section)

| Symbol (macro) | Meaning | Kind | Public/Private |
|---|---|---|---|
| `𝓟` (`\prov`) | model provider — owns the proprietary checkpoint | party | — |
| `𝓤` (`\usr`) | user — owns LoRA data/adapter, prompt/context, and generated result | party | — |
| `𝓢` (`\cloud`) | cloud service — administrative party containing the TEE and GPU | party | — |
| `𝓒` (`\ctrl`) | trusted execution environment (TEE) | cloud component | trusted |
| `𝓖` (`\acc`) | untrusted GPU | cloud component | honest-but-curious |
| `𝓐` (`\adv`) | confidentiality adversary (= honest-but-curious `𝓖`) | adversary | — |
| `θ` (`\params`) | proprietary parameter set: embeddings, projection/output-head weights, norm gains, and learned biases | parameters | **private** |
| `γ` (`\rmsgain`) | RMSNorm/LayerNorm gains (part of `θ`) | vector per norm | **private** |
| `Ω` (`\maskset`) | mask family drawn per call/session: `{N_in,N_out,N_Q,N_K,N_V,R,P,U}` | secret randomness | **private** |
| `Φ` (`\arch`) | public architecture tuple `(L,d,h,h_kv,d_h,m,V)` | constants | public |
| `L,d,h,h_kv,d_h,m,V` | #layers, model dim, #query heads, #KV heads, head dim, FFN dim, vocab | constants | public |
| `x`, `n` | input token sequence and its length `n` | input / scalar | value private; `n` public |
| `𝕊` (`\Ssec`) | secret-resident set (never leaves `𝓒`): `θ,Ω,T,(A,B)`, optimizer state, `Rec` | set | private |
| `𝕋` (`\Stra`) | transient-plaintext set (materialized in `𝓒`, erased after a pass): `X,Q,K,V,`logits`,G,dA,dB` | set | private |
| `𝕆` (`\Sobs`) | observable set (crosses to `𝓖`): `X̃,W̃,Ã,B̃,K̃,Ṽ,Ỹ` | set | masked (allowed) |
| `ℙ` (`\Spub`) | admitted metadata: `Φ`, shapes, `n`, cache length, `r_pad`, call count, coarse operation schedule | set | public |
| `τ` (`\transcript`) | accelerator transcript = the sequence of `𝕆` tensors + `ℙ` metadata that `𝓖` observes | adversary view | observable |

**Collision note:** the base glyph `P` appears as three distinct objects in three fonts — permutation `P` (`\Perm`, bold), public set `ℙ` (`\Spub`, blackboard), provider `𝓟` (`\prov`, calligraphic). Similarly `T`: pad `T` (`\Tpad`, bold) vs transient set `𝕋` (`\Stra`, blackboard); `G`: gradient `G` (`\Gmat`, bold) vs accelerator `𝓖` (`\acc`, calligraphic). Fonts disambiguate; never mix.

## 1. Symbol table

| Symbol (macro) | Meaning | Dimensions | Public/Private | Side |
|---|---|---|---|---|
| `X` (`\X`) | plaintext hidden state / input activation to a Linear | `[T,d_in]` (or `[B,T,d_in]`) | private | trusted (accelerator sees `X̃`) |
| `X̃` (`\Xt`) | masked hidden state `=(X−T)N_in` | `[T,d_in]` | masked (allowed leakage) | accelerator-visible |
| `W` (`\W`) | base weight | `[d_in,d_out]` | **private (proprietary); only `W̃` crosses** (canonical, CF-1 resolved) | trusted |
| `W̃` (`\Wt`) | masked weight `=N_in⁻¹WN_out` | `[d_in,d_out]` | masked | accelerator-visible |
| `b` | learned bias, when present (part of `θ`) | `[d_out]` | **private** | trusted (only `bN_out` crosses) |
| `Y` (`\Y`) | plaintext Linear output `XW+b` | `[T,d_out]` | private | trusted |
| `Ỹ` (`\Yt`) | masked output `=YN_out` | `[T,d_out]` | masked | accelerator-visible |
| `T` (`\Tpad`) | **additive one-time pad** (boundary translation) | shape of `X` | private | trusted (only `TN_in`, `TWN_out` cross) |
| `Z` (`\Z`) | generic intermediate in a nonlinear island | — | private | trusted (`Z̃` visible) |
| `N_in,N_out` (`\Nin,\Nout`) | boundary right-action masks | `[d_in,d_in]`,`[d_out,d_out]` | private | trusted |
| `N_Q,N_K,N_V` (`\NQ,\NK,\NV`) | per-head Q/K/V masks; require `N_Q N_Kᵀ=I` | `[d_h,d_h]` | private | trusted |
| `R` (`\Rmask`) | residual-space (orthogonal) mask | `[d_model,d_model]` | private | trusted |
| `P` (`\Perm`) | channel/head permutation in nonlinear island | permutation | private | trusted |
| `U` (`\Umix`) | rank-space LoRA mixer (invertible/orthogonal) | `[r_pad,r_pad]` | private | trusted |
| `A` (`\Amat`) | LoRA down-projection | `[d_in,r]` | private | trusted |
| `B` (`\Bmat`) | LoRA up-projection | `[r,d_out]` | private | trusted |
| `Ã,B̃` (`\At,\Bt`) | masked LoRA factors `Ã=N_in⁻¹AU, B̃=U⁻¹BN_out` | `[d_in,r_pad]`,`[r_pad,d_out]` | masked | accelerator-visible |
| `A_pad,B_pad` (`\Apad,\Bpad`) | rank-padded factors, `A_pad B_pad=AB` | `[d_in,r_pad]`,`[r_pad,d_out]` | masked | accelerator-visible |
| `A_dummy,B_dummy` (`\Adummy,\Bdummy`) | dummy padding blocks, `A_dummy B_dummy=0` | — | private/masked | trusted-designed |
| `r` (`\rtrue`) | true LoRA rank | scalar | **hidden from shape** | — |
| `r_pad` (`\rpad`) | padded rank, `r_pad≥r` | scalar | **VISIBLE (allowed leakage, U6)** | accelerator-visible |
| `α` | LoRA scaling | scalar | public | trusted |
| `Q,K,V` (`\Qmat,\K,\Vmat`) | attention query/key/value per head | per-head `[·,d_h]` | private | trusted (`Q̃,K̃,Ṽ` visible) |
| KV cache | stored masked: `K_{1:t}N_K`, `V_{1:t}N_V` | grows on token axis | masked | accelerator-visible |
| `G` (`\Gmat`) | upstream gradient `∂L/∂Y`; `G̃=GN_out⁻ᵀ` | `[T,d_out]` | private | trusted (`G̃` visible) |
| `dA,dB` | per-step LoRA gradients | `A/B` shapes | private | trusted (`dÃ,dB̃` visible) |
| `C_T` (`\Ct`) | transformed compensation `C_T=TWN_out` | `[·,d_out]` | masked/observable | produced by TEE; consumed by GPU |
| `Rec` (`\Recover`) | trusted recovery `Y=ỸN_out⁻¹` | operator | — | **trusted only** |
| `φ` | pointwise activation (GELU/ReLU/SiLU) | — | — | accelerator (on masked input) |
| `RMSCore/LNCore` | orthogonally-invariant norm cores | — | — | mixed (wired: trusted recompute) |

---

## 2. Symbol collisions / inconsistencies to RESOLVE before authoring (do NOT paper over)

1. **`T` overloaded — HIGHEST PRIORITY.** `T` = sequence length in `X∈[T,d_in]` **and** `T` = additive pad tensor. Direct collision in the same file. **Fix:** use `L` (or `S`) for sequence length; reserve `T` for the pad (this table already reserves `T` for the pad).
2. **LoRA rank-space mixer has THREE incompatible symbols:** `U` (notation/design: `Ã=N_in⁻¹AU`), `M` (theory T11–T12: `Ã=N_xᵀAM`, boundaries `N_x,N_y`), `R` (audit-v2: `Ã=M⁻¹AR`, with `M`=input mask). **Fix:** adopt `U` for the mixer and `N_in/N_out` for boundaries (this table's convention); rewrite theory/audit-v2 formulas to match.
3. **`M` overloaded:** generic operator mask (notation) vs rank mixer (theory) vs input boundary mask (audit-v2). **Fix:** retire `M`.
4. **`U` overloaded:** LoRA inner mixer (notation) vs generic nonlinear-island input activation in theory §12 Amulet (`UN→φ(U)N`). **Fix:** rename the Amulet activation to `Z`.
5. **`P` vs `Π`:** draft writes permutation `P`; theory writes `Π` and uses `P` for an orthogonal mask (and `P=I` in the Amulet contract). **Fix:** `P` = permutation everywhere; use `R` for residual/orthogonal mask.
6. **Head-space mask name:** `N_Q/N_K/N_V` (draft) vs `N_h` (theory). Residual mask `N_res` (theory) vs `R` (here). **Fix:** standardize on `N_Q/N_K/N_V` + `R`.
7. **`r` vs `r_true`:** same meaning, different symbol across ledgers. **Fix:** `r` = true rank, `r_pad` = padded rank.
8. **Inverse vs transpose convention:** design uses `N⁻¹` ("invertible"); theory uses `Nᵀ` ("orthogonal"). **Layer-dependent:** Thm 1 (linear) needs only **invertible**; Thm 4/5/6 (norm/attention) need **orthogonal**. State the required object type PER theorem (see `macros.tex` object-type comments).

---

## 3. Per-equation audit (dimensions, order, transpose, freshness, trusted-side, recovery, assumptions, impl)

For each inherited equation, the 8-point check. ✔ = verified against artifact; ⚠ = issue to resolve.

| Eq | Statement | Dims | Order/transpose | Freshness | Trusted-side | Recovery | Assumptions | Impl (✔/⚠) |
|---|---|---|---|---|---|---|---|---|
| E1 Linear+pad | `((X−T)N_in)(N_in⁻¹WN_out)+TWN_out+bN_out=(XW+b)N_out` | ✔ `[T,d_in]·[d_in,d_out]` | ✔ right-action; `N_in⁻¹` cancels | ✔ boundary masks fresh per call | `T,C_T=TWN_out` trusted | `ỸN_out⁻¹` | `N_in,N_out` invertible | ✔ `ops/linear.py` (C-COR-01) |
| E2 Attention | `(QN_Q)(KN_K)ᵀ=QKᵀ` | ✔ per-head `[·,d_h]` | ✔ `N_Q N_Kᵀ=I` | ✔ per-call | V-mask absorbed by o_proj | — | orthogonal-tied, block-diagonal | ✔ `ops/attention.py` (C-COR-02) |
| E3 RoPE | `RoPE(xN)=RoPE(x)N` | ✔ head_dim even | ⚠ only 2×2-block masks; dense FAILS | ✔ | — | — | adjacent-pair convention | ✔ but **wired path masks POST-RoPE** (C-COR-03) |
| E4 KV-append | `[K₁N_K;…]=[K₁;…]N_K` | ✔ token-axis concat | ✔ | ⚠ **KV mask FIXED within session** | — | — | fixed-KV | ✔ `cache/kv_cache.py`; ⚠ **CF-5** design says "fresh per token" (C-COR-05) |
| E5 RMSNorm island | `RMSCore(XN)=RMSCore(X)N` | ✔ | ✔ orthogonal | ✔ | ⚠ **wired: trusted recompute** | re-mask | orthogonal + **γ folded** | ⚠ vector-γ FAILS (err 9.68) (C-COR-06) |
| E6 SwiGLU | `(XW_aP)⊙SiLU(XW_bP)=((XW_a)⊙SiLU(XW_b))P` | ✔ | ✔ **paired/shared** P | ✔ | — | — | shared permutation | ✔ EXACT (C-COR-07) |
| E7 LoRA fwd | `Ỹ+(α/r)TABN_out=YN_out`, `Ã=N_in⁻¹AU,B̃=U⁻¹BN_out` | ✔ `[d_in,r_pad]·[r_pad,d_out]` | ✔ `U⁻¹U=I` | ✔ | scaling `α/r` (not `α/r_pad`) | `ỸN_out⁻¹` | invertible masks | ✔ `ops/lora.py` (C-LORA-FWD) |
| E8 LoRA bwd | `G̃=GN_out⁻ᵀ`; `dA=N_in⁻ᵀdÃUᵀ`, `dB=U⁻ᵀdB̃N_outᵀ` | ⚠ verify transpose placement | ⚠ **terse — expand (COR-01)** | ✔ | loss+optimizer trusted | un-mask grads | linear layer | ✔ fp64 (C-LORA-BWD); nonlinear bwd ⚠ not impl |
| E9 Rank pad | `A_pad B_pad=AB`, `A_dummy B_dummy=0` | ✔ | ✔ cancellation | init-time | dummy design trusted | — | exact-cancellation | ✔; ⚠ strategy count "four"/six/**seven** wrong (C-LORA-RANK) |
| E10 Masked SGD | `θ_t−lr·(PPᵀ)g_t(QᵀQ)`; orthogonal ⇒ `=P(θ−lr·g)Q` | ✔ | ✔ **linear** update commutes | per-step | — | un-mask | orthogonal masks | ✔ (C-OPT-SGD); ⚠ γ-fold breaks orthogonality → O1-C |
| E11 Masked AdamW | `√(PgQ)≠P√gQ`, `1/(PvQ)≠P(1/v)Q` | — | ⚠ **elementwise nonlinearity does NOT commute** | — | O1-C trusted correction | — | — | ⚠ GPU dense-mask UNSUPPORTED; O1-C exact fp32 (C-OPT-*) |

---

## 4. Unresolved mathematical issues (record, do NOT repair speculatively)

- **MI-1** (E8): LoRA-backward transpose/inverse cancellation is stated tersely; the exact `dA,dB` recovery with `U⁻¹`/`N_out⁻¹` placement must be re-derived and dimension-checked (reviewer risk COR-01). Artifact confirms numeric recovery (fp64 ~1e-16) but the paper derivation is incomplete.
- **MI-2** (E4/CF-5): reconcile "fresh masks per token" (boundary) with "KV mask fixed within session" — state precisely which masks refresh and which are session-bound.
- **MI-3** (E5): RMSNorm orthogonal-island identity holds only with scalar γ or γ folded into adjacent projections; the wired implementation sidesteps it via trusted recompute. The paper must not present E5 as the executed path without the deviation note.
- **MI-4** (E10/E11): "SGD is the only exact GPU path" is conditional on **orthogonal** masks; the RMSNorm γ-fold makes q/k/v/gate/up non-orthogonal (cond up to 8e5) → the O1-C hybrid is required. State this dependency explicitly.
- **MI-5** (E9): fix the dummy-strategy count to the actual **seven** and add the tracked-Δ correction for `noise_injected_cancellation` (which is not zero-contribution).
- **MI-6** (object types): every theorem must name whether its mask is invertible / orthogonal / permutation / paired-permutation / signed-permutation — several inherited statements say only "mask."
