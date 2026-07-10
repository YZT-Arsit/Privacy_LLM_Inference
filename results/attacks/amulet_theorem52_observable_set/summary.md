# Theorem 5.2 / Observable-Set Investigation (b)

**Goal:** locate the exact Theorem 5.2 observable set and verify main-text vs
appendix consistency; then compare variants A/B/C. **Finding: the premise does not
hold in this repository** — there is no `I(X;O)=0` theorem, and the Amulet paper
is not present. Details below, followed by the corrected comparison table.

## 1. What "Theorem 5.2" actually is in this repo

- `paper_draft/latex/sections/05_correctness.tex` is **`\section{Correctness
  Analysis}` (Section 5)**. Its theorems number 5.1, 5.2, …:
  - **5.1** — *Linear boundary with pad compensation*.
  - **5.2** — *Pointwise activation permutation island*: "Let `φ:ℝ→ℝ` be pointwise
    and `P` a permutation matrix. Then for every `Z`, `φ(ZP) = φ(Z)P`."
- **Theorem 5.2 is a correctness / invariance statement, not a security theorem.**
  It does **not** define an observable set `O = {X̃, M1, M2, M3, M4, R2}`, and it
  makes **no** `I(X;O)=0` claim. Its "observable" is simply `X̃ = Z·P` and
  `φ(Z·P)`, and the claim is functional (the permutation commutes with `φ`).

## 2. There is no `I(X;O)=0` island theorem anywhere in the repo

- Repo-wide search for `I(X;·)` / "mutual information" / information-theoretic
  privacy over the island returns **only negative statements**: "Kronecker does
  **not** information-theoretically remove norm/Gram", the masked-boundary mode
  "does **not** prove formal information-theoretic privacy", `security_profile =
  "proxy-evaluated, not formal"`.
- `paper_draft/latex/sections/06_security_analysis.tex` explicitly states: "we do
  **not** claim semantic, cryptographic, or formal indistinguishability." All
  security rows are **proxy-supported** with risk levels
  (`low`/`needs_more_evaluation`/`medium`/`high`), not proofs.
- The dense-Kronecker island doc (`docs/amulet_right_mask_nonlinear_islands.md`)
  sets **`formal_security_claim: false`** and labels itself a
  "nonlinear_island_correctness_experiment".

## 3. The Amulet paper is not in the repo

`papers/` = {Arrow, BRE, EDNN, ObfuscaTune-AAAI, PIA, Permutation, STIP-NDSS}.
There is no Amulet PDF/tex. **So the original Amulet "Theorem 5.2" and its
appendix cannot be located here, and I cannot verify a main-text-vs-appendix
observable-set consistency for it.** If the `O = {X̃, M1..M4, R2}, I(X;O)=0`
statement is from the original Amulet paper, it must be checked against that text,
which is not available in this tree.

The `{X̃, M1, M2, M3, M4, R2}` observable set *does* correspond to the
**dense-Kronecker experiment** (`ops/amulet_right_mask_islands.py`,
`docs/…_nonlinear_islands.md` §"Island operators") — but in this repo that
construction carries `formal_security_claim: false`, i.e. it is not backed by an
`I(X;O)=0` theorem here.

## 4. Two corrections to my earlier audits

1. **Not a left-mask-over-sequence break.** The repo's island is already
   specialised to **`P = I`** (doc line 22: "specialised to `P=I, Q=N`"). The
   `π1`/`π3` inside `M1 = π3(π1⊗R1)`, `M3 = π1ᵀE1π3ᵀ` are **transient internal
   permutations that are undone** (net output left-mask is `I`), and the activation
   island is **position-wise** (it sits in the FFN / RMSNorm path, which is not the
   KV cache). So the paper-faithful island is **generation-compatible**; the cost is
   regenerating the row-side factors per sequence length `m`, not a causality break.
   My prior "variant B breaks KV cache via a left token mask" was **over-claimed** —
   the KV-append counterexample the paper cites is about the *general* Amulet's
   static left mask on the *residual/KV* path, not this FFN island.
2. **The `M4→N` / `R==1` leaks are conditional, not a theorem break.** "`M4` leaks
   `N` up to permutation" assumes `M1..M4` are shipped to the GPU (one offload
   split); "`valid` recoverable by `R==1` scan" is the *deployed `amulet_migrated`*
   implementation, not the paper-faithful dense-single-one `R_bar`. Neither
   contradicts a repo theorem, because **no `I(X;O)=0` island theorem exists here.**

## 5. Threat-model framing (decisive for classification)

Our paper's threat model (`sec:threat`): **base-model weights are PUBLIC**; the
protected asset is the **user's runtime data** (prompt, hidden states, KV cache,
LoRA adapter, gradients). Model-weight extraction is **out of scope**.

Consequence: "the pre-activation values are present in the GPU-visible lift" is
**not** a violation of a model-privacy theorem (there is none to violate, and
weights are public anyway). For **our project** it must be classified as a
**user-data / activation-privacy** concern — which the paper already reports as
proxy-only, `needs_more_evaluation`, not `low`. Even the clean permutation island
(Thm 5.2) reveals the activation **value multiset** (permuted columns preserve
norm/Gram), so activation-value privacy is a known open item, not a formal guarantee.

## 6. Comparison table

| property | A · `amulet_migrated` (deployed) | B · paper-faithful dense-Kronecker island | C · gen-aware right-mask-only |
|---|---|---|---|
| observable tensors match repo **Thm 5.2** | **no** (Thm 5.2 = permutation island `φ(ZP)=φ(Z)P`; A is a per-feature `U⊗R` selector-lift) | **no** — matches the `{X̃,M1..M4,R2}` **dense-Kronecker experiment** (`formal_security_claim:false`), *not* Thm 5.2 | **no** (Thm 5.2 is permutation; C is a column-only dense lift) |
| raw X / pre-activation visible | **yes** (no mask; valid col == X) | **values present in lift** (inherent) | **values present in `z`** (24/24, inherent) |
| E1/E2 or `valid` indices visible | **yes** (`gather(2,valid)` on device) | folded into `M3/M4`; if `M1..M4` shipped, `M3` = 0/1 permuted selector, `M4` support exposed | **no** (`b`/perm folded into TEE `L`/`S`) |
| valid recoverable from visible tensors | **yes** (`R==1` scan — impl bug) | **not** by `R==1` (dense single-one `R_bar`); if `M1..M4` shipped, `N` recoverable up to permutation from `M4` | **no** `R==1`; statistical attack on `z` open → see (a) |
| left mult touches token/time dim | no (per-feature) | **internally** (`π1`/`π3` transient), **net `P=I`** — no persistent left mask | **no** (column-only; rows untouched) |
| compatible with decoder KV cache | yes | **yes** (position-wise FFN, `P=I`; cost = per-`m` row-factor regen) | **yes** (cleanest; no row ops, no per-`m` regen) |
| protects model weights only, or user data too | neither (no mask; compute relocation) | targets **user data** under public weights; residual activation value-multiset leak; **no formal claim** | targets **user data**; residual activation value-multiset leak (a) |

## 7. Verdict for (b)

- The exact `Theorem 5.2 → O = {X̃,M1..M4,R2}, I(X;O)=0` statement **cannot be
  located in this repo**: the Amulet paper is absent, our Thm 5.2 is a
  *permutation-island correctness* theorem, and no `I(X;O)=0` island theorem
  exists (security is proxy-only, `formal_security_claim:false`).
- The `{X̃,M1..M4,R2}` observable belongs to the **dense-Kronecker experiment**,
  which this repo does **not** back with an information-theoretic theorem. If that
  observable is used (ship `M1..M4`), `M4` leaks `N` up to permutation; the
  faithful split (ship only `Z`,`φ(Z)`) avoids that but still exposes the
  activation value multiset.
- **Do not call the pre-activation-in-`z` finding a break of Amulet's theorem.**
  Under our public-weights threat model it is a **user-data / activation-privacy**
  item, already reported by the paper as proxy-only and `needs_more_evaluation`.
- Whether even that residual (selector hiding when `z` is visible) survives is an
  empirical question → pressure-tested in (a).
