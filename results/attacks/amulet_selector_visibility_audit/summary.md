# Amulet/Kronecker Nonlinear Island — Selector-Visibility Audit

**Question:** in the real Amulet/Kronecker forward-island constructions, is the
selector (the secret unit coordinate `(a,b)` / valid-column index) actually
invisible to the untrusted GPU, and is the pre-activation hidden?

**Scope:** code audit of the three constructions in the tree; adversary = the
untrusted accelerator that executes the nonlinear op and sees whatever operators
/ lifted tensors are handed to it. All numeric leaks below were reproduced at
float64. No production path modified. Not committed.

**Bottom line — the dichotomy (holds across all three constructions):**

> **A hidden selector ⟹ the TEE computes the whole nonlinear (no offload).
> An offloaded nonlinear ⟹ the selector and/or the pre-activation is exposed.**

There is no implemented construction that both offloads the nonlinear to the GPU
*and* provably hides the selector+activation. This is the concrete, code-level
instance of the CP-uniqueness obstruction (a fixed nonlinear op has no
private-weight cover, so its offloaded operator cannot hide the mask/selector
beyond a permutation/monomial residual — see `nonlinear_backward_primitive_audit`
and the prior masked-Hadamard theory note).

---

## Construction 1 — `ops/amulet_right_mask_islands.py` (experimental island)

The elaborate island: `U n → phi(U) n` via a Kronecker lift with secret unit
coordinate `(a,b)`, permutations `π1..π4`, selectors `E1=I⊗e_aᵀ`, `E2=I⊗e_b`,
dense single-one `R_bar=R1R2R3`. Operators `M1..M4` built in
[`_island_operators`](../../../src/pllo/ops/amulet_right_mask_islands.py#L362).

- **Not wired to any worker.** `grep` of `src/pllo/protocol/` for the island
  symbols is empty; it is used only in experiments and `ops/two_sided_island.py`.
  So it does **not** currently demonstrate a GPU-side island at all — as written
  it is a monolithic function holding the full secret `params`.
- **If offloaded (M1..M4 sent to GPU), the mask/selector leak — verified:**
  - `M4 = π4ᵀ · E2 · π2ᵀ · n`: its `d` nonzero rows are exactly the rows of
    `π2ᵀ n`. **Recovered 6/6 rows of `n` up to a row-permutation** directly from
    `M4`'s nonzero rows.
  - `M2 == kron(n_inv·π2, r3) · π4` **exactly** (verified) → `n_inv` recoverable
    up to permutation via nearest-Kronecker-product.
  - `M3 = π1ᵀ E1 π3ᵀ` is a **0/1 matrix with exactly one 1 per row** (verified) —
    a permuted selector; its support is exposed.
- **Conclusion:** offloading this island collapses the dense mask `n` to a
  permutation residual and exposes the selector support — exactly the CP
  prediction. The `secret (a,b)` field is never *printed*, but the *operators the
  GPU would need* leak the mask up to permutation.

## Construction 2 — `nonlinear/amulet_backend.py :: amulet_migrated` (**deployed**)

The **actually-wired** worker backend (`gpu_worker.py:392` genuinely executes it).
It is *not* the elaborate island — it is a much simpler selector-lift
([`_selector_lift`](../../../src/pllo/nonlinear/amulet_backend.py#L88)):
`lift = U ⊗ R` `[m,h,k]`, `Af = act(lift)`, then `out = Af.gather(2, valid)`.

1. **Selector GPU-visible?** **Yes.** `valid` (per-feature valid-column index,
   [:80](../../../src/pllo/nonlinear/amulet_backend.py#L80)) is used in
   `Af.gather(2, valid)` at [:102-103](../../../src/pllo/nonlinear/amulet_backend.py#L102),
   and `_selector_lift` "runs on whatever accelerator holds `x` (… CUDA on the
   H800)". So the selector index is a plaintext tensor on the accelerator.
2. **Permutations GPU-visible?** N/A — there are no permutations; nothing hides
   the selector.
3. **`valid` recoverable from `R`?** **Yes, trivially.** The valid column is the
   unique one with `R[i,c] == 1.0` (decoys ∈ [0.5,1.5],
   [:83](../../../src/pllo/nonlinear/amulet_backend.py#L83)) — one equality scan
   of `R` recovers the entire selector, even without the on-device `gather`.
4. **Equivalent bilinear tensor?** Nothing to reconstruct — there is **no mask on
   the activation**. `amulet_migrated` claims numeric parity with `current` =
   `F.silu(x)` ([:16-18](../../../src/pllo/nonlinear/amulet_backend.py#L16);
   [current_backend.py:43](../../../src/pllo/nonlinear/current_backend.py#L43)),
   so its input `x` is the raw pre-activation, visible to the accelerator.
5. **Raw pre-activation exposed?** **Yes** — the lift is built directly from the
   unmasked `U = x`.
- **Self-labeled** `security_status = "not_formally_claimed"`,
  `security_claim_status = "under_discussion"`, with a security_note that
  explicitly flags selector leakage
  ([:41-47](../../../src/pllo/nonlinear/amulet_backend.py#L41)). The audit
  **confirms** that caveat. It **cannot** support a formal secure-nonlinear-offload
  claim.

## Construction 3 — `nonlinear/amulet_secure_r_backend.py :: amulet_secure_R` (follow-up audit)

The "hardened" backend
([`secure_r_activation`](../../../src/pllo/nonlinear/amulet_secure_r_backend.py#L75)):
`lifted = kron(u, R_bar)`, secret full shuffles `p_row [mk]`, `p_col [hk]`,
`z = shuffle(lifted)` (GPU-visible), `s = act(z)`, then un-shuffle + squeeze the
secret `(a,b)`. `R_bar` is dense single-one; `_assert_secure_rbar` enforces **no
zero decoys** and **no off-`(a,b)` entry ≈ 1**
([:57-72](../../../src/pllo/nonlinear/amulet_secure_r_backend.py#L57)).

Audited against the 5 questions:

1. **Selectors / indices GPU-visible?** **Improved vs `amulet_migrated`.** There is
   no explicit `valid` index used in a GPU `gather`; the squeeze uses secret
   `(a,b)` + inverse shuffles, intended to be folded offline. In the **code as
   written it is still monolithic** (lift+shuffle+act+unshuffle+squeeze in one
   function) — the "folded offline" trusted squeeze is *claimed, not enforced* by
   a serialization boundary.
2. **Decoy/scale factors sufficient to hide the valid column?** The two
   `amulet_migrated` tells are removed: dense decoys (no zeros) and no decoy equal
   to 1, so there is no `R==1` scale signature and no zero-structure. This is a
   real hardening of *which-column* hiding.
3. **`valid` recoverable from `R` / runtime tensors?** `R_bar` is not exposed; the
   only GPU artifact is `z = shuffle(kron(u,R_bar))` under **full unknown
   permutations** of `mk`/`hk`, which makes naive nearest-Kronecker /
   equality-scan recovery hard. Recovery of `(a,b)` is therefore blocked **only
   under the unproven "secure-R assumption"** — the backend states exactly this:
   `security_status = "claimed_under_secure_R_assumption"`, "NOT a completed
   formal proof" ([:23-31, :129-139](../../../src/pllo/nonlinear/amulet_secure_r_backend.py#L23)).
4. **Equivalent selector/lift tensor reconstructable?** Harder than Construction 1
   (there `M2` was an *exact* visible Kronecker with a one-sided permutation; here
   the shuffle is two-sided and full). Exact reconstruction is not trivial — but
   this is heuristic hardness, not a theorem.
5. **Raw pre-activation exposed, or only masked activation?** **Raw
   pre-activation values are exposed.** There is **no linear mask on `u`**; at the
   `R_bar==1` block position `kron(u,R_bar)=u`, so **every** pre-activation value
   appears in the GPU-visible `z`. **Verified: 24/24 raw pre-activation values
   present in `z`** (shuffled + decoy-mixed). The obfuscation is *combinatorial*
   (which entries are real is hidden), **not** a linear mask — the pre-activation
   *content* (value multiset, hence its distribution/outliers) is on the GPU.

**Verdict on `amulet_secure_R`:** genuinely hardens *selector-position* hiding vs
`amulet_migrated` (no on-device `valid`, dense single-one R, two-sided full
shuffles), and its security is **honestly labeled as an assumption, not a proof**.
But it still (i) exposes the raw pre-activation *values* in the GPU-visible lift,
and (ii) does not enforce the "folded offline" squeeze as a trusted boundary in
code. It therefore still sits on the *offload* side of the dichotomy: it bets that
the shuffle hides the selector, while the activation content is exposed.

---

## Cross-construction summary

| | Construction 1 (islands) | Construction 2 (`amulet_migrated`, deployed) | Construction 3 (`amulet_secure_R`) |
|---|---|---|---|
| Wired to worker | no (experiment) | **yes** | selectable design |
| Selector index on GPU | in `M3/M4` if offloaded | **yes** (`gather(2,valid)`) | no explicit index (shuffled) |
| Selector recoverable | mask `n` → perm (verified) | trivial (`R==1` scan) | only-if secure-R assumption breaks |
| Decoys | dense R_bar | scales, **`R==1` tell** | dense single-one, **no tells** |
| Raw pre-activation on GPU | masked `Un` | **yes (unmasked)** | **yes (values in `z`)** |
| Security label | experiment-only | `not_formally_claimed` | `claimed_under_secure_R_assumption` |

## Allowed claims
- "The deployed `amulet_migrated` backend exposes the selector (`valid`) and the
  raw pre-activation to the untrusted accelerator; it is self-labeled
  `not_formally_claimed` and cannot back a formal secure-nonlinear-offload claim."
- "The experimental right-mask island, if offloaded, leaks its dense mask `n` up
  to a permutation (verified: 6/6 rows recovered from `M4`; `M2` is an exact
  visible Kronecker of `n_inv`)."
- "`amulet_secure_R` hardens selector-position hiding (no on-device index, dense
  single-one R, two-sided shuffles) and its security is stated as an assumption,
  not a proof; it still exposes the raw pre-activation values (24/24 present in the
  GPU-visible lift) and does not enforce the folded squeeze as a trusted boundary."
- "Dichotomy: hidden selector ⟹ TEE computes the nonlinear (no offload);
  offloaded nonlinear ⟹ selector and/or pre-activation exposed."

## Disallowed claims
- ❌ "The Amulet island hides the selector from the GPU." (deployed backend: `valid`
  on device; island: mask leaks up to permutation if offloaded)
- ❌ "Amulet nonlinear offload is secure / privacy-preserving." (`not_formally_claimed`
  for the deployed backend; `claimed_under_assumption` for secure_R — neither proven)
- ❌ "`amulet_secure_R` hides the activation." (raw pre-activation values are in `z`)
- ❌ "The nonlinear can be offloaded and the selector hidden simultaneously." (no
  implemented construction achieves both; contradicts the dichotomy)
