# Baseline audit: ObfuscaTune & STIP on Qwen2

Correctness + boundary audit of the two obfuscation baselines. Run:
`python scripts/attacks/run_baseline_audit.py` →
`outputs/baseline_audit/{obfuscatune,stip}_audit.{json,md}`.
Audit code: `src/pllo/baselines/obfuscatune/audit.py`,
`src/pllo/baselines/stip/audit.py`.

## STIP-on-Qwen2 — status: **correct** (with fixes)

STIP = feature-dimension permutation (no TEE). Our Qwen2 port is **numerically
verified**: a genuine transformed-weight forward through the real HF RoPE/GQA
path recovers plaintext logits (`F_{θ'}(xπ)π_cᵀ = F_θ(x)`) to **~3e-8 (fp64)**,
and all seven per-linear transforms match to **~1e-16**.

**Fixes we applied that the paper omits (all flagged in the audit):**
1. **RoPE.** An unrestricted feature permutation breaks `Q'K'ᵀ=QKᵀ`. We use a
   **whole-head permutation** for `π_{i,1}=π_{i,2}` (identity within `head_dim`)
   so RoPE (applied identically per head) commutes. The global residual `π` can
   be arbitrary (undone by `πᵀ` before Q/K/V).
2. **GQA.** `π_{i,1}` is a **per-kv-group** whole-head permutation broadcast to
   the matching Q heads (mirrors `repeat_kv`), so K/V (non-square) stay aligned.
3. **SwiGLU bug.** The paper's `W_1'=πᵀW_1` (no `π_{i,3}`) misaligns the gate·up
   product; we use `W_1'=πᵀW_1 π_{i,3}` (both gate & up share `π_{i,3}`).
4. **π_{i,1}=π_{i,2}.** Under multi-head, Q/K and V internal permutations must be
   the same whole-head perm (independent only in the paper's single-head model).
5. **QKV biases.** Qwen2 has q/k/v bias; permuted with the output perm.

**Leakage (confirmed, honestly labelled):** attention scores `Q'K'ᵀ=QKᵀ` are
**unpermuted** (softmax map leaks), per-token norm/mean/variance and the
coordinate **multiset** are preserved, and the KV cache holds permuted plaintext
K/V → `kv_cache_protected=false`. No silent fallback. Prefill supported; the
relabel-forward here is prefill-oriented (decode via the transformed model).

**Not fixed (out of scope, flagged):** tied-embedding models (Qwen2-0.5B/1.5B)
need the embedding/LM-head untied for the device/cloud split.

## ObfuscaTune-on-Qwen2 — status: **correct**

Verified: orthogonal `Q` has condition number ≈ 1 and error-free inverse
(`QQ⁻¹≈I` to ~1e-16); the input (`XW=(XR)(R⁻¹W)`) and output
(`HW=(H(WR))R⁻¹`) linear equivalences hold to ~1e-16; the full-model orthogonal
obfuscation matches plaintext logits (0.0 fp64). Boundary labels are honest:
RMSNorm and SiLU are **not** obfuscation-safe under a general mask (verified) so
they stay in the (simulated) TEE; RoPE/softmax in TEE; SwiGLU gate/up
input-obfuscated with SiLU in TEE; GQA handled; **Q/K/V and attention scores
exposed in plaintext** outside the TEE; **`kv_cache_protected=false`**. No silent
fallback. Both baselines produce the intermediate representations the attacks
consume.

## Cross-baseline note
Both leak per-token norm and (STIP) the attention map / coordinate multiset —
the same structural-leakage class our own self-audit found. This is why the
security table shows STIP with high `multiset_permutation_leakage` and
`frequency_distribution`, and why ArrowMatch recovers STIP's permutation under
weight leakage.
