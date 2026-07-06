# Attack & baseline paper audit

Audit of every PDF in `papers/`, mapping each to an attack or baseline, the
algorithm details found, what is missing, and the resulting implementation
status. Read in full via the Read tool (page ranges). Nothing was downloaded.

| paper file | maps to | our status |
|---|---|---|
| STIP-NDSS.pdf | **STIP baseline** (feature permutation) | baseline: **full** (whole-head RoPE-safe variant); audit: **full** |
| ObfuscaTune-AAAI.pdf | **ObfuscaTune baseline** | baseline: full (prior stage); audit: **full** |
| Permutation.pdf | **permutation/multiset attack** | attack: **full** (Alg.1–4) + structural probe |
| Arrow.pdf | **ArrowMatch weight alignment** | attack: **full** (Eq.1 + S2) |
| PIA.pdf | **PIA prompt inversion** | attack: **best_effort** (Alg.1+2 white-box; no oracle-LLM S_s) |
| BRE.pdf | **BRE/BiSR continuous embedding** | attack: forward=**best_effort**, backward(GMA)=**blocked** (needs training gradients) |
| EDNN攻击.pdf | **NN inversion (differential variant)** | attack: **full** (differential-NN) |

---

## STIP — Secure Transformer Inference Protocol (Yuan, Zhang, Li; arXiv:2312.00025 / NDSS)

**Mechanism (§5.1, Fig.4).** A **random feature-dimension permutation** (NOT
orthogonal, NOT TEE). "Semi-symmetric": a global `π` (shared with the user) plus
per-layer developer-only `π_{i,1}, π_{i,2}, π_{i,3}` and a classifier `π_c`.
Weight transforms:
```
W_q'=πᵀW_q π_{i,1}   W_k'=πᵀW_k π_{i,1}   W_v'=πᵀW_v π_{i,2}
W_o'=π_{i,2}ᵀW_o π   W_1'=πᵀW_1 π_{i,3}   W_2'=π_{i,3}ᵀW_2 π
γ'=γπ (RMSNorm)      W_c'=πᵀW_c π_c
```
Correctness **Theorem 1**: `F_{θ'}(xπ)π_cᵀ = F_θ(x)` — feature permutation is
equivariant through RMSNorm/softmax/element-wise ops. `π_{i,1}` cancels in `QKᵀ`,
`π_{i,2}` in `V·o_proj`, `π_{i,3}` in the MLP. Embedding runs on the trusted
device; KV cache stores permuted `K'=Kπ_{i,1}, V'=Vπ_{i,2}`. Protocol = Alg.1.

**Leakage (§2.3, §5.3).** Because `π_{i,1}` cancels exactly, the cloud's attention
scores `Q'K'ᵀ = QKᵀ` are **bit-identical to plaintext** (softmax map leaks
unpermuted). Per-token RMS norm / mean / variance and the coordinate **multiset**
are preserved. KPA resistance rests on the developer-only `π_{i,*}` (recovering
`π` alone does not expose weights); key rotation / one-time transform suggested.

**Missing / buggy for a faithful Qwen2 port (audit-critical):**
1. **RoPE never mentioned.** An arbitrary feature permutation does NOT commute
   with RoPE, so `Q'K'ᵀ = QKᵀ` breaks. Only **whole-head permutations** (reorder
   heads, identity within `head_dim`) commute with RoPE (same rotation per head).
2. **Multi-head / GQA.** A global `π_{i,1}` crosses head boundaries; must be
   head-structured, and GQA (`n_kv < n_q`, non-square K/V) needs a per-kv-group
   permutation broadcast to matching Q heads.
3. **SwiGLU transform bug.** Paper writes `W_1'=πᵀW_1` (no `π_{i,3}`), which
   misaligns the gate·up Hadamard product; the correct transform is
   `W_1'=πᵀW_1 π_{i,3}` (both gate and up share `π_{i,3}`).
4. **QKV biases.** Qwen2 has q/k/v bias; needs `b_q'=b_qπ_{i,1}` etc. — undocumented.
5. **Tied embeddings** (Qwen2-0.5B/1.5B) conflict with device-embed / cloud-head split.

**Our implementation** (`src/pllo/baselines/stip/`): a **numerically exact**
Qwen2 STIP using a **whole-head permutation** for `π/π_{i,1}/π_{i,2}` (RoPE-safe,
GQA-aware) and an arbitrary intermediate permutation for `π_{i,3}` (SwiGLU fixed
per point 3), QKV biases permuted (point 4). The audit verifies `F_{θ'}(xπ)π_cᵀ
= F_θ(x)` and explicitly flags points 1–5 as the gap between the paper's naive
spec and a correct Qwen2 port.

---

## Permutation attack (Thomas et al., arXiv:2505.18332)

**Threat model (§2–3).** Semi-honest party observes layer-`l` hidden states
(permuted plaintext). Knows model weights, embedding table, can run forwards
`M_{≤l}`; decoder-only (causal) required. Relaxable for STIP/Centaur (§6).

**Algorithm — autoregressive greedy decode, O(V·N)** (causal structure gives a
positional anchor). Four variants (Alg.1–4):
- **Alg.1 (unpermuted):** for each position, forward each candidate `v`, match
  last row to `h_i` by **L1** within threshold `ε`; proposal model orders `V`.
- **Alg.2 (sequence-perm):** maintain remaining-set `H`, greedy assign+remove.
- **Alg.3 (hidden-dim perm, = STIP):** match by **sorted-L1** (sorting cancels
  coordinate permutation → matches the coordinate multiset).
- **Alg.4 (factorized 2D):** sorted-L1 over remaining-set `H`.
Token + permutation recovery are **joint** (each matched `v` is the token). `ε`
tuned by ternary search on ~50 prompts. KV-cache-across-candidates speedup.

**Metrics:** % prompts perfectly decoded (Tables 1–2, ~97–100%). Only
permutation+Gaussian σ=0.01 defends (Table 3).

**Missing:** exact `ε` objective/bounds, layer/hidden-state convention,
cross-candidate KV batching, greedy-vs-Hungarian tie-breaking, BOS handling.

**Our implementation** (`permutation_multiset_attack.py`): (a) a **structural
leakage probe** (coordinate-multiset / sorted-vector + per-token norm
preservation, permutation recovery by sorted-L1 Hungarian matching against a
plaintext reference) — full; (b) the **autoregressive sorted-L1 greedy decode**
(Alg.3) on toy / tiny Qwen — full, `ε`-thresholded.

---

## ArrowMatch — Game of Arrows (Wang et al., USENIX Sec '25)

**Threat model (§3).** Attacker has the **obfuscated offloaded weights** and the
**public pre-trained weights** (victim = fine-tuned from public; Obs1: directions
barely move). Not the victim weights, not the keys. Weight-leakage worst case.

**Algorithm (§5).** Per weight matrix, column view `W=[w_1..w_n]`.
- **S1 (Eq.1):** `σ(i) = argmin_j cos_dist(w_obf^i, w_pre^j)` — per-column greedy
  argmin (not Hungarian). Recovers the permutation. Cosine is invariant to column
  permutation + positive scaling; broken only by direction-changing matrix mixing.
- **S2:** scale `ŝ_i = ‖w_pre^{σ(i)}‖/‖w_obf^i‖`; init `W_init`, fine-tune small.

**Metrics:** `Acc_σ` (permutation recovery, >99%), `Sim` (length), surrogate
accuracy (≈ white-box). Breaks permutation/scaling/shuffle/additive schemes.

**Missing:** cosine = 1−cos_sim; sign handling; `l(·)`=L2; S2 fine-tune budget.

**Our implementation** (`arrowmatch_attack.py`): S1 (per-matrix cosine argmin +
optional Hungarian) + S2 (norm-ratio scaling) — **full**; reports `Acc_σ`,
alignment accuracy, length `Sim`. Also loads external ArrowMatch results.

---

## PIA — Prompt Inversion Attack (Qu et al., arXiv:2503.09022, IEEE S&P '25)

**Threat model (§3).** Collaborative/split inference; attacker observes an
intermediate activation `A ∈ ℝ^{s×h}` and has white-box preceding layers `F` +
embedding `E`. No logits/gradients.

**Algorithm (Alg.1–3).** Phase 1 constrained optimization on continuous `v̂`
(Eq.13): `min ‖F(v̂)−A‖² + λ Σ_i min_{t∈V} ‖v̂_i − E(t)‖²`, clip to per-dim
`[L_j,R_j]`. Phase 2 adaptive discretization (Eq.14–17): greedy autoregressive
`x̂_j = argmin_{t∈S_e∪S_s} ‖F(E(x̂_{<j},t))_j − A_j‖²`, `S_e`=top-K nearest
embeddings, `S_s`=oracle-LLM top-Y next tokens. β=0.1, λ=0.1, N=2000, K=Y=10.

**Metrics:** token accuracy (88.4% headline), BLEU, NERR.

**Requires** white-box `F` + `E` + differentiable forward + the activation `A`;
the oracle-LLM `S_s` is optional (Y=0 degrades).

**Our implementation** (`pia_prompt_inversion.py`): Phase 1 (constrained opt +
clip) + Phase 2 (embedding-candidate autoregressive activation calibration) —
**best_effort** (we omit the oracle-LLM `S_s` semantic speculation; labelled).

---

## BRE / BiSR (Chen et al., ACM CCS '24, arXiv:2409.00960)

**Threat model (§2.4).** Split *fine-tuning*; server observes smashed data
`x̃_btm` (forward) AND activation gradients `grad(x̃_trk)` (backward). Semi-white-
box: has pre-trained Bottom/Top mimics.

**Algorithm (§3).** SIP (learned GRU inverter on auxiliary data) → BRE:
- **Forward (Eq.7–8):** `e* = argmin_{e'} 1−cos(f̂_btm(e'), x̃_btm)`, then project
  `x* = argmin ‖emb(x') − e*‖`. (Continuous embedding, cosine loss.)
- **Backward (Eq.9):** gradient matching on a dummy label `y'` (vocab dist):
  `β‖∇Ldmy − ∇L‖₂ + (1−β)‖·‖₁` — **requires observed training gradients**.

**Metrics:** ROUGE-L, METEOR, token recovery.

**Our implementation** (`bre_bisr_attack.py`): the **forward** smashed-data
matching (Eq.7–8, cosine, NN projection) — **best_effort**; the **backward**
gradient-matching component is **blocked** (an inference-only harness has no
training gradients `grad(x̃_trk)`; needs an SL fine-tuning loop — see
`server_run_plan.md`). SIP GRU inverter is optional/not the core.

---

## EDNN (Lin et al., Bytedance, EMNLP '24)

**Mechanism.** Breaks the **glide-reflection** embedding obfuscation of
SentinelLMs. Glide-reflection preserves the **within-vector element-wise
difference** `e'_i[k1]−e'_i[k2] = e_i[k1]−e_i[k2]`.

**Algorithm (Alg.1).** For each obfuscated column `ẽ_i`, differential
`ẽ_i − lshift(ẽ_i)`; L2-match to each pretrained column's differential
`e_j − lshift(e_j)`; nearest neighbour → recovered token. 100% recovery on
glide-reflection; a general random mask defeats it.

**Our implementation:** a **differential-NN** mode of `nn_embedding_inversion.py`
(`metric="differential"`) — **full**. Applies only to difference-preserving
obfuscations; on general masks it degrades (reported honestly).

---

## ObfuscaTune (Frikha et al., AAAI-W)

Covered by the prior stage (`docs/obfuscatune_baseline.md`). Audit
(`src/pllo/baselines/obfuscatune/audit.py`) verifies: orthogonal Q (κ≈1), the
input/output linear equivalences, that RMSNorm/RoPE/SiLU/softmax are NOT run in
the general obfuscated domain, SwiGLU handling, GQA, KV-cache label = plaintext
(never `protected=true`), plaintext-logit alignment, and no silent fallback.
