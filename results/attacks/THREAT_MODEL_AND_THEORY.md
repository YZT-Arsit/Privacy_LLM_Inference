# Threat model, impossibility theorem, and minimal-trust defense

Paper-facing writeup backing the attack/defense results in `FINDINGS.md`. All empirical
claims are validated on real Qwen2.5-7B (and GPT-2 / Llama-3.2-1B for the reference scheme);
see `FINDINGS.md` A3–A6 and `results/attacks/*.json`.

## 1. Threat model

**Setting.** Open-weight, single trusted-runtime + untrusted-accelerator split. The model
architecture and *public* pretrained weights are known to everyone. Deployment folds the
**exact public weights** (no fine-tuning) with secret invertible masks and runs the bulk of
the forward pass on an untrusted accelerator (GPU).

**Adversary.** Honest-but-curious operator of the untrusted GPU. It follows the protocol but
inspects everything it holds:
- the architecture and public weights;
- the deployed folded/masked weights `W̃ = N_in⁻¹ W N_out`;
- every GPU-visible intermediate: the masked residual stream `hℓ·N0`, masked `q/k/v`, the
  attention logits `Q̂K̂ᵀ`, and activation inputs.
It may run arbitrary offline computation (e.g. build fingerprint/inversion tables from public
weights). **Goal:** recover the user's input tokens (and generated tokens).

**Trusted base (minimal).** A TEE (Intel TDX, remotely attested) holds the masks/secrets and
performs: token embedding, the boundary mask/unmask (single entry, single exit at logits
recovery), and — for the defense — exactly the **layer-0 attention block**. All other
nonlinearities (softmax, RMSNorm, SiLU/SwiGLU of layers 1..L-1) run on the untrusted GPU with
**zero trusted crossings**. The GPU never sees raw input, masks, `N_inv`, pad, or full logits.

**Masks (ours).** Orthogonal & *structured* (required to keep GPU nonlinearities exact without
crossings): residual/input = signed permutation; attention Q/K = pairwise rotation
(`Q̂K̂ᵀ = QKᵀ`); SwiGLU = shared channel permutation.

## 2. Impossibility: structure-preserving obfuscation cannot hide token identity

**Theorem (reduction-leak).** Let a scheme run on an untrusted device D that, from the
operands it holds, computes the model's exact output using invertible masks on hidden states
and weights, with zero trusted crossings for the nonlinear reductions (softmax, RMSNorm).
Then D can compute a context-free per-token fingerprint `g(token_p)` at layer 0 that identifies
`token_p` by nearest-neighbour against a public table. Hence token-level input confidentiality
is impossible for any such scheme.

*Proof.*
1. Correctness of attention requires D to compute the row-normalised weights
   `A_{p,·} = softmax(L_{p,·})` exactly (output `O_p = Σ_s A_{ps} v_s`).
2. `softmax` is injective up to an additive per-row constant: `A_{p,·}` determines `L_{p,·}` up
   to `+c_p·1`. Thus D knows every within-row difference `L_{pp} − L_{ps}`.
3. D computes `L` from the operands it holds; correctness forces the masks to preserve it
   (`Q̂K̂ᵀ = QKᵀ`), so D's `L` equals the **true** attention logits.
4. At layer 0 the hidden state `h_p = embed(token_p)` is a function of `token_p` alone (no
   context). Hence `L_{pp}` and the within-row differences are **context-free** functions of
   `token_p`.
5. D reconstructs `g` over the whole vocabulary from public weights and NN-matches the observed
   `g` → recovers `token_p`.
6. Any output-preserving boundary transform `T` is a fixed (deterministic) function of
   `token_p`, so D re-tables `g∘T`. The only escape is a *per-occurrence random* `T`, which
   changes the output unless corrected downstream — i.e. a second trusted crossing, contradicting
   the zero-crossing hypothesis. ∎

**Corollaries.**
- *(Norm)* Orthogonal masks preserve `‖h_p‖ = ‖embed(token_p)‖`, itself a context-free per-token
  statistic → a 1-D fingerprint; same conclusion.
- *(Structured masks leak more)* A signed permutation additionally preserves the multiset
  `{|h_p[i]|}` → sorted-`|coords|` is a context-free fingerprint. A dense rotation destroys this
  multiset, but the theorem already denies privacy to both.
- *(Fine-tuning does not escape)* Fine-tuning changes the deterministic map but a *usable* LM
  must keep tokens separable; that separability is the fingerprint. Empirically recovery falls
  to random only as perplexity diverges (Pareto: 98.5%@ppl 1.3 → 0.03%@ppl 387).

**Scope.** The theorem subsumes CONJFORMER (which acknowledges attention logits remain visible),
amulet/UNSIX (reductions run on masked state), and our A_rightmul. Empirical confirmation:
attention-fingerprint top-1 = 100% (GPT-2), 99.8% (Llama-3.2-1B), 98.5% (Qwen-7B); sorted-abs
inversion 98.6% and norm 98.5% on Qwen-7B.

## 3. Minimal-trust defense: k=1 is necessary and sufficient

Call a position **user-content** if it has ≥1 preceding non-template token; the first tokens are
the fixed, public chat template (system prompt + role headers).

**Proposition.** Relocating the first `k` attention blocks (RMSNorm+attn+residual) into the TEE:
- **k=0 (necessary):** layer-0's diagonal logit and raw masked embedding are context-free →
  100% recovery of every token (fingerprint / inversion / norm). Some relocation is required.
- **k=1 (sufficient):** removing layer 0 hides `hidden_states[0]` and the layer-0 logits. For
  ℓ≥1 and a user-content position, the GPU-visible state depends on the full prefix, so no
  context-free per-token table matches. Measured user-content recovery: fingerprint 0.05%,
  inversion 0.0%, norm ~1% (random ≈ 0.01%).
- **k>1 (no gain):** relocating more early layers leaves user-content recovery unchanged at
  ~random (k=1..6 identical); the only residual is `pos0`/early tokens, which are context-free
  at *every* layer (their K/V must stay GPU-visible so later tokens can attend) and cannot be
  zeroed by relocation — but they are the public template.

**Therefore k=1 is necessary, sufficient, and minimal for user-token privacy.** It is
bit-identical (same arithmetic, relocated), adds **no extra TEE round-trip** (the single handoff
moves one block deeper), and keeps ~27/28 attention blocks + all MLPs on the untrusted GPU.
Accounting change only: layer-0 softmax/RMSNorm become 2 trusted ops.

**Measured cost (Qwen2.5-7B, RTX 5090; `guardrail_latency.py`).** The relocated block = one
layer-0 attention (RMSNorm+q/k/v/o+SDPA+residual):

| TEE type | prefill S=512 | prefill S=1024 | decode / token |
|---|---|---|---|
| confidential GPU (best case) | 0.44 ms (1.05% of fwd) | 0.62 ms (0.77% of fwd) | 0.31 ms (2.3% of step) |
| CPU enclave (TDX CPU proxy, fp32) | 123 ms | 242 ms | ~0 (single-token attn is cheap) |

So on a confidential-GPU TEE the guardrail is **negligible (<1% prefill, ~2% decode)**. On a
CPU-only TDX enclave it is a **bounded one-time prefill cost** (≈120–240 ms for the O(S²) block,
paid once, not per generated token; decode adds only a single-token attention). Both are small
next to the scheme's existing tunnel/logits round-trip overhead and never add a round-trip.

## 4. Threat-model comparison

"w/ k=1" = with our layer-0 guardrail. ✓ = defended, ✗ = leaks/recoverable, — = not applicable.

| dimension | DP noise | CONJFORMER (split-inf, no TEE) | amulet / UNSIX (mask, no TEE) | **Ours (A_rightmul + k=1 TEE)** | TEE-everything |
|---|---|---|---|---|---|
| hardware trust anchor | none | none | none | **minimal (1 attn block + boundary)** | full model in TEE |
| architecture change | no | **yes** (scalar RMSNorm) | nonlinear islands | **no** | no |
| retraining required | no | **yes** (fine-tune) | sometimes | **no** | no |
| utility cost | high (noise) | +0.4% ppl | approx. | **0 (bit-identical)** | 0 |
| untrusted-GPU compute | all | all | all | **~27/28 attn + all MLP** | none |
| emb. inversion (naive cosine) | ✓ | ✓ (rotation) | ✓ (mask) | ✓ (perm) | — |
| emb. inversion (sorted-\|coords\|) | ✓ | ✓ (dense) | ✗ if perm-structured | ✗ raw → **✓ w/ k=1** | — |
| norm attack | ✓ | ✓ (fine-tune, D.4) | ✗ | ✗ raw → **✓ w/ k=1** | — |
| **attention fingerprint** | ✓ (noise) | **✗ (acknowledged, unmitigated)** | **✗** | ✗ raw → **✓ w/ k=1** | — |
| weight-alignment (Procrustes/Gram) | — | ✓ (block Oᵦ) | ✓ (indep. masks) | ✓ (indep. masks) | — |
| **net token-level input privacy** | ✓ (w/ utility loss) | **✗** | **✗** | **✓** | ✓ |
| scale demonstrated | small | GPT-2 / Llama-1B | small | **7B + IFEval/GSM8K/MT-Bench** | — |

**Reading.** Pure algorithmic obfuscation (CONJFORMER, amulet/UNSIX) leaves the fingerprint /
inversion / norm channels open and cannot close them without a hardware anchor; DP closes them
but at a utility cost; TEE-everything closes them but forfeits the untrusted-GPU offload. **Ours
is the only column that gets net token privacy at zero utility cost while keeping ~27/28 of the
attention and all MLPs on the untrusted GPU** — by adding the theoretically-minimal trust anchor
(k=1) that the impossibility theorem shows is unavoidable.

## 4.1 ObfuscaTune contrast (opposite TEE placement)

ObfuscaTune (Frikha et al., arXiv:2407.02960) sits in a **different threat model** — it protects a
*proprietary/secret* model + private data, not an open-weight one — so it does not slot into the
open-weight table above. It is the instructive *opposite* of our design on the one axis that matters
(where the non-linearities live), and we implement + measure it as a baseline (§A7 of `FINDINGS.md`).

| axis | ObfuscaTune | Ours (A_rightmul + k=1) |
|---|---|---|
| linear layers | obfuscated on untrusted GPU (`X*=X Ra`, `W*=Ra⁻¹W`) | folded on untrusted GPU (`W̃=N⁻¹WN`) |
| **non-linearities** | **all in TEE** (RMSNorm/softmax/SiLU) | **all on untrusted GPU** (0 nonlinear crossings) |
| TEE boundary crossings | **8/layer = 224** (28L) | **2** (single entry/exit) + k=1 guardrail |
| what the untrusted side sees | **TRUE Q/K/V** (masks cancel; measured rel-err ~1e-7) | masked `q̂/k̂` (`Q̂K̂ᵀ=QKᵀ`, `Q̂≠Q`) |
| security rests on | **weights being secret** | TEE boundary (works open-weight) |
| open-weight input privacy | **none** (read tokens off exposed Q/K/V) | **yes** (masked + k=1 closes fingerprint) |
| utility at κ=1 / exact fold | lossless (measured, GPT-2) | bit-identical |
| latency, confidential-GPU TEE | 1.05–1.09× (H800, measured) | ~1.00× |
| in-enclave nonlinear compute (real TDX, 4 vCPU, S=512/1024) | **689 / 2643 ms** | 17.8 / 78.7 ms (**30–39× less**) |

**Reading.** ObfuscaTune's "all non-linearity in the TEE" is exactly the cost our design avoids: it
buys confidentiality of *secret weights* but (i) exposes the true Q/K/V — zero input privacy in the
open-weight setting — and (ii) runs all non-linearities in the TEE, which is ~free on a confidential
GPU but on a real Intel TDX enclave (measured, `tdx_guest`=True, 4 vCPU) costs **689 ms (S=512) /
2643 ms (S=1024) of enclave compute per forward — more than the entire plaintext forward** (32 / 75 ms
on H800) before any transfer cost, because all 28 layers' non-linearities serialise through the trust
domain. Our scheme keeps every non-linearity on the untrusted GPU and adds only the theoretically-
minimal k=1 guardrail (17.8 / 78.7 ms enclave compute), for input privacy the open-weight setting
otherwise cannot get.

## 4.2 CryptoGen contrast (pure crypto, no hardware anchor)

CryptoGen (Zhang et al., arXiv:2602.08798) is the *pure-cryptographic* opposite: a hybrid HE+MPC
system with an encrypted, reusable KV cache (attention O(L²)→O(L)). No TEE; client-server semi-honest;
dual-sided (client data + model weights hidden). Linear layers run in HE (BFV CT×PT), every
non-linearity in interactive MPC (EzPC). We implement + verify its core kernels and measure real BFV
op latency (§A8 of `FINDINGS.md`).

| axis | CryptoGen | Ours (A_rightmul + k=1) |
|---|---|---|
| hardware trust anchor | **none** (HE + MPC) | minimal TEE (boundary + k=1) |
| threat model | dual-sided (data + weights hidden) | open-weight (protects user tokens) |
| non-linearities | **interactive MPC** (client↔server rounds each) | untrusted GPU, non-interactive, 0 crossings |
| real per-op cost (n=8192, measured) | CT×CT 16.9 ms, folding-sum 75 ms | — |
| per-token latency | GPT-2 ≈ **13–21 min** (12 blocks × 63–104 s, real BFV) | **11.5 ms** on Qwen-7B |
| largest model demonstrated | GPT-2 base | **Qwen-7B** |
| scales to 7B | **no** | yes |

**Reading.** CryptoGen gives strong dual-sided guarantees with *no* hardware trust, but a **single**
homomorphic reduction (75 ms) already exceeds our entire per-token 7B latency (11.5 ms); it is
~10⁴–10⁵× slower and was only ever demonstrated on GPT-2. Pure algorithmic cryptography cannot reach
7B interactive latency; a minimal hardware anchor (ours) can. Together with §4 and §4.1: DP loses
utility, split-inference/mask schemes (CONJFORMER, amulet/UNSIX) leak the reduction channels,
ObfuscaTune and TEE-everything pay the all-nonlinear-in-TEE tax, and pure crypto (CryptoGen) pays
minutes-per-token — our minimal-TEE point is the one that gets open-weight token privacy at ms latency
and 7B scale.
