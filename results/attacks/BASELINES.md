# Baselines & per-dimension evaluation plan

Companion to `FINDINGS.md` (attack/defense results) and `THREAT_MODEL_AND_THEORY.md`
(impossibility theorem + k=1 defense). This file fixes the baseline **taxonomy** (by *what is
protected*, not by mechanism), then specifies, **per evaluation dimension, exactly which schemes
we compare against and what our advantage is**. All venue/arXiv metadata below was verified
against arXiv / ACL Anthology / DBLP / proceedings (2026-07).

---

## 0. Our scope (read first — it decides who is a competitor)

- **Setting: open-weight.** Architecture + *public* pretrained weights are known to everyone,
  including the adversary. We fold the **exact public weights** (no fine-tune) with secret masks.
- **What we protect: the USER's tokens** (input prompt + generated tokens) from an
  honest-but-curious untrusted-GPU operator.
- **What we do NOT protect (yet): model weights.** They are public by assumption. Model-IP is
  out of scope, not a weakness to hide.

**Consequence for baselines:** the right axis for "same family" is **what is protected**, not the
obfuscate-and-outsource *mechanism*. Two schemes can share our mechanism yet solve a different
problem. This is why the model-weight-protection lineage is NOT our competitor set.

---

## 1. The field, partitioned by what is protected

### Group A — protect USER DATA in open/known-weight setting → **our true competitors**
These make the same claim we do (hide the user's tokens from an untrusted server) and their
security must hold when the adversary knows the weights. **We compare head-to-head here.**

| scheme | venue | arXiv | mechanism | anchor | our relationship |
|---|---|---|---|---|---|
| **CONJFORMER** (Privacy from Symmetry) | preprint 2026 | 2606.16461 | orthogonal rotation of hidden states, split-inference | none | leaks attention logits (self-acknowledged); we quantify at 7B + fix |
| **STIP** | **NDSS 2026** | 2312.00025 | three-party random **permutation** of states+weights, lossless | partial TEE | broken by ICML'25 in open-weight; our anchor closes it |
| **PermLLM** | **IJCAI 2024** | 2405.18744 | permutation + secret-sharing, ChatGLM-6B, ~3s/tok | none | broken by ICML'25; slower; no per-token anchor |
| **CENTAUR** | **ACL 2025** | 2412.10652 | permutation(weights)+SMPC(data)+private softmax | none | broken by ICML'25; MPC softmax cost |
| **GELO** | preprint 2026 | 2603.05035 | TEE per-batch **invertible mixing** A, offload AH; self-notes Gram leak | TEE (mixing only) | closest cousin; still leaks reduction channel — no k=1 relocation |
| **AloePri / Covariant Obfuscation** | preprint 2026 | 2603.01499 | "covariant" fold mask into weights (= our fold) | none | our fold construction; no hardware anchor |
| **Talaria** (Black Box) | preprint 2026 | 2603.00196 | Reversible Masked Outsourcing, client CVM + cloud GPU | CVM | heavier anchor (whole CVM) |
| **amulet / UNSIX** (our repo's ref backend) | — | — | right-mask nonlinear islands | none | subsumed by our impossibility theorem |

> **Assumption caveat:** STIP/PermLLM/CENTAUR *also* hide the weights from the server, so part of
> their user-data security borrows from weight-secrecy. In our **open-weight** setting that crutch
> is gone — which is exactly the regime ICML'25 attacks. State this honestly: we assume *strictly
> less* secrecy (public weights) and still get token privacy, via the anchor.

### Group B — protect USER DATA but different paradigm → **cite to distinguish, mostly don't run**
- **DP / noise:** DP-Forward (CCS 2023), Split-and-Denoise (ICML 2024, 2310.09130),
  InferDPT (preprint 2310.12214, DP-for-generation), TextObfuscator (ACL 2023 Findings),
  ObfusLM (ACL 2025). → protect data but **lossy**; we are bit-identical.
- **Pure crypto (HE/MPC):** CryptoGen (2602.08798, already impl §A8), Cachemir (2602.11470),
  PUMA (S&S 2025), Comet (**S&P 2025**), SHAFT (**NDSS 2025**), BumbleBee (**NDSS 2025**),
  SIGMA (PoPETs 2024), MERGE (**AAAI 2024**), "Never Autoregressively Decodes" (**ICML 2025**).
  → strongest confidentiality, **seconds–minutes/token**; the "strong-security upper bound" tier.
- **Whole-model hardware TEE (Bucket C — do NOT conflate):** PipeLLM (**ASPLOS 2025**, 2411.03357),
  Confidential Prompting/Petridish (2409.19134), Bifrost (2606.17421, TEE+FHE); GPU-TEE hardware
  (Graviton OSDI'18, NVIDIA H100-CC, Telekine NSDI'20). → trust the *whole* model region; we
  minimize the anchor to k=1.

### Group C — protect MODEL WEIGHTS (IP) against a device user → **NOT competitors; mechanistic ancestors only**
Same obfuscate-and-outsource *mechanism*, **opposite threat model** (secret weights, protect owner
from user). In our open-weight setting their goal is vacuous. **Cite as lineage in one sentence; do
not benchmark.**
- Slalom (ICLR 2019), DarKnight (MICRO 2021), Goten (AAAI 2021, protects both),
  ShadowNet (**S&P 2023**), SOTER (**USENIX ATC 2022**), GroupCover (**ICML 2024**),
  TransLinkGuard (**ACM MM 2024**), TEESlice / "No Privacy Left Outside" (**S&P 2024** + TOSEM'25),
  Amulet (2512.07495), SecureInfer (2510.19979, protects both, LLM-autoregressive).

### Attack literature we must answer (the "x-axis" for Dimension 1)
- **Hidden No More / "Attack to Break Permutation-Based ... Schemes"** — **ICML 2025**, 2505.18332.
  Breaks STIP/PermLLM/CENTAUR, 97–100% prompt recovery in open-weight. **#1 must-cite.**
- **On the (In-)Security of the Shuffling Defense** — preprint 2605.04901 (extracts weights from
  STIP-like shuffling).
- **Vulnerabilities in Partial TEE-Shielded LLM Inference w/ Precomputed Noise** — 2602.11088
  (breaks static-secret reuse; motivates our per-session mask refresh).
- **EDNN** — EMNLP 2024 (2024.emnlp-main.126), 100% inversion of glide-reflection obfuscation.
- **Vec2Text** (EMNLP 2023), **TEIA** (ACL 2024), **UnSplit** (WPES 2022) — representation/split
  inversion realism.
- **CONJFORMER attack suite** — norm / ALS-Procrustes / Gram (we already reproduce these).

---

## 2. Evaluation dimensions — ranked by emphasis, each with its baselines

**Emphasis order for a user-privacy, open-weight paper: D1 ≫ D2 > D3 ≈ D4 > D5.**
D1 is the paper. D2 is the systems novelty. D3/D4 make D1 meaningful and rebut "it's just slow/lossy."

### D1 — User-token privacy under attack  ★ PRIMARY, spend the most space here
**Claim.** In open-weight, obfuscation-only leaks user tokens through the reduction channels
(attention fingerprint / sorted-|coords| inversion / norm); our **k=1 TEE** relocation closes it to
≈random at **zero** utility cost — and the leak is *unavoidable* without an anchor (impossibility
theorem §2, corroborated by ICML'25).

**Compare against (run the SAME attack battery on each):**
- **Group A obfuscation-only:** CONJFORMER, STIP-permutation, PermLLM-permutation, CENTAUR, GELO,
  amulet/UNSIX. Show every one leaks under our fingerprint/inversion/norm attacks → this *extends*
  ICML'25's permutation break to the reduction-channel fingerprint, at **7B**.
- **Our own two columns:** `A_rightmul raw` (leaks: fp 98.5%, sorted-abs 98.6%, norm 98.5% @7B) vs
  `A_rightmul + k=1` (fp 0.05%, inversion 0.0%, norm ~1%; random ≈0.01%).
- **DP noise** (DP-Forward / SnD): closes the channel but with utility loss — the contrast that
  motivates why we don't just add noise.
- **TEE-everything**: closes it but forfeits the GPU offload (the other extreme).

**Attack methods (x-axis):** attention-fingerprint (ours), sorted-|coords| inversion, norm attack,
ALS-Procrustes + Gram alignment, EDNN, Vec2Text/TEIA, **Hidden No More permutation-break (2505.18332)**,
KV-dump recovery. Run each on the same held-out prompt set (gsm8k + chat-templated).

**Metrics:** token recovery top-1/5/10, sentence reconstruction rate, attribute inference acc,
alignment recovery error; user-content vs template-position breakdown (we already do this).

**Our advantage:** NOT "our mask is stronger" (it isn't — we leak too, honestly). It is: **the k=1
anchor closes the leak that the entire obfuscation-only family cannot, provably (theorem) and at
zero utility cost.** That is the thesis.

**Run vs cite:** RUN attacks on CONJFORMER + ObfuscaTune (done) + at least one permutation scheme
(STIP-style — it's lossless and cheap to reimplement) + ours(raw/k=1). CITE ICML'25 as the
independent confirmation that permutation-only is broken.

### D2 — Trusted-boundary minimality / TCB  ★ SECONDARY (systems novelty)
**Claim.** Our anchor = boundary (1 entry/exit) + k=1 layer-0 block = **2 trusted nonlinear ops**;
everything else (≈27/28 attention + all MLP + all softmax/RMSNorm of layers 1..L-1) runs on the
untrusted GPU with **zero crossings**.

**Compare against:**
- **ObfuscaTune** (2407.02960, already impl §A7): all nonlinearities in TEE → **224 crossings**;
  measured enclave nonlinear compute 689/2643 ms (S=512/1024) vs our 17.8/78.7 ms (**30–39× less**).
- **Petridish/Confidential Prompting**: whole model in CVM (TCB = VM+OS).
- **PipeLLM** (ASPLOS 2025): whole model in H100-CC (TCB = whole model; cite its 52.8–88.2%
  throughput drop to motivate minimizing TEE-resident compute).
- **Bifrost**: all nonlinear in CPU TEE.
- **TEE-everything** strawman column.

**Metrics:** # trusted nonlinear crossings; in-enclave nonlinear compute (ms, real TDX);
fraction of attention/MLP FLOPs on untrusted GPU; extra TEE round-trips (ours: **0**); attested TCB.

**Our advantage:** theoretically-minimal anchor (k=1 is *necessary and sufficient*, §3), so we get
the same net token privacy as TEE-everything while keeping ~96% of attention + all MLP on commodity
GPU — the offload the heavy-TEE baselines forfeit.

### D3 — Utility / losslessness  ○ SUPPORTING (establish before D1 is meaningful)
**Claim.** Bit-identical folded compute; token-identical greedy generation; no fine-tune, no
approximation.

**Compare against:**
- **DP/noise** (DP-Forward, SnD, InferDPT): utility tax — our differentiator.
- **Approximation crypto** (MPCFormer, THE-X, MERGE): polynomial-approx accuracy loss.
- **CONJFORMER**: +0.4% ppl **and requires fine-tune** (unavailable to us; we use exact public W).
- **STIP/PermLLM/CENTAUR**: also lossless → so utility does **not** distinguish us from them; it
  distinguishes us from DP/crypto/CONJFORMER. Say so.

**Metrics:** perplexity, greedy token-identical rate, sampled top-k overlap, logit MSE / KL,
downstream task acc (**IFEval / GSM8K / MT-Bench** — already in the AAAI runner).

**Our advantage:** exactness with *no fine-tune and no architecture change*, in open-weight.

### D4 — Performance / latency at 7B scale  ○ SUPPORTING
**Claim.** ms/token at 7B, near-plaintext; obfuscation is essentially free; anchor adds <1% prefill / ~2% decode.

**Compare against:**
- **vanilla plaintext** (ceiling).
- **Pure crypto** (CryptoGen impl §A8: ~13–21 min/token GPT-2; Comet/SIGMA/BumbleBee/PUMA): the
  10⁴–10⁵× gap — the "upper-bound tier."
- **Same-family** latency: STIP 31.7 ms/token, PermLLM ~3 s/token, GELO — direct comparison.
- **GPU-TEE** overhead (PipeLLM numbers).

**Metrics:** TTFT, TPOT, tok/s, prefill/decode split, KV memory, TEE round-trips (0), largest model
demonstrated (**7B** — most Group-A/crypto works stop at GPT-2 / 1B / 6B).

**Our advantage:** only scheme in the comparison that reaches **7B at ms/token with net token
privacy**; crypto can't reach 7B interactive latency, heavy-TEE pays the enclave/CC tax.

### D5 — No model change + open-weight validity + generation completeness  ○ SUPPORTING (qualitative)
**Claim.** No retrofit, no fine-tune, works on stock weights, full autoregressive loop incl. KV.

**Compare against:** CONJFORMER (equivariant retrofit + fine-tune), CENTAUR/crypto (approximations),
classification-only schemes (BOLT/Iron/TwinShield — can't generate), ObfuscaTune (evaluated on
QA/classification, not generation).

**Metrics (qualitative table):** retrofit? fine-tune? arch change? open-weight-valid? autoregressive
+ KV? scale.

**Our advantage:** deployment realism — drop-in on public Qwen-7B, exact, full generation.

---

## 3. Per-competitor "our advantage" (one honest line each)

| competitor | they have | our advantage | our honest concession |
|---|---|---|---|
| CONJFORMER | elegant O(d)-equivariance, attack suite | close the attention-logit leak they acknowledge but don't fix; no retrofit; 7B | their math framing is cleaner; they also target only data (like us) |
| STIP (NDSS'26) | lossless, 70B production, also hides weights | in open-weight (their crutch removed) permutation is broken (ICML'25); our anchor closes it | they additionally protect weights; they demo 70B |
| PermLLM (IJCAI'24) | 3s/token, also hides weights | broken by ICML'25; we are ms/token, exact | also protects weights |
| CENTAUR (ACL'25) | private softmax, also hides weights | broken by ICML'25; no MPC-softmax cost | also protects weights |
| GELO (2026) | TEE per-batch mixing, closest cousin | it still exposes the reduction channel (self-notes Gram leak); we add the k=1 relocation that actually closes it | concurrent preprint; per-batch fresh mixing is a nice idea to discuss |
| ObfuscaTune (AAAI-25 ws) | protects secret weights, published | 30–39× less enclave nonlinear compute; open-weight input privacy it structurally lacks | different threat model (secret model) |
| CryptoGen (2026) | dual-sided, no hardware trust | 10⁴–10⁵× faster, reaches 7B | strictly stronger confidentiality, no TEE needed |
| PipeLLM (ASPLOS'25) | whole-model GPU-TEE, published | minimal TCB vs whole model; keeps GPU offload | trusts less hardware only because it obfuscates instead |
| DP-Forward / SnD | formal DP | bit-identical utility | we make no formal DP guarantee |

---

## 4. Positioning paragraph (drop-in for intro / related work)

> Prior user-data protection for cloud LLM inference splits into (i) **obfuscation-only** schemes
> that permute or rotate hidden states and outsource them to an untrusted server (CONJFORMER, STIP,
> PermLLM, CENTAUR, GELO); (ii) **DP/noise** schemes that trade utility for privacy; and (iii)
> **cryptographic or whole-model-TEE** schemes that buy strong confidentiality at seconds-to-minutes
> per token or a whole-model trusted base. In the realistic **open-weight** setting, group (i) is
> not merely heuristic but *provably* broken: the reductions a correct forward pass must compute
> (softmax, RMSNorm) expose a context-free, mask-invariant per-token fingerprint (Thm. §2), which we
> confirm recovers 98.5% of tokens on Qwen-7B — independently corroborated by the ICML'25
> permutation break (2505.18332) that reverses STIP/PermLLM/CENTAUR at 97–100%. We show this leak is
> closed by a **theoretically-minimal hardware anchor** (relocating exactly the first attention
> block, k=1, into a TEE), which is *necessary and sufficient*, bit-identical, adds no round-trip,
> and leaves ~27/28 of attention and all MLPs on a commodity GPU — recovering open-weight token
> privacy at millisecond latency and 7B scale, where obfuscation-only leaks, DP loses utility, and
> crypto/whole-model-TEE pay orders of magnitude more.

---

## 5. Citation corrections / de-dupe (fix before submission)

- **CONJFORMER == "Privacy from Symmetry" (2606.16461)**; its model is literally named ConjFormer.
  One reference, not two.
- **CipherGPT** is an IACR ePrint (2023/1147), **not NeurIPS**, and is MPC (VOLE/OT), not HE+MPC.
- **AsymML**'s publication is **3LegRace, PoPETs 2022** — a **training**, data-privacy paper.
- **STIP** is **NDSS 2026** (not a loose preprint).
- **Bifrost** venue "ACM SIGOPS ATC 2026" is unverifiable; cite as arXiv 2606.17421.
- **EncFormer** "TDSC" unconfirmed; cite as arXiv 2604.09975.
- **Slalom** = ICLR 2019 (arXiv 1806.03287, 2018).
- **TEESlice** ≠ "No Privacy Left Outside" (S&P'24 original vs TOSEM'25 extension).
- Non-existent / do-not-cite: "MirageNet", "Magnitude", "Occlum-ML" (no real match found).
- **PrivacyRestore** was published at **ACL 2025** (not just 2024 arXiv); **Split-and-Denoise** is
  **ICML 2024**; **SAP** = "Split-and-Privatize".

---

## 6. Minimum experiment set to actually run (everything else = cite)

1. `vanilla` plaintext (utility + latency ceiling).
2. `plain split` no-obfuscation (weak lower bound).
3. `ObfuscaTune-like` (done, §A7) — D2 anchor-cost contrast.
4. `permutation-only` (STIP-style, lossless, cheap to reimplement) — D1 leak + D4 latency.
5. `CONJFORMER attack suite` on ours (done) + `attention-fingerprint` (done).
6. `A_rightmul raw` vs `A_rightmul + k=1` (done) — the headline D1 pair.
7. `DP baseline` (DP-Forward or SnD) — D3 utility contrast.
8. `CryptoGen` (done, §A8) — D4 upper-bound.
+ Attack battery (EDNN / Vec2Text / permutation-break 2505.18332 / KV-dump) on all obfuscation rows.

CITE-only (no run): Cachemir, PUMA, Comet, SHAFT, BumbleBee, SIGMA, MERGE, PipeLLM, Petridish,
Bifrost, PermLLM, CENTAUR, GELO, AloePri, Talaria, and the entire Group-C model-IP lineage.
