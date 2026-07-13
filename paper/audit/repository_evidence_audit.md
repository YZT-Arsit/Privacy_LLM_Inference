# Repository Evidence Audit

**Repo:** `/Users/Hoshino/Desktop/privacy_llm_obfuscation` (git; the AAAI author kit / official style files live separately at `/Users/Hoshino/Downloads/AuthorKit27`).
**Audit date:** 2026-07-13. **Method:** six parallel read-only domain audits (correctness, security/attacks, LoRA/optimizer, performance/real-machine, baselines, existing-drafts synthesis) cross-checked against Tier-1 artifacts (`src/`, `tests/`, `outputs/`, `results/`, `paper_results/`).
**Governing doc:** `.claude/skills/SKILL.md` (aaai-paper-writing). **NOTE:** no `AGENTS.md` exists in this repo; the SKILL.md is the operative instruction set.
**Git provenance caveat:** commit messages are almost all empty (`git log --oneline` shows blank subjects except a few), so per-claim "last verified commit" provenance is weak. Latest commits dated 2026-07-12/07-13. Record SHAs when the ledger is refreshed.

This document classifies evidence as **supported / proxy-only / projected / unsupported / stale / conflicting** and enumerates the non-mergeable evidence lineages and the material conflicts. It is the evidentiary basis for `../claim_ledger.md`.

---

## 0. TOP-LEVEL CONFLICTS (stop-conditions — do NOT write around these)

### CF-1 (CRITICAL): Public-base vs private-base threat model
- **Task objective** specifies **public** base weights.
- **Repo's newest drafts** (`paper_draft/threat_model_revision_patch.md`, 2026-07-10, "applied … Not committed"; `system_and_threat_model.md`; `introduction.md`; `limitations.md` item 13) have **repositioned to a PRIVATE-base model** ("only masked `W_tilde` crosses the boundary"). `results/aaai_private_base/` + `results/private_base_unified/` are a whole real-Qwen/TDX **private-base** experiment suite.
- **Older audits** (`reviewer_risk_audit.*`, `novelty_positioning_review.md`, recommendation NOV-02) still use **public-weight** wording and are flagged **superseded**.
- **Why it is decisive:** it changes the protected-asset set, the novelty argument, AND the entire security verdict (see CF-2). **Requires an author decision before Sections 4/5/7 can be authored.**

### CF-2 (CRITICAL): The security story inverts on CF-1
- Under **public-base**: structured OR dense masks provide **no token-level privacy** — 4 independent attacks recover **~98.5%** of tokens on **real Qwen2.5-7B** (`results/attacks/FINDINGS.md`), backed by an **impossibility argument** (`results/attacks/THREAT_MODEL_AND_THEORY.md` §2). Privacy is carried by the **layer-0 TEE relocation (k=1)** + gradient aggregation, NOT the algebra.
- Under **private-base**: masks "empirically resist" (S1 strict cos 0.0018, S6 masked-transfer top-1 0.003) — but ONLY because the attacker is denied the checkpoint (`results/aaai_private_base/security/security_registry.yaml` hard rule); rests on TEE/aggregation, not information destruction.
- The two regimes assume opposite attacker knowledge and **must not both be cited as "our security result."**

### CF-3: Optimizer locus prose vs artifact (two architectures conflated)
- `design.md §5.8` prose: "optimizer step (SGD or AdamW) performed entirely on the trusted side."
- Artifacts: `unified_training_config.py:__post_init__` **fail-closes** unless `optimizer_mode=='gpu_masked_sgd'` and **rejects AdamW**; the optimizer runs **GPU-masked**. Two distinct schemes (Scheme A "LoRA in trusted boundary, AdamW ok" vs Scheme B "GPU-masked, SGD-only") are conflated in prose. See ledger C-OPT-*.

### CF-4: "AdamW unsupported" (T15) vs "AdamW exact" (Gate-0.5 O1-C)
- T15: dense-mask AdamW **unsupported** (`DenseMaskedAdamWUnsupported`).
- Gate-0.5: **O1-C** (trusted-side correction) makes AdamW **exact** (fp32).
- Reconciliation (must be stated): T15 = GPU dense-mask AdamW *without* trusted assist; O1-C = *with* trusted correction. Both true, different settings.

### CF-5: KV-mask freshness
- Artifacts enforce **KV mask fixed within a session** (`kv_cache_mask_fixed_within_session=true`).
- `design.md §5.10` prose says "sample fresh masks per token" — would break the cache. Prose omits the fixed-KV restriction. See C-COR-05.

### CF-6: Four parallel claim ledgers, three status vocabularies
- Theory (T1–T15/C-PaperSafe), body (S/P/U = 8/5/8), audit-v2 (25 claims = 15/1/10), Gate-0/0.5 (real-Qwen/TDX). They disagree on counts and on AdamW. `../claim_ledger.md` unifies them under the skill's status vocabulary.

### CF-7: Baseline classification errors
- **Arrow** is recorded in RQ13 as an "unavailable defense primitive," but `papers/Arrow.pdf` is present and is the *Game of Arrows* **attack** paper (USENIX Security 2025). See C-BASE-ARROW.
- `BASELINES.md` Group C says Slalom/DarKnight/Amulet are "NOT competitors … do not benchmark," yet RQ13 benchmarks them as primitives (BAS-01, top reject risk).

---

## 1. NON-MERGEABLE EVIDENCE LINEAGES (performance/deployment)

Five distinct lineages. **Never merge; never let one borrow another's authority.**

| # | Lineage | Hardware | Model | Measured vs projected | Vintage |
|---|---|---|---|---|---|
| L1 | CPU local-emulation micro-bench | host CPU, fp64 | synthetic tiles / tiny-gpt2 | measured wall-time, tiny/illustrative | Jun 3 |
| L2 | Op-count cost-model **projection** | none (analytic) | tiny-gpt2 dims | **projected_from_op_counts** | Jun |
| L3 | H800 latency micro-bench | real H800 PCIe | **NOT real Qwen** (random-weight, Qwen-7B-shaped) | measured CUDA wall-time; **"CPU-TEE" = host CPU, not a TEE** | Jul 2 |
| L4 | Real A10 + real Intel TDX LoRA training | A10 (sm_86) + TDX guest | real Qwen (L10/L11 SGD/momentum) | measured per-step + **projected** totals | Jul 12 |
| L5 | Real H800 + real Qwen-0.5B + real TDX attestation | H800 + TDX | Gate2 synthetic; Gate3 real Qwen-0.5B | correctness + **real v5 quote**; latency partial (TDX side stubbed in GPU audit) | Jul 10–11 |

**Staleness:** `paper_results/csv/*` (Jun 3, L1/L2) **predate** all real-hardware runs (L3–L5, Jul). `gate_status.json` still `status="in_progress"` / D4 `blocked_needs_h800_tdx` while `GATE0_COMPLETE_REPORT.md` says complete — mildly inconsistent. `failed_run_registry.jsonl` is **0 bytes**. `results/aaai_private_base/o1_optimizer/{summary.md,*.csv}` are **0 bytes** (live O1 data is under `results/private_base_unified/o1_optimizer/` and `results/aaai_private_base/gate05_o1/`).

---

## 2. WHAT IS SUPPORTED (by class)

### 2.1 Functional / algebraic correctness — SUPPORTED (float-close, not exact; mostly CPU)
- Linear+pad (Thm 1), attention QK / RoPE / GQA-MQA / KV-append (Thm 6), norm islands (Thm 4/5), pointwise & SwiGLU permutation islands (Thm 2/3), LoRA fwd/bwd (Thm 7/8), rank padding (Thm 9): all **algebraically_proven + experimentally_validated**.
- Numerical envelope: **fp64 ~1e-14…1e-16; fp32 ~1e-6…1e-8; real Qwen-0.5B (CPU, fp32) recovered-logit max abs err 1.79e-7, greedy token-match 1.0.** Only a few cases are literally exact (`0.0`): permutation activation island, trusted-norm re-mask invariant, trusted RMSNorm recompute.
- **Scope caveats:** no real TEE, no CUDA, no bf16 in the correctness suite; "modern-decoder" model-level results use a **synthetic** `SyntheticLlamaModel` (fallback), not a real HF module.

### 2.2 LoRA / optimizer — SUPPORTED with a sharp boundary
- Masked forward/backward recovery: fp64 synthetic ~1e-14…1e-16 (linear only; nonlinear masked backward **not_implemented**).
- **SGD/momentum**: exact under **orthogonal** masks (algebraic + fp64 + real bf16 top-1 1.0, ΔW rel 3.6%).
- **GPU-masked Adam/AdamW/RMSProp/Adagrad**: **INCOMPATIBLE**, rejected fail-closed (elementwise nonlinearity doesn't commute with two-sided masks).
- **Trusted O1-C AdamW**: exact (fp32; **bf16 breaks, top-1 0.905** → needs fp32 master weights).
- **γ-fold refinement:** RMSNorm gain-folding makes q/k/v/gate/up masks **non-orthogonal** → naive masked SGD drifts 10–53% → requires **O1-C hybrid** (GPU-exact for o_proj/down_proj + trusted correction for the 5 γ-fed targets; the correction leaks the γ² multiset if placed on GPU).

### 2.3 Real TDX attestation — SUPPORTED but scoped
- **Genuine** real TD **Quote v5**, DCAP chain-verified, appraisal=1, report-data bound, fresh nonce, real quote binaries on disk (Gate 2 and Gate 3).
- **Caveats:** Gate 2 = **synthetic** protocol tensors (no GPU, no Qwen, no latency); Gate 3 = real Qwen-0.5B single-step but the **GPU audit run stubbed the TDX side**; **`mr_td` is NOT policy-pinned** (`allow_any_mr_td=true`) in both gates → quote verified, *measurement identity not enforced*.

### 2.4 Baselines — mixed, mostly primitive/adapted
- **CONJFORMER**: direct/faithful reproduction on **real Qwen2.5-7B, H800** (only baseline matching ours on HW+model+precision). Off-the-shelf (no fine-tune) = gibberish; inference ~1.02× plaintext.
- **STIP**: adapted reproduction (fp64 numeric, Qwen2, published formulas **corrected**); prefill-oriented, `decode_supported=False`.
- **ObfuscaTune**: adapted reproduction (**simulator** TEE, no real TEE; correctness on GPT-2/tiny; LoRA smoke-only).
- **Slalom / DarKnight / Amulet**: primitive-level CPU toy-tensor comparisons.
- **CryptoNets**: arithmetic-skeleton only (x² in plaintext, no HE).
- **Gazelle / Delphi / SecureML / MiniONN**: **cost-model-only, NOT executed** (`runtime=nan`, `runtime_directly_comparable=False`) — correctly fenced.

---

## 3. WHAT IS PROXY-ONLY

- All **security/leakage** results carry `formal_security_claim=false` / `security_profile="proxy-evaluated, not formal"`. The only non-proxy security-adjacent theorem is the **impossibility** result (which says masks *cannot* hide token identity).
- Attack "resistance" verdicts (S1–S6, unified matrix, timing 0.5124 under proxy-equalized) are **proxy_evaluated** under specific attacker configs.
- Rank-hiding *security* is **proxy_evaluated** (true-rank spectral inference `needs_more_evaluation`, NOT "low"); a "high" gradient-side rank leak at r=8 is likely a `seq_len=8` saturation artifact (reported uncaveated — flag).
- Timing side-channel numbers are `wall_time_source=projected_from_op_counts, implemented=False`.

---

## 4. WHAT IS PROJECTED (must NOT be read as measured wall-time)

- `workload_summary.csv` / `workload_profile.json` carry BOTH `measured_wall_time_ms` and `projected_wall_time_ms` columns; the projected column is a self-declared "simulated cost model, not real SGX." For `ours_current`, projected 37.38 ms is **5.5×** its own measured 6.81 ms.
- `system_comparison.md` §A "Nx slowdown" table: numerator/denominator = `latency[config]/latency[plaintext]` on the **same random-weight synthetic** Qwen-shaped model; **"CPU-TEE" rows use the host CPU as a TDX proxy** (no enclave); ObfuscaTune is a re-implementation. GPU-TEE 1.00–1.07×; host-CPU-proxy 2.4–3.0× (ours) / 82–114× (ObfuscaTune).
- **"~9× per token"** (`system_comparison.md` §C) has **no matching artifact number** — artifacts give 39.6 ms/token local + ~71 ms tunnel RTT, 3.66 tok/s prototype. **UNDERSPECIFIED — do not use without full numerator/denominator/HW/workload/boundary.**
- A10+TDX projected totals (`L12_total_hours=4.97`, `TOTAL_SST2_hours=10.33`) = per-step × steps × seeds; clearly marked `analytically_estimated`.

---

## 5. WHAT IS UNSUPPORTED (must NOT appear as positive claims)

U1 formal/cryptographic/semantic/DP security · U2 real-TEE/GPU wall-time · U3 hardware side-channel · U4 full production Qwen/LLaMA fine-tune (largest real = Qwen-0.5B feasibility loop, base weights **plaintext on GPU** there) · U5 PEFT/DeepSpeed/vLLM/FlashAttention · U6 padded rank hidden (r_pad **visible**) · U7 loss/optimizer fully outsourced to GPU · U8 compromised-TEE protection · plus: **sampling/top-k/top-p/beam correctness** (only greedy validated) · **m-RoPE/MoE/speculative decoding** · **fp16/bf16/int8/int4 real kernels** (simulated only).

---

## 6. GOOD-DISCIPLINE ITEMS TO PRESERVE

`is_real_tee_wall_time=false` flags; `wall_time_source` columns; algorithmic-vs-prototype-transport separation (`system_comparison.md §C`); explicit `uses_real_gpu/qwen/tee` booleans in TDX JSONs; `stage_7_6_claims_consistency` (149 files, **0 unsafe-wording present**, all 268 tracked phrases are disclaimers); honest "needs_more_evaluation" labels not re-graded to "low"; cost-model baselines fenced with `runtime_directly_comparable=False`.

---

## 7. KEY EVIDENCE PATHS (anchor set)

Correctness: `outputs/intermediate/{static_correctness,rope_gqa_probe,kv_cache_correctness,norm_experiments,nonlinear_island_experiments,generation_correctness,modern_decoder_generation_correctness}.json`, `outputs/_cpu_validation/eval_full_layer_0_5b_realprompts*.json`, `src/pllo/ops/*`, `tests/*`.
LoRA/opt: `outputs/lora/*`, `outputs/masked_gradient_lora_training.*`, `results/private_base_unified/o1_optimizer/*`, `results/aaai_private_base/gate05_o1/*`, `results/real_qwen_tdx_audit/qwen05b/sgd_mode/*`, `paper_results/csv/lora_training_summary.csv`.
Security: `results/attacks/{FINDINGS.md,THREAT_MODEL_AND_THEORY.md,ours_gram/,ours_pad/,*_qwen7b.json}`, `results/aaai_private_base/security/S{1..6}/*`, `outputs/paper_security/*`, `outputs/attacks/*`.
Perf/TEE: `results/aaai_private_base/system_comparison/system_comparison.md`, `results/baselines/obfuscatune_latency_h800.json`, `outputs/intermediate/workload_profile.json`, `paper_results/csv/{measured_runtime,workload_summary,ours_runtime_api_validation}.csv`, `results/real_qwen_tdx_audit/tdx_instance/gate2_real_tdx/*`, `results/real_qwen_tdx_audit/qwen05b/{claims.md,sgd_mode/*}`, `results/aaai_private_base/alicloud_a10_runs/utility_dataplane/PHASE_profiling_gate.json`, `results/aaai_private_base/gpu_execution_audit/audit_summary.json`.
Baselines: `papers/{STIP-NDSS,ObfuscaTune-AAAI,Arrow,Permutation,PIA,BRE,EDNN攻击}.pdf`, `outputs/baseline_audit/*`, `outputs/baselines/direct_prior_work_comparison.*`, `results/baselines/{conjformer/,BASELINES.md}`, `docs/baseline_audit_obfuscatune_stip.md`.
Drafts/audits: `docs/{PAPER_EVALUATION_MAP,PAPER_THEORY_OUTLINE}.md`, `outputs/stage_7_6_claims_consistency.*`, `outputs/paper/paper_claims_audit_v2.md`, `paper_draft/{claims_mapping,reviewer_risk_audit,unsafe_wording_review,*}.md`, `results/aaai_private_base/claim_experiment_map.md`.
