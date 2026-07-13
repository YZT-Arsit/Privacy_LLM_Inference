# Claim Ledger

Single source of truth for every substantive paper claim. **A claim may appear in the abstract, contributions, or conclusion ONLY if it has an entry here** with status ≥ `algebraically_proven` / `experimentally_validated` / `measured_real_system` and calibrated wording matching the body.

**Status vocabulary** (skill S4): `definition`, `algebraically_proven`, `formally_proven_under_model`, `experimentally_validated`, `measured_real_system`, `proxy_evaluated`, `analytically_estimated`, `cost_proxy_only`, `observed_limitation`, `unsupported`, `future_work`. Distinguish: algebraic correctness ≠ numerical equivalence ≠ empirical attack resistance ≠ leakage reduction ≠ formal security ≠ cryptographic security ≠ hardware isolation.

**Provenance:** evidence paths are relative to repo root `/Users/Hoshino/Desktop/privacy_llm_obfuscation`. "Last verified commit" = HEAD `86da258` (2026-07-13) unless noted; **commit subjects are empty in git history, so provenance is coarse** — re-verify on refresh. Legacy-ledger cross-refs: T# = `PAPER_THEORY_OUTLINE.md`; S#/P#/U# = `claims_mapping.md`; Gate# = `claim_experiment_map.md`.

**Reconciles four legacy ledgers** (see `audit/repository_evidence_audit.md` CF-6). Numerical convention: "fp64 ~1e-14" means float-close at that magnitude, **NOT** symbolic exactness.

---

## A. THREAT MODEL & TRUSTED BOUNDARY (C-THREAT / C-BND)

| ID | Exact claim (calibrated) | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-THREAT-01 | The untrusted accelerator observes only transformed tensors + public metadata (shapes, seq-len, per-call timing proxy, output tokens). | definition | trusted component holds secrets/masks/pads/recovery/sampling | `system_and_threat_model.md`; `src/pllo/tee/*` | trusted side is CPU **emulation**, not a real TEE |
| **C-THREAT-02** | **Base-weight visibility is UNRESOLVED**: task says public; repo's newest drafts say private (`W_tilde` only). | **observed_limitation / CONFLICT CF-1** | — | `threat_model_revision_patch.md` (2026-07-10) vs task | **decides Sec 4/5/7; author decision required** |
| C-THREAT-03 | Allowed leakage (visible by construction): tensor/batch shapes, **sequence length**, output tokens+length, per-call timing, permutation-invariant statistics, norms, Gram, padded rank r_pad. | definition | structured/orthogonal masks | `security_registry.yaml` documented_leaks; §7.2 | do NOT claim any of these hidden |
| C-BND-01 | The trusted component performs mask/pad generation, selected nonlinear/boundary ops, recovery, sampling, auditing; accelerator does heavy linear algebra on masked tensors. | definition (measured on emulated boundary) | — | `src/pllo/tee/{simulated,process}_runtime.py`; `trusted_boundary_accounting.py` | boundary "cost" = op-count proxy, not latency |
| C-BND-02 | Integrity is a probabilistic spot-check only; compromised-TEE and availability out of scope. | definition / unsupported(beyond) | — | audit-v2 `integrity_only_probabilistic_spot_check` | U8 |

## B. FUNCTIONAL / ALGEBRAIC CORRECTNESS (C-COR)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-COR-01 | Linear + boundary additive pad recovers the plaintext linear map: `((X−T)N_in)(N_in⁻¹WN_out)+TWN_out+bN_out = (XW+b)N_out`. | algebraically_proven + experimentally_validated + measured_real_system | `N_in,N_out` invertible; pad compensated before any nonlinear op | `outputs/intermediate/static_correctness.json:metrics.max_abs_error=1.07e-14` (fp64); fp32 6.68e-6; real Qwen-0.5B `outputs/_cpu_validation/eval_full_layer_0_5b_realprompts*.json:recovered_logits_max_abs_error=1.79e-7, token_match=1.0` (T1/S1) | float-close; CPU only; no bf16 |
| C-COR-02 | Attention scores preserved: `(QN_Q)(KN_K)ᵀ = QKᵀ` iff `N_Q N_Kᵀ=I` per head (block-diagonal, no cross-head mixing); V-mask absorbed by o_proj. | algebraically_proven + experimentally_validated | per-head invertible masks tied `N_K=(N_Q⁻¹)ᵀ` | `outputs/intermediate/rope_gqa_probe.json:mha.score_max_abs_error=4.26e-14`; qk_constraint 6.66e-16 (T3) | synthetic tensor probe |
| C-COR-03 | RoPE commutes with masking ONLY for 2×2-block masks on adjacent RoPE pairs (pairwise rotation / complex-scaling); dense masks do NOT commute. Wired path uses **post-RoPE masking**. | algebraically_proven + experimentally_validated; dense-fail = observed_limitation | head_dim even; adjacent-pair convention | `rope_gqa_probe.json:rope_commutation_max_error=6.66e-16`; `modern_decoder_generation_correctness.json:rope_mode="post_rope_masking"` (T6) | RoPE-compatible family weaker than dense; no security guarantee |
| C-COR-04 | GQA/MQA: shared KV masked once per group; `q_mask[h]=key_masks[h//g]⁻ᵀ` preserves scores across the group. | algebraically_proven + experimentally_validated | group-consistent tying | `rope_gqa_probe.json` gqa `score_max_abs_error=3.55e-14`; `modern_decoder...json:gqa_indexing.mask_is_per_head_not_hidden_size=true` (T7) | synthetic; `repeat_kv` on tensors not real HF module |
| C-COR-05 | KV-cache append distributes over token axis: `[K₁N_K;…;K_tN_K]=[K₁;…;K_t]N_K`; masked-cache decode == full recompute. | algebraically_proven + experimentally_validated | **KV mask FIXED within a session** | `kv_cache_correctness.json:max_key_error=4.32e-6` (fp32); modern-decoder fp64 8.88e-16; `kv_cache_mask_fixed_within_session=true` (T4) | **CF-5**: design.md "fresh per token" contradicts fixed-KV restriction |
| C-COR-06 | Norm islands: RMSNorm commutes with orthogonal `N` (Thm 4); LayerNorm with mean-preserving orthogonal `N` (Thm 5). | algebraically_proven + experimentally_validated | orthogonal `N`; **gain must be folded** | `norm_experiments.json` (fp32 1.5e-5); A7 audit fp64 <1e-8 (T5) | **vector-γ RMSNorm does NOT commute (err 9.68)** unless γ folded; **wired path uses TRUSTED recompute** (`rmsnorm_mode="trusted_fallback_with_repad"`), not a GPU island |
| C-COR-06b | Exact input-independent RMSNorm-covariant obfuscation forces `TTᵀ=cI` ⇒ per-token norm leakage unavoidable. | analytically_estimated / observed_limitation | scoped (linear, input-independent) | `docs/rmsnorm_exact_norm_impossibility.md` | NOT a universal impossibility |
| C-COR-07 | Pointwise activation island `φ(ZP)=φ(Z)P`; SwiGLU needs a **paired/shared** permutation. | algebraically_proven + experimentally_validated (some EXACT) | permutation `P`; shared across SwiGLU gate/up | `nonlinear_island_experiments.json:activation_island_cells[0].max_abs_error=0.0`; SwiGLU fp64 1.78e-15; mismatched-perm NEG control MUST fail (T2/T8) | — |
| C-COR-08 | Nonlinear-execution modes: `current`=nonlinear runs **trusted-side**; `trusted_shortcut`=runs **on accelerator** (Amulet lift; name misleading); `compatible_right_multiply`=accelerator, 0 crossings. | definition + experimentally_validated | — | `deployment/folded_nonlinear.py`; `outputs/amulet_right_mask_nonlinear_experiments.json:max_abs_error_overall=2.02e-11, audit.nonlinear_executed_on_gpu=true` | **do NOT present `current` correctness as accelerator-side nonlinear execution**; no formal security for `trusted_shortcut` |
| C-COR-GEN | Masked execution reproduces the plaintext token sequence under **GREEDY** decoding. | experimentally_validated | argmax decoding | `generation_correctness.json:sequence_exact_match=1.0, stage3_scope.sampling=false`; real Qwen-0.5B greedy 1.0 (S1) | **sampling/top-k/top-p/beam NOT validated — unsupported** |
| C-COR-CONV | Row-vector convention; object types (invertible/orthogonal/permutation/paired-perm/additive pad) are distinct. | definition | — | `notation.md` | COR-01 reviewer risk: LoRA-backward proof terse |

## C. LoRA ADAPTATION (C-LORA)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-LORA-FWD | Masked LoRA forward recovers plaintext output: `Ỹ+(α/r)TABN_out=YN_out` with `Ã=N_in⁻¹AU, B̃=U⁻¹BN_out`. | algebraically_proven + experimentally_validated | invertible masks; adapter never merged | `outputs/lora/lora_training_experiments.json:forward_max_abs_err=1.62e-14`; `masked_gradient_lora_training.json` fwd ≤1.03e-14 (T11) | fp64 synthetic; `grad_a/b` not GPU-visible |
| C-LORA-BWD | Masked backward recovers `dA,dB,dX` via `dA=N_in⁻ᵀdÃUᵀ` etc.; `G̃=GN_out⁻ᵀ`; loss+optimizer trusted-side. | algebraically_proven + experimentally_validated | linear layer | `outputs/lora/lora_backward_experiments.json:max_grad_a_err=8.33e-17, max_grad_b_err=9.99e-16` (fp64) (T12) | **nonlinear masked backward NOT implemented**; single-step; synthetic |
| C-LORA-BWD-LEAK | Naive dual-mask backward leaks cross-Gram `X̃G̃ᵀ=XGᵀ` (corr 1.0); independent-mask breaks exact equality but residual structure remains. | algebraically_proven (leak) / observed_limitation | — | `results/attacks/masked_lora_backward_audit/cross_gram_audit.csv` A0 corr 1.0 (T13-leakage) | not a zero-leak proof |
| C-LORA-RANK | Cancellation rank-padding: `A_pad B_pad = A B` (dummy contribution norm ~0); **true rank removed from shape, padded rank r_pad VISIBLE**. | algebraically_proven + experimentally_validated (correctness); rank-hiding = proxy_evaluated | exact-cancellation dummy strategies | `outputs/lora/lora_rank_padding_experiments.json:max_dummy_contribution_norm=0.0`; `masked_gradient_lora_training.json:padded_rank_visible=true` (T14) | **U6**: r_pad leaks; strategy count inconsistent ("four"/six/**seven**) — fix to 7; some grad-proxy modules leak r from shape |
| C-LORA-RANK-INF | True-rank inference from adapter SVD ≈ chance (acc 0.0, r∈{2,4,8}); labeled **needs_more_evaluation** not "low". Gradient-side "high" at r=8 likely a `seq_len=8` artifact. | proxy_evaluated | — | `lora_rank_security_proxy.json:spectral_rank_inference.rank_inference_accuracy=0.0`; `security_proxy_summary.csv` 7.2 high | rank-space `U` is a gauge (rank-count preserving), does not hide rank spectrally |

## D. OPTIMIZER TREATMENT (C-OPT)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-OPT-SGD | GPU-masked **SGD/momentum** is exact under **orthogonal** masks (linear update commutes with two-sided mask). | algebraically_proven + experimentally_validated + measured_real_system | orthogonal masks | fp64 `masked_sgd_audit` <1e-9; `o1_optimizer/trajectory.csv` 1.78e-15; real Qwen-0.5B bf16 ΔW rel 3.6%, top-1 1.0 (T13/Gate) | bf16 close not exact; **not globally exact under γ-fold** (see C-OPT-O1C) |
| C-OPT-ADAMW-GPU | GPU dense-masked **Adam/AdamW/RMSProp/Adagrad** are **INCOMPATIBLE** (elementwise nonlinearity doesn't commute); rejected fail-closed. | algebraically_proven (impossibility) + observed_limitation | dense masks | `real_tdx_training_service.py:_REJECTED_SGD_OPTIMIZERS`; `masked_gradient_lora_training.json:adamw_dense_mask_unsupported="explicitly_raised_as_designed"` (T15) | **CF-4** reconcile with O1-C |
| C-OPT-O1C | **Trusted-side O1-C** correction makes AdamW/momentum exact (incl. buffer); γ-fed targets (q/k/v/gate/up) need trusted correction; o_proj/down_proj GPU-exact. | formally_proven_under_model + measured_real_system | fp32 master weights; trusted correction | `gate05_o1/per_target_results.csv`; A10 fp32 KL 5.79e-9, top-1 1.0 | **bf16 AdamW breaks (top-1 0.905)**; O1-B exact but leaks γ² spectrum (paper-unsafe); O1-D infeasible; **CF-3** prose vs artifact |

## E. LEAKAGE & EMPIRICAL ATTACKS (C-SEC)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-SEC-LEAK | Structured/orthogonal masks preserve (leak) exactly: per-token norm, pairwise/cross-Gram, coordinate value-multiset, singular spectrum/rank, attention logits, confidence multiset (perm-only). | experimentally_validated (leakage) | — | S1 `norm_preservation_max_abs_gap`≈0; `permutation_invariant_leakage.md` sorted_mse=0.0; `masked_lora_backward_audit` attention_scores_plain_visible=true (T9/T8a) | these are NOT hidden |
| C-SEC-PIINV | Permutation-invariant statistics (row norms/sorted/quantiles) invariant under column permutation. | algebraically_proven + experimentally_validated | — | `permutation_invariant_leakage.json` (T9) | coordinate hiding ≠ value hiding |
| **C-SEC-PUBLIC-BREAK** | **Under public-base weights, masks provide NO token-level privacy: ~98.5% token recovery on real Qwen2.5-7B via 4 independent attacks (Gram-inversion, pad-peel, attention-fingerprint, CONJFORMER-suite).** | experimentally_validated (real GPU) | attacker holds public weights (CF-1) | `results/attacks/FINDINGS.md`; `ours_gram/qwen7b_gram_attack.json` perm 1.0→token 98.5%; `real_attention_fingerprint_qwen7b.json` top-1 98.5% | **CF-2**: this is evidence AGAINST the masks |
| C-SEC-IMPOSS | Structure-preserving zero-crossing obfuscation cannot hide token identity (impossibility). | analytically_estimated | public weights, structure-preserving | `results/attacks/THREAT_MODEL_AND_THEORY.md` §2 | scoped |
| C-SEC-TEE-K1 | The **layer-0 TEE relocation (k=1)** reduces user-content token recovery to ≈0.08% (≈random); k=1 necessary & sufficient. | experimentally_validated (real 7B) | layer-0 in TEE | `task2_layer0_guardrail_result.json`; task2c user-content 0.0% | privacy carried by TEE, not algebra |
| C-SEC-S1..S6 | Under private-base (secret oracle), masks empirically resist: S1 strict cos 0.0018; S2 plaintext-dW rel 1.41 (uninformative); S4 DLG basis-invariant (defense=aggregation); S5 MIA AUC 0.61; S6 KV top-1 0.54 pt / 0.536 masked-adapted. | proxy_evaluated | attacker denied checkpoint | `results/aaai_private_base/security/S{1..6}/*` | **CF-1/CF-2**: rests on TEE/aggregation; known-pairs → S1 cos 0.929 |
| C-SEC-UNIFIED | Toy/CPU matrix: fresh-pad KPA 0.68→0.0; nn-inversion 1.0→0.0; multiset-leak 1.0/1.0/0/0; **frequency 1.0 across ALL incl. ours**. | experimentally_validated (synthetic) | toy tensors | `outputs/paper_security/security_attack_table_measured.csv` | synthetic; EIA/BRE/PIA simplified |
| C-SEC-BLOCKED | Setup-blocked attacks (bre_backward, arrowmatch worst-case, nonlinear masked backward not_implemented, synthetic-fallback) are **NOT defeated attacks**. | observed_limitation | — | `outputs/attacks/*` blocked flags | do NOT report as security (skill S7) |
| C-SEC-TIMING | Decode-step timing classifier 0.749→0.254 under proxy-equalized. | proxy_evaluated / cost_proxy_only | op-count proxy | `stronger_attackers.md` (P2) | `wall_time_source=projected`, no real wall-time; SEC-04 "low"≠"solved" |

## F. EFFICIENCY / COST (C-EFF)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-EFF-CPU | Local-emulation micro-bench: masked-LoRA fwd 0.26 ms, bwd 0.12 ms, multilayer step 4.61 ms (CPU, fp64, tiny). | measured_real_system (local emulation) | — | `paper_results/csv/measured_runtime.csv:mean_ms`; `is_real_tee_wall_time=false` | **NOT TEE/GPU wall-time**; tiny tiles; 5 reps |
| C-EFF-PROJ | Op-count projection column (workload_profile) is a cost model, not latency (`ours_current` projected 37.4 ms vs measured 6.8 ms). | cost_proxy_only | — | `outputs/intermediate/workload_profile.json:projected_wall_time_ms` | **never cite as latency**; L2 lineage |
| C-EFF-BND | Boundary-call reduction 36→16→4 (op-count accounting). | cost_proxy_only | — | `ours_runtime_api_validation.csv:boundary_calls`; `trusted_boundary_accounting.py` | op counts, not timing |
| C-EFF-H800 | H800 slowdown vs plaintext (same random-weight Qwen-shaped model): GPU-TEE 1.00–1.07×; host-CPU-proxy 2.4–3.0× (ours) / 82–114× (ObfuscaTune). | measured_real_system (GPU) / analytically_estimated (CPU-TEE rows) | denom=plaintext same model | `results/baselines/obfuscatune_latency_h800.json:prefill.512.*.slowdown_vs_plaintext` | **not real Qwen; "CPU-TEE"=host CPU not enclave; ObfuscaTune=re-impl; reps not recorded** |
| C-EFF-9X | "~9× per token" | **unsupported (underspecified)** | — | no matching artifact; artifacts: 39.6 ms local +71 ms tunnel, 3.66 tok/s prototype | **do NOT use without full num/denom/HW/workload/boundary** |
| C-EFF-A10 | A10+TDX LoRA training: measured 0.459 s/batch (L12 bf16), 2999 MB peak, 1.14 MB/example; **projected** 10.33 h SST2 total. | measured_real_system (per-step) + analytically_estimated (total) | batch16 bf16 | `alicloud_a10_runs/utility_dataplane/PHASE_profiling_gate.json` | single run (3-seed matrix deferred) |

## G. REAL-SYSTEM / ATTESTATION (C-REAL)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-REAL-QWEN | Real Qwen2.5-0.5B single-step masked SGD aligns plaintext: loss abs diff 2.2e-5, top-1 1.0, ΔW cos 0.9994 (bf16); 50-step loss strictly decreasing. | measured_real_system | real H800+TDX; **base weights plaintext on GPU** | `results/real_qwen_tdx_audit/qwen05b/sgd_mode/single_step_exactness.json` | bf16 float-close (ΔW rel 3.6%); **private-base NOT claimed here**; gradA_cos degenerate (B=0 init) |
| C-REAL-ATTEST | Real Intel TDX attestation: TD Quote v5, DCAP chain-verified, appraisal=1, report-data bound, fresh nonce. | measured_real_system | — | `gate2_real_tdx/verification.json`; Gate3 `sgd_mode/tdx_attestation.json` | **mr_td NOT policy-pinned (allow_any_mr_td)**; Gate2 synthetic tensors; Gate3 GPU-audit stubbed TDX side |

## H. BASELINES (C-BASE)

| ID | Exact claim | Status | Assumptions | Evidence path / key | Limitation |
|---|---|---|---|---|---|
| C-BASE-CONJ | CONJFORMER (direct reproduction, real Qwen2.5-7B, H800): equivariance verified (logit top-1 1.0); off-the-shelf no-fine-tune = gibberish; inference ≈1.02× plaintext. | experimentally_validated (direct reproduction) | same HW/model/precision as ours | `results/baselines/conjformer/SUMMARY.md`; `qwen7b_all_bf16.json` | only baseline matching ours on HW+model |
| C-BASE-STIP | STIP (adapted reproduction, fp64, Qwen2): recovery max abs err 3.67e-8; published formulas **corrected** (RoPE/GQA/SwiGLU-π₃/bias). | experimentally_validated (adapted) | Qwen2; corrections documented | `outputs/baseline_audit/stip_audit.json` | prefill-only (`decode_supported=False`); no latency head-to-head |
| C-BASE-OBF | ObfuscaTune (adapted reproduction, **simulated TEE**): correctness fp64 logits err 0.0 (GPT-2/Qwen2); no real TEE; LoRA smoke-only. | experimentally_validated (adapted, simulator) | simulated TEE bookkeeping | `docs/obfuscatune_baseline.md`; `outputs/baseline_audit/obfuscatune_audit.md` | `full_system_reproduced=False`; H800 cost = model, not running system |
| C-BASE-PRIM | Slalom/DarKnight/Amulet/CryptoNets = primitive-level CPU toy comparisons; Gazelle/Delphi/SecureML/MiniONN = **cost-model-only, not executed**. | analytically_estimated / cost_proxy_only | — | `outputs/baselines/direct_prior_work_comparison.csv` | **BAS-01**: ours=full-system vs theirs=primitive (top reject risk); label as primitive comparison |
| C-BASE-ARROW | Arrow is an **attack** paper (Game of Arrows, USENIX Security 2025), used as ArrowMatch — NOT an "unavailable defense primitive". | observed_limitation (misclassification) | — | `papers/Arrow.pdf`; RQ13 row `arrow_direct_primitive_or_unavailable` | **CF-7**: fix RQ13 misclassification |

## I. NON-CLAIMS (C-NONCLAIM) — must be stated explicitly, never contradicted

| ID | Non-claim | Evidence |
|---|---|---|
| C-NONCLAIM-01 | No formal / cryptographic / semantic / DP / indistinguishability security. | `formal_security_claim=false` everywhere; `stage_7_6_claims_consistency` 0 unsafe |
| C-NONCLAIM-02 | No real-TEE / real-GPU wall-clock latency claim (costs are CPU emulation + op-count projection; "CPU-TEE" rows use host CPU). | L1/L2/L3 lineages |
| C-NONCLAIM-03 | No hardware side-channel / compromised-TEE protection. | U3/U8 |
| C-NONCLAIM-04 | No full production Qwen/LLaMA fine-tune; no PEFT/DeepSpeed/vLLM/FlashAttention; no fp16/bf16/int8/int4 real kernels; no MoE/m-RoPE/speculative decoding. | U4/U5; audit-v2 unsupported set |
| C-NONCLAIM-05 | Correctness NOT generalized beyond greedy decoding. | C-COR-GEN |

---

## HIGHEST-RISK CLAIMS (rank-ordered — do not enter abstract/contributions without resolution)

1. **C-THREAT-02 / CF-1** — public vs private base weights unresolved; blocks Sec 4/5/7.
2. **C-SEC-PUBLIC-BREAK / C-SEC-IMPOSS (CF-2)** — under public weights the masks are a full break; the security narrative must center the TEE relocation (C-SEC-TEE-K1), not the masks.
3. **C-EFF-H800 / C-EFF-9X** — "CPU-TEE"=host CPU, random-weight model, ObfuscaTune re-impl, "9×" unsupported; every ratio needs num/denom/HW/workload/boundary.
4. **C-OPT-ADAMW-GPU vs C-OPT-O1C (CF-3/CF-4)** — AdamW "unsupported" vs "exact" must be reconciled by architecture.
5. **C-COR-06 / C-COR-08** — RMSNorm wired as trusted recompute (not GPU island); `current` nonlinear runs trusted-side — do not sell as accelerator-side.
6. **C-COR-GEN** — greedy-only; do not generalize to sampling.
7. **C-LORA-RANK** — r_pad visible; strategy count wrong; rank-hiding proxy-only.
8. **C-REAL-QWEN / C-REAL-ATTEST** — real run keeps base weights plaintext (private-base not claimed); mr_td not pinned.

## STATUS COUNTS (this ledger)
- Entries: 34. By dominant status — algebraically_proven/experimentally_validated (correctness+LoRA+SGD): ~14; measured_real_system: ~5 (C-COR-01 real, C-OPT-*, C-EFF-A10, C-REAL-*); proxy_evaluated: ~7 (security); cost_proxy_only/analytically_estimated: ~5 (efficiency); observed_limitation/unsupported/conflict: ~8 (incl. C-THREAT-02, C-EFF-9X, C-BASE-ARROW).
- **No entry is `formally_proven_under_model` for SECURITY** — the only formal-security-adjacent result is an *impossibility* (C-SEC-IMPOSS).
