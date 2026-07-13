# Phase 0 — Code-Path Audit (Real Generative LoRA stage)

Audited 2026-07-13. Every path is categorized; categories are never mixed in the final table.

## Category legend
- **RG+RT** — real GPU (A10) + real TDX. Paper-facing protected cell.
- **RG+LD** — real GPU + local trusted diagnostic (trusted math on the A10, no separate TDX enclave).
- **CPU-SIM** — CPU simulation (no GPU, no TDX). Correctness only.
- **SYNTH** — synthetic-correctness-only (toy tensors).
- **PF-REAL** — paper-facing real result (must have script+raw output+manifest+source hash+hw profile).

## Live hardware (this session)
- A10 GPU: `39.107.123.173` pub / `172.30.25.154` priv, NVIDIA A10 23 GB, torch 2.8.0+cu128, CUDA 12.8, py3.10.12.
- Intel TDX: `39.96.43.122` pub / `172.30.25.153` priv, `/dev/tdx_guest` present, py3.10.20, torch 2.8.0+cpu.
- Data plane: A10↔TDX over **private VPC** 172.30.25.153 (~0.5 ms RTT). Mac = control-plane only.

## Existing scripts and their role in this stage

| Script | Role | Supports generation? | Category | Notes |
|---|---|---|---|---|
| `scripts/a10_batch_runner.py` | protected LoRA **training** loop (cls / clm CE) on A10; TDX ce_batch→dlogits; GPU-monomial + TDX-exact AdamW | **No** — eval is teacher-forced CE only | RG+RT (training) | Has Phase-1 attestation gate + `--require-attestation`. **A10 host copy is STALE** (`11f168fd`) vs local `63a7572`; MUST sync before protected run. |
| `scripts/tdx_persistent_service.py` | TDX enclave service: CE/dlogits, exact trusted AdamW (un-fold/step/re-fold), batch-HMAC ledger, attestation gate | n/a | RG+RT | Trusted side of the protocol. |
| `scripts/h800_unified_worker.py` | reference masked forward: `forward`/`forward_batch`→`logits_masked`; `MaskVerifier.unpermute_logits` | **Forward only** (no decode loop, no KV cache) | RG+LD / CPU-SIM | **Key:** vocab mask is a pure permutation ⇒ `masked_logits[:, perm] == plaintext_logits`. Basis for protected decode. |
| `scripts/native_profile/adapter_handoff.py` | transformed-adapter build + fail-closed generation loader (v1.0) | export/load only | CPU (build) | Reused for Phase 7 handoff; extend to bind E2E run. |
| `scripts/gate0_a10_batch_orchestrator.py`, `gate0_a10_l12_orchestrator.py` | Mac control-plane: provision keys/package/code, launch runner, collect | n/a | control-plane | Public control IPs must be updated (39.107.123.173 / 39.96.43.122); private data-plane IPs unchanged. |
| `scripts/lora_gen_gsm8k/run_gsm8k_lora_eval.py` | prior GSM8K LoRA gen eval (MPS) | yes (HF generate) | CPU-SIM (Mac MPS) | GSM8K@0.5B was task-floor; kept as parity/limitation only, NOT the main generative result. |

## Gaps this stage must fill (to be built, then run on RG+RT)
1. **Protected autoregressive greedy decode** for the private-base 0.5B package + trained adapter.
   Approach: KV-cached decoder forward on A10 → per-step **last-position masked logits** → TDX
   `unpermute_logits`+argmax → true next-token id returned. Reuses the existing permutation identity;
   one trusted argmax-boundary call per generated token (no plaintext logits or labels on the GPU beyond
   the masked vector). To be implemented as a new generation worker + TDX op.
2. **Plaintext-coordinate LoRA reference (G1)** trainer+greedy-generator on the A10 GPU in plain torch,
   loading the plaintext `/root/qwen25_05b` checkpoint (real GPU baseline, NOT protected).
3. **E2E NLG deterministic data pipeline** (Phase 2) — shared by G0/G1/G2.
4. **BLEU / ROUGE-L / chrF** offline metrics on the Mac over pulled generation jsonl.

## Category discipline
- G0 base / G1 plaintext LoRA: **real GPU (A10), plaintext checkpoint** — NOT protected, NOT TDX. Utility reference.
- G2 exact protected LoRA: **RG+RT, PF-REAL** — the protected paper-facing cell (training on A10+TDX,
  protected generation via TDX unmask-argmax boundary, real attestation evidence).
- Text metrics: offline on Mac (no compute path). Never labeled as a GPU/TDX result.
