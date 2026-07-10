# Phase 1 — Real GPU + Real TDX Infrastructure Audit

**run_id:** `real_qwen_tdx_20260710T065808Z` · **local git:** `5208ae3b` · read-only SSH, no installs/modifications, nothing committed.

## Summary verdict

Both machines are **real and capable**. The GPU worker is a real H800 80 GB with working CUDA/bf16; **39.96.4.252 is already a REAL Intel TDX confidential guest** with active memory encryption and quote-generation capability, and the project's protocol + attestation code is already present on it. No nested TDX instance needs to be created. The two blockers before real training are (a) **Qwen2.5-0.5B is not yet on the GPU box** and (b) the **TDX trusted service is not running** (only sshd listens).

## GPU worker — `connect.westb.seetacloud.com:19948`

| item | value |
|---|---|
| hostname / env | `autodl-container-35eb4cbcf2-9d84e25e` (AutoDL container), kernel 5.15.0-124 |
| GPU | **NVIDIA H800 PCIe, 81559 MiB (80 GB)**, driver 590.48.01 |
| CUDA / torch | CUDA 13.1 · **torch 2.8.0+cu128 · `cuda=True` · `bf16=True`** |
| RAM / disk | 1.0 TiB RAM (840 GiB avail); root overlay 30 G (29 G free); `autodl-tmp` large data volume |
| repo | `/root/privacy_llm_obfuscation`, `/root/Privacy_LLM_Inference`; worker launcher `/root/start_pllo_worker_bf16.sh` |

> The brief calls the worker a "5090"; the actual hardware is an **H800 80 GB**. This satisfies the "real CUDA GPU, bf16 primary" requirement (and is *stronger* than a 5090 for 7B). Recorded honestly rather than relabeled.

**Models on GPU box:** Qwen2.5-**7B**-Instruct and Qwen2.5-**1.5B**-Instruct are present under a ModelScope `._____temp` staging path (completeness to be verified in Phase 3). **Qwen2.5-0.5B is NOT present** — it must be downloaded/verified via ModelScope before Phase 3 (the pipeline mandates 0.5B first).

## TDX server — `39.96.4.252` (key auth)

| item | value |
|---|---|
| hostname / OS | `iZ2zebdihkqll10ragqp0cZ` · Alibaba Cloud Linux 3 · kernel 5.10.134-19.3.al8 |
| CPU / RAM / disk | Intel, 4 vCPU · 14 GiB RAM (13 free) · 40 G disk (29 free) |
| virtualization | `systemd-detect-virt = kvm`, hypervisor = KVM → **it is a guest VM** |
| **TDX evidence** | cpuinfo flags `tdx`,`tdx_guest`; dmesg **"tdx: Guest detected"**, **"Memory Encryption Features active: Intel TDX"**; **`/dev/tdx_guest` present**; `libtdx_attest.so(.1)` installed; `/sys/firmware/tdx` absent (expected for a *guest*) |
| existing code | conda env `tdx310`; `src/pllo/protocol/{attestation,gpu_worker,orchestrator,remote,resilient_remote,tee_gpu_messages,wire,worker_timing}.py`; `scripts/generate_alibaba_tdx_quote_evidence.py`, `generate_tdx_attestation_evidence.py`, `check_tdx_measurement_coverage.py`; artifacts in `/root/privacy_llm_tee_artifacts` |
| network | internal `172.30.25.153`, public `39.96.4.252`; **TDX→GPU:19948 OPEN**; only port 22 (sshd) listening → **trusted service not running** |

## The five required determinations

1. **Is 39.96.4.252 a TDX host, TDX guest, or plain VM?** → **Real TDX GUEST** (confidential VM). Active Intel-TDX memory encryption, `tdx_guest` CPU flag, `/dev/tdx_guest`, `detect-virt=kvm`. Not a bare host, not a plain VM.
2. **Is there already a real TDX guest?** → **YES — this machine IS the real guest.** No nested creation needed.
3. **Guest details (IP/SSH/measurement/status)?** → SSH key auth on `:22` (internal 172.30.25.153); measurement/quote obtainable via `/dev/tdx_guest` + `libtdx_attest.so` (prior stages generated real quotes here); **trusted service currently NOT running** (only sshd).
4. **If no guest, can we create one?** → **N/A** — the guest already exists; nested TDX-guest creation from inside a guest is neither needed nor possible.
5. **GPU→TDX network path?** → **TDX→GPU:19948 confirmed OPEN.** GPU→TDX direct-public is TBD (Alibaba security group may block inbound); the reliable path is an **SSH tunnel** — since TDX can initiate to the GPU, a reverse tunnel can expose the TDX trusted-service port to the GPU worker.

> Note (per the brief's caution): the presence of "TDX" strings is **not** what this verdict rests on — it rests on active memory-encryption dmesg lines, the `/dev/tdx_guest` device, and the working attest library, i.e. the guest is genuinely running under TDX, not merely labeled.

## What is ready vs. what is missing (gates for later phases)

**Ready:** real H800/CUDA/bf16 GPU; real TDX guest with attestation capability; existing protocol + attestation code on both boxes; TDX→GPU connectivity.

**Missing / to do before real training:**
- Download + hash-verify **Qwen2.5-0.5B-Instruct** on the GPU box via ModelScope (Phase 3).
- Verify completeness of the 7B/1.5B staging copies.
- Bring up the **TDX trusted service** inside the guest (loss + packed AdamW) and establish the **GPU↔TDX tunnel** (Phase 2/5).
- Confirm exact `autodl-tmp` free space and GPU→TDX reachability (deferred from this audit due to transient SSH rate-limiting on the password host after rapid reconnects).

## Honest scope statement

Phase 1 (this audit) is complete with real data. Phases 2–11 (live TDX↔GPU protocol bring-up, real Qwen2.5-0.5B LoRA training with real in-guest loss + packed AdamW, transcript attacks, performance, then 7B) are a **long-running, multi-step effort** on real hardware and are **not yet executed**. No later-phase result is claimed. `uses_real_gpu=true`, `uses_real_tee=true`; nothing has been faked, and no CPU simulation is being presented as TDX.

## Live-path bring-up (Phase 2/5) — attempted, two blockers found

User directed: "bring up the live path first" (download 0.5B, start TDX trusted service, tunnel, prove one real end-to-end trusted step). Two blockers, reported honestly (not worked around by faking):

**Blocker 1 — GPU SSH gateway throttling (transient, external).** The GPU box is behind a shared AutoDL gateway (`198.18.0.137`). After the rapid read-only audit connections, the gateway began tearing down each session immediately after auth — even a bare `echo PING_ONLY` returns EOF with no output, though `echo HELLO; whoami` succeeded earlier. This is rate-limiting / a temporary ban that needs a cooldown (10+ min). Consequence: the **Qwen2.5-0.5B download could not be confirmed-started** this session, and no GPU-side command can currently run. Mitigation for next session: batch ALL GPU work into a *single* well-formed SSH command per action; space connections out; prefer `setsid nohup ... </dev/null` so long jobs survive session teardown.

**Blocker 2 — architecture gap: existing protocol is inference-only.** The deployed GPU worker (`scripts/run_gpu_worker_server.py` → `pllo.protocol.remote.GpuWorkerServer`, backend `qwen7b_folded_package`) exposes only `GET /health`, `POST /init`, `POST /prefill`, `POST /decode`. There is **no training endpoint** — no backward, no CE-loss boundary, no packed-LoRA-gradient / in-TDX-AdamW step. The real-Qwen **LoRA-training-with-real-TDX-loss+AdamW** loop required by Phases 5–6 is therefore a genuine BUILD on top of the existing generation protocol:
- new GPU-side training client (real Qwen masked forward+backward, collect masked LoRA grads, pack once);
- new in-TDX trusted training service (`/train/logits_loss` → CE + logit-grad in-guest; `/train/packed_update` → unmask + real AdamW + remask in-guest), attestation-bound, fail-closed;
- reuse the CPU-verified masked-LoRA-backward + trusted-AdamW math ([[permutation-nonlinear-island-verified]], `masked_lora_backward_audit`, `multilayer_lora_training`) but EXECUTED INSIDE THE TDX GUEST.

Existing reusable pieces confirmed present on both boxes: `src/pllo/protocol/{gpu_worker,remote,orchestrator,tee_gpu_messages,wire,attestation,resilient_remote,worker_timing}.py`, folded-package deployment, real TDX quote scripts. What does NOT yet exist is the training protocol above.

**Net Phase-2/5 status: NOT up.** TDX guest is real and ready (Phase 2 infra satisfied); the trusted *training* service is unwritten, and the GPU channel is throttled. No live end-to-end trusted training step was performed — and none is claimed.
