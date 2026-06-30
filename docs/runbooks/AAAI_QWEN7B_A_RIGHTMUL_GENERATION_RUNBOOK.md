# AAAI Runbook — Qwen2.5-7B + A_rightmul generation (H800 worker + Alibaba TDX boundary)

This is the **single AAAI mainline**. The only nonlinear design is **A_rightmul**
(compatible right-multiply: every nonlinearity runs on the untrusted accelerator
over a *compatible-mask* state, with **zero** trusted-boundary crossings for the
nonlinearity). The compatibility assumption is **checked and enforced** in the
real build/worker path — it is not a metadata tag.

Scope (AAAI default experiment):

- **Model:** Qwen2.5-7B-Instruct (mainline). **GPT-2** is the compatibility sanity check only.
- **Datasets:** IFEval, GSM8K, MT-Bench.
- **Decode:** `seq_len = 1024`, `max_new_tokens = 512`, **EOS stopping ON**.
- **Topology:** H800 GPU worker (untrusted) + Alibaba Cloud **TDX** guest (trusted boundary client).
- **Attestation:** the TDX quote must be **real**; `--simulate` / off-TDX is **non-paper-facing**.

> **NOT in the AAAI default experiment:** LoRA (private fine-tuning) and the
> `amulet_secure_R` nonlinear design are **future / USENIX** work. They do not
> appear in any command below and are excluded from the AAAI default matrix.
> Legacy `current` / `trusted_shortcut` are debug baselines only and are rejected
> by every paper-facing gate.

No passwords appear in any command, script, or log: machine access uses SSH keys /
SSH aliases only.

---

## 0. Environment (both machines)

```bash
# repo + PYTHONPATH (each host)
cd <REPO_DIR>                       # e.g. /root/privacy_llm_obfuscation
export PYTHONPATH=$PWD/src:$PYTHONPATH
export D=A_rightmul                 # the ONLY AAAI nonlinear design
export MODEL=<QWEN25_7B_INSTRUCT_LOCAL_PATH>     # local checkpoint, no download
export PKG=<FOLDED_PACKAGE_DIR>     # e.g. /root/autodl-tmp/packages/qwen7b_folded_$D
export EMB=<BOUNDARY_EMBEDDING_ARTIFACT_DIR>
export OUT=outputs/aaai
export MRTD=<EXPECTED_MR_TD_HEX>    # the trusted-setup TD measurement
```

---

## 1. Trusted setup — build the A_rightmul folded package (paper-facing)

Run inside the trusted setup environment (TDX guest or attested setup). The
`--paper-facing` build **forces the compatible mask family**, verifies the REAL
generated masks, and binds the audit into the manifest. It **fails** (exit 3) on
`current`/`trusted_shortcut`, on a missing Linear-boundary pad, or on a
non-compatible mask family (`dense_orthogonal` residual / `pairwise_complex_scaling`
attention).

```bash
python scripts/build_qwen7b_folded_package.py \
  --model-path $MODEL --model-name Qwen2.5-7B-Instruct \
  --output-dir $PKG --seq-len 1024 --num-layers 28 \
  --dtype bfloat16 --device cuda \
  --nonlinear-backend $D --paper-facing \
  --mask-mode signed_permutation --attention-mask-family pairwise_rotation \
  --linear-boundary-pad --shard-by-layer true --write-manifest true \
  --created-by tdx_trusted_setup --mr-td $MRTD \
  --output-json $OUT/build_$D.json
```

The build report MUST contain (else the build failed):

```
compatible_masks_verified=True
residual_mask_is_signed_permutation=True
attention_qk_scores_preserved=True
swiglu_shared_channel_permutation=True
arbitrary_dense_mask_rejected=True
paper_facing=True   paper_facing_pad_coverage_verified=True
linear_pad_coverage = {q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head all True}
```

`raw_pad_visible_to_gpu=False`, `raw_mask_visible_to_gpu=False` — the GPU only
ever sees the composed artifacts `xpad_tilde = T·N_in` and `cpad_tilde = T·W·N_out`.

**GPT-2 compatibility sanity** (mainline is Qwen2.5-7B; GPT-2 only proves the
LayerNorm/GELU compatible-mask path on a non-Qwen architecture):

```bash
python scripts/build_qwen7b_folded_package.py \
  --model-path <GPT2_LOCAL_PATH> --model-name gpt2 \
  --output-dir <PKG_GPT2> --seq-len 1024 --num-layers <N> --dtype float32 \
  --device cuda --nonlinear-backend $D --paper-facing \
  --mask-mode signed_permutation --attention-mask-family pairwise_rotation \
  --linear-boundary-pad --output-json $OUT/build_gpt2_$D.json
```

---

## 2. TDX attestation evidence (real quote, bound to A_rightmul)

Run inside the **real** Alibaba TDX guest. The runtime hash binds the nonlinear
backend, so an A_rightmul quote cannot be replayed for another design.

```bash
python scripts/generate_tdx_attestation_evidence.py \
  --nonlinear-backend $D --expected-mr-td $MRTD \
  --quote-command '<REAL_TDX_QUOTE_CMD>' \
  --attest-command '<REAL_TDX_ATTEST_CMD>' \
  --output-dir $OUT/$D/tdx_$D \
  --output-evidence $OUT/$D/attestation_evidence_$D.json
```

Verify (no real quote ⇒ NOT paper-facing): `tee == tdx`,
`td_attributes.debug == false`, JWT has 3 parts, `report_data == runtime_hash`,
`runtime_hash_binds_nonlinear_backend == True`, `mr_td == $MRTD`. **`--simulate`
stamps `paper_facing=false` and must never be used for an AAAI result.**

---

## 3. Start the untrusted H800 GPU worker

On the H800 host (untrusted; holds only the public folded package, no masks):

```bash
python scripts/run_gpu_worker_server.py \
  --backend qwen7b_folded_package --folded-package-path $PKG \
  --nonlinear-backend $D --device cuda --dtype bfloat16 \
  --host 0.0.0.0 --port 18082
# health (from the TDX guest, over an SSH-key tunnel):
curl -s http://127.0.0.1:18082/health | python -m json.tool
```

The worker **refuses to start A_rightmul** unless the package manifest certifies
`compatible_masks_verified == True` (the build in §1). `/health` reports
`compatible_mask_family`, `compatible_masks_verified=True`, `tee_used_on_gpu=false`,
and the measured `nonlinear_execution_evidence` (zero trusted nonlinear calls).

SSH-key tunnel from the TDX guest to the H800 (no passwords):

```bash
ssh -i ~/.ssh/<KEY> -N -L 18082:127.0.0.1:18082 <H800_SSH_ALIAS>
```

---

## 4. Generation — IFEval / GSM8K / MT-Bench (paper-facing)

Run on the **trusted TDX guest** (the boundary client never loads the full 7B
weights). One command per dataset via the unified runner; `--paper-facing-generation`
enforces the full AAAI contract and exits non-zero on any unmet condition.

```bash
COMMON="--backend folded_remote --model-path $MODEL \
  --model-name Qwen2.5-7B-Instruct --gpu-worker-url http://127.0.0.1:18082 \
  --embedding-path $EMB --nonlinear-backend $D \
  --seq-len 1024 --max-new-tokens 512 --dtype bfloat16 --device cuda \
  --use-chat-template --require-real --audit \
  --tdx-boundary-client --trusted-runtime tdx_guest --tee-mode real_tdx \
  --h800-worker-ssh-alias <H800_SSH_ALIAS> \
  --attestation-evidence-json $OUT/$D/attestation_evidence_$D.json \
  --precompute-obfuscation-schedule --schedule-proof-mode online_deterministic \
  --paper-facing-generation"

# IFEval
python scripts/run_generation_benchmark.py --dataset ifeval \
  --input-jsonl <IFEVAL_JSONL> --output-dir $OUT/$D/ifeval $COMMON

# GSM8K (adds exact-match scoring over the responses)
python scripts/run_generation_benchmark.py --dataset gsm8k \
  --input-jsonl <GSM8K_JSONL> --output-dir $OUT/$D/gsm8k $COMMON

# MT-Bench (two-turn; writes a judge-ready JSONL)
python scripts/run_generation_benchmark.py --dataset mt_bench \
  --input-jsonl <MTBENCH_JSONL> --output-dir $OUT/$D/mt_bench $COMMON
```

**`--paper-facing-generation` requires (any failure ⇒ `paper_ready=False`, exit ≠ 0):**
nonlinear backend `A_rightmul`; `seq_len=1024`; `max_new_tokens=512`; EOS stop ON
(`--disable-eos-stop` is forbidden); `--require-real`; `folded_remote` +
`--tdx-boundary-client`; attestation evidence attached with
`runtime_hash_binds_nonlinear_backend=True`; worker `/health` readable;
`nonlinear_trusted_calls=0`; `compatible_masks_verified=True`;
`schedule_full_coverage_verified=True`.

> GPT-2 sanity generation: same commands with `--model-name gpt2`,
> `--model-path <GPT2>`, `--gpu-worker-url <GPT2_WORKER>`, **without**
> `--paper-facing-generation` (sanity, not an AAAI headline number).

---

## 5. End-to-end validation (fail-stop)

```bash
python scripts/validate_tee_gpu_e2e.py $OUT/$D --expected-mr-td $MRTD --require all
echo "e2e exit=$?"     # MUST be 0
```

Asserts across all collected JSON: Linear-pad coverage on all 8 families;
A_rightmul paper-facing + executed (not tag-only) + `nonlinear_trusted_calls=0` +
single TEE entry/exit + **`compatible_masks_verified=True`** (and the four mask
audit booleans); TEE boundary 1/1/0; real TDX quote bound to the nonlinear
backend; H800 worker health; TDX boundary client; remote-decode exactness. Off-TDX
/ simulated evidence is rejected.

---

## 6. Two-machine quick reference (no passwords)

| Role | Host | Command |
|---|---|---|
| Trusted setup | TDX guest | §1 build (`--paper-facing`) |
| Attestation | TDX guest | §2 real quote |
| GPU worker | H800 (untrusted) | §3 `run_gpu_worker_server.py ... --nonlinear-backend A_rightmul` |
| Boundary client | TDX guest | §4 `run_generation_benchmark.py ... --tdx-boundary-client --paper-facing-generation` |
| Validate | TDX guest | §5 `validate_tee_gpu_e2e.py ... --require all` |

All cross-host access is via SSH keys / aliases (`ssh -i ~/.ssh/<KEY>`,
`<H800_SSH_ALIAS>`); credentials are never written into scripts, args, or logs.
