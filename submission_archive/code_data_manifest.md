# Code and Data Supplement Manifest

Status date: 2026-07-14. This is a submission-time evidence inventory, not a
promise of future release. “Ready” means the current repository contains a
script and compact output; archive packaging still requires anonymization and a
verified redistribution license. Paths are repository-relative.

## Current Main-Paper Experimental Results

### R1 — Masked linear numerical correctness

- **Paper claim/location:** C-COR-01; Correctness, finite-precision paragraph.
- **Raw source:** `outputs/intermediate/static_correctness.json`.
- **Script/command:** historical manifest names
  `scripts/run_static_correctness.py`; the script is absent from the current
  tree, so the exact command is **unsupported** and must be restored or the
  result removed.
- **Model/dataset:** synthetic $4\times16$ input and $16\times32$ linear; no
  external dataset.
- **Hardware/software:** CPU, PyTorch float64; package lower bound is recorded in
  `pyproject.toml`, but the exact historical version and CPU model are missing.
- **Seeds/repetitions:** seed 0; one generated instance.
- **Evidence type/runtime:** experimentally validated synthetic probe; expected
  runtime below one minute once the runner is restored.
- **Validation:** `allclose=true`, max error
  `1.0658141036401503e-14`; source SHA-256
  `ba59796f74af383132317e8c57414cd010e4ded16f3febf86d7446e73926393d`.
- **Licensing:** synthetic output is redistributable in principle; repository
  code license is absent, so code redistribution is not yet cleared.
- **Archive status:** blocked on command, environment, license, and path
  sanitization.

### R2 — Attention, RoPE, and GQA numerical correctness

- **Paper claim/location:** C-COR-02/03/04; Correctness theorem 2 and validation.
- **Raw source:** `outputs/intermediate/rope_gqa_probe.json`.
- **Script/command:** historical command `python scripts/run_rope_gqa_probe.py`;
  that wrapper is absent. Implementation remains under
  `src/pllo/experiments/rope_gqa_probe.py`.
- **Model/dataset:** synthetic MHA/GQA tensors; no external dataset.
- **Hardware/software:** CPU, float64, PyTorch; exact version/CPU missing.
- **Seeds/repetitions:** seed 2027; one configured probe with 128 leakage samples.
- **Evidence type/runtime:** synthetic experimentally validated result; expected
  below one minute.
- **Validation:** `all_allclose=true`; source SHA-256
  `61506a9f259df899313e2af25b87d8a4d645eeac112b761eda9eaadb6e6a94cf`.
- **Licensing:** no external data; code license unresolved.
- **Archive status:** blocked on restored runner, exact environment, license, and
  removal of non-paper leakage language if the compact schema is reduced.

### R3 — KV-cache numerical correctness and greedy generation

- **Paper claim/location:** C-COR-05/C-COR-GEN; Correctness theorem 2 and
  validation paragraph.
- **Raw sources:** `outputs/intermediate/kv_cache_correctness.json` and
  `outputs/intermediate/modern_decoder_generation_correctness.json`.
- **Script/command:** historical runners
  `scripts/run_kv_cache_correctness.py` and
  `scripts/run_modern_decoder_generation_correctness.py` are absent; exact
  commands are unsupported.
- **Model/dataset:** synthetic two-layer decoder; random prompts, no external
  dataset.
- **Hardware/software:** CPU, fp32 and fp64 PyTorch; exact versions missing.
- **Seeds/repetitions:** seeds 42 and 2026--2028; one run per configuration.
- **Evidence type/runtime:** synthetic numerical validation; expected under five
  minutes.
- **Validation:** `cache_invariant_metrics.allclose=true` and
  `greedy_token_match_rate=1.0`; SHA-256 values
  `d94c5dd2604b7c098645c0d52376e52e4a059668dd4e4c47a1584407f399c258`
  and `56d5bd746972d7fc714f885df13d0212521869e874874a55de5c4f9292fd9e4d`.
- **Licensing:** synthetic output; code license unresolved.
- **Archive status:** blocked on runners and exact environment.

### R4 — LoRA forward/backward/update correctness

- **Paper claim/location:** C-LORA-FWD/BWD; Correctness theorem 3 and validation.
- **Raw sources:** `outputs/lora/lora_training_experiments.json` and
  `outputs/lora/lora_backward_experiments.json`.
- **Scripts/commands:** `python scripts/run_lora_training_experiments.py
  --output-dir outputs/lora`; `python scripts/run_lora_backward_experiments.py
  --output-dir outputs/lora --recover-grad-x`.
- **Model/dataset:** synthetic rank-4 LoRA linear, batch 4; no external dataset.
- **Hardware/software:** CPU, PyTorch float64; exact version/CPU missing.
- **Seeds/repetitions:** seed 2026; five sequential optimizer steps; not five
  independent trials.
- **Evidence type/runtime:** synthetic experimentally validated; below one minute.
- **Validation:** per-step `allclose`/update errors; SHA-256 values
  `f40d0e8ec32c5d6428bba0ecd3c2aa730824a954dcc41e4ec145a587726e596f`
  and `902b5477d20a5aaf7b8528058fec5b6dfb253884436b0706d0720418f8dc1d2f`.
- **Licensing:** no external data; repository code license unresolved.
- **Archive status:** runnable but blocked on exact environment, license, and
  sanitizing embedded absolute output paths.

### R5 — Qwen proxy recovered-logit check

- **Paper claim/location:** C-COR-01/C-COR-GEN; final Correctness paragraph.
- **Raw source:** `outputs/_cpu_validation/audit_realprompts_none.json` (the
  $1.788\times10^{-7}$ recovered-logit result); the top-conference runner also
  emits a related mixed-precision artifact.
- **Script/command:** `bash scripts/run_topconf_experiments.sh` documents the
  ModelScope workflow, but requires a private machine cache path and emits to a
  different directory; the exact command for this copied CPU-validation artifact
  is not established.
- **Model/dataset:** Qwen2.5-0.5B-Instruct evaluation proxy; staged real-prompt
  JSONL. Qwen is not a deployment assumption.
- **Hardware/software:** artifact labeled CPU fp32 in the claim ledger; exact
  CPU, OS, PyTorch, Transformers, and ModelScope versions are missing.
- **Seeds/repetitions:** not fully recorded; no independent repetition count.
- **Evidence type/runtime:** proxy-model numerical validation; runtime unknown.
- **Validation:** recovered-logit max error `1.7881393432617188e-7`; source
  SHA-256 `2289d72cadbb663f9f49283377397662471cf4b8df4a4c2a36e544eabe21d871`.
- **Licensing:** Qwen model license and prompt redistribution require review;
  model weights must not be bundled by default.
- **Archive status:** blocked on provenance, environment, prompt/model licensing,
  and path sanitization.

## Candidate Evaluation Results Not Yet Authorized for Main-Paper Use

These rows are inventories only because `paper/sections/08_evaluation.tex` is a
TODO. Selecting them requires claim-to-result review and complete captions.

### R6 — Local-emulation runtime

- **Claim:** C-EFF-CPU. **Source:** `paper_results/csv/measured_runtime.csv`.
- **Generation command:** not captured in the CSV; likely paper-export pipeline,
  therefore unsupported until reconstructed.
- **Scope:** synthetic CPU fp64; five measured repetitions after two warmups.
- **Validation/checksum:** units and `wall_time_source=measured_local_emulation`;
  SHA-256 `a706a13257b944c8e23d49d4f9a02b7ad57fb90e79d7264950266b5f5a94dcbe`.
- **Runtime/license:** short local run; exact environment and code license absent.

### R7 — A10 plus TDX per-batch profile

- **Claim:** C-EFF-A10. **Source:**
  `results/aaai_private_base/alicloud_a10_runs/utility_dataplane/PHASE_profiling_gate.json`.
- **Script/command:** converged driver is `scripts/run_converged_sst2.py`; the
  exact profiling command and environment are not frozen in an archive-relative
  command file.
- **Model/dataset/hardware:** Qwen proxy, SST-2, NVIDIA A10 and Intel TDX; bf16,
  physical/effective batch 16.
- **Seeds/repetitions:** planned seeds 1234/2025/7; profile is a 200-batch
  characterization, not three independent training runs.
- **Evidence type:** measured per-batch profile plus projected 10.33-hour total.
- **Validation/checksum:** SHA-256
  `01fa7c2aba42189928ddb2a5d247977f97b11c5c9b06ad5abe0948cbe8c26e1b`.
- **Licensing/status:** SST-2 and Qwen terms require review; raw cloud/TDX fields
  require sanitization; not archive-ready.

## Anonymization Checklist

- [ ] Exclude `.git/`, remotes, reflogs, commit signatures, and branch metadata.
- [ ] Remove identifying repository URLs; the current Git remote is identifying.
- [ ] Replace usernames and absolute paths in JSON, Markdown, shell scripts, and
      log headers with archive-relative paths.
- [ ] Remove hostnames, SSH targets, cloud account/instance IDs, institutional
      server names, and model-cache locations.
- [ ] Remove credentials, tokens, webhooks, private keys, cookies, and shell
      history fragments; scan both text and binary metadata.
- [ ] Minimize TDX records to claim-relevant quote/appraisal fields; remove host
      inventory and network metadata.
- [ ] Review comments, generated headers, exception traces, and timestamps for
      identity or infrastructure leakage.
- [ ] Sanitize main, supplement, and checklist PDF metadata and verify no author,
      creator username, local path, link, attachment, or bookmark remains.
- [ ] Review self-citations and acknowledgments; omit acknowledgments for review.
- [ ] Verify model, dataset, and code licenses before copying any artifact.
- [ ] Record post-sanitization SHA-256 values and validate every archive-relative
      command in a clean environment.
