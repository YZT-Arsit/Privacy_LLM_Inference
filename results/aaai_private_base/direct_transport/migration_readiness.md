# Direct-transport stack — migration readiness

The persistent direct H800↔TDX stack is **validated for correctness** on the current environment but
the AutoDL path is **unsuitable for long runs** (TDX→H800 ≈ 0.1 MB/s). This is the portable bundle to
move to a GPU+TDX environment with adequate bandwidth. **Nothing here is committed.**

## Portable components (all under `scripts/` + `results/aaai_private_base/`)
| Component | File(s) |
|---|---|
| H800 package-native worker (masked forward/backward, on CUDA) | `scripts/h800_unified_worker.py`, `scripts/h800_d4_worker.py` |
| Persistent TDX service (framed, HMAC, fail-closed, attest-once, bf16 dlogits) | `scripts/tdx_persistent_service.py` |
| Forced-command configuration (pinned service, `restrict`, no-pty/forward) | TDX `authorized_keys` entry for `h800-direct-tdx` (documented in `security_model.md`) |
| Attestation verifier (quote/JWT, reportdata bind, debug=false) | `scripts/gate0_d4_attestation.py` |
| O1-C correction service (γ-bundle → gram_inv in-enclave; q/k/v/gate/up A) | inside `tdx_persistent_service.py` (`correct` op) + `scripts/build_correction_bundle.py` |
| Momentum support (buffer sidecar, accumulates corrected grad) | `scripts/h800_direct_runner.py` (`--momentum`) |
| Persistent H800 runner (in-memory model+LoRA, one channel, no ferry) | `scripts/h800_direct_runner.py` |
| Control-plane orchestrator (setup/launch/collect only) | `scripts/gate0_direct_orchestrator.py` |
| Effective-equivalence verifier (vs plaintext HF) | `scripts/d4_trusted_verifier.py` |
| Dataset manifests / run registries | `results/aaai_private_base/full_lora_matrix/`, `empirical_matrix_closure/registry.json` |
| Fail-closed tests | `scripts/test_direct_transport_faults.py` |
| Transport benchmark (source/sink, HMAC) | `scripts/tdx_bench_service.py`, `scripts/transport_bench.py` |
| Numeric compare (direct vs ferried) | `scripts/compare_direct_vs_ferried.py` |
| GPU-execution audit | `scripts/gpu_execution_audit.py`, `scripts/gpu_audit_sample.sh` |
| **Migration preflight gate** | `scripts/migration_preflight.py` |

## Code / package hashes (bind on the new env)
- Package root hash: expected = `compute_root_hash(PKG)` == `EXPECTED_ROOT_HASH` in `h800_unified_worker.py`
  (bfd578b8… for the current gpu_package).
- Code baseline: `results/aaai_private_base/code_baseline_closure/head.txt` (`9773d9c…`) +
  `source_file_hashes.sha256`. Re-hash the migrated tree and re-bind the attestation manifest.
- Runner/service/orchestrator hashes enter the attestation `binding_manifest`
  (`worker_runtime_hash`, `tdx_service_hash`, `code_revision_or_worktree_hash`).

## Preflight (must pass before any long run)
`scripts/migration_preflight.py` gates on:
1. real GPU available;
2. real TDX quote with **DEBUG=false** (SUCCESS + reportdata-bound);
3. package + code hash match;
4. persistent direct channel (framed handshake over one SSH channel);
5. **no Mac data ferry** (`transport_profile == direct_h800_tdx`);
6. **sustained throughput ≥ 20 MB/s BOTH directions** (16 MB probe over one channel).

Exit 0 only if all pass. On the current AutoDL env, gate 6 **fails** (~0.1 MB/s ≪ 20 MB/s) →
long runs are refused. The 20 MB/s threshold makes a 12.76 MB dlogits message cost ~0.6 s instead of
~123 s, so network wait would no longer dominate the step.

## Security to re-establish on the new env
- Re-provision the forced-command key (`command="… tdx_persistent_service.py …",restrict`).
- Re-bind attestation to the new code/package hashes; verify DEBUG=false.
- HMAC key committed into `report_data` (`hmac_key_commitment`), service refuses on mismatch.
- Keep the Mac as control-plane only; never ferry per-step payloads.
- Do NOT reuse the temporary benchmark key — it was scoped, used all-zero HMAC, and has been removed.
