# Artifact misuse-resistance R0--R2

- View: **SIMULATED_PROTOCOL_VIEW**
- Overall enumerated-control pass: **False**
- Observed GPU: `NVIDIA A10` (`GPU-209b577b-b17f-22ab-eed7-4914b5b6c7a5`)
- Package root: `bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1`
- Transformed tensors initialized on GPU: 242

This supports only observed fail-closed behavior for the enumerated controls. It is not a cryptographic impossibility claim and is not a REAL_TDX_BACKED_VIEW result.

## Conditions

- R0_package_only: PASS (R0 excludes runtime and trusted/session capabilities by definition)
- R1_package_plus_runtime: PASS (session_read_before_package/model forward)
- R2_plus_local_inputs: PASS (session_read_before_package/model forward)

## Negative controls

- valid_manifest_positive_control: PASS — all pinned fields and inventory accepted
- adapter_blob_hash_positive_control: PASS — 1e29821f0b4cf4eb90bd748f171a14c2d2607b6e086b7d23bd2ad18b16e90cd2
- native_runtime_disallows_unmanifested_raw_adapter: FAIL — FAIL: copied native entrypoint contains --adapter torch.load path without adapter-package manifest validation
- wrong_base_hash: PASS — base_package_root_hash mismatch
- wrong_adapter_hash: PASS — adapter_sha256 mismatch
- wrong_target_set: PASS — target_modules mismatch
- wrong_rank: PASS — rank mismatch
- wrong_alpha: PASS — alpha mismatch
- missing_tensor: PASS — tensor inventory mismatch
- modified_tensor: PASS — per-tensor hash mismatch
- modified_manifest_digest: PASS — rank mismatch
- stale_manifest: PASS — manifest stale
- cross_run_session: PASS — session_binding mismatch
- plaintext_named_tensor: PASS — tensor inventory mismatch
