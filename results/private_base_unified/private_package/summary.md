# Private-base GPU package — summary

Implementation: `src/pllo/deployment/private_base_package.py`
(`PrivateBasePackager` + `scan_package_for_plaintext` + `assert_private_base_or_abort`).
Tests: `tests/test_private_base_package.py` — 10/10.

## What the packager guarantees (fail-closed at build time)
The trusted offline packager (or TDX provisioning env) is the ONLY place plaintext
weights + mask secrets exist. The GPU package it emits contains **transformed
artifacts only**. `add_transformed_tensor` REFUSES, at write time:
- any artifact whose name is not `*_tilde` (untransformed tensor);
- any name matching a mask-secret pattern (`n_res`, `_inv`, `n_in`, `n_out`, `r_mask`,
  `s_mask`, `u_rank`, `pi`, `perm`, `d_vocab`, `inverse`, …);
- any name containing a plaintext model-tensor hint (`embed_tokens.weight`,
  `q_proj.weight`, `lm_head.weight`, …).
The manifest asserts `contains_plaintext_base_weights = contains_plaintext_embeddings
= contains_plaintext_lm_head = contains_mask_secrets = False` and records the four
mask **domains** by name only (no secrets).

## Independent plaintext-absence scan (`plaintext_absence_scan.json`)
`scan_package_for_plaintext` re-scans the finished package dir and flags: plaintext
safetensors / original shards (`model-*-of-*.safetensors`, `pytorch_model*.bin`,
`*.ckpt`), plaintext embedding/head/projection tensors, serialized masks/inverses,
model caches (`models--*`, `modelscope_cache`, `blobs/`, `snapshots/`), temp/`~`
files, and debug dumps. Tests confirm it FLAGS a planted plaintext shard and a
planted `n_res_inv.pt`, and passes a clean package.

## Hard runtime guard (`assert_private_base_or_abort`)
`private_base_required = True`: called at worker startup with the paths the GPU
process will read (model dir / cache dir / package dir). If any path looks like a
plaintext checkpoint or contains an untransformed `*.safetensors/.bin/.pth`, it
**aborts** the run. Tests confirm abort on a planted `model.safetensors` and pass on
a clean package.

## This directory (synthetic harness — NOT the real checkpoint)
The emitted `package_manifest.json` / `artifact_hashes.json` / `package_size.json` /
`plaintext_absence_scan.json` here are built from **synthetic tiny transformed
tensors** to exercise the packager + scanner on CPU. The real package is produced by
the trusted packager against the real Qwen2.5-0.5B checkpoint at run time (behind the
TDX/offline boundary), then scanned before any GPU load. `artifact_count = 17`
(synthetic), scan `clean = True`.

## Not yet done (needs the real run, gated to after this checkpoint)
- Build the real transformed package from the actual Qwen2.5-0.5B weights inside the
  trusted packager (produces the real `*_tilde` shards + masked embedding + masked
  head), then scan it and pin its hash into the attestation `report_data`.
- Wire `assert_private_base_or_abort` into the real H800 worker startup.
