# SOTA comparison: privacy-preserving LLM inference

This document explains the comparison axes used to position **this work** (folded
masked inference with a trusted nonlinear boundary) against prior families of
privacy-preserving LLM inference. The machine-readable source of truth is
[`baselines/privacy_inference_methods.yaml`](../../baselines/privacy_inference_methods.yaml);
the table below is **illustrative** and is regenerated from that file with
[`scripts/render_sota_comparison_tables.py`](../../scripts/render_sota_comparison_tables.py).

## Honesty convention

- Every value is either a citable / confidently-known fact or **`null`**.
- **`null` means "not-claimed / unknown", NOT "false".** In rendered tables
  `null` shows as `?` (Markdown/CSV) or `--` (LaTeX).
- `reported_latency` is `null` for prior work unless a specific figure can be
  cited, and is `null` for our row as well (our latency is measured and reported
  in the experiments section, not asserted here).
- `source_type` is one of `cited | reproduced | estimated | ours`. Prior-work
  rows are **category-level** representatives of real method families; the
  `notes` field carries the honest caveat for each.

## Comparison axes

- **protects_input / protects_logits / protects_kv / protects_lora** — whether
  the serving (untrusted) party is prevented from seeing the user input, the
  output logits, the KV cache, and the (private) LoRA adapter, respectively.
- **requires_gpu_tee** — whether a confidential-computing GPU (e.g. H100 CC) is
  required for the protected compute.
- **requires_mpc_fhe** — whether secure multiparty computation or homomorphic
  encryption is required.
- **tee_holds_full_model** — whether the entire model must live in plaintext
  inside an enclave. (Our design does **not** require this.)
- **runs_real_7b** — whether the method is demonstrated on a real 7B-scale model.
- **real_attestation** — whether a real hardware attestation binds the
  deployment.

## Illustrative table

> Illustrative; numbers are either cited or `null` — see
> [`baselines/privacy_inference_methods.yaml`](../../baselines/privacy_inference_methods.yaml).
> `?` = not-claimed / unknown (not false). Abridged columns for readability; the
> full table (all 15 fields) is produced by the render script.

| method | protects_input | requires_gpu_tee | requires_mpc_fhe | tee_holds_full_model | runs_real_7b | real_attestation | source_type |
|---|---|---|---|---|---|---|---|
| MPC-based secure inference (CrypTen-style) | yes | no | yes | no | ? | no | cited |
| FHE-based private inference (CKKS) | yes | no | yes | no | ? | no | cited |
| GPU-TEE full-model (H100 CC) | yes | yes | no | yes | yes | yes | cited |
| CPU-TEE shielding of full model (SGX/SEV) | yes | no | no | yes | ? | yes | cited |
| Split inference / split learning | ? | no | no | no | ? | no | cited |
| Permutation / masking obfuscation | ? | no | no | no | ? | no | cited |
| DP prompting / output perturbation | ? | no | no | no | ? | no | cited |
| Hybrid TEE + MPC | yes | no | yes | no | ? | yes | cited |
| Plaintext cloud (no protection) | no | no | no | no | yes | no | estimated |
| **Ours (folded masked + trusted nonlinear)** | yes | no | no | no | yes | yes | ours |

## Where this work sits

Our row is the only one that simultaneously claims input/logits/KV/LoRA
protection while requiring **neither a GPU TEE nor MPC/FHE** and **without
holding the full model in plaintext inside an enclave**: only a small nonlinear
boundary is trusted, and folded (masked) weights run on an untrusted GPU. The
caveats and security-status of the nonlinear designs themselves (e.g. the
`trusted_shortcut`/Amulet-migrated alternative whose security is *not formally
claimed*) are tracked in `src/pllo/experiments/nonlinear_designs.py`.
