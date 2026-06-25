# Real TDX minimum-deployment-evidence runbook (post-H800 phase)

The H800 phase already produced the bulk of the matrix (correctness, scaling,
public benchmarks, latency, security) — see
[`REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md`](REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md).
This runbook does **not** repeat that matrix. It produces the *minimum* real
TDX-attested deployment evidence per nonlinear design and then runs the final
gate.

Official design names (use EXACTLY these, no aliases): `current`,
`trusted_shortcut`.

Key invariant: **the runtime hash binds the nonlinear design.** You must generate
a separate runtime hash and a separate TD Quote / evidence for each design;
evidence bound for `current` MUST fail for `trusted_shortcut` (we verify this).

```
DESIGNS="current trusted_shortcut"
PKG=/root/autodl-tmp/privacy_llm_packages
OUT=outputs/tdx
MRTD=<EXPECTED_MR_TD>
# H800 folded workers (started in the H800 phase), reached over SSH tunnels:
#   current          -> 18082
#   trusted_shortcut -> 18084
```

## 1. Sync the SAME code version to the TDX guest

The runtime hash measures boundary source files, so the TDX guest must run the
identical commit as H800.

```
git -C /path/to/privacy_llm_obfuscation rev-parse HEAD     # record on H800
# on TDX: git fetch && git checkout <same-commit>; verify:
python scripts/check_tdx_measurement_coverage.py           # must print OK
```

## 2. Sync ONLY the boundary artifacts to TDX (never the folded package)

The TDX guest holds only the small trusted boundary artifact per design; the 26GB
folded package stays on the untrusted H800 worker.

```
for D in $DESIGNS; do
  rsync -av h800:$PKG/qwen7b_boundary_artifact_$D/ $PKG/qwen7b_boundary_artifact_$D/
done
```

## 3. Open SSH tunnels to the H800 workers (18082 + 18084)

Do NOT rely on `ss` (not installed on H800). Use a portable Python socket probe +
`curl /health`.

```
ssh -fNT -L 18082:127.0.0.1:18082 -L 18084:127.0.0.1:18084 h800
# portable listen check (no ss):
python - <<'PY'
import socket
for p in (18082, 18084):
    s = socket.socket(); s.settimeout(2)
    print(p, "open" if s.connect_ex(("127.0.0.1", p)) == 0 else "CLOSED")
    s.close()
PY
curl -fsS http://127.0.0.1:18082/health && echo
curl -fsS http://127.0.0.1:18084/health && echo
```

## 4. Generate the runtime hash SEPARATELY per design

The design is folded into the runtime identity, so each design has its own hash.

```
declare -A PORT=( [current]=18082 [trusted_shortcut]=18084 )
for D in $DESIGNS; do
  python scripts/write_tee_boundary_runtime_hash.py \
    --boundary-backend process --gpu-backend qwen7b_folded_package \
    --nonlinear-backend $D --expected-mr-td $MRTD \
    --output $OUT/$D/runtime_hash_$D.txt --manifest $OUT/$D/runtime_manifest_$D.json
done
# sanity: the two hashes MUST differ
diff <(cat $OUT/current/runtime_hash_current.txt) \
     <(cat $OUT/trusted_shortcut/runtime_hash_trusted_shortcut.txt) \
  && echo "ERROR: hashes identical (design not bound)" || echo "OK: per-design hashes differ"
```

## 5. Generate the TD Quote / evidence SEPARATELY per design

Bind each design's runtime hash into that quote's `report_data`.

```
for D in $DESIGNS; do
  python scripts/generate_tdx_attestation_evidence.py \
    --runtime-hash $(cat $OUT/$D/runtime_hash_$D.txt) \
    --output-json $OUT/$D/attestation_evidence_$D.json
done
```

## 6. Verify current evidence FAILS for trusted_shortcut (negative control)

Run the design-A boundary with design-B evidence and confirm the binding fails.

```
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --boundary-backend process \
  --nonlinear-backend trusted_shortcut --embedding-path $PKG/qwen7b_boundary_artifact_trusted_shortcut \
  --gpu-worker-url http://127.0.0.1:18084 \
  --attestation-evidence $OUT/current/attestation_evidence_current.json \
  --expected-mr-td $MRTD --max-new-tokens 4 \
  --output-json $OUT/negative_control_cross_design.json
# EXPECT: boundary_attested=false, runtime_hash_bound=false,
#         binding_mismatch_reason set (stale/wrong-design binding).
```

## 7. TDX-attested no-LoRA decode per design

```
for D in $DESIGNS; do
  python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
    --gpu-backend qwen7b_folded_package --boundary-backend process \
    --nonlinear-backend $D --embedding-path $PKG/qwen7b_boundary_artifact_$D \
    --gpu-worker-url http://127.0.0.1:${PORT[$D]} \
    --attestation-evidence $OUT/$D/attestation_evidence_$D.json \
    --expected-mr-td $MRTD --write-runtime-manifest $OUT/$D/runtime_manifest_$D.json \
    --max-new-tokens 16 \
    --output-json $OUT/$D/tdx_attested_decode_$D.json
done
```

Confirm for each: `boundary_attested=true`, `runtime_hash_bound=true`,
`binding_mismatch_reason=None`.

## 8. (Optional) TDX-attested E9 for MMLU, BoolQ, AG News per design

Use the converted real data, never fixtures. `--require-real` will refuse a
fixture/tiny path and will fail unless the evidence binds THIS design.

```
CONV=/root/autodl-tmp/datasets/privacy_llm_benchmarks/converted
for D in $DESIGNS; do
  for DS in mmlu boolq ag_news; do
    python scripts/run_e9_task_utility_benchmark.py --require-real \
      --dataset-jsonl $CONV/${DS}.jsonl --backend tdx_attested_remote \
      --nonlinear-backend $D --model-path $MODEL \
      --gpu-worker-url http://127.0.0.1:${PORT[$D]} \
      --embedding-path $PKG/qwen7b_boundary_artifact_$D \
      --attestation-evidence $OUT/$D/attestation_evidence_$D.json \
      --expected-mr-td $MRTD \
      --output-json $OUT/$D/e9_${DS}_tdx_attested_$D.json
  done
done
```

## 9. Pack TDX evidence and copy back to H800

```
tar -czf $OUT/tdx_evidence.tar.gz -C $OUT .
scp $OUT/tdx_evidence.tar.gz h800:outputs/tdx_evidence.tar.gz
```

## 10. Final claim validator + final submission gate (on H800, all evidence)

Tag the attested claims per design; `trusted_shortcut` may back
correctness/utility but NEVER a proven-security claim (the gate enforces this and
emits `design_B_security_not_formally_claimed` /
`trusted_shortcut_cannot_support_formal_security_claim`).

```
python scripts/validate_paper_claims.py \
  $(for f in outputs/**/*.json $OUT/**/*.json; do echo --result-json $f; done) \
  --required-claims \
"no_lora_tdx_attested_remote_package_decode[current],\
no_lora_tdx_attested_remote_package_decode[trusted_shortcut],\
public_benchmark_utility_preserved[current],\
public_benchmark_utility_preserved[trusted_shortcut]" \
  --output-json outputs/final/paper_claim_validation.json

python scripts/final_submission_gate.py \
  $(for f in outputs/**/*.json $OUT/**/*.json; do echo --result-json $f; done) \
  --nonlinear-backends current,trusted_shortcut \
  --final-artifact-tar outputs/final/final_artifacts.tar.gz \
  --output-json outputs/final/final_gate.json --output-md outputs/final/final_gate.md
```

---
**Do not:** claim proven security for `trusted_shortcut`; reuse one design's TD
Quote for the other; send the folded package or any mask secret to the TDX guest
beyond the boundary artifact; use fixture/tiny datasets for `--require-real`.
