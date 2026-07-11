# D4 one-step (real H800 + real Intel TDX)  --  PASS

run_id D4-1783795544-acf0dc3c

- Fresh attestation VERIFIED: appraisal SUCCESS, reportdata binds D4 manifest True, DEBUG=False, mr_td 8a56b29ae530d23e...
- Real TDX loss boundary: TDX CE 0.45415 vs independently-recomputed expected 0.45414 (dCE 1.4e-05); TDX dlogits vs expected re-masked 1.4e-06; HMAC True
- Effective equivalence vs HF plaintext + un-folded LoRA: top1 1.0, KL 1.1e-09
- LoRA: 168/168 targets received gradients; base tensors require_grad False; step finite True
- Forbidden plaintext counters all zero; logical trusted invocations/step = 2
- Package-native worker; NO plaintext HF forward; NO original safetensors read

**D4 PASS: True**
