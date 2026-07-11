# Protected-adapter deployment smoke  --  PASS

Frozen masked adapter 99e40193e0eefbba... loaded into the SAME package-native
forward (root hash bfd578b809ef2313...). Held-out GSM8K-test (4
prompts): mean CE 3.881, ppl 48.5 (SMOKE only -- 10
steps on one prompt; NOT a task-utility claim).

Assertions: {"same_transformed_package": true, "same_base_weight_view": true, "same_qkv_forward": true, "same_nonlinear_backend": true, "same_residual_domains": true, "training_adds_backward_only": true, "adapter_reencoding_required": false, "plaintext_adapter_materializations": 0, "plaintext_base_materializations": 0, "base_tensors_require_grad": false}

**DEPLOYMENT SMOKE PASS: True**
