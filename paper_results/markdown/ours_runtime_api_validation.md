# Deployable Runtime API Validation

| component | trusted_methods_used | accelerator_methods_used | boundary_calls | transcript_sanitized | raw_secret_leaked | correctness_error | allclose | backend | tee_gpu_ready_interface | remaining_backend_work |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_pad_compensation | sample_mask,sample_pad,transform_linear,recover_output | linear | 1 | True | False | 1.0880185641326534e-14 | True | local_cpu | True | no real TEE/GPU backend implemented |
| nonlinear_island | (island mask sampled trusted-side) | activation:gelu | 2 | True | False | 0.0 | True | local_cpu | True | no real TEE/GPU backend implemented |
| modern_decoder_full_forward | (islands assembled trusted-side) | rmsnorm_core,linear,activation:silu | 4 | True | False | 0.0 | True | local_cpu | True | no real TEE/GPU backend implemented |
| modern_decoder_prefill | sample_mask,transform_linear,recover_output | linear | 1 | True | False | 1.0658141036401503e-14 | True | local_cpu | True | no real TEE/GPU backend implemented |
| modern_decoder_decode_step | sample_mask,(N^{-1} recovery) | append_kv_cache | 3 | True | False | 1.1102230246251565e-15 | True | local_cpu | True | no real TEE/GPU backend implemented |
| modern_decoder_greedy_generation | sample_mask,(N^{-1} recovery) | append_kv_cache,linear,softmax | 3 | True | False | 1.1102230246251565e-15 | True | local_cpu | True | no real TEE/GPU backend implemented |
| lora_forward | sample_mask,sample_pad,transform_linear,transform_lora_adapter,recover_output | lora_forward | 1 | True | False | 1.4210854715202004e-14 | True | local_cpu | True | no real TEE/GPU backend implemented |
| lora_backward | sample_mask,transform_lora_adapter,recover_lora_gradients | lora_backward | 1 | True | False | 3.1086244689504383e-15 | True | local_cpu | True | no real TEE/GPU backend implemented |
| rank_padding | (dummy construction trusted-side) | matmul | 0 | True | False | 0.0 | True | local_cpu | True | no real TEE/GPU backend implemented |
| multilayer_lora_training_step | sample_mask,sample_pad,transform_linear,transform_lora_adapter,recover_output | lora_forward (x2) | 2 | True | False | 1.0658141036401503e-14 | True | local_cpu | True | no real TEE/GPU backend implemented |
