# Correctness Summary

| stage | component | architecture | scope | metric | value | allclose | artifact_path | notes |
|---|---|---|---|---|---|---|---|---|
| 1-4 | ours_current_greedy | sshleifer/tiny-gpt2 | gpt2_model_level_greedy | measured_wall_time_ms | 6.196050000289688 | see token_match in generation_correctness | outputs/workload_profile.json | GPT-2 model-level greedy measured wall-time. |
| 5.2 | compatible_nonlinear_islands | cross_architecture | tensor_level | max_max_abs_error | see artifact | None | outputs/nonlinear_island_experiments.json |  |
| 6.4b | modern_decoder_block_wrapper | modern_decoder | block_level | max_abs_error | see artifact | None | outputs/modern_decoder_block_wrapper_smoke.json |  |
| 6.4c | modern_decoder_model_wrapper | modern_decoder | model_level_greedy | sequence_exact_match | see artifact | False | outputs/modern_decoder_model_wrapper_smoke.json |  |
| 7.0 | lora_forward | synthetic_linear | single_step | max_loss_diff | None | None | outputs/lora_training_experiments.json |  |
| 7.1 | lora_backward | synthetic_linear | single_step | max_grad_a_err | None | None | outputs/lora_backward_experiments.json |  |
| 7.1 | lora_backward | synthetic_linear | single_step | max_grad_b_err | None | None | outputs/lora_backward_experiments.json |  |
| 7.2 | rank_padded_lora_forward | synthetic_linear | single_step | max_forward_err | 2.1760371282653068e-14 | True | outputs/lora_rank_padding_experiments.json |  |
| 7.2 | rank_padded_lora_backward | synthetic_linear | single_step | max_grad_a_real_err | 1.3062467774105357e-15 | True | outputs/lora_rank_padding_experiments.json |  |
| 7.2 | rank_hiding | synthetic_linear | single_step | true_rank_hidden_from_shape | True | True | outputs/lora_rank_padding_experiments.json | true_rank hidden from shape; padded_rank still visible. |
| 7.3 | multi_layer_lora_training | synthetic_multi_layer_decoder | layers=2, modules=14 | max_loss_diff | 1.0658141036401503e-14 | True | outputs/multilayer_lora_training_experiments.json |  |
| 7.3 | multi_layer_lora_training | synthetic_multi_layer_decoder | layers=2, modules=14 | max_grad_a_real_err | 1.8446615762668372e-15 | True | outputs/multilayer_lora_training_experiments.json |  |
| 7.4 | stronger_dummy::zero_dummy | synthetic_linear | single_step | max_forward_err | 2.842170943040401e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
| 7.4 | stronger_dummy::paired_cancellation_dummy | synthetic_linear | single_step | max_forward_err | 2.4868995751603507e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
| 7.4 | stronger_dummy::gaussian_matched_dummy | synthetic_linear | single_step | max_forward_err | 2.6645352591003757e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
| 7.4 | stronger_dummy::spectrum_matched_dummy | synthetic_linear | single_step | max_forward_err | 3.019806626980426e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
| 7.4 | stronger_dummy::noise_injected_cancellation_dummy | synthetic_linear | single_step | max_forward_err | 1.9539925233402755e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
| 7.4 | stronger_dummy::orthogonalized_cancellation_dummy | synthetic_linear | single_step | max_forward_err | 2.220446049250313e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
| 7.4 | stronger_dummy::mixed_dummy_ensemble | synthetic_linear | single_step | max_forward_err | 2.1316282072803006e-14 | True | outputs/lora_stronger_dummy_experiments.json |  |
