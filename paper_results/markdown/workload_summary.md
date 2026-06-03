# Workload Summary

| method | architecture | integration_level | boundary_calls | trusted_compute | gpu_compute | preprocessing_cost | online_extra_matmul_count | measured_wall_time_ms | projected_wall_time_ms | wall_time_source | artifact_path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_hf_gpu | sshleifer/tiny-gpt2 | projected | 0 | 0 | 4434424 | 0 | 0 | 2.434716999414377 | 2.434716999414377 | measured | outputs/workload_profile.json |
| tslp_trusted_nonlinear_baseline | sshleifer/tiny-gpt2 | projected | 32 | 1110230 | 4429848 | 0 | 0 | None | 37.193893690611645 | projected_from_op_counts | outputs/workload_profile.json |
| ours_current | sshleifer/tiny-gpt2 | projected | 36 | 1116310 | 4429848 | 403784 | 0 | 6.813033198704943 | 37.38195204023441 | measured | outputs/workload_profile.json |
| ours_ideal_gpu_nonlinear | sshleifer/tiny-gpt2 | projected | 4 | 1105654 | 4434424 | 403784 | 0 | None | 36.92668586094678 | projected_from_op_counts | outputs/workload_profile.json |
| ours_compatible_nonlinear_islands | sshleifer/tiny-gpt2 | model_level_smoke | 16 | 1105830 | 4434424 | 403992 | 0 | None | 36.99217314509804 | projected_from_op_counts | outputs/workload_profile.json |
| amulet_style_reference | sshleifer/tiny-gpt2 | projected | 4 | 1105654 | 4434424 | 403784 | 0 | None | 36.92668586094678 | projected_from_op_counts | outputs/workload_profile.json |
