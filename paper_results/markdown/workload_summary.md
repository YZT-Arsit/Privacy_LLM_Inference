# Workload Summary

| method | architecture | integration_level | boundary_calls | trusted_compute | gpu_compute | preprocessing_cost | online_extra_matmul_count | measured_wall_time_ms | projected_wall_time_ms | wall_time_source | artifact_path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_hf_gpu | sshleifer/tiny-gpt2 | projected | 0 | 0 | 4434424 | 0 | 0 | 2.8292586001043674 | 2.8292586001043674 | measured | outputs/workload_profile.json |
| tslp_trusted_nonlinear_baseline | sshleifer/tiny-gpt2 | projected | 32 | 1110230 | 4429848 | 0 | 0 | None | 42.52702272550677 | projected_from_op_counts | outputs/workload_profile.json |
| ours_current | sshleifer/tiny-gpt2 | projected | 36 | 1116310 | 4429848 | 403784 | 0 | 6.196050000289688 | 42.742128702196716 | measured | outputs/workload_profile.json |
| ours_ideal_gpu_nonlinear | sshleifer/tiny-gpt2 | projected | 4 | 1105654 | 4434424 | 403784 | 0 | None | 42.23986513554088 | projected_from_op_counts | outputs/workload_profile.json |
| ours_compatible_nonlinear_islands | sshleifer/tiny-gpt2 | model_level_smoke | 16 | 1105830 | 4434424 | 403992 | 0 | None | 42.306135377317766 | projected_from_op_counts | outputs/workload_profile.json |
| amulet_style_reference | sshleifer/tiny-gpt2 | projected | 4 | 1105654 | 4434424 | 403784 | 0 | None | 42.23986513554088 | projected_from_op_counts | outputs/workload_profile.json |
