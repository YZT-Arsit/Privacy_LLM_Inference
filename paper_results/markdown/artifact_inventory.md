# Artifact Inventory

| slot | artifact_name | artifact_path | status | json_error | size_bytes | top_level_keys |
|---|---|---|---|---|---|---|
| inference | workload_profile | outputs/workload_profile.json | present |  | 55668 | calibration\|config\|interaction_breakdown\|interpretation\|limitations\|methods\|module_breakdown\|paper_metrics\|wr... |
| inference | cross_architecture_summary | outputs/cross_architecture_summary.json | present |  | 58979 | architectures\|compatible_island_integration_status\|compatible_island_projection\|config\|global_summary\|stage_note... |
| inference | nonlinear_island_experiments | outputs/nonlinear_island_experiments.json | present |  | 24024 | activation_island_cells\|config\|global_summary\|mask_family_assignments\|mlp_island_cells\|norm_island_cells\|pad_pl... |
| inference | nonlinear_island_security | outputs/nonlinear_island_security.json | present |  | 14024 | config\|global_summary\|island_linkability\|limitations\|mask_family_accounting\|permutation_recovery\|threat_model |
| inference | adaptive_island_attacks | outputs/adaptive_island_attacks.json | present |  | 20536 | comparison_with_naive_proxy\|config\|limitations\|linear_inverter\|mitigation_summary\|mlp_inverter\|permutation_reco... |
| inference | modern_decoder_probe | outputs/modern_decoder_probe.json | present |  | 13228 | architecture_spec\|both_bundles_summary\|config\|global_summary\|gqa_probe\|limitations\|model_loading\|rmsnorm_probe... |
| inference | modern_decoder_block_wrapper_smoke | outputs/modern_decoder_block_wrapper_smoke.json | present |  | 17728 | block_spec\|caveats\|config\|model_loading\|per_run\|source\|summary |
| inference | modern_decoder_model_wrapper_smoke | outputs/modern_decoder_model_wrapper_smoke.json | present |  | 39584 | block_spec\|caveats\|config\|input_ids_shape\|model_loading\|per_run\|source\|summary |
| inference | real_activation_attacks | outputs/real_activation_attacks.json | present |  | 57732 | attacker_summary\|block_spec\|bundle_comparison\|config\|limitations\|model_loading\|recommendation\|source\|target_t... |
| inference | real_token_activation_attacks | outputs/real_token_activation_attacks.json | present |  | 136397 | attacker_summary\|block_spec_summary\|bundle_comparison\|comparison_with_stage_5_5\|config\|decode_step_log\|generati... |
| inference | stronger_attackers | outputs/stronger_attackers.json | present |  | 22128 | blackbox_attacker\|comparison_with_prior_stages\|config\|constant_time_decode_summary\|inter_block_closure_summary\|i... |
| lora | lora_training_experiments | outputs/lora_training_experiments.json | present |  | 9430 | config\|gradient_and_optimizer_handling\|limitations\|lora_config_fingerprint\|lora_private_training_status\|next_sta... |
| lora | lora_security_proxy | outputs/lora_security_proxy.json | present |  | 23587 | adapter_extraction_proxy\|config\|gradient_leakage_accounting\|interpretation\|limitations\|lora_security_proxy_statu... |
| lora | lora_backward_experiments | outputs/lora_backward_experiments.json | present |  | 11538 | autograd_vs_analytic_step0\|config\|gradient_handling\|limitations\|lora_backward_status\|lora_config_fingerprint\|lo... |
| lora | lora_gradient_security_proxy | outputs/lora_gradient_security_proxy.json | present |  | 25084 | config\|gradient_extraction_proxy\|gradient_leakage_accounting\|gradient_membership_style_linkability_proxy\|interpre... |
| lora | lora_rank_padding_experiments | outputs/lora_rank_padding_experiments.json | present |  | 6860 | config\|dummy_rank_strategy\|limitations\|lora_config_fingerprint\|lora_hidden_rank_status\|lora_padded_rank_visible\... |
| lora | lora_rank_security_proxy | outputs/lora_rank_security_proxy.json | present |  | 9904 | config\|gradient_rank_inference\|interpretation\|limitations\|lora_rank_security_proxy_status\|membership_style_linka... |
| lora | multilayer_lora_training_experiments | outputs/multilayer_lora_training_experiments.json | present |  | 19518 | config\|limitations\|lora_multilayer_training_status\|model_spec\|next_stage_plan\|optimizer_summary\|per_layer_metri... |
| lora | multilayer_lora_security_proxy | outputs/multilayer_lora_security_proxy.json | present |  | 12542 | config\|cross_layer_adapter_linkage\|heterogeneous_true_rank_with_shared_padded_rank\|interpretation\|limitations\|lo... |
| lora | lora_training_timing_proxy | outputs/lora_training_timing_proxy.json | present |  | 8404 | config\|constant_time_training_proxy\|interpretation\|leakage_tasks_off\|leakage_tasks_proxy_equalized\|limitations\|... |
| lora | lora_stronger_dummy_experiments | outputs/lora_stronger_dummy_experiments.json | present |  | 37062 | config\|limitations\|lora_config_fingerprint\|lora_spectral_rank_hardening_status\|lora_stronger_dummy_status\|next_s... |
| lora | lora_stronger_dummy_security_proxy | outputs/lora_stronger_dummy_security_proxy.json | present |  | 26893 | config\|cross_layer_linkage\|dummy_strategy_classification\|gradient_rank_inference\|interpretation\|limitations\|lor... |
